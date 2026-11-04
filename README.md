# Presento Server

발표 영상 분석 백엔드(FastAPI + Celery). API 명세는 [`docs/api-spec.md`](docs/api-spec.md), 작업 규칙은 [`CLAUDE.md`](CLAUDE.md).

## 로컬 개발

```bash
cp .env.example .env          # 값 채우기 (.env는 커밋하지 않는다)
python3.11 -m venv .venv
.venv/bin/pip install -e ".[dev]"

.venv/bin/pytest              # 테스트 (외부 호출 없음, .env 읽지 않음)
.venv/bin/ruff check .        # 린트

.venv/bin/uvicorn app.main:create_app --factory --reload
# → http://localhost:8000/api/v1/health, Swagger: http://localhost:8000/docs
```

## Docker로 전체 실행

```bash
docker compose up --build
```

api(8000), worker, postgres(5432), redis(6379), minio(9000, 콘솔 9001)가 뜨고, `minio-init`이 `S3_BUCKET` 버킷을 만든다.

## 평가와 시연은 Docker 안에서만

분석 결과(특히 제스처·자세 같은 비언어 지표)는 실행 환경에 따라 달라진다. 같은 영상이라도 ffmpeg 버전의
색 변환과 MediaPipe 플랫폼 빌드(macOS/Linux)가 달라 랜드마크가 조금씩 다르게 나오기 때문이다
(실측: 같은 영상에서 분당 제스처 Mac 13.33 / Docker 8.57이었다. 1초 평균 속도·시작/끝 기준이
다른 히스테리시스·가려진 손목 제외로 안정화한 뒤에도 시선은 0.23 / 0.25로 조금 다르다).
평가셋 비교와 시연은 반드시 `docker compose`로 띄운 워커에서 실행하고, Mac에서 직접 돌린 결과는 점수 비교에 쓰지 않는다.

## 점수·리포트 재계산 (LLM 재호출 없음)

분석이 끝나면 회차별 중간 결과(받아쓰기, 말하기 지표, 프레임별 비언어 관측값, content·feedback
LLM 응답)를 `analysis_artifacts` 테이블에 저장한다. 점수 기준을 바꾼 뒤 OpenAI를 다시 부르지 않고
점수와 리포트를 다시 만들 수 있다(Docker 안에서 실행).

```bash
docker compose exec worker python -m app.tools.recompute a-1234 --dry-run   # 저장 없이 비교
docker compose exec worker python -m app.tools.recompute --all              # 전부 다시 저장
docker compose exec worker python -m app.tools.recompute --stats            # 저장 용량
```

- 피드백 문장은 저장된 것을 그대로 쓴다. 지표 기준이 바뀌어 피드백 후보가 달라지면
  "피드백 근거 변경됨"을 표시한다(문장 속 수치·구간이 새 지표와 다를 수 있음).
- 파이프라인 버전(`PIPELINE_VERSION`)이 다른 중간 결과는 거부한다(프레임 추출 방식이 달라 새로 분석한
  결과와 다르다). 그래도 하려면 `--allow-old-version`.
- 용량: 1분 영상 약 40KB, 10분 영상 약 360KB(추정). 사용자 대본이 들어 있으므로 회차(N6)·발표(P5)를
  지우면 함께 지워진다.

## 배포 (AWS EC2)

대상: EC2 t4g.small(arm64, Ubuntu 24.04, 메모리 2GB), 서울 리전(ap-northeast-2). 파일은 AWS S3,
HTTPS는 Caddy(Let's Encrypt 자동 발급), 도메인은 sslip.io를 쓴다. 관련 파일:

| 파일 | 내용 |
| --- | --- |
| `docker-compose.prod.yml` | api, worker(동시 실행 1), postgres, redis, caddy. 외부 포트는 80/443만 |
| `deploy/Caddyfile` | `DOMAIN`으로 HTTPS, `/api/v1`·`/docs`·`/openapi.json`만 api로 프록시 |
| `deploy/setup.sh` | Docker 설치, 스왑 4GB(`deploy/swap.sh`), 저장소 clone, `.env.prod` 생성 |
| `deploy/update.sh` | git pull → 빌드 → DB 마이그레이션 → 재시작 → 상태 확인 |
| `deploy/iam-policy.json` | EC2 역할용 S3 정책(버킷 한정 Get/Put/Delete/List) |
| `deploy/s3-cors.json` | S3 버킷 CORS(프론트 주소에서 PUT/GET) |
| `.env.prod.example` | 운영 설정 예시(비밀값은 비어 있음) |

### 1. AWS 콘솔에서 (한 번)

1. **S3 버킷**: 반드시 서울 리전(ap-northeast-2)에 만든다(다른 리전이면 presigned URL이 301로 실패).
   이름에 점(.)을 넣지 않는다(가상 호스트 주소의 HTTPS 인증서가 맞지 않음). 퍼블릭 액세스 차단은 켠 채로
   둔다(presigned URL로만 접근).
   - 권한 → CORS에 `deploy/s3-cors.json`의 `CORSRules` 배열 내용을 붙여 넣는다.
     `https://YOUR_FRONTEND_DOMAIN`은 프론트 배포 주소로 바꾸고, 아직 없으면 그 줄을 지운다.
2. **IAM 역할**: 신뢰 대상 EC2로 역할을 만들고, `deploy/iam-policy.json`의 `YOUR_BUCKET_NAME`을 버킷
   이름으로 바꿔 인라인 정책으로 붙인다.
   - `HeadObject`(업로드 확인)는 `s3:GetObject` 권한으로 허용된다. `s3:ListBucket`이 있어야 없는
     파일에 403 대신 404가 와서 "아직 업로드 안 됨"으로 구분된다.
3. **EC2**: t4g.small, Ubuntu 24.04 arm64, 디스크 20GB 이상(이미지 빌드에 여유 필요).
   - 위 IAM 역할을 인스턴스 프로필로 연결한다.
   - 메타데이터 옵션: IMDSv2 필수, **홉 제한 2**. 컨테이너 안에서 역할 자격 증명을 받으려면 2 이상이어야
     한다(1이면 S3 호출이 자격 증명 없음으로 실패).
   - 보안 그룹 인바운드: 22(내 IP만), 80, 443(TCP·UDP). 5432·6379·8000은 열지 않는다.
4. **탄력적 IP**를 만들어 인스턴스에 연결한다. 예: `3.38.12.34` → 도메인 `api.3-38-12-34.sslip.io`.

### 2. 서버에서 (처음 한 번)

```bash
ssh ubuntu@<탄력적 IP>
curl -fsSLO https://raw.githubusercontent.com/YoungHunSaGo/Presento-server/main/deploy/setup.sh
bash setup.sh                      # Docker, 스왑 4GB, ~/Presento-server clone, .env.prod 생성
                                   # (ubuntu 사용자를 docker 그룹에 넣는다 = root와 같은 권한)
exit                               # docker 그룹 반영을 위해 다시 접속
ssh ubuntu@<탄력적 IP>
cd ~/Presento-server
nano .env.prod                     # 아래 값 채우기
bash deploy/update.sh              # 빌드·마이그레이션·실행 (첫 빌드는 10분 안팎)
curl https://api.3-38-12-34.sslip.io/api/v1/health
```

`.env.prod`에서 채울 값:

- `DOMAIN`: `api.<탄력적 IP를 대시로>.sslip.io`
- `POSTGRES_PASSWORD`: `openssl rand -hex 24`(URL에 들어가므로 영문·숫자만)
- `JWT_SECRET`: `openssl rand -hex 32`
- `OPENAI_API_KEY`, `S3_BUCKET`
- `S3_ACCESS_KEY`, `S3_SECRET_KEY`: **비워 둔다.** 둘 다 비면 boto3 기본 자격 증명(EC2 역할)을 쓴다.
  하나만 채우면 서버가 시작하지 않는다.
- `CORS_ORIGINS`: `http://localhost:5173`. 프론트를 배포하면 `,https://프론트주소`를 덧붙이고
  `bash deploy/update.sh --no-pull`로 다시 띄운다(S3 CORS에도 같은 주소 추가).

### 3. 갱신

```bash
cd ~/Presento-server && bash deploy/update.sh
```

- 진행 중인 분석이 있으면 워커는 그 분석이 끝날 때까지(최대 17분) 기다렸다가 새 버전으로 바뀐다.
- 마지막에 api 상태와 S3 접근(역할·홉 제한·버킷·권한)을 확인하고, 실패하면 이유를 출력한다.

### 운영 메모

- 로그: `docker compose -f docker-compose.prod.yml --env-file .env.prod logs -f worker`
- 재계산: `docker compose -f docker-compose.prod.yml --env-file .env.prod exec worker python -m app.tools.recompute --stats`
- 메모리 2GB라 분석은 한 번에 하나씩 처리한다(나머지는 큐에서 대기). 스왑은 분석이 몰릴 때만 쓰인다.
- 인증서는 `caddy_data` 볼륨에 있다. 볼륨을 지우면 재발급되므로(발급 횟수 제한) 지우지 않는다.
- 역할 자격 증명은 몇 시간마다 바뀐다. 바뀌기 직전에 만든 presigned URL은 만료 시간보다 일찍 무효가
  될 수 있다(재생 URL이 안 열리면 N5를 다시 부르면 된다).
- DB 백업: `docker compose -f docker-compose.prod.yml --env-file .env.prod exec -T postgres pg_dump -U presento presento | gzip > backup.sql.gz`

## 집 서버 배포 (Ubuntu + Docker, Tailscale)

AWS 대신 집 서버(RTX 3090, 같은 서버에서 HowMuch·Anima도 돎)에 띄우는 방법이다. AWS용 파일
(`docker-compose.prod.yml`, `deploy/update.sh` 등)과는 별개다.

| 파일 | 내용 |
| --- | --- |
| `docker-compose.home.yml` | 프로젝트 이름 `presento-home`: api, worker, postgres, redis, minio, minio-init(버킷 생성) |
| `.env.home.example` | 설정 예시(비밀값은 비어 있음, 비밀번호는 필수) |
| `deploy/home-update.sh` | git pull → 빌드 → DB 마이그레이션 → 재시작 → `/api/v1/health` 확인 → 저장소 확인 |

- 호스트 포트는 `API_PORT`(기본 8000)·`MINIO_PORT`(기본 9000)로 바꿀 수 있고, `BIND_ADDR`(기본
  127.0.0.1) 주소에만 연다. postgres·redis는 호스트에 열지 않는다.
- 메모리 상한: api 768m, worker 1500m(동시 분석 1개), postgres 512m, redis 128m, minio 512m.
  GPU는 쓰지 않는다(MediaPipe는 CPU, 받아쓰기·LLM은 OpenAI).
- 서버 안에서는 `http://minio:9000`으로, 브라우저용 presigned URL은 `S3_PUBLIC_ENDPOINT_URL`로
  서명한다. 서명에 호스트가 들어가므로 이 값은 브라우저가 실제로 접속하는 주소여야 한다.
- MinIO CORS는 `CORS_ORIGINS`와 같은 값으로 자동 설정된다(허용한 주소에서 업로드 PUT·재생 GET 가능,
  `ETag` 헤더 노출). MinIO는 쉼표 뒤 공백을 지우지 않으므로 `a,b`처럼 붙여 쓴다.
- 앱은 MinIO 관리자 계정이 아니라 `minio-init`이 만든 전용 계정(`MINIO_APP_USER`, 이 버킷만 읽기·쓰기·삭제)을 쓴다.

### 첫 설치

```bash
# 서버에 Docker(compose 플러그인 포함)와 Tailscale이 있어야 한다
tailscale ip -4                       # 서버의 Tailscale 주소 확인. 예: 100.101.102.103
git clone https://github.com/YoungHunSaGo/Presento-server.git ~/dev/Presento-server
cd ~/dev/Presento-server
cp .env.home.example .env.home && chmod 600 .env.home
nano .env.home                        # 아래 값 채우기
bash deploy/home-update.sh --no-pull  # 빌드·마이그레이션·실행(첫 빌드는 10분 안팎)
```

`.env.home`에서 채울 값:

- `BIND_ADDR`: 서버의 Tailscale 주소(예: `100.101.102.103`). `0.0.0.0`은 쓰지 않는다(집 네트워크 전체에
  열리고, Docker가 연 포트는 UFW 방화벽 규칙을 거치지 않는다).
- `API_PORT`, `MINIO_PORT`: HowMuch·Anima와 겹치면 바꾼다(`ss -ltnp`로 사용 중인 포트 확인).
- `S3_PUBLIC_ENDPOINT_URL`: `http://<BIND_ADDR>:<MINIO_PORT>`(예: `http://100.101.102.103:9000`)
- `MINIO_ROOT_PASSWORD`, `MINIO_APP_PASSWORD`, `POSTGRES_PASSWORD`: 각각 `openssl rand -hex 24`
  (주소에 들어가므로 영문·숫자만. `POSTGRES_PASSWORD`는 첫 실행 뒤 바꾸지 않는다)
- `JWT_SECRET`: `openssl rand -hex 32`, `OPENAI_API_KEY`
- `CORS_ORIGINS`: 프론트 주소(쉼표 구분). 친구가 자기 컴퓨터에서 `npm run dev`로 띄우면
  `http://localhost:5173` 그대로 된다.

Tailscale 주소에 포트를 묶으므로, 서버가 재부팅될 때 Docker가 Tailscale보다 먼저 뜨면 포트를 못 열어
컨테이너가 시작되지 않을 수 있다(재시작 정책과 상관없이 Docker는 다시 시도하지 않는다). 처음 한 번
아래를 설정해 둔다. 서버 전체 설정이지만 다른 서비스에는 영향이 없다.

```bash
echo 'net.ipv4.ip_nonlocal_bind=1' | sudo tee /etc/sysctl.d/99-presento-bind.conf
sudo sysctl --system
```

### 갱신

```bash
cd ~/dev/Presento-server && bash deploy/home-update.sh
```

진행 중인 분석이 있으면 워커는 그 분석이 끝날 때까지(최대 17분) 기다렸다가 새 버전으로 바뀐다.
마지막에 api(`/api/v1/health`), 저장소, 워커, 호스트 포트를 확인하고 하나라도 실패하면 로그와 함께 멈춘다.
정리는 이 프로젝트 이미지만 한다(다른 서비스의 이미지·빌드 캐시는 건드리지 않음).

### 로그 확인

```bash
cd ~/dev/Presento-server
alias pc='docker compose -f docker-compose.home.yml --env-file .env.home'
pc ps
pc logs -f --tail 100 worker          # 분석 진행 상황
pc logs -f --tail 100 api
```

### 중지

```bash
pc stop        # 잠깐 멈춤(다시 시작: pc start)
pc down        # 컨테이너 삭제(데이터는 볼륨에 남음)
```

> **주의: `pc down -v`나 `docker volume rm`/`docker volume prune`은 쓰지 않는다.**
> 회원·발표·리포트(DB)와 업로드한 영상(MinIO)이 모두 지워진다. 같은 서버의 다른 서비스를
> 정리할 때도 `docker system prune --volumes`는 쓰지 않는다.

### Tailscale로 접속하기(친구)

1. 친구도 Tailscale을 설치하고, 서버 주인이 Tailscale 관리 화면에서 서버 기기를 친구에게
   공유(Share)한다(또는 같은 tailnet에 초대).
   - 공유하면 친구는 그 서버의 모든 포트에 접속할 수 있다(HowMuch·Anima 포함). Tailscale ACL에서 친구에게는
     `API_PORT`·`MINIO_PORT`(예: tcp 8000, 9000)만 허용해 둔다.
2. 친구 컴퓨터에서 확인: `curl http://100.101.102.103:8000/api/v1/health`
3. 프론트 `.env`:

   ```bash
   VITE_API_BASE_URL=http://100.101.102.103:8000/api/v1
   ```

   영상 업로드·재생은 API가 주는 presigned URL(`http://100.101.102.103:9000/...`)로 브라우저가 직접
   MinIO에 접속하므로, 친구 기기에서 MinIO 포트도 열려 있어야 한다(같은 Tailscale 주소라 따로 할 일은 없다).

### 메모: HTTPS 프론트와 mixed content

지금 API와 MinIO는 HTTP다. 친구가 프론트를 `https://`(Vercel 등)로 배포하면 브라우저가 HTTPS 페이지에서
HTTP API·영상 주소를 막는다(mixed content, Chrome은 공개 사이트 → 사설 주소 요청도 막는다). 그때는
Cloudflare Tunnel로 API와 MinIO에 HTTPS 주소를 붙이고, `VITE_API_BASE_URL`·`S3_PUBLIC_ENDPOINT_URL`·
`CORS_ORIGINS`를 그 주소로 바꾼다.

- Cloudflare 무료 요금제는 요청 본문이 100MB까지라, 100MB가 넘는 영상 업로드(PUT)는 413으로 실패한다.
  터널로 바꿀 때 `MAX_VIDEO_BYTES`를 `99000000` 정도로 낮추거나, 업로드(MinIO)만 Tailscale로 남긴다.

## 얼굴 사진 픽스처 (선택)

`tests/test_nonverbal.py`의 실제 얼굴 인식 테스트는 사진이 없으면 건너뛴다. 돌리려면:

1. 본인 얼굴 사진 2장을 JPEG로 준비한다. 상반신(어깨까지) 나오게, 긴 변 640px 정도.
   - `front.jpg`: 카메라를 정면으로 본 사진
   - `side40.jpg`: 고개를 옆으로 약 40도 돌린 사진
2. `tests/fixtures/faces/`에 넣는다. 이 폴더는 `.gitignore`에 있어 커밋되지 않는다.
3. MediaPipe 모델 경로를 주고 테스트를 돌린다.

```bash
mkdir -p tests/fixtures/faces
cp ~/Pictures/front.jpg ~/Pictures/side40.jpg tests/fixtures/faces/
MEDIAPIPE_FACE_MODEL_PATH=/path/to/face_landmarker.task \
MEDIAPIPE_POSE_MODEL_PATH=/path/to/pose_landmarker_full.task \
  .venv/bin/pytest tests/test_nonverbal.py
```
