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
