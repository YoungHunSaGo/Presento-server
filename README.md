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
(실측: 같은 영상에서 분당 제스처 Mac 13.33 / Docker 8.57). 평가셋 비교와 시연은 반드시
`docker compose`로 띄운 워커에서 실행하고, Mac에서 직접 돌린 결과는 점수 비교에 쓰지 않는다.

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
- 용량: 1분 영상 약 40KB, 10분 영상 약 360KB(추정). 사용자 대본이 들어 있으므로 회차(N6)·발표(P5)를
  지우면 함께 지워진다.

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
