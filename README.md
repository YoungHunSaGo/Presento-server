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
