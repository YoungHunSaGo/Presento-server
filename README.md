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
