#!/usr/bin/env bash
# 집 서버 갱신: git pull → 이미지 빌드 → DB 마이그레이션 → 재시작 → /api/v1/health 확인.
# 첫 설치도 이 스크립트로 한다. 사용: bash deploy/home-update.sh [--no-pull]
#
# AWS용 deploy/update.sh와 같은 안전장치를 쓴다(AWS 파일은 따로 유지):
# - pull 뒤 바뀐 이 스크립트로 다시 실행
# - 필수 값이 비었으면 시작 전에 중단(값은 출력하지 않음)
# - DB가 정상이 된 뒤 마이그레이션, 그다음 재시작
# - 워커는 진행 중인 분석이 끝날 때까지 기다렸다가 바뀐다(compose의 stop_grace_period 17m)
# - 상태 확인이 실패하면 로그를 보여 주고 실패로 끝난다
set -euo pipefail

SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
cd "$(dirname "$SELF")/.."

ENV_FILE=.env.home
if [[ ! -f $ENV_FILE ]]; then
  echo "$ENV_FILE이 없어요. cp .env.home.example $ENV_FILE 후 값을 채워 주세요." >&2
  exit 1
fi
chmod 600 "$ENV_FILE"

# 비어 있으면 안 되는 값. KEY="" 나 공백만 있는 것도 빈 값으로 본다.
for key in POSTGRES_PASSWORD JWT_SECRET OPENAI_API_KEY MINIO_ROOT_USER MINIO_ROOT_PASSWORD \
  MINIO_APP_USER MINIO_APP_PASSWORD S3_BUCKET S3_PUBLIC_ENDPOINT_URL CORS_ORIGINS; do
  if ! grep -Eq "^${key}=[\"']?[^[:space:]\"']" "$ENV_FILE"; then
    echo "$ENV_FILE의 ${key}가 비어 있어요." >&2
    exit 1
  fi
done

dc() { docker compose -f docker-compose.home.yml --env-file "$ENV_FILE" "$@"; }

if [[ "${1:-}" != "--no-pull" ]]; then
  echo "== git pull =="
  git pull --ff-only
  # 이 스크립트도 바뀌었을 수 있으니 새 내용으로 다시 실행한다.
  exec bash "$SELF" --no-pull
fi
git log --oneline -1

echo "== 빌드 =="
dc build api  # api와 worker는 같은 이미지(presento-server:home)

echo "== DB·저장소 준비, 마이그레이션 =="
dc up -d --wait postgres redis minio  # 정상(healthy)이 될 때까지 기다린다
dc run --rm --no-deps api alembic upgrade head

echo "== 재시작 =="
# minio-init(버킷 생성)도 여기서 실행된다. 진행 중인 분석이 있으면 워커는 끝날 때까지(최대 17분)
# 기다렸다가 바뀐다. 실패해도 아래 상태 확인에서 로그와 함께 판단한다.
dc up -d --remove-orphans || true

echo "== 상태 확인 =="
healthy=false
for _ in $(seq 1 30); do
  if dc exec -T api python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health')" 2>/dev/null; then
    healthy=true
    break
  fi
  sleep 2
done
dc ps
if [[ $healthy != true ]]; then
  echo "api가 60초 안에 정상 응답하지 않았어요. 로그:" >&2
  dc logs --tail 50 api minio-init >&2
  exit 1
fi
echo "api 정상"

echo "== 저장소 확인 =="
# MinIO 계정·버킷을 서버 안 주소로 확인한다(없는 파일 조회 → None이면 정상).
if ! dc exec -T api python -c "
from app.core.config import get_settings
from app.services.storage import S3Storage
assert S3Storage(get_settings()).head('deploy-check/none') is None
" >/dev/null 2>&1; then
  echo "MinIO에 접근하지 못했어요. MINIO_APP_USER·MINIO_APP_PASSWORD·S3_BUCKET과 minio-init 로그를 확인해 주세요." >&2
  dc logs --tail 20 minio-init >&2
  exit 1
fi
echo "저장소 정상"

echo "== 워커 확인 =="
# 워커가 계속 재시작하면 업로드는 되는데 분석이 대기열에 쌓이기만 한다.
worker_ok=false
for _ in $(seq 1 15); do
  if dc exec -T worker celery -A app.workers.celery_app inspect ping --timeout 5 >/dev/null 2>&1; then
    worker_ok=true
    break
  fi
  sleep 2
done
if [[ $worker_ok != true ]]; then
  echo "워커가 응답하지 않아요. 로그:" >&2
  dc logs --tail 50 worker >&2
  exit 1
fi
echo "워커 정상"

echo "== 호스트 포트 확인 =="
bind="$(grep -E '^BIND_ADDR=' "$ENV_FILE" | tail -1 | cut -d= -f2- || true)"
port="$(grep -E '^API_PORT=' "$ENV_FILE" | tail -1 | cut -d= -f2- || true)"
url="http://${bind:-127.0.0.1}:${port:-8000}/api/v1/health"
if command -v curl >/dev/null 2>&1 && ! curl -fsS -o /dev/null --max-time 5 "$url"; then
  echo "컨테이너는 정상인데 $url 로 접속되지 않아요. BIND_ADDR·API_PORT(다른 서비스와 겹침)를 확인해 주세요." >&2
  exit 1
fi
echo "접속 주소: $url"

# 이 프로젝트가 빌드하다 남긴 이미지만 정리한다(같은 서버의 HowMuch·Anima 이미지·빌드 캐시는 건드리지
# 않는다).
docker image prune -f --filter label=com.docker.compose.project=presento-home >/dev/null
