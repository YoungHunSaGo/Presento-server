#!/usr/bin/env bash
# 집 서버 갱신: git pull → 이미지 빌드 → DB 마이그레이션 → 재시작 → /api/v1/health 확인.
# 첫 설치도 이 스크립트로 한다. 사용: bash deploy/home-update.sh [--no-pull]
#
# AWS용 deploy/update.sh와 같은 안전장치를 쓴다(AWS 파일은 따로 유지):
# - pull 뒤 바뀐 이 스크립트로 다시 실행
# - 필수 값이 비었거나 위험한 값(BIND_ADDR=0.0.0.0 등)이면 시작 전에 중단(비밀값은 출력하지 않음)
# - DB·MinIO·초기화(minio-init)가 정상이 된 뒤 마이그레이션, 그다음 재시작
# - 워커는 진행 중인 분석이 끝날 때까지 기다렸다가 바뀐다(compose의 stop_grace_period 17m)
# - 상태 확인이 실패하면 로그를 보여 주고 실패로 끝난다
set -euo pipefail

if [[ $# -gt 1 || ( $# -eq 1 && "$1" != "--no-pull" ) ]]; then
  echo "사용법: bash deploy/home-update.sh [--no-pull]" >&2
  exit 2
fi

SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
cd "$(dirname "$SELF")/.."

ENV_FILE=.env.home
if [[ ! -f $ENV_FILE ]]; then
  echo "${ENV_FILE}이 없어요. cp .env.home.example ${ENV_FILE} 후 값을 채워 주세요." >&2
  exit 1
fi
chmod 600 "$ENV_FILE"

# .env.home에서 마지막 KEY=... 줄의 값을 꺼낸다(앞뒤 공백·따옴표·인라인 주석 제거).
env_val() {
  local raw
  raw="$(grep -E "^${1}=" "$ENV_FILE" | tail -1 | cut -d= -f2- || true)"
  raw="${raw#"${raw%%[![:space:]]*}"}"
  if [[ "$raw" =~ ^\"([^\"]*)\"[[:space:]]*(#.*)?$ ]]; then
    printf '%s' "${BASH_REMATCH[1]}"
  elif [[ "$raw" =~ ^\'([^\']*)\'[[:space:]]*(#.*)?$ ]]; then
    printf '%s' "${BASH_REMATCH[1]}"
  else
    raw="${raw%%#*}"
    raw="${raw%"${raw##*[![:space:]]}"}"
    printf '%s' "$raw"
  fi
}

# 비어 있으면 안 되는 값. KEY="" 나 공백·주석만 있는 것도 빈 값으로 본다.
for key in POSTGRES_PASSWORD JWT_SECRET OPENAI_API_KEY MINIO_ROOT_USER MINIO_ROOT_PASSWORD \
  MINIO_APP_USER MINIO_APP_PASSWORD S3_BUCKET S3_PUBLIC_ENDPOINT_URL CORS_ORIGINS; do
  if [[ -z "$(env_val "$key")" ]]; then
    echo "${ENV_FILE}의 ${key}가 비어 있어요." >&2
    exit 1
  fi
done

bind="$(env_val BIND_ADDR)"
bind="${bind:-127.0.0.1}"
if [[ "$bind" == "0.0.0.0" || "$bind" == "::" || "$bind" == "[::]" || "$bind" == "*" ]]; then
  echo "${ENV_FILE}의 BIND_ADDR에 ${bind}은 사용할 수 없어요(외부 노출 위험). 127.0.0.1 또는 Tailscale 주소를 넣어 주세요." >&2
  exit 1
fi

jwt_secret="$(env_val JWT_SECRET)"
if [[ ${#jwt_secret} -lt 32 ]]; then
  echo "${ENV_FILE}의 JWT_SECRET은 32자 이상이어야 해요." >&2
  exit 1
fi

for url_key in POSTGRES_PASSWORD MINIO_ROOT_USER MINIO_ROOT_PASSWORD; do
  if [[ ! "$(env_val "$url_key")" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "${ENV_FILE}의 ${url_key}에는 영문·숫자만 써 주세요(URL에 직접 들어가는 값이에요)." >&2
    exit 1
  fi
done

s3_bucket="$(env_val S3_BUCKET)"
if [[ ! "$s3_bucket" =~ ^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$ ]]; then
  echo "${ENV_FILE}의 S3_BUCKET 형식이 올바르지 않아요(소문자·숫자·하이픈, 3~63자)." >&2
  exit 1
fi

retention_days="$(env_val VIDEO_RETENTION_DAYS)"
retention_days="${retention_days:-7}"
if [[ ! "$retention_days" =~ ^[1-9][0-9]*$ ]]; then
  echo "${ENV_FILE}의 VIDEO_RETENTION_DAYS는 1 이상의 정수여야 해요." >&2
  exit 1
fi

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
dc run --rm minio-init                # 버킷과 앱 전용 MinIO 계정 준비
dc run --rm --no-deps api alembic upgrade head

echo "== 재시작 =="
# 진행 중인 분석이 있으면 워커는 끝날 때까지(최대 17분) 기다렸다가 바뀐다.
dc up -d --remove-orphans

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
http_check() {
  local target="$1"
  if command -v curl >/dev/null 2>&1; then
    curl -fsS -o /dev/null --max-time 5 "$target"
  else
    python3 -c "import sys, urllib.request; urllib.request.urlopen(sys.argv[1], timeout=5)" "$target" >/dev/null 2>&1
  fi
}

port="$(env_val API_PORT)"
url="http://${bind}:${port:-8000}/api/v1/health"
if ! http_check "$url"; then
  echo "컨테이너는 정상인데 $url 로 접속되지 않아요. BIND_ADDR·API_PORT(다른 서비스와 겹침)를 확인해 주세요." >&2
  exit 1
fi

public_s3="$(env_val S3_PUBLIC_ENDPOINT_URL)"
minio_url="${public_s3%/}/minio/health/live"
if ! http_check "$minio_url"; then
  echo "브라우저용 저장소 주소($minio_url)로 접속되지 않아요. S3_PUBLIC_ENDPOINT_URL·BIND_ADDR·MINIO_PORT를 확인해 주세요." >&2
  exit 1
fi
echo "접속 주소: $url"

# 이 프로젝트가 빌드하다 남긴 이미지만 정리한다(같은 서버의 HowMuch·Anima 이미지·빌드 캐시는 건드리지
# 않는다).
docker image prune -f --filter label=com.docker.compose.project=presento-home >/dev/null
