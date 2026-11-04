#!/usr/bin/env bash
# 최신 코드로 갱신: git pull → 이미지 빌드 → DB 마이그레이션 → 재시작 → 상태 확인.
# 첫 배포도 이 스크립트로 한다. 사용: bash deploy/update.sh [--no-pull]
set -euo pipefail

SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
cd "$(dirname "$SELF")/.."

if [[ ! -f .env.prod ]]; then
  echo ".env.prod가 없어요. cp .env.prod.example .env.prod 후 값을 채워 주세요." >&2
  exit 1
fi
chmod 600 .env.prod

# 비어 있으면 안 되는 값(값 자체는 출력하지 않는다). KEY="" 나 공백만 있는 것도 빈 값으로 본다.
for key in DOMAIN POSTGRES_PASSWORD JWT_SECRET OPENAI_API_KEY S3_BUCKET; do
  if ! grep -Eq "^${key}=[\"']?[^[:space:]\"']" .env.prod; then
    echo ".env.prod의 ${key}가 비어 있어요." >&2
    exit 1
  fi
done

dc() { docker compose -f docker-compose.prod.yml --env-file .env.prod "$@"; }

if [[ "${1:-}" != "--no-pull" ]]; then
  echo "== git pull =="
  git pull --ff-only
  # 이 스크립트도 바뀌었을 수 있으니 새 내용으로 다시 실행한다.
  exec bash "$SELF" --no-pull
fi
git log --oneline -1

echo "== 빌드 =="
dc build api  # api와 worker는 같은 이미지(presento-server:prod)

echo "== DB 준비·마이그레이션 =="
dc up -d --wait postgres redis  # 정상(healthy)이 될 때까지 기다린다
dc run --rm --no-deps api alembic upgrade head

echo "== 재시작 =="
# 진행 중인 분석이 있으면 워커는 끝날 때까지(최대 17분) 기다렸다가 바뀐다.
# api가 정상이 안 되면 caddy 의존성 때문에 실패하므로, 판단은 아래 상태 확인에 맡긴다.
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
  dc logs --tail 50 api >&2
  exit 1
fi
echo "api 정상"

echo "== S3 접근 확인 =="
# 역할 자격 증명(IMDS 홉 제한), 버킷 이름·리전, ListBucket 권한을 한 번에 확인한다.
# 없는 파일을 조회해 None(404)이 나오면 정상. 권한·자격 증명 문제면 오류로 끝난다.
if ! dc exec -T api python -c "
from app.core.config import get_settings
from app.services.storage import S3Storage
assert S3Storage(get_settings()).head('deploy-check/none') is None
" >/dev/null 2>&1; then
  echo "S3에 접근하지 못했어요. IAM 역할 연결, 메타데이터 홉 제한 2, S3_BUCKET·S3_REGION을 확인해 주세요." >&2
  echo "(역할을 방금 연결했다면 api·worker를 다시 시작: bash deploy/update.sh --no-pull)" >&2
  exit 1
fi
echo "S3 정상"
domain="$(grep -E '^DOMAIN=' .env.prod | cut -d= -f2-)"
echo "확인: curl https://${domain}/api/v1/health  (첫 실행은 인증서 발급에 몇십 초 걸릴 수 있어요)"

# 이전 빌드에서 남은 이미지 정리(디스크가 작다)
docker image prune -f >/dev/null
docker builder prune -f --filter until=168h >/dev/null
