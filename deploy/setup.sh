#!/usr/bin/env bash
# 새 EC2(Ubuntu 24.04, arm64)에서 한 번 실행: Docker 설치, 스왑 4GB, 저장소 clone, .env.prod 준비.
# 사용: curl로 받거나 복사해서  bash setup.sh [저장소 URL] [설치 경로]
# 끝나면 .env.prod 값을 채우고 deploy/update.sh를 실행한다(README "배포" 참고).
set -euo pipefail

REPO_URL="${1:-https://github.com/YoungHunSaGo/Presento-server.git}"
APP_DIR="${2:-$HOME/Presento-server}"

if [[ $EUID -eq 0 ]]; then
  echo "root가 아닌 사용자(ubuntu)로 실행해 주세요. 필요한 곳만 sudo를 씁니다." >&2
  exit 1
fi

echo "== 1/4 Docker 설치 =="
if ! command -v docker >/dev/null 2>&1; then
  # 첫 부팅 때는 자동 업데이트가 dpkg 잠금을 잡고 있어 최대 10분 기다린다.
  apt=(sudo apt-get -o DPkg::Lock::Timeout=600)
  "${apt[@]}" update -y
  "${apt[@]}" install -y ca-certificates curl git
  sudo install -m 0755 -d /etc/apt/keyrings
  sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
  sudo chmod a+r /etc/apt/keyrings/docker.asc
  # shellcheck disable=SC1091
  codename="$(. /etc/os-release && echo "$VERSION_CODENAME")"
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $codename stable" \
    | sudo tee /etc/apt/sources.list.d/docker.list >/dev/null
  "${apt[@]}" update -y
  "${apt[@]}" install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
  sudo systemctl enable --now docker
else
  echo "Docker가 이미 있어요: $(docker --version)"
fi
# docker 그룹은 root와 같은 권한이다(이 서버 전용 사용자에게만 준다).
sudo usermod -aG docker "$USER"

echo "== 2/4 스왑 =="
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$SCRIPT_DIR/swap.sh" ]]; then
  sudo bash "$SCRIPT_DIR/swap.sh" 4G
fi

echo "== 3/4 저장소 =="
if [[ -d "$APP_DIR/.git" ]]; then
  echo "이미 있어요: $APP_DIR"
else
  git clone "$REPO_URL" "$APP_DIR"
fi
# 저장소를 받기 전에 이 스크립트만 따로 실행한 경우 스왑을 여기서 만든다.
if ! swapon --show=NAME --noheadings | grep -q .; then
  sudo bash "$APP_DIR/deploy/swap.sh" 4G
fi

echo "== 4/4 .env.prod =="
cd "$APP_DIR"
if [[ ! -f .env.prod ]]; then
  cp .env.prod.example .env.prod
  echo ".env.prod를 만들었어요. 비어 있는 값을 채워 주세요."
fi
chmod 600 .env.prod

cat <<MSG

설치가 끝났어요. 다음 순서:
  1) docker 그룹 반영을 위해 로그아웃 후 다시 접속(또는 newgrp docker)
  2) nano $APP_DIR/.env.prod  # DOMAIN, POSTGRES_PASSWORD, JWT_SECRET, OPENAI_API_KEY, S3_BUCKET, CORS_ORIGINS
  3) bash $APP_DIR/deploy/update.sh
MSG
