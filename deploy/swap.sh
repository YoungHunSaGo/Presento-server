#!/usr/bin/env bash
# 스왑 파일을 만들고 부팅 때마다 켜지게 한다(메모리 2GB 인스턴스용). 여러 번 실행해도 안전하다.
# 사용: sudo bash deploy/swap.sh [크기, 기본 4G]
set -euo pipefail

SIZE="${1:-4G}"
SWAPFILE=/swapfile

if [[ $EUID -ne 0 ]]; then
  echo "sudo로 실행해 주세요" >&2
  exit 1
fi

if swapon --show=NAME --noheadings | grep -qx "$SWAPFILE"; then
  echo "스왑이 이미 켜져 있어요: $(swapon --show=NAME,SIZE --noheadings | tr -s ' ')"
else
  # 전에 중간에 멈춰 남은 파일이 있을 수 있어 매번 새로 만든다(켜져 있지 않은 파일만).
  rm -f "$SWAPFILE"
  fallocate -l "$SIZE" "$SWAPFILE" || dd if=/dev/zero of="$SWAPFILE" bs=1M count=$(( $(numfmt --from=iec "$SIZE") / 1048576 ))
  chmod 600 "$SWAPFILE"
  mkswap "$SWAPFILE"
  swapon "$SWAPFILE"
fi

grep -q "^$SWAPFILE " /etc/fstab || echo "$SWAPFILE none swap sw 0 0" >> /etc/fstab

# 메모리가 남아 있으면 되도록 스왑을 쓰지 않는다(분석이 몰릴 때만 넘치게).
cat > /etc/sysctl.d/99-presento-swap.conf <<'CONF'
vm.swappiness=10
vm.vfs_cache_pressure=50
CONF
sysctl -q --system

free -h
