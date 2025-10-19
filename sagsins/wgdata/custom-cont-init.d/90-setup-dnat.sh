#!/usr/bin/env bash
set -euo pipefail

echo "[WG] DNAT init starting..."

# 0) tools (nếu image thiếu iptables)
apk add --no-cache iptables 2>/dev/null || true
apt-get update && apt-get install -y iptables 2>/dev/null || true

# 1) Đợi sagsins-core lên & có IP trong docker DNS (tối đa 60s)
for i in {1..60}; do
  CORE_IP="$(getent hosts sagsins-core | awk '{print $1}')"
  if [ -n "${CORE_IP:-}" ]; then
    echo "[WG] sagsins-core IP: $CORE_IP"
    break
  fi
  echo "[WG] waiting sagsins-core DNS... ($i)"
  sleep 1
done
[ -n "${CORE_IP:-}" ] || { echo "[ERR] cannot resolve sagsins-core"; exit 1; }

# 2) Route dải 192.168.0.0/16 về core (để tới ipB)
ip route replace 192.168.0.0/16 via "$CORE_IP" || true

# 3) Đợi file map do core sinh (tối đa 60s)
MAP_FILE="/data/runtime/link_map.sh"
for i in {1..60}; do
  if [ -s "$MAP_FILE" ]; then
    echo "[WG] found $MAP_FILE"
    break
  fi
  echo "[WG] waiting $MAP_FILE ... ($i)"
  sleep 1
done
[ -s "$MAP_FILE" ] || { echo "[ERR] $MAP_FILE not found or empty"; exit 1; }

# 4) Nạp map link_id -> ipB
declare -A LINK_IPB
add_map(){ LINK_IPB["$1"]="$2"; }
# shellcheck disable=SC1090
source "$MAP_FILE"
echo "[WG] loaded ${#LINK_IPB[@]} map entries"

# 5) Khai báo client -> link (bổ sung nếu có thêm peer)
declare -A CLIENT_LINK
CLIENT_LINK["10.10.0.2"]="LINK_AIR_GROUND_01"

# 6) Áp iptables (idempotent cho từng client)
ETH="eth0"   # outbound của container
for CIP in "${!CLIENT_LINK[@]}"; do
  LID="${CLIENT_LINK[$CIP]}"
  IPB="${LINK_IPB[$LID]:-}"
  if [ -z "$IPB" ]; then
    echo "[WARN] missing ipB for $LID (client $CIP)"; continue
  fi

  # Xóa rule cũ nếu có (tránh nhân bản), rồi thêm lại
  iptables -t nat -D PREROUTING -i wg0 -s "$CIP" -p tcp --dport 8080 -j DNAT --to-destination "$IPB":8080 2>/dev/null || true
  iptables     -D FORWARD    -s "$CIP" -d "$IPB" -p tcp --dport 8080 -j ACCEPT 2>/dev/null || true

  iptables -t nat -A PREROUTING -i wg0 -s "$CIP" -p tcp --dport 8080 -j DNAT --to-destination "$IPB":8080
  iptables     -A FORWARD    -s "$CIP" -d "$IPB" -p tcp --dport 8080 -j ACCEPT

  echo "[WG][DNAT] $CIP -> $LID ($IPB:8080)"
done

# Masquerade chiều về (nếu chưa có)
iptables -t nat -C POSTROUTING -o "$ETH" -j MASQUERADE 2>/dev/null || iptables -t nat -A POSTROUTING -o "$ETH" -j MASQUERADE

echo "[WG] DNAT ready."