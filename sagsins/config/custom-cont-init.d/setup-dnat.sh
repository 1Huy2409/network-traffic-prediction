#!/usr/bin/env bash
set -euo pipefail

# 0) iptables tool (tuỳ base image alpine/debian)
apk add --no-cache iptables 2>/dev/null || true
apt-get update && apt-get install -y iptables 2>/dev/null || true || true

echo "[WG] Resolving sagsins-core IP..."
CORE_IP="$(getent hosts sagsins-core | awk '{print $1}' || true)"
if [ -z "${CORE_IP:-}" ]; then
  echo "[ERR] Cannot resolve sagsins-core. Are containers on the same network?"
  exit 1
fi
echo "[WG] sagsins-core IP: $CORE_IP"

# 1) Route dải 192.168.0.0/16 (các link /30) về core
ip route replace 192.168.0.0/16 via "$CORE_IP" || true

# 2) Nạp map link_id -> ipB do core sinh
declare -A LINK_IPB
add_map(){ LINK_IPB["$1"]="$2"; }  # hàm được dùng bởi file map

MAP_FILE="/data/runtime/link_map.sh"
if [ ! -f "$MAP_FILE" ]; then
  echo "[ERR] $MAP_FILE not found. Ensure sagsins-core has generated it."
  exit 1
fi

# shellcheck disable=SC1090
source "$MAP_FILE"
echo "[WG] Loaded $(wc -l < "$MAP_FILE") link map entries."

# 3) Map IP client (WG) -> link_id (giữ như bạn đang dùng)
declare -A CLIENT_LINK
CLIENT_LINK["10.10.0.2"]="LINK_AIR_GROUND_01"
CLIENT_LINK["10.10.0.3"]="LINK_SATELLITE_GROUND_13"
CLIENT_LINK["10.10.0.4"]="LINK_SATELLITE_GROUND_14"
CLIENT_LINK["10.10.0.5"]="LINK_UAV_GROUND_15"
CLIENT_LINK["10.10.0.6"]="LINK_SHIP_GROUND_16"
CLIENT_LINK["10.10.0.7"]="LINK_SHIP_GROUND_17"
CLIENT_LINK["10.10.0.8"]="LINK_GROUND_GROUND_18"
CLIENT_LINK["10.10.0.9"]="LINK_GROUND_GROUND_19"

# 4) Apply iptables DNAT on wg0
ETH0="eth0"
echo "[WG] Flushing old NAT rules..."
iptables -t nat -F PREROUTING || true
iptables -F FORWARD || true
iptables -t nat -F POSTROUTING || true

for CIP in "${!CLIENT_LINK[@]}"; do
  LID="${CLIENT_LINK[$CIP]}"
  IPB="${LINK_IPB[$LID]:-}"
  if [ -z "$IPB" ]; then
    echo "[WARN] Missing ipB for $LID (client $CIP). Skipping."
    continue
  fi
  echo "[WG][DNAT] $CIP -> $LID ($IPB:8080)"
  iptables -t nat -A PREROUTING -i wg0 -s "$CIP" -p tcp --dport 8080 \
           -j DNAT --to-destination "$IPB":8080
  iptables -A FORWARD -s "$CIP" -d "$IPB" -p tcp --dport 8080 -j ACCEPT
done

# 5) Masquerade chiều về để conntrack trả đúng client
iptables -t nat -A POSTROUTING -o "$ETH0" -j MASQUERADE

echo "[WG] DNAT ready."
