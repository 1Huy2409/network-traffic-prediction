set -euo pipefail

echo "[CORE] Updating base packages..."
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends \
  iproute2 iputils-ping python3 python3-pip net-tools curl jq iptables nftables ca-certificates
rm -rf /var/lib/apt/lists/*

echo "[CORE] Installing Python deps (fastapi/uvicorn/pandas/numpy)..."
pip3 install --no-cache-dir fastapi uvicorn pandas numpy >/dev/null

echo "[CORE] Enable IPv4 forwarding..."
sysctl -w net.ipv4.ip_forward=1 || true

# ========== Paths ==========
mkdir -p /data/runtime
TOPO=${TOPO:-/data/topology_data.csv}
IFMAP=/data/runtime/ifmap.csv
LINK_MAP=/data/runtime/link_map.sh
IPB_LIST=/data/runtime/ipb.list

# ========== 1) Build topology (ifmap.csv) ==========
echo "[CORE] Building topology from ${TOPO} ..."
python3 /core/boot_topology.py "$TOPO"

if [ ! -s "$IFMAP" ]; then
  echo "[ERR] Missing $IFMAP after boot_topology.py"
  exit 1
fi
echo "[CORE] ifmap.csv generated at $IFMAP"

# ========== 2) Generate link_map.sh (link_id -> ipB) ==========
echo "[CORE] Generating $LINK_MAP ..."
: > "$LINK_MAP"
python3 - <<'PY'
import pandas as pd
df = pd.read_csv('/data/runtime/ifmap.csv')
out = '/data/runtime/link_map.sh'
with open(out, 'w') as f:
    # yêu cầu cột 'link_id' và 'ipB' trong ifmap.csv
    for _, r in df.iterrows():
        lid = str(r['link_id']).strip()
        ipb = str(r['ipB']).split('/')[0].strip()
        if lid and ipb:
            f.write(f'add_map "{lid}" "{ipb}"\n')
print("[CORE] Wrote", out)
PY
chmod +x "$LINK_MAP"

# ========== 3) Extract ipB list for uvicorn bind ==========
echo "[CORE] Extracting ipB list to $IPB_LIST ..."
python3 - <<'PY'
import pandas as pd
df = pd.read_csv('/data/runtime/ifmap.csv')
ips = []
for ipb in df['ipB']:
    ip = str(ipb).split('/')[0].strip()
    if ip:
        ips.append(ip)
with open('/data/runtime/ipb.list', 'w') as f:
    for ip in ips:
        f.write(ip + '\n')
print("[CORE] Wrote /data/runtime/ipb.list with", len(ips), "IPs")
PY

# ========== 4) Start FastAPI (bind trên từng ipB:8080) ==========
echo "[CORE] Starting FastAPI echo_api:app on each ipB:8080 ..."
cd /core
# Nếu bạn đã có echo_api.py, dùng luôn; nếu file khác tên thì sửa lại ở dưới.
while read -r ip; do
  [ -z "$ip" ] && continue
  echo "  -> uvicorn echo_api:app --host $ip --port 8080"
  # mỗi IP một process; 1 worker là đủ cho demo/echo
  uvicorn echo_api:app --host "$ip" --port 8080 --workers 1 --log-level warning &
done < "$IPB_LIST"

sleep 1
echo "[CORE] Listening sockets:"
ss -lntp | grep 8080 || true

# ========== 5) Done ==========
echo "[CORE] Ready. (tail to keep container alive)"
tail -f /dev/null
