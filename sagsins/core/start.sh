#!/usr/bin/env bash
set -e

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y \
  iproute2 iputils-ping python3 python3-pip net-tools curl jq iptables nftables

pip3 install pandas numpy fastapi uvicorn

# Bật IP forwarding
sysctl -w net.ipv4.ip_forward=1 || true

# ===================== 1) KHỞI TẠO TOPOLOGY =====================
export TOPO=/data/topology_data.csv        # CSV topo của bạn (đã có nhiều link đổ về GROUND_GATEWAY_01)
python3 /core/boot_topology.py "$TOPO"   # script này sẽ sinh /data/runtime/ifmap.csv

# ===================== 2) TẠO FASTAPI ECHO SERVER =====================
cat >/core/echo_api.py <<'PY'
from fastapi import FastAPI, Request
app = FastAPI()
@app.get("/")
def root():
    return {"status":"ok","service":"sagsins-echo"}
@app.post("/ingest")
async def ingest(req: Request):
    j = await req.json()
    return {"ok": True, "echo": j}
PY

# ===================== 3) CHẠY SERVER CHO CÁC LINK ĐỔ VỀ GROUND_GATEWAY_01 =====================
export SERVER_NODE="GROUND_GATEWAY_01"
export IFMAP="/data/runtime/ifmap.csv"

python3 - <<'PY'
import csv, subprocess, os
TOPO = os.environ.get("TOPO", "/data/topology_data.csv")
SERVER_NODE = os.environ["SERVER_NODE"]
IFMAP = os.environ["IFMAP"]

def sh(cmd): subprocess.run(cmd, shell=True, check=True)

# 1) lấy danh sách các link đổ về SERVER_NODE
server_links = set()
with open(TOPO, newline='') as f:
    for row in csv.DictReader(f):
        if row["destination_node"] == SERVER_NODE:
            server_links.add(row["link_id"])

# 2) đọc ifmap để tra ipB cho từng link
link2ipB = {}
with open(IFMAP, newline='') as f:
    for row in csv.DictReader(f):
        lid = row["link_id"]
        ipB = row["ipB"].split("/")[0]
        link2ipB[lid] = ipB

# 3) khởi động uvicorn và ghi map
open("/core/link_map.sh","w").close()
started = []
for lid in sorted(server_links):
    ipB = link2ipB.get(lid)
    if not ipB:
        print(f"[WARN] không thấy ipB cho {lid} trong {IFMAP}")
        continue
    # chạy uvicorn bind vào ipB:8080 (mỗi IP riêng, nên cùng port 8080 không xung đột)
    sh(f'nohup uvicorn --app-dir /core echo_api:app --host {ipB} --port 8080 >/tmp/echo_{lid}.log 2>&1 &')
    with open("/core/link_map.sh","a") as o:
        o.write(f'add_map "{lid}" "{ipB}"\n')
    started.append(lid)

print("[OK] started servers for:", ",".join(started))
PY

# Nạp map link_id -> ipB vào shell
declare -A LINK_IPB
add_map(){ LINK_IPB["$1"]="$2"; }
source /core/link_map.sh || true

# ===================== 4) TẠO DNAT CHO CÁC CLIENT (WireGuard) =====================
# Mỗi client WireGuard thật sẽ có 1 IP riêng (10.10.0.x)
# Ta map IP đó vào link tương ứng (đều đổ về GROUND_GATEWAY_01)
declare -A CLIENT_LINK
CLIENT_LINK["10.10.0.2"]="LINK_AIR_GROUND_01"       # UAV_01
CLIENT_LINK["10.10.0.3"]="LINK_SATELLITE_GROUND_13" # SATELLITE_02
CLIENT_LINK["10.10.0.4"]="LINK_SATELLITE_GROUND_14" # SATELLITE_03
CLIENT_LINK["10.10.0.5"]="LINK_UAV_GROUND_15"       # UAV_02
CLIENT_LINK["10.10.0.6"]="LINK_SHIP_GROUND_16"      # SHIP_01
CLIENT_LINK["10.10.0.7"]="LINK_SHIP_GROUND_17"      # SHIP_02
CLIENT_LINK["10.10.0.8"]="LINK_GROUND_GROUND_18"    # GROUND_GATEWAY_02
CLIENT_LINK["10.10.0.9"]="LINK_GROUND_GROUND_19"    # GROUND_GATEWAY_03

ETH0=$(ip route show default | awk '/default/ {print $5; exit}')
iptables -t nat -F PREROUTING || true
iptables -F FORWARD || true
iptables -t nat -F POSTROUTING || true

for CIP in "${!CLIENT_LINK[@]}"; do
  LID="${CLIENT_LINK[$CIP]}"
  IPB="${LINK_IPB[$LID]}"
  if [ -z "$IPB" ]; then
    echo "[WARN] Không tìm thấy IP_B cho $LID (client $CIP)"
    continue
  fi
  echo "[MAP] $CIP → $LID ($IPB)"
  iptables -t nat -A PREROUTING -i "$ETH0" -s "$CIP" -p tcp --dport 8080 \
           -j DNAT --to-destination "$IPB":8080
  iptables -A FORWARD -s "$CIP" -d "$IPB" -p tcp --dport 8080 -j ACCEPT
done

iptables -t nat -A POSTROUTING -o "$ETH0" -j MASQUERADE

# ===================== 5) CHẠY COLLECTOR GHI TRAFFIC =====================
mkdir -p /data/runtime
python3 /core/collector.py --interval 10 --outdir /data/runtime
