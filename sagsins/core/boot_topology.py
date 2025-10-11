# import sys, csv, subprocess

# CSV = sys.argv[1]

# def sh(cmd):
#     subprocess.run(cmd, shell=True, check=True)

# # Init network device name from name node
# def dev(a,b,link_id,side):
#     return f"veth-{link_id}-{side}-{a[:6]}-{b[:6]}".replace("_","-")
#     # veth-LINK_SPACE_GROUND_01-A-SATTELLITE_01-GROUND_GATEWAY_01
#     # veth-LINK_SPACE_GROUND_01-B-SATTELLITE_01-GROUND_GATEWAY_01
# with open(CSV, newline='') as f:
#     reader = csv.DictReader(f)
#     for row in reader:
#         link_id = row["link_id"] # get link id from topologies file
#         cap = int(float(row["capacity_bps"])) # get capacity_bps
#         delay_ms = float(row["base_latency_milliseconds"]) # get base latency (ms)
#         src = row["source_node"]; dst = row["destination_node"]
#         # Init veth name device for this link
#         a = dev(src,dst,link_id,"A")
#         b = dev(src,dst,link_id,"B")

#         # tạo cặp veth
#         sh(f"ip link del {a} >/dev/null 2>&1 || true")
#         sh(f"ip link del {b} >/dev/null 2>&1 || true")
#         sh(f"ip link add {a} type veth peer name {b}")
#         sh(f"ip link set {a} up"); sh(f"ip link set {b} up")

#         # shaping 2 chiều
#         mbit = max(int(cap/1e6), 1) # convert bit => megabit for capacity_bps of link
#         sh(f"tc qdisc replace dev {a} root handle 1: netem delay {delay_ms}ms")
#         sh(f"tc qdisc add dev {a} parent 1:1 handle 10: tbf rate {mbit}mbit burst 64kb latency 400ms")
#         sh(f"tc qdisc replace dev {b} root handle 1: netem delay {delay_ms}ms")
#         sh(f"tc qdisc add dev {b} parent 1:1 handle 10: tbf rate {mbit}mbit burst 64kb latency 400ms")

#         # gán IP /30 duy nhất cho mỗi link
#         base = (abs(hash(link_id)) % 200) + 10
#         ipA = f"192.168.{base}.1/30"
#         ipB = f"192.168.{base}.2/30"
#         sh(f"ip addr add {ipA} dev {a} || true")
#         sh(f"ip addr add {ipB} dev {b} || true")

#         print(f"[OK] {link_id}: {a} <-> {b}  delay={delay_ms}ms rate={mbit}mbit  IPs={ipA},{ipB}")

#!/usr/bin/env python3
import sys, csv, subprocess, hashlib, os

CSV = sys.argv[1]
OUTMAP = "/data/runtime/ifmap.csv"

def sh(cmd):
    subprocess.run(cmd, shell=True, check=True)

def short_name(prefix, link_id, side):
    # vA-<8hex> hoặc vB-<8hex> (<= 12 ký tự)
    h = hashlib.sha1(f"{link_id}-{side}".encode()).hexdigest()[:8]
    return f"{prefix}{h}"

os.makedirs(os.path.dirname(OUTMAP), exist_ok=True)
rows_map = []

with open(CSV, newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        lid = row["link_id"]
        cap = int(float(row["capacity_bps"]))
        delay_ms = float(row["base_latency_milliseconds"])

        a = short_name("vA-", lid, "A")
        b = short_name("vB-", lid, "B")

        # dọn cũ nếu tồn tại
        sh(f"ip link del {a} >/dev/null 2>&1 || true")
        sh(f"ip link del {b} >/dev/null 2>&1 || true")

        # tạo cặp veth
        sh(f"ip link add {a} type veth peer name {b}")
        sh(f"ip link set {a} up")
        sh(f"ip link set {b} up")

        # shaping 2 chiều (netem + tbf)
        mbit = max(int(cap/1e6), 1)
        sh(f"tc qdisc replace dev {a} root handle 1: netem delay {delay_ms}ms")
        sh(f"tc qdisc add dev {a} parent 1:1 handle 10: tbf rate {mbit}mbit burst 64kb latency 400ms")
        sh(f"tc qdisc replace dev {b} root handle 1: netem delay {delay_ms}ms")
        sh(f"tc qdisc add dev {b} parent 1:1 handle 10: tbf rate {mbit}mbit burst 64kb latency 400ms")

        # gán IP /30 ổn định theo hash(lid)
        base = (int(hashlib.sha1(lid.encode()).hexdigest(), 16) % 200) + 10
        ipA = f"192.168.{base}.1/30"
        ipB = f"192.168.{base}.2/30"
        sh(f"ip addr add {ipA} dev {a} || true")
        sh(f"ip addr add {ipB} dev {b} || true")

        rows_map.append((lid, a, b, ipA, ipB))
        print(f"[OK] {lid}: {a}<->{b} delay={delay_ms}ms rate={mbit}mbit IPs={ipA},{ipB}")

# ghi bảng ánh xạ để start.sh tra cứu
with open(OUTMAP, "w") as o:
    o.write("link_id,ifA,ifB,ipA,ipB\n")
    for lid,a,b,ipA,ipB in rows_map:
        o.write(f"{lid},{a},{b},{ipA},{ipB}\n")
print(f"[MAP] written {OUTMAP}")
