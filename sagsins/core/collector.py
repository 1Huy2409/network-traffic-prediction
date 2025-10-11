import argparse, os, time, math, pandas as pd, datetime as dt

ap = argparse.ArgumentParser()
ap.add_argument("--interval", type=int, default=10)
ap.add_argument("--outdir", type=str, default="/data/runtime")
args = ap.parse_args()

os.makedirs(args.outdir, exist_ok=True)

def parse_ifaces():
    m = {}
    for name in os.listdir("/sys/class/net"):
        if name.startswith("veth-"):
            parts = name.split("-")
            link_id = "-".join(parts[1:-3])
            m.setdefault(link_id, []).append(name)
    return m

def read_bytes(dev):
    base = f"/sys/class/net/{dev}/statistics"
    with open(f"{base}/rx_bytes") as f: rx=int(f.read())
    with open(f"{base}/tx_bytes") as f: tx=int(f.read())
    return rx, tx

def load_topology(csv_path="/data/topology_data.csv"):
    df = pd.read_csv(csv_path)
    df = df[["link_id","source_layer","destination_layer","capacity_bps"]].drop_duplicates("link_id")
    return { r.link_id: dict(source_layer=r.source_layer, destination_layer=r.destination_layer, capacity_bps=r.capacity_bps)
             for _,r in df.iterrows() }

META = load_topology()
IFMAP = parse_ifaces()
STATE = {}
BUFF = {lid: [] for lid in IFMAP.keys()}

print("[collector] links:", list(IFMAP.keys()))

def features(ts_local):
    hour = ts_local.hour
    dow  = ts_local.weekday()
    hour_sin = math.sin(2*math.pi*hour/24)
    hour_cos = math.cos(2*math.pi*hour/24)
    day_sin  = math.sin(2*math.pi*dow/7)
    day_cos  = math.cos(2*math.pi*dow/7)
    return hour, dow, int(dow>=5), hour_sin, hour_cos, day_sin, day_cos

while True:
    ts = dt.datetime.now().replace(tzinfo=dt.timezone.utc)
    ts_local = ts.astimezone(dt.timezone(dt.timedelta(hours=7)))
    for lid, devs in IFMAP.items():
        try:
            total_bps = 0
            for d in devs:
                rx, tx = read_bytes(d)
                key = f"{lid}:{d}"
                if key in STATE:
                    drx = rx - STATE[key][0]
                    dtx = tx - STATE[key][1]
                    total_bps += (drx + dtx) * 8 / args.interval
                STATE[key] = (rx, tx)
            cap = float(META.get(lid,{}).get("capacity_bps", 1.0))
            util = total_bps / cap if cap>0 else 0.0
            hour, dow, is_wk, hsin, hcos, dsin, dcos = features(ts_local)
            row = {
              "timestamp": ts.isoformat(),
              "bitrate_bps": total_bps,
              "rtt_milliseconds": None,
              "loss_rate": None,
              "jitter_milliseconds": None,
              "capacity_bps": cap,
              "source_layer": META[lid]["source_layer"],
              "destination_layer": META[lid]["destination_layer"],
              "link_id": lid,
              "hour": hour, "day_of_week": dow, "is_weekend": is_wk,
              "hour_sin": hsin, "hour_cos": hcos, "day_sin": dsin, "day_cos": dcos,
              "utilization": util
            }
            BUFF[lid].append(row)
            if len(BUFF[lid]) % 12 == 0:
                pd.DataFrame(BUFF[lid][-360:]).to_csv(f"{args.outdir}/{lid}_traffic_processed.csv", index=False)
        except Exception as e:
            print("collect error", lid, e)
    time.sleep(args.interval)
