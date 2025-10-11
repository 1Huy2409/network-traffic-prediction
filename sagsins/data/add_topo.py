import pandas as pd

df = pd.read_csv("topology_data.csv")
server_node = "GROUND_GATEWAY_01"

# Tạo các link mới từ các node chưa đổ về gateway này
new_links = []
for src in ["SATELLITE_02","SATELLITE_03","UAV_02","SHIP_01","SHIP_02","GROUND_GATEWAY_02","GROUND_GATEWAY_03"]:
    link_id = f"LINK_{src.split('_')[0]}_GROUND_{len(df)+len(new_links)+1:02d}"
    layer_map = {
        "SATELLITE":"space",
        "UAV":"air",
        "GROUND":"ground",
        "SHIP":"sea"
    }
    src_layer = layer_map[src.split('_')[0]]
    dst_layer = "ground"
    new_links.append({
        "link_id": link_id,
        "source_node": src,
        "destination_node": server_node,
        "source_layer": src_layer,
        "destination_layer": dst_layer,
        "capacity_bps": 30_000_000,
        "base_latency_milliseconds": 80 + 20*len(new_links),
        "geo_information": "{}",
        "min_capacity_bps": 20_000_000,
        "max_capacity_bps": 40_000_000,
        "reliability_score": 0.9,
        "priority_level": 2,
        "distance_km": 1000 + 500*len(new_links),
        "backup_links": ""
    })

df_new = pd.concat([df, pd.DataFrame(new_links)], ignore_index=True)
df_new.to_csv("topology_data_expanded.csv", index=False)
print("✅ Đã sinh thêm link, tổng số:", len(df_new))
