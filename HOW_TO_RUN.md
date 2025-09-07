# PBL4 – Network Traffic Prediction

## HƯỚNG DẪN CHẠY (Windows / PowerShell)

1. Tạo môi trường ảo và cài dependencies

---

python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt

2. Chạy tiền xử lý dữ liệu (Preprocessing)

---

python preprocessing.py

3. Pipeline chính đã làm gì

---

- Resample dữ liệu theo từng link, cửa sổ 10 giây
  - bytes_sent -> SUM
  - bitrate_bps, rtt, loss_rate, jitter, latency -> MEAN
  - capacity_bps -> LAST / FFILL
- Tạo features:
  - hour, is_weekend
  - utilization (tính theo băng thông, chuẩn hóa)
  - throughput_mbps
  - quality_score (dựa trên loss + jitter)
  - efficiency = utilization \* quality_score
- Pivot dữ liệu thành snapshot dạng "timestamp x link (wide)"
  - dùng chung cho LSTM & VAE
  - giữ thứ tự link cố định (link_index.json)
- Chuẩn hóa dữ liệu:
  - Fit MinMaxScaler theo từng feature
  - Chỉ fit trên TRAIN, sau đó transform toàn bộ
- Sinh chuỗi LSTM:
  - sequence_length = 24 (24 bước = 4 phút nếu bước 10s)
  - horizon = 1 (dự báo bước kế tiếp)
- Chia train/val/test theo thời gian (không shuffle)
  - Train = 70%
  - Val = 15%
  - Test = 15%

4. Các file kết quả quan trọng

---

**_Dữ liệu cho LSTM_**

- data/X_train.npy, data/y_train.npy
- data/X_val.npy, data/y_val.npy
- data/X_test.npy, data/y_test.npy

**_Dữ liệu cho VAE_**

- data/vae_snapshots.npy : toàn bộ snapshot đã scale
- data/vae_columns.json : tên cột (feature x link) theo thứ tự cố định

**_Tái lập / Inference_**

- data/features.json : danh sách feature đã chọn
- data/link_index.json : thứ tự link cố định
- data/timestamp_splits.json: thông tin chia train/val/test (theo thời gian)
- models/wide_scalers.pkl : scaler MinMax cho từng feature, fit trên train

**_Phân tích chất lượng dữ liệu_**

- data/missing_mask.npy : ma trận True/False cùng shape với vae_snapshots
  - True = giá trị gốc bị thiếu (NaN trước khi interpolate/ffill)
  - Dùng để:
    - Kiểm tra chất lượng dữ liệu, xem link nào hay thiếu
    - Thiết kế masked loss (không phạt nặng tại điểm thiếu)
    - Debug model (nếu dự đoán kém ở đoạn dữ liệu thiếu nhiều)

**_Tham chiếu_**

- data/traffic_processed.csv: dữ liệu đã resample + feature engineering, trước khi pivot

xem 4 ảnh để hiểu dữ liệu mà lstm vs vae nhận vào như thế nào nha
