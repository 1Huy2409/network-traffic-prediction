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
## 4. Chạy LSTM Baseline Model

---

```bash
python lstm_baseline.py
```

### 📊 **Chi tiết từng bước của LSTM Baseline:**

#### **4.1. Khởi tạo và Setup (dòng 400-415)**
```python
# Tạo thư mục results/ và models/
# Kiểm tra GPU/CPU và chọn device phù hợp
# In thông tin device đang sử dụng
```

#### **4.2. Load Data (dòng 417-418)**
```python
# Load các file .npy đã được preprocessing:
# - X_train.npy: (6024, 24, 72) - 6024 sequences, 24 timesteps, 72 features
# - y_train.npy: (6024, 12) - 6024 targets, 12 links (utilization)
# - X_val.npy, y_val.npy: (1296, 24, 72), (1296, 12)
# - X_test.npy, y_test.npy: (1296, 24, 72), (1296, 12)
# - features.json: 6 features được chọn
# - link_index.json: 12 tên links theo thứ tự
```

#### **4.3. Tạo DataLoaders (dòng 420-424)**
```python
# Convert numpy arrays thành PyTorch tensors
# Tạo DataLoader với batch_size:
# - GPU: batch_size = 64
# - CPU: batch_size = 32
# - shuffle=True cho train, False cho val/test
```

#### **4.4. Model Configuration (dòng 426-443)**
```python
# input_size = 72 (6 features × 12 links)
# output_size = 12 (12 links utilization)
# sequence_length = 24 (4 phút với 10s intervals)
# LSTM Model:
#   - hidden_size = 128
#   - num_layers = 3
#   - dropout = 0.3
#   - Fully connected: 128 → 64 → 12
```

#### **4.5. Model Architecture (dòng 16-69)**
```python
# LSTMModel.forward():
# 1. Input: [batch_size, 24, 72]
# 2. LSTM layers: 3 layers, 128 hidden units
# 3. Lấy output cuối cùng: [batch_size, 128]
# 4. FC layers: 128 → 64 → 12
# 5. Output: [batch_size, 12] (utilization cho 12 links)
```

#### **4.6. Training Process (dòng 453-459)**
```python
# LSTMTrainer.train():
# - Optimizer: Adam (lr=0.001, weight_decay=1e-5)
# - Loss: MSE (Mean Squared Error)
# - Scheduler: ReduceLROnPlateau (patience=5, factor=0.5)
# - Epochs: 200 (với early stopping)
# - Patience: 15 epochs
# - Gradient clipping: max_norm=1.0
```

#### **4.7. Training Loop (dòng 127-169)**
```python
# Mỗi epoch:
# 1. train_epoch(): Forward pass, backward pass, update weights
# 2. validate_epoch(): Forward pass trên validation set
# 3. Learning rate scheduling dựa trên validation loss
# 4. Early stopping nếu validation loss không cải thiện
# 5. Save best model khi validation loss giảm
# 6. Print progress mỗi 10 epochs
```

#### **4.8. Evaluation (dòng 464-486)**
```python
# LSTMEvaluator.predict():
# 1. Load best model
# 2. Forward pass trên test set
# 3. Collect predictions và true values
# 4. Calculate metrics: MSE, RMSE, MAE, R²
# 5. Per-link performance analysis
```

#### **4.9. Metrics Calculation (dòng 197-224)**
```python
# Overall metrics:
# - MSE: Mean Squared Error
# - RMSE: Root Mean Squared Error
# - MAE: Mean Absolute Error
# - R²: R-squared (coefficient of determination)

# Per-link metrics:
# - MSE và MAE cho từng link riêng biệt
# - So sánh performance giữa các loại link
```

#### **4.10. Visualizations (dòng 226-303)**
```python
# 1. plot_training_curves():
#    - Training loss vs Validation loss
#    - Smoothed curves với rolling window

# 2. plot_results():
#    - So sánh predicted vs true values
#    - Plot 4 links đầu tiên (2x2 subplot)
#    - 200 samples đầu tiên

# 3. plot_scatter():
#    - Scatter plot predicted vs true
#    - Perfect prediction line (y=x)
#    - Flatten tất cả links thành 1D
```

#### **4.11. Results Saving (dòng 488-522)**
```python
# Lưu vào results/:
# - lstm_results.json: Detailed metrics và config
# - training_curves.png: Loss curves
# - lstm_results.png: Prediction plots
# - lstm_scatter.png: Scatter plot

# Lưu vào models/:
# - best_lstm_model.pth: Best trained model weights
```

### 🎯 **Kết quả mong đợi:**

#### **Input của LSTM:**
- **X**: (batch_size, 24, 72) - 24 timesteps của 6 features cho 12 links
- **Features**: utilization, bitrate_bps, loss_rate, jitter_milliseconds, rtt_milliseconds, capacity_bps
- **Links**: 12 SAGSINs links (space, air, ground, sea)

#### **Output của LSTM:**
- **y_pred**: (batch_size, 12) - Utilization prediction cho 12 links
- **Range**: [0, 1] (đã được normalize)
- **Horizon**: 1 bước tiếp theo (10 giây sau)

#### **Performance Metrics:**
- **MSE**: Thấp hơn = tốt hơn
- **RMSE**: Căn bậc 2 của MSE
- **MAE**: Trung bình absolute error
- **R²**: Gần 1 = tốt hơn (perfect = 1.0)

#### **Files được tạo:**