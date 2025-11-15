# 📊 Prediction Dashboard Guide

## 🎯 Giới thiệu

Dashboard trực quan hiển thị kết quả prediction real-time từ AI models (VAE + LSTM) cho network traffic utilization.

## ✨ Tính năng

### 1. **Real-time Monitoring**

- 🔴 Live status của prediction service
- 📊 Utilization trend chart (last 20 predictions)
- 🎯 Model comparison (VAE vs LSTM)
- 📋 Predictions history table

### 2. **Visualizations**

- **Line Chart**: Trend của VAE, LSTM, Average qua thời gian
- **Bar Chart**: So sánh output của 2 models cho prediction mới nhất
- **Status Indicators**: Color-coded status (LOW/MEDIUM/HIGH)

### 3. **Auto-update**

- **WebSocket mode**: Real-time updates khi có prediction mới (nếu Node.js backend đang chạy)
- **Polling mode**: Tự động refresh mỗi 3 giây (fallback)

---

## 🚀 Cách sử dụng

### Bước 1: Start Prediction Service

```bash
cd D:\HuyCoding\PBL4\PBL4-Network-Traffic-Prediction
python prediction_service.py --port 5000
```

Hoặc dùng script:

```bash
./start-prediction-service.bat
```

### Bước 2: Start Docker Containers

```bash
cd D:\HuyCoding\PBL4\SAGSINs-System\docker
docker-compose up -d
```

### Bước 3: Start Node.js Backend

```bash
cd D:\HuyCoding\PBL4\SAGSINs-System\wep-app\backend
npm start
```

### Bước 4: Start Frontend (Web App)

```bash
cd D:\HuyCoding\PBL4\SAGSINs-System\wep-app\frontend
npm run dev
```

### Bước 5: Mở Dashboard

Có 2 cách:

#### Option 1: Truy cập qua Prediction Service

```
http://localhost:5000
```

#### Option 2: Mở file trực tiếp

```
Mở file: D:\HuyCoding\PBL4\PBL4-Network-Traffic-Prediction\dashboard.html
```

---

## 📡 Kiến trúc Dataflow

```
┌──────────────┐
│   Web App    │ (Frontend - React)
│   Port 5173  │
└──────┬───────┘
       │ Socket.IO
       ↓
┌──────────────┐
│   Node.js    │ (Backend - Socket.IO)
│   Port 3001  │
└──────┬───────┘
       │ HTTP POST
       ↓
┌──────────────┐
│   Docker     │ (Flask Server)
│   Port 8080  │
└──────┬───────┘
       │ HTTP POST
       ↓
┌──────────────┐         ┌──────────────┐
│  Prediction  │────────→│  Dashboard   │
│  Service     │ Serve   │  (HTML/JS)   │
│  Port 5000   │         │  Port 5000   │
└──────────────┘         └──────────────┘
       ↑
       │ WebSocket (Real-time updates)
       │
┌──────┴───────┐
│   Node.js    │
│   Backend    │
└──────────────┘
```

### Data Flow khi gửi packet:

1. **User** gửi packet từ Web App (React)
2. **Node.js Backend** nhận qua Socket.IO
3. **Docker Server** tạo traffic metrics và lưu CSV
4. **Prediction Service** được gọi bởi Docker server
5. **AI Models** (VAE + LSTM) predict utilization
6. **Docker Server** trả prediction về Node.js
7. **Node.js** broadcast prediction qua WebSocket
8. **Dashboard** nhận và update charts real-time

---

## 🎨 Dashboard Components

### Status Cards

- **Service Status**: Online/Offline/Unhealthy
- **Latest Prediction**: Average utilization (%)
- **Active Link**: Link ID đang được predict
- **Total Predictions**: Tổng số predictions đã tạo

### Charts

1. **Utilization Trend** (Line Chart)

   - X-axis: Thời gian
   - Y-axis: Utilization (%)
   - 3 lines: VAE (blue), LSTM (pink), Average (cyan)
   - Keep last 20 predictions

2. **Model Comparison** (Bar Chart)
   - So sánh output của VAE vs LSTM vs Average
   - Update mỗi khi có prediction mới

### Predictions Table

- **Columns**: Timestamp, Link ID, VAE, LSTM, Average, Status
- **Status Badge**:
  - 🟢 LOW: < 60%
  - 🟡 MEDIUM: 60-80%
  - 🔴 HIGH: > 80%
- Show 10 recent predictions (newest first)

---

## 🔧 API Endpoints

### Prediction Service (Port 5000)

#### 1. `GET /`

Dashboard UI (HTML page)

#### 2. `GET /api`

API information

```json
{
  "service": "PBL4 Prediction Service",
  "version": "1.0.0",
  "status": "running"
}
```

#### 3. `POST /predict`

Run prediction

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "csv_path": "D:/HuyCoding/PBL4/SAGSINs-System/docker/data/traffic_data.csv",
    "use_latest": true
  }'
```

Response:

```json
{
  "status": "success",
  "prediction": {
    "link_id": "LINK_AIR_GROUND_01",
    "timestamp": "2025-11-15T16:10:57.793",
    "vae": {
      "utilization": 0.5642,
      "utilization_percent": 56.42,
      "status": "LOW"
    },
    "lstm": {
      "utilization": 0.2816,
      "utilization_percent": 28.16,
      "status": "LOW"
    },
    "average": {
      "utilization": 0.4229,
      "utilization_percent": 42.29,
      "status": "LOW"
    }
  }
}
```

#### 4. `GET /predictions?limit=50`

Get recent predictions

```bash
curl http://localhost:5000/predictions?limit=20
```

Response:

```json
{
  "status": "success",
  "total": 100,
  "returned": 20,
  "predictions": [...]
}
```

#### 5. `GET /health`

Health check

```bash
curl http://localhost:5000/health
```

Response:

```json
{
  "status": "healthy",
  "predictor": "loaded",
  "models": {
    "vae": true,
    "lstm": true
  },
  "total_predictions": 42
}
```

---

## 🧪 Testing

### Test 1: Health Check

```bash
curl http://localhost:5000/health
```

Expected: `"status": "healthy"`

### Test 2: Send Packet từ Web App

1. Mở Web App: http://localhost:5173
2. Register node: `UAV_01`
3. Send packet: `UAV_01` → `GROUND_GATEWAY_01`
4. Check logs:
   - Node.js: `✅ Traffic data saved`
   - Prediction service: `[SUCCESS] Prediction complete`
   - Dashboard: Chart tự động update

### Test 3: View Dashboard

1. Mở: http://localhost:5000
2. Kiểm tra:
   - ✅ Service Status = "Online"
   - ✅ Charts hiển thị data
   - ✅ Table có predictions
   - ✅ Real-time updates khi gửi packet mới

---

## 🐛 Troubleshooting

### Dashboard không load

**Triệu chứng**: Blank page hoặc 404  
**Nguyên nhân**: `dashboard.html` không được serve  
**Giải pháp**:

```bash
# Đảm bảo dashboard.html cùng folder với prediction_service.py
ls D:\HuyCoding\PBL4\PBL4-Network-Traffic-Prediction\dashboard.html

# Restart prediction service
python prediction_service.py --port 5000
```

### Service Status = "Offline"

**Triệu chứng**: Red indicator, "Offline" status  
**Nguyên nhân**: Prediction service không chạy  
**Giải pháp**:

```bash
cd D:\HuyCoding\PBL4\PBL4-Network-Traffic-Prediction
python prediction_service.py --port 5000
```

### Charts không update

**Triệu chứng**: Data cũ, không có real-time updates  
**Nguyên nhân**: WebSocket không kết nối  
**Giải pháp**:

```bash
# Check Node.js backend đang chạy
curl http://localhost:3001/nodes

# Nếu không chạy:
cd D:\HuyCoding\PBL4\SAGSINs-System\wep-app\backend
npm start
```

**Fallback**: Dashboard tự động polling mỗi 3 giây nếu WebSocket fail

### Predictions = 0

**Triệu chứng**: "No predictions yet"  
**Nguyên nhân**: Chưa gửi packet  
**Giải pháp**: Gửi ít nhất 1 packet từ Web App

---

## 📚 Technology Stack

- **Frontend**: Vanilla JavaScript, HTML5, CSS3
- **Charts**: Chart.js 4.4.0
- **Real-time**: Socket.IO 4.5.4
- **Backend**: FastAPI (Python 3.13)
- **AI Models**: PyTorch (VAE + LSTM)

---

## 🎓 Demo Scenario

```
Scenario: Monitor network congestion cho UAV → Ground link

1. User mở Dashboard (http://localhost:5000)
   → Status: Online, 0 predictions

2. User gửi large packet (10KB) từ Web App
   UAV_01 → GROUND_GATEWAY_01

3. Docker server:
   ✅ Generate traffic metrics
   ✅ Save to CSV
   ✅ Call prediction service

4. AI Models predict:
   📊 VAE:  73.2%
   📊 LSTM: 74.5%
   📊 AVG:  73.8% (MEDIUM)

5. Dashboard auto-updates:
   ✅ Status card shows 73.8%
   ✅ Line chart adds new point
   ✅ Bar chart updates
   ✅ Table adds new row with 🟡 MEDIUM badge

6. User continues sending packets
   → Dashboard tracks trend over time
   → Alert nếu utilization > 80% (HIGH)
```

---

## 🚀 Next Steps

### Enhancements:

- [ ] Alert notifications khi HIGH utilization
- [ ] Export predictions to CSV/JSON
- [ ] Historical charts (last 24 hours)
- [ ] Per-link filtering
- [ ] Dark mode theme
- [ ] Mobile responsive design
- [ ] Authentication/login

### Integration:

- [ ] Prometheus metrics export
- [ ] Grafana dashboard
- [ ] Email/SMS alerts
- [ ] Database persistence (PostgreSQL)

---

**🎯 Enjoy monitoring your network predictions!** 🚀
