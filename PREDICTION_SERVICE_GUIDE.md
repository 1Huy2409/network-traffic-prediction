# 🔮 Prediction Service - Hướng dẫn Setup (Architecture mới)

## 🎯 Architecture

### Trước (chạy prediction trong Docker):

```
Web App → Docker Server (Flask + PyTorch) → Save CSV + Predict
                ❌ Nặng, phải cài PyTorch trong container
                ❌ Build lâu (5-10 phút)
                ❌ Prediction chậm
```

### Sau (Prediction Service riêng biệt):

```
Web App → Docker Server (Flask only) → Save CSV
                ↓
         HTTP Request
                ↓
         Prediction Service (Local - FastAPI)
         ✅ Chạy trên host machine
         ✅ PyTorch đã có sẵn
         ✅ Prediction nhanh
         ✅ Dễ debug và restart
```

---

## 📦 Setup Instructions

### Bước 1: Install Prediction Service dependencies

```bash
cd PBL4-Network-Traffic-Prediction
pip install -r requirements-prediction-service.txt
```

**Dependencies:**

- FastAPI (Web framework)
- Uvicorn (ASGI server)
- PyTorch (đã có từ training)
- NumPy, Pandas, scikit-learn

### Bước 2: Start Prediction Service

#### Windows:

```bash
cd PBL4-Network-Traffic-Prediction
start-prediction-service.bat
```

#### Linux/Mac:

```bash
cd PBL4-Network-Traffic-Prediction
chmod +x start-prediction-service.sh
./start-prediction-service.sh
```

#### Manual:

```bash
cd PBL4-Network-Traffic-Prediction
python prediction_service.py --port 5000
```

**Expected output:**

```
======================================
🚀 PBL4 Prediction Service
======================================
Host: 0.0.0.0
Port: 5000
URL:  http://localhost:5000
======================================

INFO:     Started server process
INFO:     Uvicorn running on http://0.0.0.0:5000
```

### Bước 3: Rebuild Docker (không cần PyTorch nữa)

```bash
cd SAGSINs-System/docker
docker-compose down
docker-compose build sagsins-server  # ✅ Nhanh hơn nhiều!
docker-compose up -d
```

⏱️ **Lưu ý**: Build chỉ mất ~30 giây (không phải 5-10 phút như trước)

### Bước 4: Test

#### Test Prediction Service trực tiếp:

```bash
curl http://localhost:5000/health
```

Expected:

```json
{
  "status": "healthy",
  "predictor": "loaded",
  "models": {
    "vae": true,
    "lstm": true
  }
}
```

#### Test qua simulator:

1. Send packet từ Web App
2. Check Docker logs: `docker logs -f sagsins-server`
3. Sẽ thấy:

```
📦 Received packet: SATELLITE_01 -> GROUND_GATEWAY_01
🔮 Calling prediction service at http://host.docker.internal:5000...
✅ Received prediction from service
```

---

## 🔧 Configuration

### Environment Variables (trong docker-compose.yml):

```yaml
environment:
  # Prediction Service URL
  - PREDICTION_SERVICE_URL=http://host.docker.internal:5000

  # Enable/disable predictions
  - PREDICTION_ENABLED=true

  # CSV path trên host machine
  - HOST_TRAFFIC_CSV=D:/HuyCoding/PBL4/SAGSINs-System/docker/data/traffic_data.csv
```

### Tắt Prediction (nếu cần):

```yaml
environment:
  - PREDICTION_ENABLED=false
```

---

## 📊 API Endpoints

### Prediction Service (Port 5000)

#### `GET /`

Root endpoint - service info

#### `GET /health`

Health check

Response:

```json
{
  "status": "healthy",
  "predictor": "loaded",
  "models": {
    "vae": true,
    "lstm": true
  }
}
```

#### `POST /predict`

Run prediction

Request:

```json
{
  "csv_path": "D:/HuyCoding/PBL4/SAGSINs-System/docker/data/traffic_data.csv",
  "use_latest": true
}
```

Response:

```json
{
  "status": "success",
  "prediction": {
    "link_id": "LINK_SPACE_GROUND_01",
    "timestamp": "2025-11-07T10:30:00",
    "vae": {
      "utilization": 0.732,
      "utilization_percent": 73.2,
      "status": "MEDIUM"
    },
    "lstm": {
      "utilization": 0.745,
      "utilization_percent": 74.5,
      "status": "MEDIUM"
    },
    "average": {
      "utilization": 0.7385,
      "utilization_percent": 73.85,
      "status": "MEDIUM"
    }
  }
}
```

---

## 🧪 Testing

### Test Script (Python):

```python
import requests

# Test health
response = requests.get('http://localhost:5000/health')
print(response.json())

# Test prediction
response = requests.post(
    'http://localhost:5000/predict',
    json={
        'csv_path': 'D:/HuyCoding/PBL4/SAGSINs-System/docker/data/traffic_data.csv',
        'use_latest': True
    }
)
print(response.json())
```

### Test với curl:

```bash
# Health check
curl http://localhost:5000/health

# Prediction
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"csv_path":"D:/HuyCoding/PBL4/SAGSINs-System/docker/data/traffic_data.csv","use_latest":true}'
```

---

## 🔍 Troubleshooting

### ❌ "Cannot connect to prediction service"

**Triệu chứng:**

```
⚠️  Cannot connect to prediction service at http://host.docker.internal:5000
```

**Nguyên nhân**: Prediction service chưa chạy

**Giải pháp:**

```bash
cd PBL4-Network-Traffic-Prediction
python prediction_service.py --port 5000
```

---

### ❌ "host.docker.internal not resolved"

**Triệu chứng:**

```
Cannot resolve host.docker.internal
```

**Nguyên nhân**: Docker không hỗ trợ `host.docker.internal`

**Giải pháp:**

**Option 1**: Dùng host IP

```yaml
environment:
  - PREDICTION_SERVICE_URL=http://192.168.1.100:5000 # Your host IP
```

**Option 2**: Dùng `network_mode: host` (Linux only)

```yaml
services:
  sagsins-server:
    network_mode: host
```

---

### ❌ "Model not found"

**Triệu chứng:**

```
FileNotFoundError: models/simple_vae_best.pth
```

**Nguyên nhân**: Models chưa được train

**Giải pháp:**

```bash
cd PBL4-Network-Traffic-Prediction
python train_vae_simple.py
python train_lstm.py
```

---

### ⚠️ "Prediction timeout"

**Triệu chứng:**

```
⚠️  Prediction service timeout
```

**Nguyên nhân**: Prediction mất quá 10 giây

**Giải pháp**: Tăng timeout trong `app.py`

```python
response = requests.post(
    ...,
    timeout=30  # Tăng lên 30 giây
)
```

---

## 🚀 Workflow hoàn chỉnh

### 1. Start Prediction Service (Local)

```bash
cd PBL4-Network-Traffic-Prediction
python prediction_service.py --port 5000
```

Keep this running in terminal 1.

### 2. Start Docker Containers

```bash
cd SAGSINs-System/docker
docker-compose up -d
```

### 3. Start Web App

```bash
cd SAGSINs-System/wep-app/frontend
npm run dev
```

### 4. Send Packet & Watch Logs

Terminal 1 (Prediction Service):

```
🔮 Running prediction for LINK_SPACE_GROUND_01...
   📈 VAE:  73.20% (MEDIUM)
   📈 LSTM: 74.50% (MEDIUM)
   📊 AVG:  73.85% (MEDIUM)
✅ Prediction complete
```

Terminal 2 (Docker Server):

```bash
docker logs -f sagsins-server

# Output:
📦 Received packet: SATELLITE_01 -> GROUND_GATEWAY_01
🔮 Calling prediction service...
✅ Received prediction from service
```

---

## ✅ Advantages

### ✨ So với architecture cũ:

| Tiêu chí             | Cũ (In Docker) | Mới (Separate Service) |
| -------------------- | -------------- | ---------------------- |
| **Build time**       | 5-10 phút      | 30 giây                |
| **Container size**   | ~2GB           | ~200MB                 |
| **Prediction speed** | Chậm           | Nhanh                  |
| **Debug**            | Khó            | Dễ                     |
| **Restart**          | Phải rebuild   | Chỉ restart service    |
| **Resource**         | Nặng           | Nhẹ                    |

### 🎯 Benefits:

1. ✅ **Không cần rebuild Docker** khi thay đổi model code
2. ✅ **Debug dễ dàng** - chỉ cần restart Python script
3. ✅ **Prediction nhanh hơn** - chạy trực tiếp trên host
4. ✅ **Độc lập** - có thể dùng cho nhiều clients khác
5. ✅ **Scale dễ** - có thể deploy service lên cloud riêng

---

## 📈 Next Steps (Optional)

### 1. Deploy Prediction Service lên Cloud

```bash
# Example: Deploy to Heroku
heroku create pbl4-prediction-service
git push heroku main
```

Update docker-compose.yml:

```yaml
environment:
  - PREDICTION_SERVICE_URL=https://pbl4-prediction-service.herokuapp.com
```

### 2. Add Caching

```python
from functools import lru_cache

@lru_cache(maxsize=100)
def predict_cached(csv_hash):
    # Cache predictions for same traffic data
    pass
```

### 3. Add Authentication

```python
from fastapi.security import HTTPBearer

@app.post("/predict")
def predict(request: PredictRequest, token: str = Depends(HTTPBearer())):
    # Verify token
    pass
```

---

## 📚 Files Created/Modified

### New Files:

- ✅ `PBL4-Network-Traffic-Prediction/prediction_service.py`
- ✅ `PBL4-Network-Traffic-Prediction/requirements-prediction-service.txt`
- ✅ `PBL4-Network-Traffic-Prediction/start-prediction-service.bat`
- ✅ `PBL4-Network-Traffic-Prediction/start-prediction-service.sh`

### Modified Files:

- ✅ `SAGSINs-System/docker/server/app.py` - Call HTTP API thay vì local prediction
- ✅ `SAGSINs-System/docker/server/requirements.txt` - Bỏ PyTorch
- ✅ `SAGSINs-System/docker/docker-compose.yml` - Add env vars, remove PBL4 mount

---

**🎉 Architecture mới: Nhẹ hơn, nhanh hơn, dễ maintain hơn!**

---

**Version**: 2.0 (Separate Prediction Service)  
**Date**: 2025-11-07  
**Author**: PBL4 Team
