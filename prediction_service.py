#!/usr/bin/env python3
"""
Prediction Service - FastAPI Server
====================================
Chạy prediction service riêng trên local machine (không trong Docker)

Usage:
    python prediction_service.py
    
    # Hoặc chỉ định port khác:
    python prediction_service.py --port 5000

API Endpoints:
    POST /predict
        Body: {
            "csv_path": "/path/to/traffic_data.csv",
            "use_latest": true
        }
        
    GET /health
        Health check endpoint

Author: PBL4 Team
Date: 2025-11-07
"""

import sys
import io
import logging
from pathlib import Path

# Configure logging to file instead of stdout
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('prediction_service.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Safe print function that works in all environments
def safe_print(*args, **kwargs):
    """Print that handles closed stdout/stderr gracefully"""
    try:
        print(*args, **kwargs, flush=True)
    except Exception:
        # Silently ignore all print errors (stdout/stderr closed, etc.)
        pass

# Do NOT reconfigure stdout/stderr - let uvicorn handle it
# Reconfiguring causes "I/O operation on closed file" in batch scripts

import argparse
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Import predictor
from demo_predict_from_simulator import SimulatorPredictor

# =============================================
# FastAPI App
# =============================================
app = FastAPI(
    title="PBL4 Prediction Service",
    description="Network Traffic Prediction Service using VAE + LSTM models",
    version="1.0.0"
)

# Global predictor instance (lazy load)
predictor = None

# =============================================
# Request/Response Models
# =============================================
class PredictRequest(BaseModel):
    csv_path: str
    use_latest: bool = True

class PredictionResult(BaseModel):
    link_id: str
    timestamp: str
    vae: dict = None
    lstm: dict = None
    average: dict = None

class PredictResponse(BaseModel):
    status: str
    prediction: PredictionResult = None
    error: str = None

# =============================================
# Helper Functions
# =============================================
def get_predictor():
    """Lazy load predictor"""
    global predictor
    
    if predictor is None:
        logger.info("[INIT] Initializing predictor...")
        predictor = SimulatorPredictor(
            vae_model_path='models/simple_vae_best.pth',
            lstm_model_path='models/best_lstm_model.pth',
            scalers_path='models/wide_scalers.pkl',
            features_json='data/features.json',
            link_index_json='data/link_index.json',
            test_data_path='data/X_test.npy'
        )
        logger.info("[INIT] Predictor initialized successfully")
    
    return predictor

# =============================================
# API Endpoints
# =============================================
@app.get("/")
def root():
    """Root endpoint"""
    return {
        "service": "PBL4 Prediction Service",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "POST /predict": "Run prediction on traffic data",
            "GET /health": "Health check"
        }
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    try:
        pred = get_predictor()
        return {
            "status": "healthy",
            "predictor": "loaded" if pred is not None else "not loaded",
            "models": {
                "vae": pred.vae_model is not None if pred else False,
                "lstm": pred.lstm_model is not None if pred else False
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """
    Run prediction on traffic data
    
    Args:
        csv_path: Path to traffic_data.csv
        use_latest: Use latest record (default: True)
    
    Returns:
        Prediction results with VAE, LSTM, and average
    """
    try:
        # Validate CSV path
        csv_path = Path(request.csv_path)
        if not csv_path.exists():
            raise HTTPException(
                status_code=404, 
                detail=f"CSV file not found: {csv_path}"
            )
        
        # Get predictor
        pred = get_predictor()
        
        # Load simulator data
        logger.info(f"[DATA] Loading data from: {csv_path}")
        sim_features, simulator_link_id = pred.load_simulator_csv(
            csv_path, 
            use_latest=request.use_latest
        )
        
        # Check if link exists
        if not simulator_link_id or simulator_link_id not in pred.link_index:
            raise HTTPException(
                status_code=400,
                detail=f"Link {simulator_link_id} not found in model. Available: {pred.link_index}"
            )
        
        # Get link index
        link_idx = pred.link_index.index(simulator_link_id)
        
        # Get history
        history_full = pred.X_test[-1, :, :].copy()  # (96, 132)
        
        # Update link features
        link_start = link_idx * pred.num_features
        link_end = (link_idx + 1) * pred.num_features
        history_full[-1, link_start:link_end] = sim_features
        
        # Predict
        logger.info(f"[PREDICT] Running prediction for {simulator_link_id}...")
        preds = pred.predict(history_full, model_type='both')
        
        # Format results
        from datetime import datetime
        
        result = {
            'link_id': simulator_link_id,
            'timestamp': datetime.now().isoformat()
        }
        
        # VAE
        if 'vae' in preds:
            util_vae = float(preds['vae'][link_idx])
            result['vae'] = {
                'utilization': util_vae,
                'utilization_percent': util_vae * 100,
                'status': 'HIGH' if util_vae > 0.8 else 'MEDIUM' if util_vae > 0.6 else 'LOW'
            }
            logger.info(f"   [VAE]  {util_vae*100:.2f}% ({result['vae']['status']})")
        
        # LSTM
        if 'lstm' in preds:
            util_lstm = float(preds['lstm'][link_idx])
            result['lstm'] = {
                'utilization': util_lstm,
                'utilization_percent': util_lstm * 100,
                'status': 'HIGH' if util_lstm > 0.8 else 'MEDIUM' if util_lstm > 0.6 else 'LOW'
            }
            logger.info(f"   [LSTM] {util_lstm*100:.2f}% ({result['lstm']['status']})")
        
        # Average
        if 'vae' in result and 'lstm' in result:
            avg_util = (result['vae']['utilization'] + result['lstm']['utilization']) / 2
            result['average'] = {
                'utilization': avg_util,
                'utilization_percent': avg_util * 100,
                'status': 'HIGH' if avg_util > 0.8 else 'MEDIUM' if avg_util > 0.6 else 'LOW'
            }
            logger.info(f"   [AVG]  {avg_util*100:.2f}% ({result['average']['status']})")
        
        logger.info(f"[SUCCESS] Prediction complete")
        
        return PredictResponse(
            status="success",
            prediction=result
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[ERROR] Error in prediction: {e}", exc_info=True)
        
        return PredictResponse(
            status="error",
            error=str(e)
        )

# =============================================
# Main
# =============================================
def main():
    """Start prediction service"""
    parser = argparse.ArgumentParser(description='PBL4 Prediction Service')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('--reload', action='store_true', help='Auto-reload on code changes')
    
    args = parser.parse_args()
    
    safe_print("=" * 70)
    safe_print("PBL4 Prediction Service")
    safe_print("=" * 70)
    safe_print(f"Host: {args.host}")
    safe_print(f"Port: {args.port}")
    safe_print(f"URL:  http://localhost:{args.port}")
    safe_print("=" * 70)
    
    logger.info(f"Starting server on {args.host}:{args.port}")
    
    # Start server - Let uvicorn handle all logging
    uvicorn.run(
        "prediction_service:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
        access_log=False
    )

if __name__ == '__main__':
    main()