"""
Demo Prediction from Simulator Traffic
=======================================
Script đơn giản: Đưa traffic_data.csv từ simulator → predict ngay

Workflow:
1. Load 95 sequences cuối từ test set (X_test.npy)
2. Load 1 record mới nhất từ simulator CSV
3. Ghép thành 96 sequences
4. Predict utilization 30s ahead
5. Hiển thị kết quả

Usage:
    python demo_predict_from_simulator.py
    python demo_predict_from_simulator.py --simulator-csv ../SAGSINs-Simulator/docker/data/traffic_data.csv

Author: PBL4 Team
Date: 2025-11-06
"""

import sys
import io
if sys.platform == 'win32':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except (AttributeError, ValueError):
        # stdout/stderr already configured or buffer closed (e.g., in bash)
        pass

import torch
import numpy as np
import pandas as pd
import json
import pickle
import argparse
from pathlib import Path
from datetime import datetime

# Import models
from train_vae_simple import SimpleVAE
from train_lstm import LSTMModel


class SimulatorPredictor:
    """
    Predictor đơn giản cho simulator traffic (hỗ trợ cả VAE và LSTM)
    """
    
    def __init__(self, 
                 vae_model_path='models/simple_vae_best.pth',
                 lstm_model_path='models/best_lstm_model.pth',
                 scalers_path='models/wide_scalers.pkl',
                 features_json='data/features.json',
                 link_index_json='data/link_index.json',
                 test_data_path='data/X_test.npy'):
        """
        Initialize predictor
        
        Args:
            vae_model_path: Path to trained VAE model
            lstm_model_path: Path to trained LSTM model
            scalers_path: Path to scalers
            features_json: Path to features config
            link_index_json: Path to link index
            test_data_path: Path to test set (for history)
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Device: {self.device}")
        
        self.vae_model_path = vae_model_path
        self.lstm_model_path = lstm_model_path
        
        # Load features config
        with open(features_json, 'r') as f:
            config = json.load(f)
            self.model_features = config['model_features']
            self.num_features = len(self.model_features)
        
        # Load link index
        with open(link_index_json, 'r') as f:
            self.link_index = json.load(f)
            self.num_links = len(self.link_index)
        
        # Calculate actual input dim (wide format: num_features * num_links)
        self.input_dim = self.num_features * self.num_links
        
        print(f"✅ Features: {self.num_features} base features × {self.num_links} links = {self.input_dim} dims")
        print(f"✅ Base features: {self.model_features}")
        
        # Load scalers (optional)
        self.wide_scalers = None
        if Path(scalers_path).exists():
            with open(scalers_path, 'rb') as f:
                self.wide_scalers = pickle.load(f)
            print(f"✅ Loaded scalers from {scalers_path}")
        else:
            print(f"⚠️  Scalers not found at {scalers_path}")
            print(f"   → Will use pre-scaled data from X_test (OK for demo)")
        
        # Load VAE model
        self.vae_model = None
        if Path(vae_model_path).exists():
            self.vae_model = SimpleVAE(
                input_dim=self.input_dim,
                latent_dim=96,
                hidden_dim=256,
                num_links=self.num_links,
                dropout=0.5
            ).to(self.device)
            
            checkpoint = torch.load(vae_model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.vae_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.vae_model.load_state_dict(checkpoint)
            self.vae_model.eval()
            print(f"✅ Loaded VAE model from {vae_model_path}")
        else:
            print(f"⚠️  VAE model not found at {vae_model_path}")
        
        # Load LSTM model
        self.lstm_model = None
        if Path(lstm_model_path).exists():
            self.lstm_model = LSTMModel(
                input_size=self.input_dim,
                hidden_size=256,
                num_layers=2,
                dropout=0.2,
                output_size=self.num_links,
                features_per_link=self.num_features,
                num_links=self.num_links,
                attn_layers=2,
                attn_heads=4
            ).to(self.device)
            
            checkpoint = torch.load(lstm_model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.lstm_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.lstm_model.load_state_dict(checkpoint)
            self.lstm_model.eval()
            print(f"✅ Loaded LSTM model from {lstm_model_path}")
        else:
            print(f"⚠️  LSTM model not found at {lstm_model_path}")
        
        # Load test data (for history)
        self.X_test = np.load(test_data_path)
        print(f"✅ Loaded test data: {self.X_test.shape}")
    
    def load_simulator_csv(self, csv_path, use_latest=True):
        """
        Load traffic từ simulator CSV và extract features cho 1 link
        
        Args:
            csv_path: Path to simulator traffic_data.csv
            use_latest: Lấy record mới nhất (default: True)
        
        Returns:
            tuple: (features_base, link_id)
            - features_base: numpy array shape (11,) - base features cho 1 link
            - link_id: str - link ID từ simulator
        """
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Simulator CSV not found: {csv_path}")
        
        # Read CSV
        df = pd.read_csv(csv_path)
        print(f"📄 Read simulator CSV: {len(df)} records")
        
        if use_latest:
            df_latest = df.iloc[[-1]].copy()
            print(f"   ✅ Using latest record: {df_latest['timestamp'].iloc[0]}")
        else:
            df_latest = df
        
        # Get link_id từ simulator
        simulator_link_id = df_latest['link_id'].iloc[0] if 'link_id' in df_latest.columns else None
        if simulator_link_id:
            print(f"   🔗 Link: {simulator_link_id}")
        
        # Extract 11 base features
        missing = [f for f in self.model_features if f not in df_latest.columns]
        if missing:
            raise ValueError(f"Missing features in simulator CSV: {missing}")
        
        # Get feature values (11 features)
        features_df = df_latest[self.model_features]
        features_base = features_df.values[0]  # shape: (11,)
        
        print(f"   ✅ Extracted {self.num_features} base features")
        
        return features_base, simulator_link_id  # shape: (11,), link_id
    
    def get_history_from_test(self, n_sequences=95):
        """
        Get n timesteps cuối từ test set sequence cuối cùng
        
        Args:
            n_sequences: Số timesteps cần lấy (default: 95)
        
        Returns:
            numpy array shape (n_sequences, input_dim)
        """
        # X_test shape: (num_samples, 96, 132)
        # Lấy sample cuối và 95 timesteps cuối từ nó
        last_sample = self.X_test[-1, :, :]  # shape: (96, 132)
        history = last_sample[-n_sequences:, :]  # shape: (95, 132)
        print(f"📊 Loaded history: {history.shape} (last {n_sequences} timesteps from test set)")
        return history
    
    def preprocess_and_scale(self, X_raw):
        """
        Scale dữ liệu sử dụng wide_scalers (nếu có)
        
        Args:
            X_raw: numpy array shape (seq_len, num_features)
        
        Returns:
            numpy array shape (seq_len, num_features) - scaled
        """
        if self.wide_scalers is None:
            # No scaler available, assume data already scaled
            print("   ℹ️  No scaler - using raw data (assumed pre-scaled)")
            return X_raw
        
        # X_raw shape: (96, 11)
        seq_len = X_raw.shape[0]
        
        # Scale từng feature
        X_scaled = np.zeros_like(X_raw)
        for i, feature_name in enumerate(self.model_features):
            if feature_name in self.wide_scalers:
                scaler = self.wide_scalers[feature_name]
                # Reshape to (seq_len, 1) for scaler
                X_scaled[:, i] = scaler.transform(X_raw[:, i].reshape(-1, 1)).flatten()
            else:
                # No scaler, use raw
                X_scaled[:, i] = X_raw[:, i]
        
        return X_scaled
    
    def predict(self, X_input, model_type='both'):
        """
        Predict utilization 30s ahead
        
        Args:
            X_input: numpy array shape (seq_len, num_features)
            model_type: 'vae', 'lstm', or 'both'
        
        Returns:
            dict: {'vae': pred_vae, 'lstm': pred_lstm} or single array
        """
        # Scale
        X_scaled = self.preprocess_and_scale(X_input)
        
        # To tensor
        X_tensor = torch.FloatTensor(X_scaled).unsqueeze(0).to(self.device)  # (1, 96, 132)
        
        results = {}
        
        # VAE prediction
        if model_type in ['vae', 'both'] and self.vae_model is not None:
            with torch.no_grad():
                pred_vae, mu, logvar = self.vae_model(X_tensor)
                pred_vae_np = pred_vae.cpu().numpy()[0]  # (12,)
                
                # Clip to [0, 1]
                pred_vae_np = np.clip(pred_vae_np, 0, 1)
                results['vae'] = pred_vae_np
        
        # LSTM prediction
        if model_type in ['lstm', 'both'] and self.lstm_model is not None:
            with torch.no_grad():
                pred_lstm = self.lstm_model(X_tensor)
                pred_lstm_np = pred_lstm.cpu().numpy()[0]  # (12,)
                
                # Clip to [0, 1]
                pred_lstm_np = np.clip(pred_lstm_np, 0, 1)
                results['lstm'] = pred_lstm_np
        
        # Return based on model_type
        if model_type == 'both':
            return results
        elif model_type == 'vae':
            return results.get('vae', None)
        else:  # lstm
            return results.get('lstm', None)
    
    def run_demo(self, simulator_csv):
        """
        Main demo workflow
        
        Args:
            simulator_csv: Path to simulator traffic_data.csv
        """
        print("\n" + "=" * 70)
        print("🚀 Demo: Predict from Simulator Traffic")
        print("=" * 70)
        
        # Step 1: Load history (96 sequences with all 12 links)
        print("\n📌 Step 1: Load history from test set (all 12 links)")
        # Get FULL history with all 12 links from test set
        history_full = self.X_test[-1, :, :].copy()  # (96, 132) - last sample from test
        print(f"   ✅ Loaded history: {history_full.shape} (96 timesteps × 132 dims)")
        
        # Step 2: Load simulator record
        print("\n📌 Step 2: Load latest record from simulator")
        sim_features, simulator_link_id = self.load_simulator_csv(simulator_csv, use_latest=True)
        
        # Check if simulator link exists in model
        if not simulator_link_id or simulator_link_id not in self.link_index:
            print(f"\n⚠️  Warning: Link {simulator_link_id} not found in model")
            print(f"   Available links: {self.link_index}")
            return None
        
        # Get link index
        link_idx = self.link_index.index(simulator_link_id)
        
        # Step 3: Update ONLY simulator link in last timestep
        print(f"\n📌 Step 3: Update link {simulator_link_id} (index {link_idx}) in last timestep")
        # Calculate feature range for this link in wide format
        link_start = link_idx * self.num_features  # e.g., link 7: 7*11 = 77
        link_end = (link_idx + 1) * self.num_features  # e.g., link 7: 8*11 = 88
        
        # Replace ONLY simulator link features in last timestep
        history_full[-1, link_start:link_end] = sim_features
        print(f"   ✅ Updated features [{link_start}:{link_end}] with simulator data")
        print(f"   ℹ️  Other 11 links keep test set values (realistic network state)")
        
        # Step 4: Predict với cả 2 models
        print("\n📌 Step 4: Predict utilization 30s ahead")
        preds = self.predict(history_full, model_type='both')
        
        # Step 5: Display results (CHỈ link mô phỏng)
        print("\n" + "=" * 70)
        print("📊 PREDICTION RESULTS (30s ahead)")
        print("=" * 70)
        
        print(f"\n🎯 Link: {simulator_link_id}")
        print("-" * 70)
        
        # Display both predictions
        if 'vae' in preds:
            util_vae = preds['vae'][link_idx]
            util_vae_percent = util_vae * 100
            
            # Status
            if util_vae > 0.8:
                status_vae = "🔴 HIGH"
            elif util_vae > 0.6:
                status_vae = "🟡 MEDIUM"
            else:
                status_vae = "🟢 LOW"
            
            print(f"\n📈 VAE Model:")
            print(f"   Predicted Utilization: {util_vae_percent:>6.2f}%  {status_vae}")
        else:
            print(f"\n⚠️  VAE model not available")
            util_vae = None
        
        if 'lstm' in preds:
            util_lstm = preds['lstm'][link_idx]
            util_lstm_percent = util_lstm * 100
            
            # Status
            if util_lstm > 0.8:
                status_lstm = "🔴 HIGH"
            elif util_lstm > 0.6:
                status_lstm = "🟡 MEDIUM"
            else:
                status_lstm = "🟢 LOW"
            
            print(f"\n📈 LSTM Model:")
            print(f"   Predicted Utilization: {util_lstm_percent:>6.2f}%  {status_lstm}")
        else:
            print(f"\n⚠️  LSTM model not available")
            util_lstm = None
        
        # Comparison
        if util_vae is not None and util_lstm is not None:
            diff = abs(util_vae - util_lstm) * 100
            avg = (util_vae + util_lstm) / 2 * 100
            
            print(f"\n📊 Comparison:")
            print(f"   Difference:  {diff:.2f}%")
            print(f"   Average:     {avg:.2f}%")
            
            if diff < 5:
                print(f"   ✅ Models agree well (diff < 5%)")
            elif diff < 10:
                print(f"   ⚠️  Models have some difference (5% < diff < 10%)")
            else:
                print(f"   🔴 Models disagree significantly (diff > 10%)")
        
        # Recommendation (dựa trên average nếu có cả 2)
        if util_vae is not None and util_lstm is not None:
            util_avg = (util_vae + util_lstm) / 2
        elif util_vae is not None:
            util_avg = util_vae
        elif util_lstm is not None:
            util_avg = util_lstm
        else:
            print("\n❌ No predictions available")
            return None
        
        print(f"\n💡 Recommendation:")
        if util_avg > 0.8:
            print(f"   🔴 Link utilization is HIGH ({util_avg*100:.1f}%)!")
            print(f"   → Consider load balancing or capacity upgrade")
        elif util_avg > 0.6:
            print(f"   ⚠️  Link utilization is moderate ({util_avg*100:.1f}%)")
            print(f"   → Monitor closely")
        else:
            print(f"   ✅ Link utilization is healthy ({util_avg*100:.1f}%)")
            print(f"   → No action needed")
        
        print("\n✅ Prediction complete!")
        print("=" * 70)
        
        return preds


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Demo prediction from simulator traffic')
    parser.add_argument('--simulator-csv', 
                       default='../SAGSINs-System/docker/data/traffic_data.csv',
                       help='Path to simulator traffic_data.csv')
    parser.add_argument('--vae-model', 
                       default='models/simple_vae_best.pth',
                       help='Path to trained VAE model')
    parser.add_argument('--lstm-model',
                       default='models/best_lstm_model.pth',
                       help='Path to trained LSTM model')
    parser.add_argument('--scalers',
                       default='models/wide_scalers.pkl',
                       help='Path to scalers')
    
    args = parser.parse_args()
    
    # Check files exist
    simulator_csv = Path(args.simulator_csv)
    if not simulator_csv.exists():
        print(f"❌ Error: Simulator CSV not found: {simulator_csv}")
        print(f"\nPlease check the path or run simulator first to generate traffic_data.csv")
        return
    
    # Check at least one model exists
    vae_exists = Path(args.vae_model).exists()
    lstm_exists = Path(args.lstm_model).exists()
    
    if not vae_exists and not lstm_exists:
        print(f"❌ Error: No models found!")
        print(f"   VAE:  {args.vae_model}")
        print(f"   LSTM: {args.lstm_model}")
        print(f"\nPlease train at least one model first")
        return
    
    if not vae_exists:
        print(f"⚠️  VAE model not found: {args.vae_model}")
        print(f"   → Will only use LSTM model")
    
    if not lstm_exists:
        print(f"⚠️  LSTM model not found: {args.lstm_model}")
        print(f"   → Will only use VAE model")
    
    # Run prediction
    predictor = SimulatorPredictor(
        vae_model_path=args.vae_model,
        lstm_model_path=args.lstm_model,
        scalers_path=args.scalers,
        features_json='data/features.json',
        link_index_json='data/link_index.json',
        test_data_path='data/X_test.npy'
    )
    
    predictor.run_demo(simulator_csv)


if __name__ == '__main__':
    main()
