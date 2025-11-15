# ================== LSTM TRAINER (Per-Link + Cross-Link Attention, GPU/AMP) ==================
# Input từ preprocessing.py:
#   data/X_train.npy [N, T, D], D = F * L
#   data/y_train.npy [N, L]
#   data/features.json  -> {"model_features":[...], "target_feature":"utilization"}
#   data/link_index.json -> ["LINK_...", ...]
#   models/wide_scalers.pkl (optional, để inverse-scale khi báo cáo metric thật)
#
# Save:
#   models/best_lstm_model.pth, models/last_checkpoint.pth
#   results/training_curves.png, lstm_results.png, lstm_scatter.png
#   results/lstm_results.json, results/y_true.npy/csv, results/y_pred.npy/csv
# =================================================================================================

import os
import json
import random
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ---------------- Repro ----------------
SEED = 41
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
try: torch.use_deterministic_algorithms(False)
except: pass
try: torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
except: pass


# ==============================
# Model (GIỮ TÊN: LSTMModel)
# ==============================
class LSTMModel(nn.Module):
    """
    Nâng cấp bên trong:
      - Reshape x: [B, T, D=F*L] -> [B, T, L, F]
      - LSTM "sharing weights" cho từng link: (B*L, T, F) -> (B*L, H)
      - Gộp về [B, L, H] rồi chạy self-attention (TransformerEncoder) across links
      - Head per-link -> [B, L]
    """
    def __init__(self,
                 input_size,              # D = F*L (giữ tham số cũ để tương thích): số chiều của 1 timestamp (số feature * số link)
                 hidden_size=256,
                 num_layers=2,
                 dropout=0.2,
                 output_size=None,        # = L
                 features_per_link=None,  # F
                 num_links=None,          # L
                 attn_layers=2,
                 attn_heads=4):
        super().__init__()
        assert features_per_link is not None and num_links is not None, \
            "Cần truyền features_per_link (F) và num_links (L) vào LSTMModel."

        self.D = input_size
        self.F = features_per_link
        self.L = num_links
        self.H = hidden_size
        self.num_layers = num_layers

        if output_size is None:
            output_size = self.L  # dự đoán 1 target per link

        # Per-link temporal encoder
        self.lstm = nn.LSTM(
            input_size=self.F,
            hidden_size=self.H,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        # Cross-link self-attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.H, nhead=attn_heads, dim_feedforward=self.H * 2,
            dropout=0.2,  # ✅ Về lại 0.2 (original)
            activation='gelu', batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=attn_layers)

        # Head per-link
        self.head = nn.Sequential(
            nn.LayerNorm(self.H),
            nn.Linear(self.H, self.H // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.H // 2, 1)  # -> 1 target per link
        )

    def forward(self, x):
        # x: [B, T, D]
        B, T, D = x.shape
        assert D == self.F * self.L, f"Input D={D} không khớp F*L={self.F*self.L}"

        # [B, T, L, F]
        x = x.view(B, T, self.L, self.F)

        # Gom link vào batch: [B*L, T, F]
        x = x.permute(0, 2, 1, 3).contiguous().view(B * self.L, T, self.F)

        # Per-link LSTM
        out, _ = self.lstm(x)      # [B*L, T, H]
        last = out[:, -1, :]       # [B*L, H]

        # [B, L, H]
        h = last.view(B, self.L, self.H)

        # Cross-link attention
        h = self.encoder(h)        # [B, L, H]

        # Head per-link -> [B, L]
        y = self.head(h).squeeze(-1)
        return y


# =================================
# Trainer (GIỮ TÊN: LSTMTrainer)
# =================================
class LSTMTrainer:
    def __init__(self, model, device='cpu', use_amp=True):
        self.model = model.to(device)
        self.device = device
        self.use_amp = use_amp and (device.type == 'cuda')
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.train_losses = []
        self.val_losses = []

    def _step(self, xb, yb, criterion, train=False, optimizer: optim.Optimizer = None):
        if train:
            self.model.train()
            optimizer.zero_grad(set_to_none=True)
            # ✅ Không thêm noise (original)
        else:
            self.model.eval()

        with torch.cuda.amp.autocast(enabled=self.use_amp):
            pred = self.model(xb)
            loss = criterion(pred, yb)

        if train:
            self.scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(optimizer); self.scaler.update()

        return loss.item()

    def train_epoch(self, loader, optimizer, criterion):
        total = 0.0
        for xb, yb in tqdm(loader, desc="Training", leave=False):
            xb, yb = xb.to(self.device, non_blocking=True), yb.to(self.device, non_blocking=True)
            total += self._step(xb, yb, criterion, train=True, optimizer=optimizer)
        return total / max(1, len(loader))

    def validate_epoch(self, loader, criterion):
        total = 0.0
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(self.device, non_blocking=True), yb.to(self.device, non_blocking=True)
                total += self._step(xb, yb, criterion, train=False, optimizer=None)
        return total / max(1, len(loader))

    def train(self, train_loader, val_loader, epochs=300, lr=1e-3, patience=30, weight_decay=1e-4):
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.MSELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=6, factor=0.5)

        best_val = float('inf'); bad = 0
        print(f"Starting training for {epochs} epochs on {self.device} | AMP={self.use_amp}")

        for ep in range(1, epochs + 1):
            tr = self.train_epoch(train_loader, optimizer, criterion)
            va = self.validate_epoch(val_loader, criterion)
            scheduler.step(va)

            self.train_losses.append(tr); self.val_losses.append(va)

            os.makedirs('models', exist_ok=True)
            torch.save({'epoch': ep,
                        'model_state': self.model.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                        'train_loss': tr, 'val_loss': va}, 'models/last_checkpoint.pth')

            if va < best_val - 1e-7:
                best_val = va; bad = 0
                torch.save(self.model.state_dict(), 'models/best_lstm_model.pth'); flag = " (best)"
            else:
                bad += 1; flag = ""

            if ep == 1 or ep % 10 == 0 or flag:
                print(f"[{ep:03d}] train={tr:.6f} | val={va:.6f} | best={best_val:.6f} | bad={bad:02d} "
                      f"| lr={optimizer.param_groups[0]['lr']:.2e}{flag}")

            if bad >= patience:
                print(f"Early stopping at epoch {ep}")
                break

        self.model.load_state_dict(torch.load('models/best_lstm_model.pth', map_location=self.device))
        print("Training finished. Best model restored.")
        return self.train_losses, self.val_losses


# ==================
# Evaluator helpers
# ==================
class LSTMEvaluator:
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device

    def predict(self, loader):
        self.model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for xb, yb in tqdm(loader, desc="Predicting"):
                xb = xb.to(self.device, non_blocking=True)
                out = self.model(xb).cpu().numpy()
                preds.append(out); trues.append(yb.numpy())
        y_pred = np.vstack(preds); y_true = np.vstack(trues)
        return y_true, y_pred

    def evaluate_metrics(self, y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        rmse = float(np.sqrt(mse))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        if y_true.ndim == 2 and y_true.shape[1] > 1:
            per_link_mse = np.mean((y_true - y_pred) ** 2, axis=0)
            per_link_mae = np.mean(np.abs(y_true - y_pred), axis=0)
        else:
            per_link_mse = np.array([mse]); per_link_mae = np.array([mae])
        return {
            'mse': float(mse), 'rmse': rmse, 'mae': float(mae), 'r2': float(r2),
            'per_link_mse': per_link_mse, 'per_link_mae': per_link_mae
        }

    def plot_results(self, y_true, y_pred, link_names=None, save_path='results/lstm_results.png'):
        os.makedirs('results', exist_ok=True)
        plt.style.use('default'); sns.set_palette("husl")

        if y_true.ndim == 2 and y_true.shape[1] > 1:
            n = min(4, y_true.shape[1])
            fig, axes = plt.subplots(2, 2, figsize=(15, 10)); axes = axes.flatten()
            for i in range(n):
                ax = axes[i]; n_s = min(200, len(y_true))
                ax.plot(y_true[:n_s, i], label='True', alpha=0.9, lw=1.5)
                ax.plot(y_pred[:n_s, i], label='Pred', alpha=0.9, lw=1.5)
                ax.set_title(link_names[i] if link_names else f'Link {i}', fontweight='bold')
                ax.set_xlabel('Time'); ax.set_ylabel('Target'); ax.grid(True, alpha=0.3); ax.legend()
            plt.tight_layout(); plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.show()
        else:
            plt.figure(figsize=(12, 6)); n_s = min(200, len(y_true))
            plt.plot(y_true[:n_s], label='True', alpha=0.9, lw=1.5)
            plt.plot(y_pred[:n_s], label='Pred', alpha=0.9, lw=1.5)
            plt.title('LSTM Predictions vs True', fontweight='bold'); plt.xlabel('Time'); plt.ylabel('Target')
            plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.show()

    def plot_scatter(self, y_true, y_pred, save_path='results/lstm_scatter.png'):
        os.makedirs('results', exist_ok=True)
        plt.figure(figsize=(10, 8))
        t, p = y_true.flatten(), y_pred.flatten()
        plt.scatter(t, p, alpha=0.4, s=3)
        mn, mx = min(t.min(), p.min()), max(t.max(), p.max())
        plt.plot([mn, mx], [mn, mx], 'r--', lw=2, label='Perfect')
        plt.xlabel('True'); plt.ylabel('Pred'); plt.title('Predictions vs True')
        plt.grid(True, alpha=0.3); plt.legend()
        plt.tight_layout(); plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.show()


# ==================
# Data utils
# ==================
def load_data():
    print("Loading preprocessed data...")
    X_train = np.load('data/X_train.npy'); y_train = np.load('data/y_train.npy')
    X_val   = np.load('data/X_val.npy');   y_val   = np.load('data/y_val.npy')
    X_test  = np.load('data/X_test.npy');  y_test  = np.load('data/y_test.npy')

    with open('data/features.json', 'r') as f: feat_meta = json.load(f)
    model_features = feat_meta.get('model_features', [])
    target_feature = feat_meta.get('target_feature', 'utilization')

    with open('data/link_index.json', 'r') as f: link_names = json.load(f)

    print("Shapes:")
    print(f"  X_train: {X_train.shape} | y_train: {y_train.shape}")
    print(f"  X_val  : {X_val.shape}   | y_val  : {y_val.shape}")
    print(f"  X_test : {X_test.shape}  | y_test : {y_test.shape}")
    print(f"  features({len(model_features)}): {model_features}")
    print(f"  target_feature: {target_feature}")
    print(f"  links({len(link_names)}): first 5 -> {link_names[:5]}")
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), model_features, target_feature, link_names


def create_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size=128):
    X_train_t, y_train_t = torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float()
    X_val_t,   y_val_t   = torch.from_numpy(X_val).float(),   torch.from_numpy(y_val).float()
    X_test_t,  y_test_t  = torch.from_numpy(X_test).float(),  torch.from_numpy(y_test).float()
    tr = TensorDataset(X_train_t, y_train_t); va = TensorDataset(X_val_t, y_val_t); te = TensorDataset(X_test_t, y_test_t)
    pin = True
    train_loader = DataLoader(tr, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=pin)
    val_loader   = DataLoader(va, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin)
    test_loader  = DataLoader(te, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin)
    return train_loader, val_loader, test_loader


def plot_training_curves(train_losses, val_losses, save_path='results/training_curves.png'):
    os.makedirs('results', exist_ok=True)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train', alpha=0.9)
    plt.plot(val_losses, label='Val', alpha=0.9)
    plt.title('Loss (MSE)'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.grid(True, alpha=0.3); plt.legend()

    plt.subplot(1, 2, 2)
    window = max(3, min(10, len(train_losses) // 10)) if len(train_losses) > 10 else 3
    plt.plot(pd.Series(train_losses).rolling(window, min_periods=1).mean(), label=f'Train (smooth {window})', alpha=0.95)
    plt.plot(pd.Series(val_losses).rolling(window, min_periods=1).mean(),   label=f'Val (smooth {window})',   alpha=0.95)
    plt.title('Smoothed'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.grid(True, alpha=0.3); plt.legend()
    plt.tight_layout(); plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.show()


# ==================
# Main
# ==================
def main():
    print("===== LSTM (Per-Link + Cross-Link Attention) =====")
    os.makedirs('models', exist_ok=True); os.makedirs('results', exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")

    # >>> FIXED unpack (KHÔNG có dấu phẩy thừa) <<<
    (X_train, y_train), (X_val, y_val), (X_test, y_test), model_features, target_feature, link_names = load_data()

    # Dims
    D = X_train.shape[-1]
    L = y_train.shape[-1]
    assert D % L == 0, f"input_size={D} không chia hết cho L={L}"
    F = D // L
    T = X_train.shape[1]

    batch_size = 128 if device.type == 'cuda' else 64
    train_loader, val_loader, test_loader = create_dataloaders(
        X_train, y_train, X_val, y_val, X_test, y_test, batch_size=batch_size
    )

    print(f"Model dims: F={F} | L={L} | T={T} | D={D} | batch={batch_size}")

    # Build model (GIỮ TÊN class)
    model = LSTMModel(
        input_size=D,
        hidden_size=256,  # ✅ Về lại 256 (original)
        num_layers=2,     # ✅ Về lại 2 layers (original)
        dropout=0.2,      # ✅ Về lại 0.2 (original)
        output_size=L,
        features_per_link=F,
        num_links=L,
        attn_layers=2,    # ✅ Về lại 2 (original)
        attn_heads=4      # ✅ Về lại 4 (original)
    )

    # Param stats
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: total={total_params:,} | trainable={trainable_params:,}")

    # Train (GIỮ TÊN class)
    trainer = LSTMTrainer(model, device=device, use_amp=True)
    train_losses, val_losses = trainer.train(
        train_loader, val_loader,
        epochs=300,       # ✅ Về lại 300 (original)
        lr=1e-3,          # ✅ Về lại 1e-3 (original)
        patience=30,      # ✅ Về lại 30 (original)
        weight_decay=1e-4 # ✅ Về lại 1e-4 (original)
    )
    plot_training_curves(train_losses, val_losses)

    # Evaluate
    print("\nEvaluating on TEST ...")
    evaluator = LSTMEvaluator(model, device=device)
    y_true, y_pred = evaluator.predict(test_loader)

    # Metrics (scaled)
    metrics = evaluator.evaluate_metrics(y_true, y_pred)
    print("\nTest (scaled): "
          f"MSE={metrics['mse']:.6f} | RMSE={metrics['rmse']:.6f} | "
          f"MAE={metrics['mae']:.6f} | R2={metrics['r2']:.6f}")

    # Optional: real-scale metrics (nếu có scaler)
    real_metrics = None
    try:
        import joblib
        scalers = joblib.load('models/wide_scalers.pkl')
        scaler = scalers.get(target_feature, None)
        if scaler is not None:
            y_true_real = scaler.inverse_transform(y_true)
            y_pred_real = scaler.inverse_transform(y_pred)
            real_metrics = evaluator.evaluate_metrics(y_true_real, y_pred_real)
            print("Test (real):  "
                  f"MSE={real_metrics['mse']:.6f} | RMSE={real_metrics['rmse']:.6f} | "
                  f"MAE={real_metrics['mae']:.6f} | R2={real_metrics['r2']:.6f}")
        else:
            print(f"(Skip real metrics: no scaler for '{target_feature}')")
    except Exception as e:
        print(f"(Skip real metrics: {e})")

    # Plots
    evaluator.plot_results(y_true, y_pred, link_names=link_names)
    evaluator.plot_scatter(y_true, y_pred)

    # Save arrays + json
    np.save('results/y_true.npy', y_true); np.save('results/y_pred.npy', y_pred)
    pd.DataFrame(y_true, columns=link_names).to_csv('results/y_true.csv', index=False)
    pd.DataFrame(y_pred, columns=link_names).to_csv('results/y_pred.csv', index=False)

    results = {
        'metrics_scaled': {
            'mse': metrics['mse'],
            'rmse': metrics['rmse'],
            'mae': metrics['mae'],
            'r2': metrics['r2'],
            'per_link_mse': metrics['per_link_mse'].tolist(),
            'per_link_mae': metrics['per_link_mae'].tolist(),
        },
        'metrics_real': None if real_metrics is None else {
            'mse': real_metrics['mse'],
            'rmse': real_metrics['rmse'],
            'mae': real_metrics['mae'],
            'r2': real_metrics['r2'],
            'per_link_mse': real_metrics['per_link_mse'].tolist(),
            'per_link_mae': real_metrics['per_link_mae'].tolist(),
        },
        'model_config': {
            'arch': 'LSTM(per-link)+Transformer(cross-link)',
            'features_per_link': int(F),
            'num_links': int(L),
            'hidden_size': 256,
            'num_layers': 2,
            'attn_layers': 2,
            'attn_heads': 4,
            'dropout': 0.2,
        },
        'training_config': {
            'epochs': len(train_losses),
            'batch_size': batch_size,
            'lr': 1e-3,
            'patience': 30,
            'weight_decay': 1e-4,
            'device': str(device),
            'amp': device.type == 'cuda',
        },
        'data_info': {
            'features': model_features,
            'target_feature': target_feature,
            'link_names': link_names,
            'seq_len': int(T),
            'train_samples': int(len(X_train)),
            'val_samples': int(len(X_val)),
            'test_samples': int(len(X_test)),
        }
    }
    with open('results/lstm_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\nSaved:")
    print("  models/best_lstm_model.pth, models/last_checkpoint.pth")
    print("  results/lstm_results.json, y_true|y_pred.(npy|csv)")
    print("  results/training_curves.png, lstm_results.png, lstm_scatter.png")


if __name__ == "__main__":
    main()
