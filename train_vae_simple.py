# -*- coding: utf-8 -*-
"""
Simple VAE for Network Traffic Prediction
-----------------------------------------
Minimalist architecture focusing on prediction only:
- Simple Bi-LSTM encoder
- No reconstruction decoder (prediction-only VAE)
- Single timestep prediction (match LSTM task)
- ~500K params (vs 8.57M in AdvancedHybridVAE)

Target: R² > 0.60 (realistic for VAE)
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.amp import autocast, GradScaler
import numpy as np
import json
import os
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ============================================================================
# Simple VAE Architecture
# ============================================================================

class SimpleVAE(nn.Module):
    """
    Minimalist VAE focusing on prediction
    
    Architecture:
    1. Bi-LSTM Encoder → latent (μ, σ)
    2. NO Decoder (no reconstruction)
    3. Simple Predictor: latent → future utilization (1 timestep)
    
    Total params: ~500K (vs 8.57M in AdvancedHybridVAE)
    """
    def __init__(self, input_dim, latent_dim=32, hidden_dim=128, 
                 num_links=12, dropout=0.5):
        super().__init__()
        
        # 1. Bi-LSTM Encoder (simple, proven to work)
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=3,        # ✅ Tăng từ 2 -> 3 layers (deeper)
            dropout=dropout,
            batch_first=True,
            bidirectional=True
        )
        
        self.norm1 = nn.LayerNorm(hidden_dim * 2)
        
        # 2. Attention (optional but helpful)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,
            num_heads=8,         # ✅ Tăng từ 4 -> 8 heads (more attention)
            dropout=dropout,
            batch_first=True
        )
        
        # 3. Latent projection
        self.fc_mu = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 2, latent_dim)
        
        # 4. Simple predictor (latent → prediction)
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_links)
        )
        
    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        """
        x: (batch, seq_len=96, input_dim)
        Returns: pred, mu, logvar
        """
        # 1. Encode
        encoded, _ = self.encoder(x)  # (batch, 96, hidden*2)
        encoded = self.norm1(encoded)
        
        # 2. Attention
        attn_out, _ = self.attention(encoded, encoded, encoded)
        attn_out = attn_out + encoded  # Residual
        
        # 3. Pool to get sequence representation
        pooled = attn_out.mean(dim=1)  # (batch, hidden*2)
        
        # 4. Latent
        mu = self.fc_mu(pooled)          # (batch, latent_dim)
        logvar = self.fc_logvar(pooled)  # (batch, latent_dim)
        
        # 5. Sample latent
        z = self.reparameterize(mu, logvar)
        
        # 6. Predict (single timestep)
        pred = self.predictor(z)  # (batch, num_links)
        
        return pred, mu, logvar


# ============================================================================
# Training
# ============================================================================

class SimpleVAETrainer:
    """
    Single-stage training (no multi-stage complexity)
    Loss = prediction_loss + β * KL_loss
    """
    def __init__(self, model, train_loader, val_loader, device='cuda', 
                 lr=1e-3, weight_decay=1e-3, beta=0.1):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.beta = beta
        
        # AMP for CUDA
        self.use_amp = (device == 'cuda')
        if self.use_amp:
            self.scaler = GradScaler('cuda')
            print(f"   ⚡ Mixed precision enabled")
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True
        )
        
        # History
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_pred': [], 'train_kl': [],
            'val_r2': [], 'val_mae': []
        }
        self.best_r2 = -float('inf')
        self.best_val_loss = float('inf')
        
    def compute_loss(self, pred, y_true, mu, logvar):
        """
        Loss = prediction_loss + β * KL_loss
        """
        batch_size = pred.size(0)
        
        # 1. Prediction loss (MSE)
        pred_loss = F.mse_loss(pred, y_true, reduction='mean')
        
        # 2. KL divergence (regularization)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_loss / (batch_size * mu.size(1))
        
        # Total
        total_loss = pred_loss + self.beta * kl_loss
        
        return total_loss, pred_loss, kl_loss
    
    def train_epoch(self):
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        total_pred = 0
        total_kl = 0
        
        for X_batch, y_batch in tqdm(self.train_loader, desc="Training", leave=False):
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with autocast('cuda'):
                    pred, mu, logvar = self.model(X_batch)
                    loss, pred_loss, kl_loss = self.compute_loss(pred, y_batch, mu, logvar)
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                pred, mu, logvar = self.model(X_batch)
                loss, pred_loss, kl_loss = self.compute_loss(pred, y_batch, mu, logvar)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            
            total_loss += loss.item()
            total_pred += pred_loss.item()
            total_kl += kl_loss.item()
        
        n = len(self.train_loader)
        return {
            'loss': total_loss / n,
            'pred': total_pred / n,
            'kl': total_kl / n
        }
    
    def validate(self):
        """Validation"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for X_batch, y_batch in self.val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                pred, mu, logvar = self.model(X_batch)
                loss, _, _ = self.compute_loss(pred, y_batch, mu, logvar)
                
                total_loss += loss.item()
                all_preds.append(pred.cpu().numpy())
                all_targets.append(y_batch.cpu().numpy())
        
        preds = np.concatenate(all_preds, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        
        r2 = r2_score(targets.flatten(), preds.flatten())
        mae = mean_absolute_error(targets.flatten(), preds.flatten())
        
        return total_loss / len(self.val_loader), r2, mae
    
    def train(self, epochs=100, patience=20):
        """Main training loop"""
        print(f"Starting training for {epochs} epochs on {self.device} | AMP={self.use_amp}")
        
        no_improve = 0
        
        for epoch in range(1, epochs + 1):
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_loss, val_r2, val_mae = self.validate()
            
            # Log
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_pred'].append(train_metrics['pred'])
            self.history['train_kl'].append(train_metrics['kl'])
            self.history['val_loss'].append(val_loss)
            self.history['val_r2'].append(val_r2)
            self.history['val_mae'].append(val_mae)
            
            # Early stopping based on R²
            if val_r2 > self.best_r2:
                self.best_r2 = val_r2
                no_improve = 0
                os.makedirs('models', exist_ok=True)
                torch.save(self.model.state_dict(), 'models/simple_vae_best.pth')
                flag = " (best)"
            else:
                no_improve += 1
                flag = ""
            
            # Print progress (giống LSTM)
            if epoch == 1 or epoch % 10 == 0 or flag:
                print(f"[{epoch:03d}] train={train_metrics['loss']:.6f} "
                      f"(pred={train_metrics['pred']:.6f}, kl={train_metrics['kl']:.4f}) | "
                      f"val={val_loss:.6f} | R²={val_r2:.4f} | MAE={val_mae:.4f} | "
                      f"bad={no_improve:02d}{flag}")
            
            # Scheduler
            self.scheduler.step(val_loss)
            
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        self.model.load_state_dict(torch.load('models/simple_vae_best.pth', map_location=self.device))
        print("Training finished. Best model restored.")
        
        return self.best_r2
    
    def plot_history(self, save_dir='results-vae'):
        """Plot training history (giống LSTM format)"""
        os.makedirs(save_dir, exist_ok=True)
        
        fig = plt.figure(figsize=(12, 5))
        
        # Loss
        plt.subplot(1, 2, 1)
        plt.plot(self.history['train_loss'], label='Train', alpha=0.9)
        plt.plot(self.history['val_loss'], label='Val', alpha=0.9)
        plt.title('Loss (Total)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Smoothed
        plt.subplot(1, 2, 2)
        window = max(3, min(10, len(self.history['train_loss']) // 10)) if len(self.history['train_loss']) > 10 else 3
        plt.plot(pd.Series(self.history['train_loss']).rolling(window, min_periods=1).mean(), 
                label=f'Train (smooth {window})', alpha=0.95)
        plt.plot(pd.Series(self.history['val_loss']).rolling(window, min_periods=1).mean(), 
                label=f'Val (smooth {window})', alpha=0.95)
        plt.title('Smoothed')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\n📊 Training curves saved to {save_dir}/training_curves.png")
    
    def plot_results(self, y_true, y_pred, link_names=None, save_dir='results-vae'):
        """Plot predictions (giống LSTM format)"""
        os.makedirs(save_dir, exist_ok=True)
        plt.style.use('default')
        sns.set_palette("husl")
        
        if y_true.ndim == 2 and y_true.shape[1] > 1:
            n = min(4, y_true.shape[1])
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes = axes.flatten()
            for i in range(n):
                ax = axes[i]
                n_s = min(200, len(y_true))
                ax.plot(y_true[:n_s, i], label='True', alpha=0.9, lw=1.5)
                ax.plot(y_pred[:n_s, i], label='Pred', alpha=0.9, lw=1.5)
                ax.set_title(link_names[i] if link_names else f'Link {i}', fontweight='bold')
                ax.set_xlabel('Time')
                ax.set_ylabel('Target')
                ax.grid(True, alpha=0.3)
                ax.legend()
            plt.tight_layout()
            plt.savefig(f'{save_dir}/vae_results.png', dpi=300, bbox_inches='tight')
            plt.close()
        print(f"📊 Results plot saved to {save_dir}/vae_results.png")
    
    def plot_scatter(self, y_true, y_pred, save_dir='results-vae'):
        """Plot scatter (giống LSTM format)"""
        os.makedirs(save_dir, exist_ok=True)
        plt.figure(figsize=(10, 8))
        t, p = y_true.flatten(), y_pred.flatten()
        plt.scatter(t, p, alpha=0.4, s=3)
        mn, mx = min(t.min(), p.min()), max(t.max(), p.max())
        plt.plot([mn, mx], [mn, mx], 'r--', lw=2, label='Perfect')
        plt.xlabel('True')
        plt.ylabel('Pred')
        plt.title('VAE Predictions vs True')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{save_dir}/vae_scatter.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 Scatter plot saved to {save_dir}/vae_scatter.png")


# ============================================================================
# Main
# ============================================================================

def main():
    set_seed(42)
    
    print("===== Simple VAE (Prediction-Only) =====")
    
    # Load data
    print("\nLoading preprocessed data...")
    X_train = np.load('data/X_train.npy')
    y_train = np.load('data/y_train.npy')
    X_val = np.load('data/X_val.npy')
    y_val = np.load('data/y_val.npy')
    X_test = np.load('data/X_test.npy')
    y_test = np.load('data/y_test.npy')
    
    with open('data/features.json', 'r') as f:
        feat_meta = json.load(f)
    model_features = feat_meta.get('model_features', [])
    target_feature = feat_meta.get('target_feature', 'utilization')
    
    with open('data/link_index.json', 'r') as f:
        link_names = json.load(f)
    
    print("Shapes:")
    print(f"  X_train: {X_train.shape} | y_train: {y_train.shape}")
    print(f"  X_val  : {X_val.shape}   | y_val  : {y_val.shape}")
    print(f"  X_test : {X_test.shape}  | y_test : {y_test.shape}")
    print(f"  features({len(model_features)}): {model_features}")
    print(f"  target_feature: {target_feature}")
    print(f"  links({len(link_names)}): first 5 -> {link_names[:5]}")
    
    input_dim = X_train.shape[2]
    num_links = y_train.shape[1]
    
    # Create dataloaders
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, 
                             num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, 
                           num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, 
                            num_workers=0, pin_memory=True)
    
    # Model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    if device == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    model = SimpleVAE(
        input_dim=input_dim,
        latent_dim=96,
        hidden_dim=256,
        num_links=num_links,
        dropout=0.25
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: total={num_params:,} | trainable={trainable_params:,}")
    
    # Train
    trainer = SimpleVAETrainer(
        model, train_loader, val_loader, device,
        lr=5e-4,
        weight_decay=2e-4,
        beta=0.02
    )
    
    best_r2 = trainer.train(epochs=200, patience=30)
    
    # Plot training curves
    trainer.plot_history()
    
    # Test evaluation
    print("\nEvaluating on TEST ...")
    model.load_state_dict(torch.load('models/simple_vae_best.pth', weights_only=True))
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in tqdm(test_loader, desc="Predicting"):
            X_batch = X_batch.to(device)
            pred, _, _ = model(X_batch)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y_batch.numpy())
    
    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_targets, axis=0)
    
    # Metrics
    test_r2 = r2_score(y_true.flatten(), y_pred.flatten())
    test_mae = mean_absolute_error(y_true.flatten(), y_pred.flatten())
    test_mse = mean_squared_error(y_true.flatten(), y_pred.flatten())
    test_rmse = np.sqrt(test_mse)
    
    # Per-link metrics
    per_link_mse = np.mean((y_true - y_pred) ** 2, axis=0)
    per_link_mae = np.mean(np.abs(y_true - y_pred), axis=0)
    
    print("\nTest (scaled): "
          f"MSE={test_mse:.6f} | RMSE={test_rmse:.6f} | "
          f"MAE={test_mae:.6f} | R2={test_r2:.6f}")
    
    # Plots
    trainer.plot_results(y_true, y_pred, link_names=link_names)
    trainer.plot_scatter(y_true, y_pred)
    
    # Save arrays + csv
    os.makedirs('results-vae', exist_ok=True)
    np.save('results-vae/y_true.npy', y_true)
    np.save('results-vae/y_pred.npy', y_pred)
    pd.DataFrame(y_true, columns=link_names).to_csv('results-vae/y_true.csv', index=False)
    pd.DataFrame(y_pred, columns=link_names).to_csv('results-vae/y_pred.csv', index=False)
    
    # Save JSON results (giống LSTM format)
    results = {
        'metrics_scaled': {
            'mse': float(test_mse),
            'rmse': float(test_rmse),
            'mae': float(test_mae),
            'r2': float(test_r2),
            'per_link_mse': per_link_mse.tolist(),
            'per_link_mae': per_link_mae.tolist(),
        },
        'metrics_real': None,
        'model_config': {
            'arch': 'SimpleVAE(Bi-LSTM+Attention+Predictor)',
            'features_per_link': int(input_dim),
            'num_links': int(num_links),
            'hidden_dim': 256,
            'latent_dim': 96,
            'num_layers': 3,
            'attn_heads': 8,
            'dropout': 0.25,
        },
        'training_config': {
            'epochs': len(trainer.history['train_loss']),
            'batch_size': 128,
            'lr': 5e-4,
            'patience': 30,
            'weight_decay': 2e-4,
            'beta': 0.02,
            'device': str(device),
            'amp': device == 'cuda',
        },
        'data_info': {
            'features': model_features,
            'target_feature': target_feature,
            'link_names': link_names,
            'seq_len': int(X_train.shape[1]),
            'train_samples': int(len(X_train)),
            'val_samples': int(len(X_val)),
            'test_samples': int(len(X_test)),
        }
    }
    
    with open('results-vae/vae_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nSaved:")
    print("  models/simple_vae_best.pth")
    print("  results-vae/vae_results.json, y_true|y_pred.(npy|csv)")
    print("  results-vae/training_curves.png, vae_results.png, vae_scatter.png")


if __name__ == '__main__':
    main()
