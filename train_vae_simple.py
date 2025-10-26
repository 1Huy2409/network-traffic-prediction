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
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.amp import autocast, GradScaler
import numpy as np
import json
import os
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt


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
            num_layers=2,
            dropout=dropout,
            batch_first=True,
            bidirectional=True
        )
        
        self.norm1 = nn.LayerNorm(hidden_dim * 2)
        
        # 2. Attention (optional but helpful)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,
            num_heads=4,
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
        
        for X_batch, y_batch in self.train_loader:
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
        print(f"\n{'='*70}")
        print(f"🚀 SIMPLE VAE TRAINING")
        print(f"{'='*70}")
        print(f"Device: {self.device}")
        print(f"Epochs: {epochs}, Patience: {patience}")
        print(f"Beta (KL weight): {self.beta}")
        print(f"{'='*70}\n")
        
        no_improve = 0
        
        for epoch in range(epochs):
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
            
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train: {train_metrics['loss']:.4f} "
                  f"(pred={train_metrics['pred']:.4f}, kl={train_metrics['kl']:.4f}) | "
                  f"Val: {val_loss:.4f} | "
                  f"R²: {val_r2:.4f} | "
                  f"MAE: {val_mae:.4f}")
            
            # Scheduler
            self.scheduler.step(val_loss)
            
            # Early stopping based on R²
            if val_r2 > self.best_r2:
                self.best_r2 = val_r2
                no_improve = 0
                os.makedirs('models', exist_ok=True)
                torch.save(self.model.state_dict(), 'models/simple_vae_best.pth')
                print(f"   ✅ New best R²: {self.best_r2:.4f}")
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"\n   ⚠️ Early stopping (no improve for {patience} epochs)")
                    break
        
        print(f"\n{'='*70}")
        print(f"🎉 TRAINING COMPLETE!")
        print(f"   Best Val R²: {self.best_r2:.4f}")
        print(f"{'='*70}")
        
        return self.best_r2
    
    def plot_history(self, save_dir='results-vae'):
        """Plot training history"""
        os.makedirs(save_dir, exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(self.history['train_loss'], label='Train')
        axes[0, 0].plot(self.history['val_loss'], label='Val')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training & Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Loss components
        axes[0, 1].plot(self.history['train_pred'], label='Pred Loss')
        axes[0, 1].plot(self.history['train_kl'], label='KL Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Loss Components')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # R²
        axes[1, 0].plot(self.history['val_r2'], label='Val R²', color='green')
        axes[1, 0].axhline(y=0.60, color='orange', linestyle='--', label='Target (0.60)')
        axes[1, 0].axhline(y=0.894, color='r', linestyle='--', label='LSTM (0.894)')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('R²')
        axes[1, 0].set_title('Validation R²')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # MAE
        axes[1, 1].plot(self.history['val_mae'], color='purple')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('MAE')
        axes[1, 1].set_title('Validation MAE')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/simple_vae_history.png', dpi=150, bbox_inches='tight')
        print(f"\n📊 Training history saved to {save_dir}/simple_vae_history.png")
        plt.close()


# ============================================================================
# Main
# ============================================================================

def main():
    set_seed(42)
    
    print("=" * 70)
    print("SIMPLE VAE TRAINING")
    print("=" * 70)
    
    # Load data (use LSTM data - single timestep prediction)
    print("\n📂 Loading data...")
    X_train = np.load('data/X_train.npy')
    y_train = np.load('data/y_train.npy')
    X_val = np.load('data/X_val.npy')
    y_val = np.load('data/y_val.npy')
    X_test = np.load('data/X_test.npy')
    y_test = np.load('data/y_test.npy')
    
    print(f"  X_train: {X_train.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  X_val:   {X_val.shape}")
    print(f"  y_val:   {y_val.shape}")
    print(f"  X_test:  {X_test.shape}")
    print(f"  y_test:  {y_test.shape}")
    
    input_dim = X_train.shape[2]
    num_links = y_train.shape[1]
    
    print(f"\n📐 Model config:")
    print(f"  Input dim: {input_dim}")
    print(f"  Num links: {num_links}")
    
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
    print(f"\n🖥️  Device: {device}")
    
    if device == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    model = SimpleVAE(
        input_dim=input_dim,
        latent_dim=32,       # Small latent
        hidden_dim=128,      # Moderate hidden
        num_links=num_links,
        dropout=0.5          # High regularization
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\n📊 Model parameters: {num_params:,}")
    print(f"   Target: ~500K (Actual: {num_params/1000:.0f}K)")
    
    # Train
    trainer = SimpleVAETrainer(
        model, train_loader, val_loader, device,
        lr=1e-3,
        weight_decay=1e-3,  # Strong regularization
        beta=0.1            # Low KL weight
    )
    
    best_r2 = trainer.train(epochs=100, patience=20)
    
    # Plot
    trainer.plot_history()
    
    # Test evaluation
    print("\n📊 Evaluating on test set...")
    model.load_state_dict(torch.load('models/simple_vae_best.pth', weights_only=True))
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            pred, _, _ = model(X_batch)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y_batch.numpy())
    
    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_targets, axis=0)
    
    # Metrics
    test_r2 = r2_score(y_true.flatten(), y_pred.flatten())
    test_mae = mean_absolute_error(y_true.flatten(), y_pred.flatten())
    test_rmse = np.sqrt(mean_squared_error(y_true.flatten(), y_pred.flatten()))
    
    print(f"\n{'='*70}")
    print(f"TEST SET RESULTS")
    print(f"{'='*70}")
    print(f"R²:   {test_r2:.4f}")
    print(f"MAE:  {test_mae:.4f}")
    print(f"RMSE: {test_rmse:.4f}")
    print(f"\nComparison:")
    print(f"  LSTM baseline: R² = 0.894")
    print(f"  Simple VAE:    R² = {test_r2:.4f}")
    print(f"  Gap:           {test_r2 - 0.894:.4f}")
    
    if test_r2 > 0.60:
        print(f"\n✅ SUCCESS: R² > 0.60 target!")
    elif test_r2 > 0.50:
        print(f"\n⚠️ Close to target, may need tuning")
    else:
        print(f"\n❌ Below target")
    print(f"{'='*70}")
    
    # Save results
    os.makedirs('results-vae', exist_ok=True)
    results = {
        'architecture': 'SimpleVAE',
        'num_params': int(num_params),
        'best_val_r2': float(best_r2),
        'test_r2': float(test_r2),
        'test_mae': float(test_mae),
        'test_rmse': float(test_rmse),
        'config': {
            'latent_dim': 32,
            'hidden_dim': 128,
            'dropout': 0.5,
            'beta': 0.1,
            'lr': 1e-3,
            'weight_decay': 1e-3
        }
    }
    
    with open('results-vae/simple_vae_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save predictions
    np.save('results-vae/simple_vae_y_pred.npy', y_pred)
    np.save('results-vae/simple_vae_y_true.npy', y_true)
    
    print(f"\n💾 Results saved to results-vae/simple_vae_results.json")


if __name__ == '__main__':
    main()
