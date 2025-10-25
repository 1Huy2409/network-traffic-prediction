"""
Train Variational Autoencoder (VAE) for Network Traffic Prediction
------------------------------------------------------------------
GenAI approach: VAE learns latent representation of traffic patterns
and generates predictions for SAGSIN network optimization.

Architecture:
- Encoder: Traffic snapshots → Latent space (μ, σ)
- Decoder: Latent samples → Reconstructed traffic  
- Predictor: Latent features → Future utilization

Comparison with LSTM baseline for research paper.
"""

import os
import json
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ==================== REPRODUCIBILITY ====================

def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ==================== ADVANCED VAE MODEL ====================

class SequenceEncoder(nn.Module):
    """
    ADVANCED: Encode entire sequence (not individual timesteps)
    Uses Bi-LSTM + Self-Attention for better temporal understanding
    """
    def __init__(self, input_dim, hidden_dim, latent_dim, dropout=0.3):
        super().__init__()
        
        # Bi-directional LSTM to capture temporal patterns
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            dropout=dropout,
            batch_first=True,
            bidirectional=True
        )
        
        # Self-attention to focus on important timesteps
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,  # bidirectional
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer norm for stability
        self.norm1 = nn.LayerNorm(hidden_dim * 2)
        self.norm2 = nn.LayerNorm(hidden_dim * 2)
        
        # Project to latent space
        self.fc_mu = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 2, latent_dim)
        
    def forward(self, x):
        """
        x: (batch, seq_len, input_dim)
        Returns: mu, logvar for latent distribution
        """
        # LSTM encoding
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden*2)
        lstm_out = self.norm1(lstm_out)
        
        # Self-attention (Q=K=V)
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = self.norm2(attn_out + lstm_out)  # Residual connection
        
        # Pool over sequence (use last timestep + mean)
        last = attn_out[:, -1, :]  # Last hidden state
        mean = attn_out.mean(dim=1)  # Mean pooling
        combined = (last + mean) / 2  # Combine both
        
        # Project to latent
        mu = self.fc_mu(combined)
        logvar = self.fc_logvar(combined)
        
        return mu, logvar, attn_out  # Return attn_out for skip connection


class HierarchicalDecoder(nn.Module):
    """
    ADVANCED: Hierarchical decoder with skip connections
    """
    def __init__(self, latent_dim, hidden_dim, seq_len, output_dim, dropout=0.3):
        super().__init__()
        
        self.seq_len = seq_len
        
        # Expand latent to sequence
        self.fc_expand = nn.Linear(latent_dim, hidden_dim * seq_len)
        
        # Decoder LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            dropout=dropout,
            batch_first=True
        )
        
        # Output projection
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, z, skip=None):
        """
        z: (batch, latent_dim)
        skip: (batch, seq_len, hidden_dim*2) from encoder
        """
        batch_size = z.size(0)
        
        # Expand to sequence
        h = self.fc_expand(z)  # (batch, hidden*seq_len)
        h = h.view(batch_size, self.seq_len, -1)  # (batch, seq_len, hidden)
        
        # Add skip connection if provided
        if skip is not None:
            # Project skip to match dimension
            skip_proj = skip[:, :, :h.size(2)]  # Take first hidden_dim channels
            h = h + skip_proj
        
        # LSTM decoding
        lstm_out, _ = self.lstm(h)
        lstm_out = self.dropout(lstm_out)
        
        # Project to output
        recon = self.fc_out(lstm_out)  # (batch, seq_len, output_dim)
        
        return recon


class HybridPredictor(nn.Module):
    """
    HYBRID: Combines VAE latent + direct raw input access (like LSTM)
    
    Best of both worlds:
    - VAE latent provides global compressed representation
    - Raw input LSTM provides temporal details (like LSTM baseline)
    
    This should achieve R² close to LSTM baseline!
    """
    def __init__(self, input_dim, latent_dim, hidden_dim, num_links, dropout=0.3):
        super().__init__()
        
        # LSTM processes raw input (like LSTM baseline does)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            dropout=dropout,
            batch_first=True,
            bidirectional=True
        )
        
        # Attention over LSTM outputs
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        # Fusion layer: combine LSTM output with VAE latent
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2 + latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Per-link prediction heads
        self.link_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1)
            )
            for _ in range(num_links)
        ])
        
    def forward(self, x, z):
        """
        x: (batch, seq_len, input_dim) - RAW INPUT (direct access!)
        z: (batch, latent_dim) - VAE latent (compressed global context)
        
        Returns: (batch, num_links) predictions
        """
        # Process raw input with LSTM (like LSTM baseline)
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden*2)
        
        # Self-attention to focus on important timesteps
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Use last timestep as sequence summary
        last_hidden = attn_out[:, -1, :]  # (batch, hidden*2)
        
        # Fuse LSTM output with VAE latent
        combined = torch.cat([last_hidden, z], dim=1)  # (batch, hidden*2 + latent)
        fused = self.fusion(combined)  # (batch, hidden)
        
        # Per-link predictions
        preds = []
        for head in self.link_heads:
            pred = head(fused)  # (batch, 1)
            preds.append(pred)
        
        return torch.cat(preds, dim=1)  # (batch, num_links)


class HybridTrafficVAE(nn.Module):
    """
    HYBRID VAE: Two-Stage Training with Hybrid Predictor
    
    Stage 1: Pre-train Encoder + Decoder (unsupervised reconstruction)
    Stage 2: Train Hybrid Predictor with frozen encoder (supervised prediction)
    
    Hybrid Predictor = VAE latent + Raw input LSTM (best of both worlds!)
    """
    def __init__(self, input_dim, seq_len, latent_dim, hidden_dim, num_links, dropout=0.3):
        super().__init__()
        
        self.encoder = SequenceEncoder(input_dim, hidden_dim, latent_dim, dropout)
        self.decoder = HierarchicalDecoder(latent_dim, hidden_dim, seq_len, input_dim, dropout)
        self.predictor = HybridPredictor(input_dim, latent_dim, hidden_dim, num_links, dropout)
        
    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x, predict=True):
        """
        x: (batch, seq_len, input_dim)
        Returns: recon, mu, logvar, pred (if predict=True)
        """
        # Encode sequence
        mu, logvar, encoder_out = self.encoder(x)
        
        # Sample latent (for decoder stochasticity)
        z = self.reparameterize(mu, logvar)

        # Decode with skip connection
        recon = self.decoder(z, skip=encoder_out)

        # Predict if needed - Hybrid predictor uses RAW INPUT + latent!
        pred = None
        if predict:
            pred = self.predictor(x, mu)  # Pass raw input x + deterministic latent mu
        
        return recon, mu, logvar, pred


# Alias for compatibility
TrafficVAE = HybridTrafficVAE


def vae_loss_function(recon, x, mu, logvar, pred, y_target, 
                      beta=1.0, pred_weight=1.0, stage=1):
    """
    VAE Loss = Reconstruction Loss + β * KL Divergence + Prediction Loss
    
    TWO-STAGE TRAINING:
    - Stage 1 (Unsupervised): recon_loss + beta * kl_loss (learn latent representation)
    - Stage 2 (Supervised): pred_loss only (fine-tune predictor with frozen encoder)
    
    Args:
    - recon: (batch, seq_len, input_dim) - reconstructed input
    - x: (batch, seq_len, input_dim) - original input
    - mu, logvar: (batch, latent_dim) - latent distribution parameters
    - pred: (batch, num_links) - predicted future traffic
    - y_target: (batch, num_links) - ground truth future traffic
    - beta: weight for KL divergence (β-VAE)
    - pred_weight: weight for prediction loss
    - stage: 1 (pre-train encoder/decoder) or 2 (fine-tune predictor)
    """
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(recon, x, reduction='mean')
    
    # KL divergence: KL(q(z|x) || p(z)) where p(z) = N(0, I)
    # KL = -0.5 * sum(1 + log(σ²) - μ² - σ²)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # Normalize by batch*latent_dim for stable scaling
    kl_loss = kl_loss / (mu.shape[0] * mu.shape[1])
    
    # Prediction loss (MSE for future traffic)
    pred_loss = F.mse_loss(pred, y_target, reduction='mean')
    
    # STAGE-SPECIFIC LOSS
    if stage == 1:
        # Stage 1: Focus on learning good latent representation
        # No prediction loss - encoder/decoder only
        total_loss = recon_loss + beta * kl_loss
    else:
        # Stage 2: Focus on prediction with frozen encoder
        # No reconstruction/KL loss - predictor only
        total_loss = pred_loss
    
    return total_loss, recon_loss, kl_loss, pred_loss


# ==================== DATASET ====================

class VAETrafficDataset(Dataset):
    """
    Dataset for VAE training
    
    Input: Sequence of traffic snapshots (wide format)
    Output: Future utilization per link
    """
    def __init__(self, X, y, seq_len=96):
        """
        X: (N, seq_len, features*links) - sequences of snapshots
        y: (N, num_links) - future utilization targets
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.seq_len = seq_len
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ==================== TRAINER ====================

class VAETrainer:
    """
    Two-Stage Trainer for VAE model
    
    Stage 1: Pre-train Encoder + Decoder (unsupervised)
    Stage 2: Fine-tune Predictor with frozen Encoder (supervised)
    """
    
    def __init__(self, model, train_loader, val_loader, device='cuda',
                 lr=1e-3, weight_decay=1e-4, beta=1.0, pred_weight=1.0, 
                 kl_warmup_epochs=10, stage=1):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Training stage (1 or 2)
        self.stage = stage
        
        # Stage-specific optimizer setup
        if stage == 1:
            # Stage 1: Train encoder + decoder only
            params = list(self.model.encoder.parameters()) + \
                     list(self.model.decoder.parameters())
            print(f'  Stage 1: Training Encoder + Decoder ({len(params)} param groups)')
        else:
            # Stage 2: Train predictor only (encoder/decoder frozen)
            self.freeze_encoder_decoder()
            params = self.model.predictor.parameters()
            print(f'  Stage 2: Training Predictor only (Encoder+Decoder frozen)')
        
        self.optimizer = torch.optim.AdamW(
            params, 
            lr=lr, 
            weight_decay=weight_decay
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
        )
        
        # AMP only for CUDA
        self.use_amp = device.startswith('cuda')
        if self.use_amp:
            self.scaler = GradScaler('cuda')
        
        self.beta = beta
        self.pred_weight = pred_weight
        self.kl_warmup_epochs = kl_warmup_epochs
        
        # History - track all 4 loss components
        self.train_losses = []
        self.val_losses = []
        self.train_recon_losses = []
        self.val_recon_losses = []
        self.train_kl_losses = []
        self.val_kl_losses = []
        self.train_pred_losses = []
        self.val_pred_losses = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.current_epoch = 0
    
    def freeze_encoder_decoder(self):
        """Freeze encoder and decoder for Stage 2"""
        for param in self.model.encoder.parameters():
            param.requires_grad = False
        for param in self.model.decoder.parameters():
            param.requires_grad = False
        print('  ✓ Encoder and Decoder frozen')
    
    def unfreeze_all(self):
        """Unfreeze all parameters"""
        for param in self.model.parameters():
            param.requires_grad = True
        print('  ✓ All parameters unfrozen')
        
    def train_epoch(self):
        """Train for one epoch (stage-aware)"""
        self.model.train()
        total_loss = 0
        total_recon = 0
        total_kl = 0
        total_pred = 0
        
        # KL annealing: warm up beta from 0 to target value (Stage 1 only)
        if self.stage == 1:
            beta_now = min(self.beta, self.beta * self.current_epoch / max(1, self.kl_warmup_epochs))
        else:
            beta_now = 0.0  # No KL loss in Stage 2
        
        for X_batch, y_batch in self.train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass with AMP if CUDA
            if self.use_amp:
                with autocast('cuda'):
                    recon, mu, logvar, pred = self.model(X_batch, predict=True)
                    loss, recon_loss, kl_loss, pred_loss = vae_loss_function(
                        recon, X_batch, mu, logvar, pred, y_batch,
                        beta=beta_now, pred_weight=self.pred_weight,
                        stage=self.stage  # Pass stage info
                    )
                
                # Backward with gradient scaling
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # CPU: no AMP
                recon, mu, logvar, pred = self.model(X_batch, predict=True)
                loss, recon_loss, kl_loss, pred_loss = vae_loss_function(
                    recon, X_batch, mu, logvar, pred, y_batch,
                    beta=beta_now, pred_weight=self.pred_weight,
                    stage=self.stage  # Pass stage info
                )
                
                # Backward without scaling
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
            
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            total_pred += pred_loss.item()
        
        n_batches = len(self.train_loader)
        return {
            'loss': total_loss / n_batches,
            'recon': total_recon / n_batches,
            'kl': total_kl / n_batches,
            'pred': total_pred / n_batches,
            'beta': beta_now
        }
    
    @torch.no_grad()
    def validate(self):
        """Validate model (stage-aware)"""
        self.model.eval()
        total_loss = 0
        total_recon = 0
        total_kl = 0
        total_pred = 0
        
        for X_batch, y_batch in self.val_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            # Forward pass
            recon, mu, logvar, pred = self.model(X_batch, predict=True)
            
            # Compute loss (stage-aware)
            beta_val = self.beta if self.stage == 1 else 0.0
            loss, recon_loss, kl_loss, pred_loss = vae_loss_function(
                recon, X_batch, mu, logvar, pred, y_batch,
                beta=beta_val, pred_weight=self.pred_weight,
                stage=self.stage  # Pass stage info
            )
            
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            total_pred += pred_loss.item()
        
        n_batches = len(self.val_loader)
        return {
            'loss': total_loss / n_batches,
            'recon': total_recon / n_batches,
            'kl': total_kl / n_batches,
            'pred': total_pred / n_batches
        }
    
    def train(self, epochs, patience=20):
        """Train model with early stopping (stage-aware)"""
        stage_name = "STAGE 1: Pre-training Encoder+Decoder" if self.stage == 1 else "STAGE 2: Fine-tuning Predictor"
        
        print(f'\n{"="*60}')
        print(f'{stage_name}')
        print(f'{"="*60}')
        print(f'Device: {self.device}')
        print(f'AMP: {self.use_amp}')
        print(f'Epochs: {epochs}, Patience: {patience}')
        
        if self.stage == 1:
            print(f'Focus: Learning latent representation (recon + KL loss)')
            print(f'Beta (KL weight): {self.beta} (warmup: {self.kl_warmup_epochs} epochs)')
        else:
            print(f'Focus: Prediction with frozen encoder (pred loss only)')
            print(f'Pred weight: {self.pred_weight}')
        
        print(f'{"="*60}\n')
        
        # Ensure save directories exist
        os.makedirs('models', exist_ok=True)
        os.makedirs('results-vae', exist_ok=True)
        
        start_time = time.time()
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            self.train_losses.append(train_metrics['loss'])
            self.train_recon_losses.append(train_metrics['recon'])
            self.train_kl_losses.append(train_metrics['kl'])
            self.train_pred_losses.append(train_metrics['pred'])
            
            # Validate
            val_metrics = self.validate()
            self.val_losses.append(val_metrics['loss'])
            self.val_recon_losses.append(val_metrics['recon'])
            self.val_kl_losses.append(val_metrics['kl'])
            self.val_pred_losses.append(val_metrics['pred'])
            
            # Learning rate scheduling
            self.scheduler.step(val_metrics['loss'])
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Print progress
            print(f"Epoch {epoch+1}/{epochs} (β={train_metrics['beta']:.4f})")
            print(f"  Train - Loss: {train_metrics['loss']:.6f}, "
                  f"Recon: {train_metrics['recon']:.6f}, "
                  f"KL: {train_metrics['kl']:.6f}, "
                  f"Pred: {train_metrics['pred']:.6f}")
            print(f"  Val   - Loss: {val_metrics['loss']:.6f}, "
                  f"Recon: {val_metrics['recon']:.6f}, "
                  f"KL: {val_metrics['kl']:.6f}, "
                  f"Pred: {val_metrics['pred']:.6f}")
            print(f"  LR: {current_lr:.6f}")
            
            # Early stopping
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.patience_counter = 0
                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_metrics['loss'],
                }, 'models/best_vae_model.pth')
                print(f"  ✓ Best model saved!")
            else:
                self.patience_counter += 1
                if self.patience_counter >= patience:
                    print(f"\n✓ Early stopping at epoch {epoch+1}")
                    break
            
            print()
        
        # Load best model with device mapping
        checkpoint = torch.load('models/best_vae_model.pth', map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        elapsed = time.time() - start_time
        print(f'\n{"="*60}')
        print(f'Training completed in {elapsed/60:.2f} minutes')
        print(f'Best val loss: {self.best_val_loss:.6f}')
        print(f'{"="*60}\n')
        
        return self.model


# ==================== EVALUATION ====================

@torch.no_grad()
def evaluate_vae(model, test_loader, device, scaler_y=None, link_names=None):
    """Evaluate VAE on test set"""
    model.eval()
    
    all_preds = []
    all_targets = []
    
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        
        # Forward pass (only prediction)
        _, _, _, pred = model(X_batch, predict=True)
        
        all_preds.append(pred.cpu().numpy())
        all_targets.append(y_batch.numpy())
    
    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_targets, axis=0)
    
    # Metrics on scaled data
    mse_scaled = mean_squared_error(y_true, y_pred)
    rmse_scaled = np.sqrt(mse_scaled)
    mae_scaled = mean_absolute_error(y_true, y_pred)
    r2_scaled = r2_score(y_true, y_pred)
    
    # Per-link metrics (scaled)
    num_links = y_true.shape[1]
    per_link_mse = [mean_squared_error(y_true[:, i], y_pred[:, i]) for i in range(num_links)]
    per_link_mae = [mean_absolute_error(y_true[:, i], y_pred[:, i]) for i in range(num_links)]
    
    print('\n' + '='*60)
    print('VAE Test Set Evaluation (Scaled Data)')
    print('='*60)
    print(f'MSE:  {mse_scaled:.6f}')
    print(f'RMSE: {rmse_scaled:.6f}')
    print(f'MAE:  {mae_scaled:.6f}')
    print(f'R²:   {r2_scaled:.6f}')
    
    # Inverse transform if scaler provided
    results = {
        'metrics_scaled': {
            'mse': float(mse_scaled),
            'rmse': float(rmse_scaled),
            'mae': float(mae_scaled),
            'r2': float(r2_scaled),
            'per_link_mse': [float(x) for x in per_link_mse],
            'per_link_mae': [float(x) for x in per_link_mae]
        },
        'y_pred_scaled': y_pred,
        'y_true_scaled': y_true
    }
    
    if scaler_y is not None:
        # Inverse transform
        y_true_real = scaler_y.inverse_transform(y_true)
        y_pred_real = scaler_y.inverse_transform(y_pred)
        
        mse_real = mean_squared_error(y_true_real, y_pred_real)
        rmse_real = np.sqrt(mse_real)
        mae_real = mean_absolute_error(y_true_real, y_pred_real)
        r2_real = r2_score(y_true_real, y_pred_real)
        
        per_link_mse_real = [mean_squared_error(y_true_real[:, i], y_pred_real[:, i]) 
                            for i in range(num_links)]
        per_link_mae_real = [mean_absolute_error(y_true_real[:, i], y_pred_real[:, i]) 
                            for i in range(num_links)]
        
        print('\n' + '='*60)
        print('VAE Test Set Evaluation (Real Scale)')
        print('='*60)
        print(f'MSE:  {mse_real:.6f}')
        print(f'RMSE: {rmse_real:.6f}')
        print(f'MAE:  {mae_real:.6f}')
        print(f'R²:   {r2_real:.6f}')
        
        if link_names:
            print('\nPer-Link MAE (Real Scale):')
            for i, name in enumerate(link_names):
                print(f'  {name}: {per_link_mae_real[i]:.6f}')
        
        results['metrics_real'] = {
            'mse': float(mse_real),
            'rmse': float(rmse_real),
            'mae': float(mae_real),
            'r2': float(r2_real),
            'per_link_mse': [float(x) for x in per_link_mse_real],
            'per_link_mae': [float(x) for x in per_link_mae_real]
        }
        results['y_pred_real'] = y_pred_real
        results['y_true_real'] = y_true_real
    
    print('='*60 + '\n')
    
    return results


def plot_vae_results(results, trainer, link_names, save_dir='results-vae'):
    """Plot VAE training curves and predictions"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Training curves - 4 loss components in 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    epochs = range(1, len(trainer.train_losses) + 1)
    
    # Plot 1: Total Loss
    axes[0].plot(epochs, trainer.train_losses, label='Train', linewidth=2, color='#1f77b4')
    axes[0].plot(epochs, trainer.val_losses, label='Val', linewidth=2, color='#ff7f0e')
    axes[0].set_xlabel('Epoch', fontsize=11)
    axes[0].set_ylabel('Loss', fontsize=11)
    axes[0].set_title('Total Loss (Recon + β×KL + λ×Pred)', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Reconstruction Loss
    axes[1].plot(epochs, trainer.train_recon_losses, label='Train', linewidth=2, color='#2ca02c')
    axes[1].plot(epochs, trainer.val_recon_losses, label='Val', linewidth=2, color='#d62728')
    axes[1].set_xlabel('Epoch', fontsize=11)
    axes[1].set_ylabel('Loss', fontsize=11)
    axes[1].set_title('Reconstruction Loss (MSE)', fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: KL Divergence Loss
    axes[2].plot(epochs, trainer.train_kl_losses, label='Train', linewidth=2, color='#9467bd')
    axes[2].plot(epochs, trainer.val_kl_losses, label='Val', linewidth=2, color='#8c564b')
    axes[2].set_xlabel('Epoch', fontsize=11)
    axes[2].set_ylabel('Loss', fontsize=11)
    axes[2].set_title('KL Divergence Loss (with β warmup)', fontsize=12, fontweight='bold')
    axes[2].legend(fontsize=10)
    axes[2].grid(True, alpha=0.3)
    
    # Plot 4: Prediction Loss
    axes[3].plot(epochs, trainer.train_pred_losses, label='Train', linewidth=2, color='#e377c2')
    axes[3].plot(epochs, trainer.val_pred_losses, label='Val', linewidth=2, color='#7f7f7f')
    axes[3].set_xlabel('Epoch', fontsize=11)
    axes[3].set_ylabel('Loss', fontsize=11)
    axes[3].set_title('Prediction Loss (Future Utilization MSE)', fontsize=12, fontweight='bold')
    axes[3].legend(fontsize=10)
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/vae_training_curves.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f'✓ Saved training curves (4 loss components) to {save_dir}/vae_training_curves.png')
    
    # 2. Predictions vs True (scatter plot)
    y_true = results['y_true_scaled']
    y_pred = results['y_pred_scaled']
    
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true.flatten(), y_pred.flatten(), alpha=0.3, s=1)
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect', linewidth=2)
    
    plt.xlabel('True')
    plt.ylabel('Pred')
    plt.title('VAE: Predictions vs True')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/vae_predictions_vs_true.png', dpi=150)
    plt.close()
    
    # 3. Per-link time series (first 200 samples)
    if 'y_pred_real' in results:
        y_true_real = results['y_true_real']
        y_pred_real = results['y_pred_real']
        
        # Plot 4 representative links
        sample_links = [0, 3, 6, 10]  # AIR_GROUND_01, GROUND_SEA_01, SPACE_AIR_02, SPACE_SPACE_01
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, link_idx in enumerate(sample_links):
            ax = axes[idx]
            
            n_samples = min(200, len(y_true_real))
            x = range(n_samples)
            
            ax.plot(x, y_true_real[:n_samples, link_idx], label='True', alpha=0.7, linewidth=1.5)
            ax.plot(x, y_pred_real[:n_samples, link_idx], label='Pred', alpha=0.7, linewidth=1.5)
            
            ax.set_xlabel('Time')
            ax.set_ylabel('Target')
            ax.set_title(link_names[link_idx])
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/vae_time_series.png', dpi=150)
        plt.close()
    
    print(f'✓ Plots saved to {save_dir}/')


# ==================== MAIN ====================

def main():
    """Main training pipeline"""
    
    # Set seed for reproducibility
    set_seed(42)
    
    print('\n' + '='*60)
    print('VAE for Network Traffic Prediction - SAGSIN')
    print('GenAI Approach: Variational Autoencoder')
    print('='*60 + '\n')
    
    # Ensure directories exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('results-vae', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # Configuration
    config = {
        # Model architecture - ADVANCED for better performance
        'latent_dim': 64,           
        'hidden_dim': 128,          
        'dropout': 0.4,             
        
        # TWO-STAGE TRAINING
        'stage1_epochs': 30,        # Stage 1: Pre-train encoder/decoder
        'stage2_epochs': 50,        # Stage 2: Fine-tune predictor
        'patience': 15,             # Early stopping patience
        
        # Training hyperparameters
        'batch_size': 128,
        'stage1_lr': 1e-3,          # LR for Stage 1
        'stage2_lr': 5e-4,          # Lower LR for Stage 2 (fine-tuning)
        'weight_decay': 5e-4,       
        
        # Loss weights
        'beta': 0.5,                # REDUCED: KL weight for Stage 1 (was 1.5)
        'pred_weight': 1.0,         # Prediction weight (not used in Stage 1)
        'kl_warmup_epochs': 15,     # KL warmup for Stage 1
        
        # Device
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print('Configuration:')
    for k, v in config.items():
        print(f'  {k}: {v}')
    print()
    
    # Load preprocessed data
    print('Loading preprocessed data...')
    
    # Load VAE sequences (already created by preprocessing)
    X_train = np.load('data/X_vae_train.npy')
    y_train = np.load('data/y_vae_train.npy')
    X_val = np.load('data/X_vae_val.npy')
    y_val = np.load('data/y_vae_val.npy')
    X_test = np.load('data/X_vae_test.npy')
    y_test = np.load('data/y_vae_test.npy')
    
    # Load metadata
    with open('data/link_index.json', 'r') as f:
        link_index = json.load(f)
    
    with open('data/features.json', 'r') as f:
        features_dict = json.load(f)
    
    # Extract feature list
    if isinstance(features_dict, dict):
        features = features_dict.get('model_features', features_dict)
    else:
        features = features_dict
    
    num_links = len(link_index)
    seq_len = X_train.shape[1]
    input_dim = X_train.shape[2]  # features * links
    
    print(f'  VAE sequences loaded from preprocessing')
    print(f'  Train: X={X_train.shape}, y={y_train.shape}')
    print(f'  Val:   X={X_val.shape}, y={y_val.shape}')
    print(f'  Test:  X={X_test.shape}, y={y_test.shape}')
    print(f'  Input dim: {input_dim} ({input_dim // num_links} features × {num_links} links)')
    print(f'  Features: {features}')
    print()
    
    # Create datasets
    train_dataset = VAETrafficDataset(X_train, y_train, seq_len)
    val_dataset = VAETrafficDataset(X_val, y_val, seq_len)
    test_dataset = VAETrafficDataset(X_test, y_test, seq_len)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                              shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], 
                            shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], 
                             shuffle=False, num_workers=0, pin_memory=True)
    
    # Create model
    print('Creating HYBRID VAE model (Two-Stage Training)...')
    print('  Architecture: VAE Encoder-Decoder + Hybrid Predictor')
    print('  Hybrid Predictor = VAE Latent + Raw Input LSTM (Best of both worlds!)')
    model = TrafficVAE(
        input_dim=input_dim,
        seq_len=seq_len,
        latent_dim=config['latent_dim'],
        hidden_dim=config['hidden_dim'],
        num_links=num_links,
        dropout=config['dropout']
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  Total parameters: {total_params:,}')
    print(f'  Trainable parameters: {trainable_params:,}')
    print()
    
    # ============================================
    # TWO-STAGE TRAINING PIPELINE
    # ============================================
    
    best_model_path = 'models/best_vae_model.pth'
    stage1_model_path = 'models/vae_stage1_encoder.pth'
    
    # Check if already trained
    if os.path.exists(best_model_path):
        print('='*60)
        print('Found existing trained model (Stage 2 complete)!')
        print('='*60)
        user_input = input('\nDo you want to:\n  [1] Use existing model (skip training)\n  [2] Train from scratch (overwrite)\nChoice (1/2): ').strip()
        
        if user_input == '1':
            print('\n✓ Loading existing model...')
            checkpoint = torch.load(best_model_path, map_location=config['device'], weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(config['device'])
            print(f'✓ Loaded model from epoch {checkpoint.get("epoch", "unknown")}')
            print(f'✓ Best val loss: {checkpoint.get("val_loss", "unknown"):.6f}\n')
            skip_training = True
        else:
            print('\n→ Training new model from scratch (2 stages)...\n')
            skip_training = False
    else:
        skip_training = False
    
    if not skip_training:
        # ============================================
        # STAGE 1: Pre-train Encoder + Decoder
        # ============================================
        print('\n' + '🔥'*30)
        print('STAGE 1: PRE-TRAINING ENCODER + DECODER')
        print('Goal: Learn good latent representation (unsupervised)')
        print('🔥'*30 + '\n')
        
        trainer_stage1 = VAETrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=config['device'],
            lr=config['stage1_lr'],
            weight_decay=config['weight_decay'],
            beta=config['beta'],
            pred_weight=1.0,  # Not used in Stage 1
            kl_warmup_epochs=config['kl_warmup_epochs'],
            stage=1  # Stage 1
        )
        
        model = trainer_stage1.train(
            epochs=config['stage1_epochs'], 
            patience=config['patience']
        )
        
        # Save Stage 1 encoder
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config
        }, stage1_model_path)
        print(f'\n✓ Stage 1 encoder saved to {stage1_model_path}\n')
        
        # ============================================
        # STAGE 2: Fine-tune Predictor (Freeze Encoder)
        # ============================================
        print('\n' + '🚀'*30)
        print('STAGE 2: FINE-TUNING PREDICTOR')
        print('Goal: Optimize prediction with frozen encoder (supervised)')
        print('🚀'*30 + '\n')
        
        trainer_stage2 = VAETrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=config['device'],
            lr=config['stage2_lr'],  # Lower LR for fine-tuning
            weight_decay=config['weight_decay'],
            beta=0.0,  # No KL loss in Stage 2
            pred_weight=config['pred_weight'],
            kl_warmup_epochs=0,  # No warmup in Stage 2
            stage=2  # Stage 2
        )
        
        model = trainer_stage2.train(
            epochs=config['stage2_epochs'], 
            patience=config['patience']
        )
        
        print(f'\n✓ Two-stage training complete!')
        print(f'✓ Final model saved to {best_model_path}\n')
        
        # Combine training histories from both stages for plotting
        combined_trainer = type('obj', (object,), {
            'train_losses': trainer_stage1.train_losses + trainer_stage2.train_losses,
            'val_losses': trainer_stage1.val_losses + trainer_stage2.val_losses,
            'train_recon_losses': trainer_stage1.train_recon_losses + trainer_stage2.train_recon_losses,
            'val_recon_losses': trainer_stage1.val_recon_losses + trainer_stage2.val_recon_losses,
            'train_kl_losses': trainer_stage1.train_kl_losses + trainer_stage2.train_kl_losses,
            'val_kl_losses': trainer_stage1.val_kl_losses + trainer_stage2.val_kl_losses,
            'train_pred_losses': trainer_stage1.train_pred_losses + trainer_stage2.train_pred_losses,
            'val_pred_losses': trainer_stage1.val_pred_losses + trainer_stage2.val_pred_losses,
        })()
        final_trainer = combined_trainer
    else:
        # Model was loaded, create dummy trainer for plotting
        final_trainer = type('obj', (object,), {
            'train_losses': [],
            'val_losses': [],
            'train_recon_losses': [],
            'val_recon_losses': [],
            'train_kl_losses': [],
            'val_kl_losses': [],
            'train_pred_losses': [],
            'val_pred_losses': [],
        })()
    
    # Load scaler for inverse transform
    import joblib
    scalers_dict = joblib.load('models/wide_scalers.pkl')
    
    # Get utilization scaler
    if 'utilization' in scalers_dict:
        scaler_y = scalers_dict['utilization']
        print('  Using existing utilization scaler from preprocessing')
    else:
        print('  ⚠️ Warning: No utilization scaler found')
        scaler_y = None
    
    print()
    
    results = evaluate_vae(
        model=model,
        test_loader=test_loader,
        device=config['device'],
        scaler_y=scaler_y,
        link_names=link_index  # link_index is already a list
    )
    
    # Save results
    print('Saving results...')
    
    # Save predictions
    np.save('results-vae/vae_y_pred.npy', results['y_pred_scaled'])
    np.save('results-vae/vae_y_true.npy', results['y_true_scaled'])
    
    if 'y_pred_real' in results:
        np.save('results-vae/vae_y_pred_real.npy', results['y_pred_real'])
        np.save('results-vae/vae_y_true_real.npy', results['y_true_real'])
    
    # Save metrics
    save_results = {
        'model': 'HybridVAE-TwoStage',
        'architecture': 'Two-Stage: (1) Pre-train VAE Encoder+Decoder → (2) Train Hybrid Predictor',
        'training_strategy': 'Stage 1: recon+KL loss | Stage 2: Hybrid Predictor (VAE latent + Raw LSTM) with frozen encoder',
        'predictor_type': 'Hybrid: VAE latent (compressed) + Raw input LSTM (temporal details)',
        'metrics_scaled': results['metrics_scaled'],
        'model_config': {
            'latent_dim': config['latent_dim'],
            'hidden_dim': config['hidden_dim'],
            'dropout': config['dropout'],
            'input_dim': input_dim,
            'seq_len': seq_len,
            'num_links': num_links
        },
        'training_config': {
            'batch_size': config['batch_size'],
            'stage1_epochs': config['stage1_epochs'],
            'stage2_epochs': config['stage2_epochs'],
            'stage1_lr': config['stage1_lr'],
            'stage2_lr': config['stage2_lr'],
            'weight_decay': config['weight_decay'],
            'beta': config['beta'],
            'pred_weight': config['pred_weight'],
            'kl_warmup_epochs': config['kl_warmup_epochs'],
            'patience': config['patience'],
            'device': config['device']
        },
        'data_info': {
            'features': features,  # Now it's a list
            'link_names': link_index,  # link_index is already a list
            'seq_len': seq_len,
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'test_samples': len(X_test)
        }
    }
    
    if 'metrics_real' in results:
        save_results['metrics_real'] = results['metrics_real']
    
    with open('results-vae/vae_results.json', 'w') as f:
        json.dump(save_results, f, indent=2)
    
    # Plot results (use combined trainer from both stages)
    plot_vae_results(results, final_trainer, link_index)  # link_index is already a list
    
    print('\n' + '='*60)
    print('VAE Training Complete!')
    print('='*60)
    print('\nResults saved to:')
    print('  - models/best_vae_model.pth')
    print('  - results-vae/vae_results.json')
    print('  - results-vae/vae_*.npy')
    print('  - results-vae/vae_*.png')
    print()
    
    # Comparison with LSTM
    if os.path.exists('results/lstm_results.json'):
        print('\n' + '='*60)
        print('Comparison: VAE vs LSTM Baseline')
        print('='*60)
        
        with open('results/lstm_results.json', 'r') as f:
            lstm_results = json.load(f)
        
        vae_r2 = results['metrics_scaled']['r2']
        lstm_r2 = lstm_results['metrics_scaled']['r2']
        
        vae_mae = results['metrics_scaled']['mae']
        lstm_mae = lstm_results['metrics_scaled']['mae']
        
        print(f'\nR² Score:')
        print(f'  LSTM (baseline): {lstm_r2:.6f}')
        print(f'  VAE (GenAI):     {vae_r2:.6f}')
        print(f'  Difference:      {vae_r2 - lstm_r2:+.6f} ({(vae_r2-lstm_r2)/lstm_r2*100:+.2f}%)')
        
        print(f'\nMAE (scaled):')
        print(f'  LSTM (baseline): {lstm_mae:.6f}')
        print(f'  VAE (GenAI):     {vae_mae:.6f}')
        print(f'  Difference:      {vae_mae - lstm_mae:+.6f} ({(vae_mae-lstm_mae)/lstm_mae*100:+.2f}%)')
        
        if 'metrics_real' in results and 'metrics_real' in lstm_results:
            vae_mae_real = results['metrics_real']['mae']
            lstm_mae_real = lstm_results['metrics_real']['mae']
            
            print(f'\nMAE (real scale):')
            print(f'  LSTM (baseline): {lstm_mae_real:.6f}')
            print(f'  VAE (GenAI):     {vae_mae_real:.6f}')
            print(f'  Difference:      {vae_mae_real - lstm_mae_real:+.6f} ({(vae_mae_real-lstm_mae_real)/lstm_mae_real*100:+.2f}%)')
        
        print('='*60 + '\n')


if __name__ == '__main__':
    main()
