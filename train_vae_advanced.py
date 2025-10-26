# -*- coding: utf-8 -*-
"""
Advanced VAE Training Script

Implements improved VAE architecture to beat LSTM baseline:
- MultiScaleEncoder: Captures temporal patterns at different scales
- Seq2SeqDecoder: Better reconstruction with teacher forcing
- AdvancedSeq2SeqPredictor: Sequence-to-sequence prediction
- MultiStageTrainer: Progressive training strategy

Target: R² > 0.80 (beat LSTM baseline ~0.75)
"""

import sys
import io
# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import json
import os
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt


# ============================================================================
# STAGE 2.1: MultiScaleEncoder
# ============================================================================

class MultiScaleEncoder(nn.Module):
    """
    Multi-scale temporal encoding inspired by WaveNet
    
    Captures patterns at different time scales:
    - Fine: 1-5 timesteps (short-term spikes)
    - Medium: 5-20 timesteps (hourly trends)
    - Coarse: 20-96 timesteps (long-term patterns)
    
    Architecture:
        Input (batch, seq_len, input_dim)
          ↓
        3 Parallel Conv1D paths (different kernel sizes)
          ↓
        Fusion layer
          ↓
        Temporal attention (learn which timesteps matter)
          ↓
        Bi-LSTM (capture long-range dependencies)
          ↓
        Output: μ, logvar (batch, latent_dim)
    """
    def __init__(self, input_dim, hidden_dim=256, latent_dim=128, dropout=0.3):
        super().__init__()
        
        # 1. Multi-scale Conv1D (parallel paths)
        self.conv_fine = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.conv_medium = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=9, padding=4),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.conv_coarse = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=21, padding=10),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 2. Fusion layer (combine 3 scales)
        self.fusion = nn.Sequential(
            nn.Conv1d(hidden_dim * 3, hidden_dim, kernel_size=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU()
        )
        
        # 3. Temporal attention (learn which timesteps matter)
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # 4. Bi-directional LSTM (capture long-range dependencies)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=2,
            dropout=dropout,
            batch_first=True,
            bidirectional=True
        )
        
        # 5. Latent projection (μ and σ)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, x):
        """
        x: (batch, seq_len=96, input_dim)
        Returns: mu, logvar, attn_weights
        """
        batch_size, seq_len, input_dim = x.shape
        
        # Transpose for Conv1D: (batch, channels, seq_len)
        x_t = x.transpose(1, 2)  # (batch, input_dim, 96)
        
        # Multi-scale convolution (parallel)
        fine = self.conv_fine(x_t)      # (batch, hidden, 96)
        medium = self.conv_medium(x_t)  # (batch, hidden, 96)
        coarse = self.conv_coarse(x_t)  # (batch, hidden, 96)
        
        # Fuse scales
        fused = torch.cat([fine, medium, coarse], dim=1)  # (batch, hidden*3, 96)
        fused = self.fusion(fused)  # (batch, hidden, 96)
        
        # Transpose back: (batch, 96, hidden)
        fused = fused.transpose(1, 2)
        
        # Temporal attention (which timesteps are important?)
        attn_out, attn_weights = self.temporal_attn(fused, fused, fused)
        attn_out = attn_out + fused  # Residual connection
        
        # LSTM for long-range dependencies
        lstm_out, (h_n, c_n) = self.lstm(attn_out)
        
        # Global pooling (mean over sequence)
        pooled = lstm_out.mean(dim=1)  # (batch, hidden)
        
        # Latent parameters
        mu = self.fc_mu(pooled)          # (batch, latent_dim)
        logvar = self.fc_logvar(pooled)  # (batch, latent_dim)
        
        return mu, logvar, attn_weights


# ============================================================================
# STAGE 2.2: Seq2SeqDecoder
# ============================================================================

class Seq2SeqDecoder(nn.Module):
    """
    Sequence-to-Sequence Decoder with Teacher Forcing
    
    Input: latent z (batch, latent_dim)
    Output: reconstructed sequence (batch, seq_len, output_dim)
    
    Teacher forcing: During training, use ground truth with probability
                     to provide better gradients
    """
    def __init__(self, latent_dim, hidden_dim, output_dim, seq_len=96, dropout=0.3):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # 1. Latent to initial hidden state
        self.latent_to_hidden = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 2. Decoder LSTM
        self.decoder_lstm = nn.LSTM(
            input_size=output_dim,  # Input: previous timestep reconstruction
            hidden_size=hidden_dim,
            num_layers=2,
            dropout=dropout,
            batch_first=True
        )
        
        # 3. Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, z, target_seq=None, teacher_forcing_ratio=0.5):
        """
        z: (batch, latent_dim)
        target_seq: (batch, seq_len, output_dim) - ground truth for teacher forcing
        
        Returns: (batch, seq_len, output_dim)
        """
        batch_size = z.size(0)
        device = z.device
        
        # Initialize hidden state from latent
        h0_c0 = self.latent_to_hidden(z)  # (batch, hidden*2)
        h0 = h0_c0[:, :self.hidden_dim].unsqueeze(0).repeat(2, 1, 1)  # (2, batch, hidden)
        c0 = h0_c0[:, self.hidden_dim:].unsqueeze(0).repeat(2, 1, 1)
        
        # Start token (zeros)
        decoder_input = torch.zeros(batch_size, 1, self.output_dim).to(device)
        
        outputs = []
        hidden = (h0, c0)
        
        for t in range(self.seq_len):
            # Decode one timestep
            lstm_out, hidden = self.decoder_lstm(decoder_input, hidden)
            output_t = self.output_proj(lstm_out)  # (batch, 1, output_dim)
            outputs.append(output_t)
            
            # Teacher forcing: use ground truth with probability
            if target_seq is not None and torch.rand(1).item() < teacher_forcing_ratio:
                decoder_input = target_seq[:, t:t+1, :]  # Ground truth
            else:
                decoder_input = output_t  # Model's own prediction
        
        # Concatenate all timesteps
        recon = torch.cat(outputs, dim=1)  # (batch, seq_len, output_dim)
        return recon


# ============================================================================
# STAGE 2.3: AdvancedSeq2SeqPredictor
# ============================================================================

class AdvancedSeq2SeqPredictor(nn.Module):
    """
    Predicts HORIZON future timesteps (sequence-to-sequence)
    
    Architecture:
    1. Encode input sequence (X) via Bi-LSTM
    2. Fuse encoder output with latent (z)
    3. Decode future sequence via LSTM
    4. Project to per-link predictions (separate heads for each link)
    
    Expected: R² 0.70-0.80 (match or beat LSTM baseline 0.75)
    """
    def __init__(self, input_dim, latent_dim, hidden_dim, num_links, 
                 horizon=12, dropout=0.3):
        super().__init__()
        self.horizon = horizon
        self.num_links = num_links
        
        # 1. Input encoder (Bi-LSTM)
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            dropout=dropout,
            batch_first=True,
            bidirectional=True
        )
        
        # 2. Attention over encoder outputs
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # 3. Fusion with latent
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2 + latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 4. Decoder LSTM (generates future)
        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            dropout=dropout,
            batch_first=True
        )
        
        # 5. Per-link output heads (separate for each link)
        # This allows model to learn link-specific patterns
        self.link_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1)
            ) for _ in range(num_links)
        ])
        
    def forward(self, x, z):
        """
        x: (batch, seq_len=96, input_dim)
        z: (batch, latent_dim)
        
        Returns: 
            pred: (batch, horizon, num_links)
            attn_weights: attention weights for analysis
        """
        batch_size = x.size(0)
        
        # 1. Encode input
        encoded, _ = self.encoder(x)  # (batch, 96, hidden*2)
        
        # 2. Attention (which timesteps matter for prediction?)
        context, attn_weights = self.attention(encoded, encoded, encoded)
        context = context + encoded  # Residual
        
        # 3. Take last context
        last_context = context[:, -1, :]  # (batch, hidden*2)
        
        # 4. Fuse with latent
        fused = self.fusion(torch.cat([last_context, z], dim=-1))  # (batch, hidden)
        
        # 5. Expand to horizon timesteps (decoder input)
        decoder_input = fused.unsqueeze(1).repeat(1, self.horizon, 1)  # (batch, horizon, hidden)
        
        # 6. Decode future sequence
        decoded, _ = self.decoder(decoder_input)  # (batch, horizon, hidden)
        
        # 7. Per-link predictions (parallel)
        link_preds = []
        for link_head in self.link_heads:
            pred_link = link_head(decoded)  # (batch, horizon, 1)
            link_preds.append(pred_link)
        
        pred = torch.cat(link_preds, dim=-1)  # (batch, horizon, num_links)
        
        return pred, attn_weights


# ============================================================================
# STAGE 2.4: Complete AdvancedHybridVAE
# ============================================================================

class AdvancedHybridVAE(nn.Module):
    """
    Complete VAE with:
    - MultiScaleEncoder (better latent representation)
    - Seq2SeqDecoder (better reconstruction)
    - AdvancedSeq2SeqPredictor (better forecast)
    
    Expected R²: 0.75-0.85 (match or beat LSTM baseline)
    """
    def __init__(self, input_dim, latent_dim=128, hidden_dim=256, 
                 num_links=10, seq_len=96, horizon=12, dropout=0.3):
        super().__init__()
        self.seq_len = seq_len
        self.horizon = horizon
        
        # Encoder (multi-scale)
        self.encoder = MultiScaleEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            dropout=dropout
        )
        
        # Decoder (seq2seq with teacher forcing)
        self.decoder = Seq2SeqDecoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            output_dim=input_dim,
            seq_len=seq_len,
            dropout=dropout
        )
        
        # Predictor (seq2seq future forecast)
        self.predictor = AdvancedSeq2SeqPredictor(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_links=num_links,
            horizon=horizon,
            dropout=dropout
        )
        
    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x, target_seq=None, predict=True, teacher_forcing_ratio=0.5):
        """
        x: (batch, 96, input_dim)
        target_seq: (batch, 96, input_dim) - for teacher forcing in decoder
        
        Returns:
            recon: (batch, 96, input_dim)
            mu: (batch, latent_dim)
            logvar: (batch, latent_dim)
            pred: (batch, horizon, num_links) if predict=True
            attn_enc: encoder attention weights
            attn_pred: predictor attention weights
        """
        # Encode
        mu, logvar, attn_enc = self.encoder(x)
        
        # Sample latent
        z = self.reparameterize(mu, logvar)
        
        # Decode (reconstruction)
        recon = self.decoder(z, target_seq, teacher_forcing_ratio)
        
        # Predict (if needed)
        if predict:
            pred, attn_pred = self.predictor(x, z)
            return recon, mu, logvar, pred, attn_enc, attn_pred
        else:
            return recon, mu, logvar, None, attn_enc, None


# ============================================================================
# STAGE 3.1: MultiStageVAETrainer
# ============================================================================

class MultiStageVAETrainer:
    """
    3-stage training for better convergence:
    
    Stage 1 (20 epochs): Reconstruction only
        → Encoder learns good latent representation
        → Loss = recon_loss + β*kl_loss (β=0.1)
    
    Stage 2 (20 epochs): Joint training
        → All components active, low pred_weight
        → Loss = recon + β*kl + 0.3*pred (β=0.5)
    
    Stage 3 (40 epochs): Prediction focus
        → High pred_weight, teacher_forcing decay
        → Loss = recon + β*kl + 1.0*pred (β=1.0)
    
    Expected: Better R² than single-stage training
    """
    def __init__(self, model, train_loader, val_loader, device='cuda', 
                 save_dir='models'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-3,
            weight_decay=1e-5
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        
        # Training state
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.best_r2 = -float('inf')
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_recon': [], 'train_kl': [], 'train_pred': [],
            'val_r2': []
        }
        
    def compute_loss(self, recon, x, mu, logvar, pred, y_target, 
                     beta=1.0, pred_weight=0.0):
        """
        Unified loss with multi-task weighting
        """
        batch_size = x.size(0)
        
        # 1. Reconstruction loss (MSE)
        recon_loss = F.mse_loss(recon, x, reduction='mean')
        
        # 2. KL divergence (regularization)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_loss / (batch_size * mu.size(1))
        
        # 3. Prediction loss (seq2seq MSE)
        if pred is not None and pred_weight > 0:
            pred_loss = F.mse_loss(pred, y_target, reduction='mean')
        else:
            pred_loss = torch.tensor(0.0).to(x.device)
        
        # Total loss
        total_loss = recon_loss + beta * kl_loss + pred_weight * pred_loss
        
        return total_loss, recon_loss, kl_loss, pred_loss
    
    def train_epoch(self, stage=1):
        """
        Train one epoch with stage-specific config
        """
        self.model.train()
        
        # Stage-specific parameters
        if stage == 1:
            # Focus on reconstruction
            beta = 0.1  # Low KL weight (allow flexibility)
            pred_weight = 0.0  # NO prediction yet
            teacher_forcing = 0.8  # High teacher forcing
        elif stage == 2:
            # Joint training
            beta = 0.5
            pred_weight = 0.3  # Start prediction
            teacher_forcing = 0.5
        else:  # stage == 3
            # Focus on prediction
            beta = 1.0
            pred_weight = 1.0  # Full prediction weight
            # Decay teacher forcing over epochs
            teacher_forcing = max(0.1, 0.8 - 0.01 * (self.epoch - 40))
        
        total_loss_sum = 0
        recon_loss_sum = 0
        kl_loss_sum = 0
        pred_loss_sum = 0
        
        for X_batch, y_batch in self.train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward
            recon, mu, logvar, pred, _, _ = self.model(
                X_batch, 
                target_seq=X_batch,  # For teacher forcing
                predict=(pred_weight > 0),
                teacher_forcing_ratio=teacher_forcing
            )
            
            # Compute loss
            loss, recon_loss, kl_loss, pred_loss = self.compute_loss(
                recon, X_batch, mu, logvar, pred, y_batch,
                beta=beta,
                pred_weight=pred_weight
            )
            
            # Backward
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Accumulate
            total_loss_sum += loss.item()
            recon_loss_sum += recon_loss.item()
            kl_loss_sum += kl_loss.item()
            pred_loss_sum += pred_loss.item()
        
        n = len(self.train_loader)
        return {
            'total': total_loss_sum / n,
            'recon': recon_loss_sum / n,
            'kl': kl_loss_sum / n,
            'pred': pred_loss_sum / n
        }
    
    def validate(self, stage=1):
        """Validation with metrics corresponding to stage"""
        self.model.eval()
        
        # Stage-specific beta/pred_weight (same as train)
        if stage == 1:
            beta, pred_weight = 0.1, 0.0
        elif stage == 2:
            beta, pred_weight = 0.5, 0.3
        else:
            beta, pred_weight = 1.0, 1.0
        
        total_loss_sum = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for X_batch, y_batch in self.val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                # Forward (NO teacher forcing in val)
                recon, mu, logvar, pred, _, _ = self.model(
                    X_batch,
                    predict=(pred_weight > 0),
                    teacher_forcing_ratio=0.0
                )
                
                loss, _, _, _ = self.compute_loss(
                    recon, X_batch, mu, logvar, pred, y_batch,
                    beta=beta,
                    pred_weight=pred_weight
                )
                
                total_loss_sum += loss.item()
                
                if pred is not None:
                    all_preds.append(pred.cpu().numpy())
                    all_targets.append(y_batch.cpu().numpy())
        
        val_loss = total_loss_sum / len(self.val_loader)
        
        # Compute R² if in prediction stages
        r2 = None
        mae = None
        if len(all_preds) > 0:
            preds = np.concatenate(all_preds, axis=0)  # (N, horizon, num_links)
            targets = np.concatenate(all_targets, axis=0)
            
            # Flatten for R² computation
            r2 = r2_score(targets.reshape(-1), preds.reshape(-1))
            mae = mean_absolute_error(targets.reshape(-1), preds.reshape(-1))
        
        return val_loss, r2, mae
    
    def train_all_stages(self, stage1_epochs=20, stage2_epochs=20, stage3_epochs=40):
        """
        Main training loop
        """
        print("=" * 70)
        print("🚀 ADVANCED VAE TRAINING - MULTI-STAGE APPROACH")
        print("=" * 70)
        
        # STAGE 1: Reconstruction Focus
        print("\n" + "=" * 70)
        print("📊 STAGE 1: Reconstruction Focus (β=0.1, pred_weight=0.0)")
        print("   Goal: Learn good latent representation")
        print("=" * 70)
        
        for epoch in range(stage1_epochs):
            self.epoch = epoch
            train_metrics = self.train_epoch(stage=1)
            val_loss, _, _ = self.validate(stage=1)
            
            self.history['train_loss'].append(train_metrics['total'])
            self.history['val_loss'].append(val_loss)
            self.history['train_recon'].append(train_metrics['recon'])
            self.history['train_kl'].append(train_metrics['kl'])
            
            print(f"Epoch {epoch+1:2d}/{stage1_epochs} | "
                  f"Train: {train_metrics['total']:.4f} "
                  f"(recon={train_metrics['recon']:.4f}, kl={train_metrics['kl']:.4f}) | "
                  f"Val: {val_loss:.4f}")
            
            self.scheduler.step(val_loss)
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), 
                          os.path.join(self.save_dir, 'vae_stage1_best.pth'))
        
        # STAGE 2: Joint Training
        print("\n" + "=" * 70)
        print("📊 STAGE 2: Joint Training (β=0.5, pred_weight=0.3)")
        print("   Goal: Activate all components, start prediction")
        print("=" * 70)
        
        for epoch in range(stage2_epochs):
            self.epoch = stage1_epochs + epoch
            train_metrics = self.train_epoch(stage=2)
            val_loss, r2, mae = self.validate(stage=2)
            
            self.history['train_loss'].append(train_metrics['total'])
            self.history['val_loss'].append(val_loss)
            self.history['train_recon'].append(train_metrics['recon'])
            self.history['train_kl'].append(train_metrics['kl'])
            self.history['train_pred'].append(train_metrics['pred'])
            if r2 is not None:
                self.history['val_r2'].append(r2)
            
            r2_str = f"{r2:.4f}" if r2 is not None else "N/A"
            mae_str = f"{mae:.4f}" if mae is not None else "N/A"
            print(f"Epoch {epoch+1:2d}/{stage2_epochs} | "
                  f"Train: {train_metrics['total']:.4f} "
                  f"(pred={train_metrics['pred']:.4f}) | "
                  f"Val: {val_loss:.4f} | "
                  f"R²: {r2_str} | "
                  f"MAE: {mae_str}")
            
            self.scheduler.step(val_loss)
        
        # STAGE 3: Prediction Focus
        print("\n" + "=" * 70)
        print("📊 STAGE 3: Prediction Focus (β=1.0, pred_weight=1.0)")
        print("   Goal: Maximize prediction performance")
        print("=" * 70)
        
        patience = 15
        no_improve = 0
        
        for epoch in range(stage3_epochs):
            self.epoch = stage1_epochs + stage2_epochs + epoch
            train_metrics = self.train_epoch(stage=3)
            val_loss, r2, mae = self.validate(stage=3)
            
            self.history['train_loss'].append(train_metrics['total'])
            self.history['val_loss'].append(val_loss)
            self.history['train_recon'].append(train_metrics['recon'])
            self.history['train_kl'].append(train_metrics['kl'])
            self.history['train_pred'].append(train_metrics['pred'])
            if r2 is not None:
                self.history['val_r2'].append(r2)
            
            print(f"Epoch {epoch+1:2d}/{stage3_epochs} | "
                  f"Train: {train_metrics['total']:.4f} | "
                  f"Val: {val_loss:.4f} | "
                  f"R²: {r2:.4f} | "
                  f"MAE: {mae:.4f}")
            
            # Early stopping based on R²
            if r2 > self.best_r2:
                self.best_r2 = r2
                no_improve = 0
                torch.save(self.model.state_dict(), 
                          os.path.join(self.save_dir, 'vae_best.pth'))
                print(f"   ✅ New best R²: {self.best_r2:.4f}")
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"   ⚠️ Early stopping (no improve for {patience} epochs)")
                    break
            
            self.scheduler.step(val_loss)
        
        print(f"\n{'='*70}")
        print(f"🎉 TRAINING COMPLETE!")
        print(f"   Best Val R²: {self.best_r2:.4f}")
        print(f"   Model saved to: {os.path.join(self.save_dir, 'vae_best.pth')}")
        print(f"{'='*70}")
        
        return self.best_r2
    
    def plot_history(self, save_path='results/vae_training_history.png'):
        """Plot training history"""
        os.makedirs('results', exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curve
        axes[0, 0].plot(self.history['train_loss'], label='Train')
        axes[0, 0].plot(self.history['val_loss'], label='Val')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Total Loss')
        axes[0, 0].set_title('Training & Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Component losses
        axes[0, 1].plot(self.history['train_recon'], label='Recon')
        axes[0, 1].plot(self.history['train_kl'], label='KL')
        if len(self.history['train_pred']) > 0:
            axes[0, 1].plot(range(20, 20+len(self.history['train_pred'])), 
                           self.history['train_pred'], label='Pred')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss Components')
        axes[0, 1].set_title('Loss Decomposition')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # R² curve
        if len(self.history['val_r2']) > 0:
            axes[1, 0].plot(range(20, 20+len(self.history['val_r2'])), 
                           self.history['val_r2'], label='Val R²', color='green')
            axes[1, 0].axhline(y=0.75, color='r', linestyle='--', 
                              label='LSTM Baseline (0.75)')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('R²')
            axes[1, 0].set_title('Validation R² (Target: >0.75)')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Stage markers
        axes[1, 1].axvspan(0, 20, alpha=0.2, color='blue', label='Stage 1: Recon')
        axes[1, 1].axvspan(20, 40, alpha=0.2, color='orange', label='Stage 2: Joint')
        axes[1, 1].axvspan(40, len(self.history['train_loss']), 
                          alpha=0.2, color='green', label='Stage 3: Pred')
        axes[1, 1].plot(self.history['train_loss'], linewidth=2, color='black')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Train Loss')
        axes[1, 1].set_title('Training Stages')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n📊 Training history saved to: {save_path}")
        
        # Show plot in popup window
        try:
            plt.show()
        except Exception as e:
            print(f"   (Could not display plot window: {e})")
        finally:
            plt.close()
    
    def plot_predictions(self, X_test, y_test, save_path='results/vae_predictions.png'):
        """Plot predictions vs ground truth on test set"""
        self.model.eval()
        os.makedirs('results', exist_ok=True)
        
        print("\n📊 Generating prediction plots...")
        
        # Get predictions
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            # Convert to DataLoader for batch processing
            test_dataset = torch.utils.data.TensorDataset(
                torch.FloatTensor(X_test),
                torch.FloatTensor(y_test)
            )
            test_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=128, shuffle=False
            )
            
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                _, _, _, pred, _, _ = self.model(X_batch, predict=True, teacher_forcing_ratio=0.0)
                
                all_preds.append(pred.cpu().numpy())
                all_targets.append(y_batch.cpu().numpy())
        
        y_pred = np.concatenate(all_preds, axis=0)  # (N, horizon, num_links)
        y_true = np.concatenate(all_targets, axis=0)
        
        # Create figure with 3 subplots
        fig = plt.figure(figsize=(18, 6))
        
        # 1. Scatter plot (all timesteps, all links)
        ax1 = plt.subplot(131)
        ax1.scatter(y_true.flatten(), y_pred.flatten(), alpha=0.3, s=1)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect', linewidth=2)
        ax1.set_xlabel('True Utilization')
        ax1.set_ylabel('Predicted Utilization')
        ax1.set_title(f'Predictions vs Ground Truth\n(R² = {self.best_r2:.4f})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Per-timestep R² (how well does it predict at each horizon step?)
        ax2 = plt.subplot(132)
        r2_per_timestep = []
        for t in range(y_true.shape[1]):  # For each horizon timestep
            r2_t = r2_score(y_true[:, t, :].flatten(), y_pred[:, t, :].flatten())
            r2_per_timestep.append(r2_t)
        
        timesteps = np.arange(1, len(r2_per_timestep) + 1)
        ax2.plot(timesteps, r2_per_timestep, marker='o', linewidth=2)
        ax2.axhline(y=0.75, color='r', linestyle='--', label='LSTM Baseline (0.75)')
        ax2.set_xlabel('Forecast Horizon (timestep)')
        ax2.set_ylabel('R²')
        ax2.set_title('Prediction Quality vs Forecast Horizon')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(timesteps)
        
        # 3. Time series for first 100 samples, middle horizon, 4 random links
        ax3 = plt.subplot(133)
        n_samples = min(100, len(y_true))
        mid_horizon = y_true.shape[1] // 2  # Middle timestep
        sample_links = [0, 3, 6, 9]  # 4 links
        
        x = range(n_samples)
        for link_idx in sample_links:
            ax3.plot(x, y_true[:n_samples, mid_horizon, link_idx], 
                    alpha=0.5, linewidth=1, linestyle='--', label=f'True Link {link_idx}')
            ax3.plot(x, y_pred[:n_samples, mid_horizon, link_idx], 
                    alpha=0.7, linewidth=1.5, label=f'Pred Link {link_idx}')
        
        ax3.set_xlabel('Sample Index')
        ax3.set_ylabel('Utilization')
        ax3.set_title(f'Time Series (horizon step {mid_horizon+1})')
        ax3.legend(fontsize=8, ncol=2)
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   Saved to: {save_path}")
        
        # Show plot
        try:
            plt.show()
        except Exception as e:
            print(f"   (Could not display plot window: {e})")
        finally:
            plt.close()
        
        return y_pred, y_true


# ============================================================================
# STAGE 3.2: Main Training Script
# ============================================================================

def main():
    print("=" * 70)
    print("ADVANCED VAE TRAINING")
    print("=" * 70)
    
    # Load data
    print("\n📂 Loading data...")
    X_train = np.load('data/X_vae_train.npy')
    y_train = np.load('data/y_vae_train.npy')
    X_val = np.load('data/X_vae_val.npy')
    y_val = np.load('data/y_vae_val.npy')
    
    print(f"  X_train: {X_train.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  X_val:   {X_val.shape}")
    print(f"  y_val:   {y_val.shape}")
    
    # Load metadata
    with open('data/features.json') as f:
        metadata = json.load(f)
        horizon = metadata.get('horizon', 1)
    
    # Check y shape
    if y_train.ndim == 2:
        print("\n⚠️ WARNING: y_vae is 2D (old format - single timestep)")
        print("   Please re-run: python preprocessing.py")
        print("   Expected: y shape should be (N, horizon, num_links)")
        return
    else:
        print(f"\n✅ y_vae is 3D (sequence prediction)")
        print(f"   Horizon: {horizon} timesteps = {horizon*30}s = {horizon*30/60:.1f} min")
    
    input_dim = X_train.shape[2]
    num_links = y_train.shape[2]
    
    print(f"\n📐 Model parameters:")
    print(f"  Input dim: {input_dim}")
    print(f"  Num links: {num_links}")
    print(f"  Horizon: {horizon}")
    
    # Create dataloaders
    print("\n🔄 Creating dataloaders...")
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(y_val)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=0)
    
    # Initialize model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n🖥️  Using device: {device}")
    
    model = AdvancedHybridVAE(
        input_dim=input_dim,
        latent_dim=128,
        hidden_dim=256,
        num_links=num_links,
        seq_len=96,
        horizon=horizon,
        dropout=0.3
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\n📊 Model parameters: {num_params:,}")
    
    # Train
    print("\n🚀 Starting training...")
    trainer = MultiStageVAETrainer(model, train_loader, val_loader, device)
    best_r2 = trainer.train_all_stages(
        stage1_epochs=20,
        stage2_epochs=20,
        stage3_epochs=40
    )
    
    # Plot history
    trainer.plot_history()
    
    # Load test data and plot predictions
    print("\n📊 Evaluating on test set...")
    X_test = np.load('data/X_vae_test.npy')
    y_test = np.load('data/y_vae_test.npy')
    print(f"  X_test: {X_test.shape}")
    print(f"  y_test: {y_test.shape}")
    
    # Plot predictions
    y_pred, y_true = trainer.plot_predictions(X_test, y_test)
    
    # Compute test metrics
    test_r2 = r2_score(y_true.flatten(), y_pred.flatten())
    test_mae = mean_absolute_error(y_true.flatten(), y_pred.flatten())
    
    print(f"\n📊 Test Set Metrics:")
    print(f"   R²:  {test_r2:.4f}")
    print(f"   MAE: {test_mae:.4f}")
    
    # Save predictions
    np.save('results/vae_y_pred_advanced.npy', y_pred)
    np.save('results/vae_y_true_advanced.npy', y_true)
    print(f"   Predictions saved to results/vae_y_pred_advanced.npy")
    
    # Final summary
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    print(f"Best Val R²: {best_r2:.4f}")
    print(f"Target (LSTM baseline): 0.75")
    if best_r2 > 0.75:
        print("✅ VAE BEATS LSTM BASELINE!")
    elif best_r2 > 0.70:
        print("⚠️ Close to baseline, may need hyperparameter tuning")
    else:
        print("❌ Below baseline, check model architecture or data")
    print("=" * 70)
    
    # Save final model
    torch.save(model.state_dict(), 'models/vae_final.pth')
    print("\n💾 Final model saved to: models/vae_final.pth")
    
    # Save results
    results = {
        'best_r2': float(best_r2),
        'test_r2': float(test_r2),
        'test_mae': float(test_mae),
        'horizon': int(horizon),
        'num_params': int(num_params),
        'architecture': 'AdvancedHybridVAE',
        'training': 'multi-stage (20+20+40 epochs)'
    }
    
    with open('results/vae_results_advanced.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("📊 Results saved to: results/vae_results_advanced.json")


if __name__ == '__main__':
    main()
