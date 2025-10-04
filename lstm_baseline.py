import os
import json
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
import warnings
warnings.filterwarnings('ignore')

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2, output_size=None):
        """
        LSTM Baseline Model for Network Traffic Prediction
        
        Args:
            input_size: Số features đầu vào (features * links)
            hidden_size: Kích thước hidden state
            num_layers: Số lớp LSTM
            dropout: Dropout rate
            output_size: Số features đầu ra (mặc định = số links)
        """
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        if output_size is None:
            output_size = input_size
            
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, features]
        batch_size = x.size(0)
        
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # Take the last output
        last_output = lstm_out[:, -1, :]  # [batch_size, hidden_size]
        
        # Fully connected layers
        output = self.fc(last_output)
        
        return output

class LSTMTrainer:
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.train_losses = []
        self.val_losses = []
        
    def train_epoch(self, dataloader, optimizer, criterion):
        self.model.train()
        total_loss = 0
        
        for batch_x, batch_y in tqdm(dataloader, desc="Training", leave=False):
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
            
        return total_loss / len(dataloader)
    
    def validate_epoch(self, dataloader, criterion):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                total_loss += loss.item()
                
        return total_loss / len(dataloader)
    
    def train(self, train_loader, val_loader, epochs=100, lr=0.001, patience=10):
        """
        Train LSTM model với early stopping
        """
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        criterion = nn.MSELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=5, factor=0.5
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"Starting training for {epochs} epochs...")
        print(f"Device: {self.device}")
        
        for epoch in range(epochs):
            # Training
            train_loss = self.train_epoch(train_loader, optimizer, criterion)
            
            # Validation
            val_loss = self.validate_epoch(val_loader, criterion)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Save losses
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                os.makedirs('models', exist_ok=True)
                torch.save(self.model.state_dict(), 'models/best_lstm_model.pth')
            else:
                patience_counter += 1
            
            # Print progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch [{epoch+1}/{epochs}]:")
                print(f"  Train Loss: {train_loss:.6f}")
                print(f"  Val Loss: {val_loss:.6f}")
                print(f"  Best Val Loss: {best_val_loss:.6f}")
                print(f"  Patience: {patience_counter}/{patience}")
                print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")
            
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        self.model.load_state_dict(torch.load('models/best_lstm_model.pth'))
        print("Training completed!")
        
        return self.train_losses, self.val_losses

class LSTMEvaluator:
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        
    def predict(self, dataloader):
        """
        Dự đoán trên test set
        """
        self.model.eval()
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch_x, batch_y in tqdm(dataloader, desc="Predicting"):
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                outputs = self.model(batch_x)
                
                predictions.append(outputs.cpu().numpy())
                targets.append(batch_y.cpu().numpy())
        
        predictions = np.vstack(predictions)
        targets = np.vstack(targets)
        
        return predictions, targets
    
    def evaluate_metrics(self, y_true, y_pred):
        """
        Tính toán các metrics đánh giá
        """
        # Overall metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Per-link metrics (assuming last dimension is links)
        if len(y_true.shape) > 1 and y_true.shape[1] > 1:
            per_link_mse = np.mean((y_true - y_pred) ** 2, axis=0)
            per_link_mae = np.mean(np.abs(y_true - y_pred), axis=0)
        else:
            per_link_mse = [mse]
            per_link_mae = [mae]
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'per_link_mse': per_link_mse,
            'per_link_mae': per_link_mae
        }
        
        return metrics
    
    def plot_results(self, y_true, y_pred, link_names=None, save_path='results/lstm_results.png'):
        """
        Visualize prediction results
        """
        os.makedirs('results', exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Nếu data có nhiều links, chỉ plot vài links đầu
        if len(y_true.shape) > 1 and y_true.shape[1] > 1:
            n_links_to_plot = min(4, y_true.shape[1])
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes = axes.flatten()
            
            for i in range(n_links_to_plot):
                ax = axes[i]
                
                # Plot subset of predictions (first 200 samples)
                n_samples = min(200, len(y_true))
                ax.plot(y_true[:n_samples, i], label='True', alpha=0.8, linewidth=1.5)
                ax.plot(y_pred[:n_samples, i], label='Predicted', alpha=0.8, linewidth=1.5)
                
                link_name = link_names[i] if link_names else f'Link {i}'
                ax.set_title(f'{link_name}', fontsize=12, fontweight='bold')
                ax.set_xlabel('Time Steps')
                ax.set_ylabel('Utilization')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
        else:
            # Single output case
            plt.figure(figsize=(12, 6))
            n_samples = min(200, len(y_true))
            plt.plot(y_true[:n_samples], label='True', alpha=0.8, linewidth=1.5)
            plt.plot(y_pred[:n_samples], label='Predicted', alpha=0.8, linewidth=1.5)
            plt.title('LSTM Predictions vs True Values', fontsize=14, fontweight='bold')
            plt.xlabel('Time Steps')
            plt.ylabel('Target Value')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_scatter(self, y_true, y_pred, save_path='results/lstm_scatter.png'):
        """
        Scatter plot of predictions vs true values
        """
        plt.figure(figsize=(10, 8))
        
        if len(y_true.shape) > 1 and y_true.shape[1] > 1:
            # For multi-output, flatten arrays
            y_true_flat = y_true.flatten()
            y_pred_flat = y_pred.flatten()
        else:
            y_true_flat = y_true
            y_pred_flat = y_pred
        
        plt.scatter(y_true_flat, y_pred_flat, alpha=0.5, s=1)
        
        # Perfect prediction line
        min_val = min(y_true_flat.min(), y_pred_flat.min())
        max_val = max(y_true_flat.max(), y_pred_flat.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.title('Predictions vs True Values')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

def load_data():
    """
    Load preprocessed data
    """
    print("Loading preprocessed data...")
    
    # Load arrays
    X_train = np.load('data/X_train.npy')
    y_train = np.load('data/y_train.npy')
    X_val = np.load('data/X_val.npy')
    y_val = np.load('data/y_val.npy')
    X_test = np.load('data/X_test.npy')
    y_test = np.load('data/y_test.npy')
    
    # Load metadata
    with open('data/features.json', 'r') as f:
        features = json.load(f)
    
    with open('data/link_index.json', 'r') as f:
        link_names = json.load(f)
    
    print(f"Data shapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  X_val: {X_val.shape}")
    print(f"  y_val: {y_val.shape}")
    print(f"  X_test: {X_test.shape}")
    print(f"  y_test: {y_test.shape}")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), features, link_names

def create_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size=32):
    """
    Create PyTorch DataLoaders
    """
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)
    
    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, test_loader

def plot_training_curves(train_losses, val_losses, save_path='results/training_curves.png'):
    """
    Plot training and validation curves
    """
    os.makedirs('results', exist_ok=True)
    
    plt.figure(figsize=(12, 5))
    
    # Loss curves
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss', alpha=0.8)
    plt.plot(val_losses, label='Validation Loss', alpha=0.8)
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Smoothed curves
    plt.subplot(1, 2, 2)
    window = min(10, len(train_losses) // 10)
    if window > 1:
        train_smooth = pd.Series(train_losses).rolling(window=window).mean()
        val_smooth = pd.Series(val_losses).rolling(window=window).mean()
        plt.plot(train_smooth, label=f'Training (smoothed, window={window})', alpha=0.8)
        plt.plot(val_smooth, label=f'Validation (smoothed, window={window})', alpha=0.8)
    else:
        plt.plot(train_losses, label='Training Loss', alpha=0.8)
        plt.plot(val_losses, label='Validation Loss', alpha=0.8)
    
    plt.title('Smoothed Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """
    Main training and evaluation pipeline
    """
    print("LSTM Baseline Model Training (PyTorch)")
    print("=" * 50)
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # Load data
    (X_train, y_train), (X_val, y_val), (X_test, y_test), features, link_names = load_data()
    
    # Create dataloaders
    batch_size = 64 if torch.cuda.is_available() else 32
    train_loader, val_loader, test_loader = create_dataloaders(
        X_train, y_train, X_val, y_val, X_test, y_test, batch_size=batch_size
    )
    
    # Model parameters
    input_size = X_train.shape[-1]  # features * links
    output_size = y_train.shape[-1]  # number of links (for utilization prediction)
    
    print(f"Model configuration:")
    print(f"  Input size: {input_size}")
    print(f"  Output size: {output_size}")
    print(f"  Sequence length: {X_train.shape[1]}")
    print(f"  Batch size: {batch_size}")
    
    # Create model
    model = LSTMModel(
        input_size=input_size,
        hidden_size=128,
        num_layers=3,
        dropout=0.3,
        output_size=output_size
    )
    
    # Model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    
    # Training
    trainer = LSTMTrainer(model, device)
    train_losses, val_losses = trainer.train(
        train_loader, val_loader,
        epochs=200,
        lr=0.001,
        patience=15
    )
    
    # Plot training curves
    plot_training_curves(train_losses, val_losses)
    
    # Evaluation
    print("\nEvaluating on test set...")
    evaluator = LSTMEvaluator(model, device)
    y_pred, y_true = evaluator.predict(test_loader)
    
    # Calculate metrics
    metrics = evaluator.evaluate_metrics(y_true, y_pred)
    
    print("\nTest Results:")
    print(f"  MSE: {metrics['mse']:.6f}")
    print(f"  RMSE: {metrics['rmse']:.6f}")
    print(f"  MAE: {metrics['mae']:.6f}")
    print(f"  R²: {metrics['r2']:.6f}")
    
    # Per-link results
    if len(metrics['per_link_mse']) > 1:
        print(f"\nPer-link Performance:")
        for i, (link, mse, mae) in enumerate(zip(link_names, metrics['per_link_mse'], metrics['per_link_mae'])):
            print(f"  {link}: MSE={mse:.6f}, MAE={mae:.6f}")
    
    # Plot results
    evaluator.plot_results(y_true, y_pred, link_names)
    evaluator.plot_scatter(y_true, y_pred)
    
    # Save results
    results = {
        'metrics': {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in metrics.items()},
        'model_config': {
            'input_size': input_size,
            'output_size': output_size,
            'hidden_size': 128,
            'num_layers': 3,
            'dropout': 0.3
        },
        'training_config': {
            'epochs': len(train_losses),
            'batch_size': batch_size,
            'learning_rate': 0.001,
            'device': str(device)
        },
        'data_info': {
            'features': features,
            'link_names': link_names,
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'test_samples': len(X_test)
        }
    }
    
    with open('results/lstm_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nTraining and evaluation completed!")
    print("Results saved in 'results/' directory:")
    print("   - lstm_results.json: Detailed metrics")
    print("   - training_curves.png: Training visualization")
    print("   - lstm_results.png: Prediction plots")
    print("   - lstm_scatter.png: Scatter plot")
    print("Best model saved as 'models/best_lstm_model.pth'")

if __name__ == "__main__":
    main()