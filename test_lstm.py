import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import the model class from the original file
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2, output_size=None):
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
        batch_size = x.size(0)
        
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # Take the last output
        last_output = lstm_out[:, -1, :]
        
        # Fully connected layers
        output = self.fc(last_output)
        
        return output

def load_data():
    """Load preprocessed data"""
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

def create_dataloaders(X_test, y_test, batch_size=32):
    """Create PyTorch DataLoader for test set"""
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return test_loader

def test_model():
    """Test the trained LSTM model"""
    print("LSTM Model Testing")
    print("=" * 50)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    (X_train, y_train), (X_val, y_val), (X_test, y_test), features, link_names = load_data()
    
    # Create test dataloader
    test_loader = create_dataloaders(X_test, y_test, batch_size=32)
    
    # Model parameters
    input_size = X_train.shape[-1]
    output_size = y_train.shape[-1]
    
    print(f"Model configuration:")
    print(f"  Input size: {input_size}")
    print(f"  Output size: {output_size}")
    print(f"  Sequence length: {X_train.shape[1]}")
    
    # Create model
    model = LSTMModel(
        input_size=input_size,
        hidden_size=128,
        num_layers=3,
        dropout=0.3,
        output_size=output_size
    )
    
    # Load trained model
    model_path = 'models/best_lstm_model.pth'
    if os.path.exists(model_path):
        print(f"Loading trained model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
    else:
        print(f"Model file not found: {model_path}")
        return
    
    # Test the model
    model.eval()
    predictions = []
    targets = []
    
    print("\nMaking predictions on test set...")
    with torch.no_grad():
        for batch_x, batch_y in tqdm(test_loader, desc="Testing"):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            
            predictions.append(outputs.cpu().numpy())
            targets.append(batch_y.cpu().numpy())
    
    predictions = np.vstack(predictions)
    targets = np.vstack(targets)
    
    # Calculate metrics
    mse = mean_squared_error(targets, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(targets, predictions)
    r2 = r2_score(targets, predictions)
    
    print("\nTest Results:")
    print(f"  MSE: {mse:.6f}")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  MAE: {mae:.6f}")
    print(f"  R²: {r2:.6f}")
    
    # Per-link results
    if len(targets.shape) > 1 and targets.shape[1] > 1:
        print(f"\nPer-link Performance:")
        per_link_mse = np.mean((targets - predictions) ** 2, axis=0)
        per_link_mae = np.mean(np.abs(targets - predictions), axis=0)
        
        for i, (link, mse, mae) in enumerate(zip(link_names, per_link_mse, per_link_mae)):
            print(f"  {link}: MSE={mse:.6f}, MAE={mae:.6f}")
    
    # Plot results
    plot_predictions(targets, predictions, link_names)
    plot_scatter(targets, predictions)
    
    # Show some sample predictions
    print(f"\nSample Predictions (first 5 samples):")
    print("True values vs Predicted values:")
    for i in range(min(5, len(targets))):
        print(f"Sample {i+1}:")
        for j in range(min(3, targets.shape[1])):  # Show first 3 links
            link_name = link_names[j] if j < len(link_names) else f"Link {j}"
            print(f"  {link_name}: True={targets[i,j]:.4f}, Pred={predictions[i,j]:.4f}")
        print()
    
    # Save detailed results
    save_results(targets, predictions, mse, rmse, mae, r2, link_names)

def plot_predictions(y_true, y_pred, link_names=None, save_path='results/test_predictions.png'):
    """Plot prediction results"""
    os.makedirs('results', exist_ok=True)
    
    plt.style.use('default')
    sns.set_palette("husl")
    
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

def plot_scatter(y_true, y_pred, save_path='results/test_scatter.png'):
    """Scatter plot of predictions vs true values"""
    plt.figure(figsize=(10, 8))
    
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
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
    plt.title('Predictions vs True Values (Test Set)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def save_results(y_true, y_pred, mse, rmse, mae, r2, link_names):
    """Save detailed test results"""
    os.makedirs('results', exist_ok=True)
    
    # Calculate per-link metrics
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        per_link_mse = np.mean((y_true - y_pred) ** 2, axis=0)
        per_link_mae = np.mean(np.abs(y_true - y_pred), axis=0)
        per_link_r2 = []
        for i in range(y_true.shape[1]):
            r2_link = r2_score(y_true[:, i], y_pred[:, i])
            per_link_r2.append(r2_link)
    else:
        per_link_mse = [mse]
        per_link_mae = [mae]
        per_link_r2 = [r2]
    
    # Create results dictionary
    results = {
        'overall_metrics': {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2)
        },
        'per_link_metrics': {
            'mse': per_link_mse.tolist() if isinstance(per_link_mse, np.ndarray) else per_link_mse,
            'mae': per_link_mae.tolist() if isinstance(per_link_mae, np.ndarray) else per_link_mae,
            'r2': per_link_r2
        },
        'link_names': link_names,
        'test_samples': len(y_true),
        'model_info': {
            'input_size': y_true.shape[1] if len(y_true.shape) > 1 else 1,
            'sequence_length': 24,
            'prediction_horizon': 12
        }
    }
    
    # Save to JSON
    with open('results/test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed results saved to 'results/test_results.json'")
    print(f"Visualization plots saved to 'results/test_predictions.png' and 'results/test_scatter.png'")

if __name__ == "__main__":
    test_model()
