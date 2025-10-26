"""
Pipeline Validation Script

Kiểm tra toàn bộ data pipeline sau khi preprocessing:
- Features configuration
- Data shapes (LSTM & VAE)
- Data quality (NaN, Inf, range)
- Split method & timestamps
"""

import numpy as np
import json
import os

def check_file_exists(filepath):
    """Check if file exists and return True/False"""
    exists = os.path.exists(filepath)
    status = "✅" if exists else "❌"
    print(f"{status} {filepath}")
    return exists

def check_data_quality(data, name):
    """Check for NaN, Inf, and data range"""
    print(f"\n{name} Quality Check:")
    
    # NaN check
    nan_count = np.isnan(data).sum()
    nan_pct = (nan_count / data.size) * 100
    print(f"  NaN: {nan_count:,} / {data.size:,} ({nan_pct:.4f}%)")
    
    # Inf check
    inf_count = np.isinf(data).sum()
    print(f"  Inf: {inf_count:,}")
    
    # Range
    print(f"  Range: [{data.min():.4f}, {data.max():.4f}]")
    print(f"  Mean: {data.mean():.4f}, Std: {data.std():.4f}")
    
    # Warnings
    if nan_count > 0:
        print(f"  ⚠️ WARNING: Found {nan_count} NaN values!")
    if inf_count > 0:
        print(f"  ⚠️ WARNING: Found {inf_count} Inf values!")
    
    return nan_count == 0 and inf_count == 0

def main():
    print("=" * 70)
    print("DATA PIPELINE VALIDATION")
    print("=" * 70)
    
    all_checks_passed = True
    
    # 1. Check files exist
    print("\n📁 File Existence Check:")
    files_to_check = [
        'data/features.json',
        'data/X_train.npy', 'data/y_train.npy',
        'data/X_val.npy', 'data/y_val.npy',
        'data/X_test.npy', 'data/y_test.npy',
        'data/X_vae_train.npy', 'data/y_vae_train.npy',
        'data/X_vae_val.npy', 'data/y_vae_val.npy',
        'data/X_vae_test.npy', 'data/y_vae_test.npy',
        'data/timestamp_splits.json'
    ]
    
    for filepath in files_to_check:
        if not check_file_exists(filepath):
            all_checks_passed = False
    
    # 2. Load and check features
    print("\n📋 Features Configuration:")
    try:
        with open('data/features.json') as f:
            features = json.load(f)
            model_features = features['model_features']
            target_feature = features['target_feature']
            horizon = features.get('horizon', 1)
        
        print(f"  Model features ({len(model_features)}): {model_features}")
        print(f"  Target feature: {target_feature}")
        print(f"  Horizon (VAE): {horizon} timesteps = {horizon * 30}s = {horizon * 30 / 60:.1f} min")
        
        if horizon == 1:
            print(f"  ⚠️ WARNING: horizon=1 (old format). VAE will predict single timestep!")
            print(f"     → Re-run preprocessing.py to use horizon=12 (sequence prediction)")
        else:
            print(f"  ✅ horizon={horizon} (sequence prediction enabled)")
    
    except Exception as e:
        print(f"  ❌ Error loading features.json: {e}")
        all_checks_passed = False
        return
    
    # 3. Check LSTM shapes
    print("\n📊 LSTM Data Shapes:")
    try:
        X_train = np.load('data/X_train.npy')
        y_train = np.load('data/y_train.npy')
        X_val = np.load('data/X_val.npy')
        y_val = np.load('data/y_val.npy')
        X_test = np.load('data/X_test.npy')
        y_test = np.load('data/y_test.npy')
        
        print(f"  X_train: {X_train.shape}  # (N, seq_len=96, features)")
        print(f"  y_train: {y_train.shape}  # (N, num_links)")
        print(f"  X_val:   {X_val.shape}")
        print(f"  y_val:   {y_val.shape}")
        print(f"  X_test:  {X_test.shape}")
        print(f"  y_test:  {y_test.shape}")
        
        # Validate dimensions
        expected_seq_len = 96
        expected_features = X_train.shape[2]
        num_links = y_train.shape[1]
        
        if X_train.shape[1] != expected_seq_len:
            print(f"  ⚠️ WARNING: seq_len={X_train.shape[1]}, expected {expected_seq_len}")
        
        print(f"\n  Derived parameters:")
        print(f"    - Sequence length: {X_train.shape[1]}")
        print(f"    - Input features: {expected_features}")
        print(f"    - Number of links: {num_links}")
        
        # Check quality
        lstm_quality_ok = (
            check_data_quality(X_train, "X_train (LSTM)") and
            check_data_quality(y_train, "y_train (LSTM)")
        )
        
        if not lstm_quality_ok:
            all_checks_passed = False
    
    except Exception as e:
        print(f"  ❌ Error loading LSTM data: {e}")
        all_checks_passed = False
        return
    
    # 4. Check VAE shapes
    print("\n📊 VAE Data Shapes:")
    try:
        X_vae_train = np.load('data/X_vae_train.npy')
        y_vae_train = np.load('data/y_vae_train.npy')
        X_vae_val = np.load('data/X_vae_val.npy')
        y_vae_val = np.load('data/y_vae_val.npy')
        X_vae_test = np.load('data/X_vae_test.npy')
        y_vae_test = np.load('data/y_vae_test.npy')
        
        print(f"  X_vae_train: {X_vae_train.shape}  # (N, seq_len=96, features)")
        print(f"  y_vae_train: {y_vae_train.shape}  # (N, horizon, num_links) or (N, num_links)")
        print(f"  X_vae_val:   {X_vae_val.shape}")
        print(f"  y_vae_val:   {y_vae_val.shape}")
        print(f"  X_vae_test:  {X_vae_test.shape}")
        print(f"  y_vae_test:  {y_vae_test.shape}")
        
        # Check if y is sequence or single timestep
        if y_vae_train.ndim == 2:
            print(f"\n  ⚠️ WARNING: y_vae is 2D (single timestep prediction)")
            print(f"     Current shape: (N={y_vae_train.shape[0]}, num_links={y_vae_train.shape[1]})")
            print(f"     → Re-run preprocessing.py to get 3D shape: (N, horizon, num_links)")
            all_checks_passed = False
        elif y_vae_train.ndim == 3:
            print(f"\n  ✅ y_vae is 3D (sequence prediction)")
            print(f"     Shape: (N={y_vae_train.shape[0]}, horizon={y_vae_train.shape[1]}, num_links={y_vae_train.shape[2]})")
            
            # Check horizon matches
            if y_vae_train.shape[1] != horizon:
                print(f"  ⚠️ WARNING: y_vae horizon={y_vae_train.shape[1]} != features.json horizon={horizon}")
                all_checks_passed = False
        
        # Check quality
        vae_quality_ok = (
            check_data_quality(X_vae_train, "X_vae_train") and
            check_data_quality(y_vae_train, "y_vae_train")
        )
        
        if not vae_quality_ok:
            all_checks_passed = False
    
    except Exception as e:
        print(f"  ❌ Error loading VAE data: {e}")
        all_checks_passed = False
        return
    
    # 5. Check splits
    print("\n📅 Data Splits:")
    try:
        with open('data/timestamp_splits.json') as f:
            splits = json.load(f)
        
        print(f"  Split method: {splits['split_method']}")
        print(f"  Train end: {splits['train_end']}")
        print(f"  Val end:   {splits['val_end']}")
        print(f"  Test end:  {splits.get('test_end', 'N/A')}")
        
        if splits['split_method'] == 'hybrid_sequential':
            print(f"  ✅ Using hybrid sequential split (train augmented with weekend data)")
        else:
            print(f"  ℹ️ Using {splits['split_method']} split")
    
    except Exception as e:
        print(f"  ❌ Error loading splits: {e}")
        all_checks_passed = False
    
    # 6. Summary
    print("\n" + "=" * 70)
    if all_checks_passed:
        print("✅ ALL CHECKS PASSED!")
        print("\nNext steps:")
        print("  1. Train LSTM: python train_lstm.py")
        print("  2. Train VAE:  python train_vae.py")
    else:
        print("❌ SOME CHECKS FAILED!")
        print("\nRecommendations:")
        print("  - If horizon=1 or y_vae is 2D: Re-run preprocessing.py")
        print("  - If NaN/Inf found: Check data cleaning in preprocessing")
        print("  - If dimension mismatch: Delete old checkpoints in models/")
    print("=" * 70)

if __name__ == '__main__':
    main()
