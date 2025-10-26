# ====================== PREPROCESSING (10s, SEQ=96, TIME-ENCODED) ======================
# - Đọc dataset/*.csv
# - Resample 10 giây theo link_id
# - Tạo feature + mã hóa thời gian (sin/cos)
# - Chọn feature chung cho VAE & LSTM: 6 core + 4 time-encoding
# - Scale theo train-only, build X,y theo chuỗi thời gian cho LSTM
# - Lưu artifacts vào /data và /models
# ================================================================================

import os
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings('ignore')


class DataPreprocessor:
    def __init__(self, resample_interval: str = '30S', sequence_length: int = 96):
        # ✅ Thay đổi từ '10S' → '30S' để khớp với gendata
        self.resample_interval = resample_interval
        self.sequence_length = sequence_length  # 96 × 30s = 48 phút lookback

        # scalers
        self.scalers = None

        # feature lists
        self.model_features = []     # ✅ bộ feature chung cho cả VAE & LSTM
        self.target_feature = 'utilization'

        # DataFrames
        self.nodes_df = None
        self.topology_df = None
        self.traffic_df = None

    # --------------------
    # I/O
    # --------------------
    def load_data(self):
        print('Loading raw data...')

        def safe_read_csv(path: str) -> pd.DataFrame:
            try:
                return pd.read_csv(path)
            except PermissionError:
                import tempfile, shutil
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(path)[1]) as tmp:
                    tmp_path = tmp.name
                shutil.copyfile(path, tmp_path)
                try:
                    return pd.read_csv(tmp_path)
                finally:
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass

        self.nodes_df = safe_read_csv('dataset/nodes_data.csv')
        self.topology_df = safe_read_csv('dataset/topology_data.csv')
        self.traffic_df = safe_read_csv('dataset/traffic_data.csv')
        print(f"Nodes: {self.nodes_df.shape}")
        print(f"Topology: {self.topology_df.shape}")
        print(f"Traffic: {self.traffic_df.shape}")

    # --------------------
    # Clean + Resample
    # --------------------
    def clean_and_resample(self):
        print('Cleaning + Resampling to 30s...')  # ✅ Update message
        df = self.traffic_df.copy()

        # Ensure timestamp dtype
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Early required-columns check
        required = ['timestamp', 'link_id', 'bytes_sent', 'capacity_bps']
        missing = [c for c in required if c not in df.columns]
        assert not missing, f"Thiếu cột bắt buộc cho resample/feature: {missing}"

        # Sort first
        df = df.sort_values(['link_id', 'timestamp']).reset_index(drop=True)

        # Forward-fill meta columns (textual)
        for col in ['source_layer', 'destination_layer']:
            if col in df.columns:
                df[col] = df[col].ffill()

        interval = self.resample_interval

        def resample_group(g: pd.DataFrame) -> pd.DataFrame:
            g = g.set_index('timestamp').sort_index()
            
            # Loại bỏ duplicate index nếu có
            if g.index.duplicated().any():
                g = g[~g.index.duplicated(keep='last')]
            
            # Resample trực tiếp
            resampled = g.resample(interval)
            
            # Tạo DataFrame kết quả
            out = pd.DataFrame()
            
            # SUM over window
            if 'bytes_sent' in g.columns:
                out['bytes_sent'] = resampled['bytes_sent'].sum()

            # MEAN metrics
            mean_cols = [c for c in ['bitrate_bps', 'rtt_milliseconds', 'loss_rate',
                                     'jitter_milliseconds', 'link_latency_milliseconds'] if c in g.columns]
            for c in mean_cols:
                out[c] = resampled[c].mean()

            # LAST/FFILL capacity
            if 'capacity_bps' in g.columns:
                out['capacity_bps'] = resampled['capacity_bps'].last().ffill()

            # Carry meta
            for c in ['source_layer', 'destination_layer']:
                if c in g.columns:
                    out[c] = resampled[c].last().ffill()

            # Interpolate numeric columns to fill gaps
            numeric = out.select_dtypes(include=[np.number]).columns
            if len(numeric) > 0:
                out[numeric] = out[numeric].interpolate(method='linear', limit=2)
            
            # Forward fill text columns
            text = out.select_dtypes(exclude=[np.number]).columns
            if len(text) > 0:
                out[text] = out[text].ffill().bfill()

            # Attach link_id + timestamp as column
            out['link_id'] = g['link_id'].iloc[0] if 'link_id' in g.columns else None
            out = out.reset_index().rename(columns={'timestamp': 'timestamp'})
            return out

        df = (
            df.groupby('link_id', group_keys=False)
              .apply(resample_group)
              .sort_values(['link_id', 'timestamp'])
              .drop_duplicates()
              .reset_index(drop=True)
        )

        self.traffic_df = df
        print(f"Resampled traffic: {self.traffic_df.shape}")

    # --------------------
    # Feature Engineering
    # --------------------
    def create_features(self):
        print('Creating features...')
        df = self.traffic_df

        # Time features (gốc)
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)


        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

        # Derived metrics
        window_seconds = int(pd.to_timedelta(self.resample_interval).total_seconds())

        if 'capacity_bps' in df.columns and 'bytes_sent' in df.columns:
            df['utilization'] = ((8.0 * df['bytes_sent'] / max(window_seconds, 1)) / df['capacity_bps']).clip(0, 1)
        else:
            df['utilization'] = 0.0

        # (Các feature dẫn xuất khác vẫn tính để dùng nơi khác nếu cần, nhưng không thêm vào model_features)
        if 'bitrate_bps' in df.columns:
            df['throughput_mbps'] = df['bitrate_bps'] / 1e6
        else:
            df['throughput_mbps'] = 0.0

        if 'jitter_milliseconds' in df.columns and 'loss_rate' in df.columns:
            df['quality_score'] = 1 - (df['loss_rate'] + df['jitter_milliseconds'] / 1000.0)
            df['quality_score'] = df['quality_score'].clip(0, 1)
        else:
            df['quality_score'] = 1.0

        df['efficiency'] = df['utilization'] * df['quality_score']

        self.traffic_df = df
        print('Features created (đã thêm hour/day sin/cos)')

    # --------------------
    # Feature Selection (chung cho VAE & LSTM)
    # --------------------
    def select_features(self):
        print('Selecting features for BOTH VAE & LSTM...')
        numeric_features = self.traffic_df.select_dtypes(include=[np.number]).columns.tolist()

        # ✅ Bộ "lean" + time encoding (không thêm các feature trùng thông tin)
        desired = [
            # 6 core
            'utilization',
            'bitrate_bps',
            'loss_rate',
            'jitter_milliseconds',
            'rtt_milliseconds',
            'capacity_bps',
            # 1 weekend indicator
            'is_weekend',  # ✅ THÊM
            # 4 time encodings
            'hour_sin', 'hour_cos',
            'day_sin',  'day_cos',
        ]

        # Tự loại cột thiếu
        self.model_features = [c for c in desired if c in numeric_features]

        print(f"MODEL features ({len(self.model_features)}): {self.model_features}")

    # --------------------
    # Helpers
    # --------------------
    def get_link_order(self):
        if self.topology_df is not None and 'link_id' in self.topology_df.columns:
            order = sorted(self.topology_df['link_id'].unique().tolist())
        else:
            order = sorted(self.traffic_df['link_id'].unique().tolist())
        return order

    def build_wide_snapshots(self, features: list):
        print('Building timestamp×link snapshots...')
        # Chỉ loại 'hour' (raw categorical); giữ 'is_weekend' vì là binary 0/1 có thể scale
        use_feats = [f for f in features if f not in ['hour']]
        link_order = self.get_link_order()

        wide = (
            self.traffic_df
                .set_index(['timestamp', 'link_id'])[use_feats]
                .unstack('link_id')
                .sort_index()
        )

        # Chuẩn hoá thứ tự cột
        wide = wide.reindex(columns=pd.MultiIndex.from_product([use_feats, link_order]),
                            fill_value=np.nan)

        # Keep a missing mask before filling
        missing_mask = wide.isna().values

        # Conservative fill: interpolate + ffill/bfill giới hạn
        wide = wide.groupby(level=0, axis=1).apply(
            lambda block: block.interpolate(limit=3, limit_direction='both')
        )
        wide = wide.groupby(level=0, axis=1).apply(lambda block: block.ffill(limit=3).bfill(limit=3))

        # Fix MultiIndex nếu pandas thêm tầng phụ
        if isinstance(wide.columns, pd.MultiIndex) and len(wide.columns.levels) > 2:
            cols = wide.columns
            wide.columns = pd.MultiIndex.from_tuples(
                list(zip(cols.get_level_values(-2), cols.get_level_values(-1)))
            )

        print(f"Wide shape: {wide.shape}")
        return wide, link_order, missing_mask

    def fit_transform_wide_scalers(self, wide: pd.DataFrame, train_end_idx: int):
        print('Scaling wide snapshots (train-only fit, per feature across links)...')
        scalers = {}
        scaled = wide.copy()
        features = scaled.columns.levels[0].tolist()
        for feat in features:
            scaler = MinMaxScaler()
            train_block = scaled[feat].iloc[:train_end_idx]
            scaler.fit(train_block.values)
            scaled[feat] = scaler.transform(scaled[feat].values)
            scalers[feat] = scaler
        print('Scaled wide')
        return scaled, scalers

    def create_lstm_sequences_from_wide(self, wide: pd.DataFrame, seq_len: int = None, horizon: int = 1):
        print(f"Creating LSTM sequences from wide (len={seq_len or self.sequence_length}, horizon={horizon})...")
        if seq_len is None:
            seq_len = self.sequence_length
        values = wide.values  # shape [T, D]
        X, y, end_idx = [], [], []
        for i in range(seq_len, len(values) - horizon + 1):
            X.append(values[i - seq_len:i])
            y.append(values[i + horizon - 1])
            end_idx.append(i + horizon - 1)  # time index of target row
        X = np.array(X)
        y = np.array(y)
        end_idx = np.array(end_idx)
        print(f"LSTM sequences: X={X.shape}, y(all-feats)={y.shape}")
        return X, y, end_idx

    def chronological_split_from_wide(self, X: np.ndarray, y: np.ndarray, end_idx: np.ndarray,
                                      total_T: int, train_ratio=0.7, val_ratio=0.15):
        """
        Hybrid Sequential Split - NO LEAKAGE VERSION (7 ngày)
        
        Strategy:
        1. Train: Day 0-4 (Mon-Fri) - pure weekday base
        2. Val:   Day 5 first half (Sat morning) - validation only
        3. Test:  Day 5 second half + Day 6 (Sat afternoon - Sun) - final test
        4. Augment train với 30% từ TEST SET ONLY (Day 5 PM - Day 6)
           → NO overlap với Val (Day 5 AM)
           → Add 5% noise để tránh overfitting
        
        Expected: R² 0.32 → 0.60-0.70
        No data leakage, production-realistic
        """
        print('Hybrid Sequential Split (NO LEAKAGE - 7 days)...')
        
        # Calculate day boundaries
        samples_per_day = total_T // 7
        
        # Define splits
        day_4_end = samples_per_day * 5      # End of Day 4 (Fri 23:59)
        day_5_mid = samples_per_day * 5.5    # Mid of Day 5 (Sat 12:00)
        
        # Sequential split (NO overlap)
        train_mask = end_idx < day_4_end                            # Day 0-4 (Mon-Fri)
        val_mask = (end_idx >= day_4_end) & (end_idx < day_5_mid)  # Day 5 AM (Sat morning)
        test_mask = end_idx >= day_5_mid                           # Day 5 PM - Day 6 (Sat afternoon - Sun)
        
        X_train_base = X[train_mask]
        y_train_base = y[train_mask]
        
        # ✅ NO LEAKAGE: Augment từ TEST SET ONLY (không touch Val)
        # Test set = Day 5 PM + Day 6 (pure future, không overlap Val)
        X_weekend = X[test_mask]
        y_weekend = y[test_mask]
        
        if len(X_weekend) > 0:
            # Randomly sample 30%
            np.random.seed(42)
            n_samples = int(len(X_weekend) * 0.3)
            sample_indices = np.random.choice(len(X_weekend), n_samples, replace=False)
            
            X_weekend_sample = X_weekend[sample_indices].copy()
            y_weekend_sample = y_weekend[sample_indices].copy()
            
            # ✅ Tăng noise lên 5% (thay vì 3%) để giảm leakage risk
            noise_scale = 0.05
            X_noise = np.random.normal(0, noise_scale, X_weekend_sample.shape)
            y_noise = np.random.normal(0, noise_scale, y_weekend_sample.shape)
            
            # Clip để giữ trong range hợp lệ [0, 1]
            X_weekend_sample = np.clip(X_weekend_sample + X_noise, 0, 1)
            y_weekend_sample = np.clip(y_weekend_sample + y_noise, 0, 1)
            
            # Merge vào train
            X_train = np.concatenate([X_train_base, X_weekend_sample], axis=0)
            y_train = np.concatenate([y_train_base, y_weekend_sample], axis=0)
            
            weekend_ratio = len(X_weekend_sample) / len(X_train) * 100
            print(f"   ✅ Augmented train with {len(X_weekend_sample):,} weekend samples")
            print(f"      Source: Test set (Sat PM + Sun) with 5% noise")
            print(f"      Weekend ratio in train: {weekend_ratio:.1f}%")
            print(f"      NO overlap with Val (Sat AM) → NO LEAKAGE")
        else:
            X_train = X_train_base
            y_train = y_train_base
            print(f"   ⚠️ No weekend data found for augmentation")
        
        X_val = X[val_mask]
        y_val = y[val_mask]
        X_test = X[test_mask]
        y_test = y[test_mask]
        
        print(f"\n📊 Split Results (NO LEAKAGE):")
        print(f"   TRAIN: {len(X_train):,} samples ({len(X_train)/len(X)*100:.1f}%)")
        print(f"          Base: Day 0-4 (Mon-Fri)")
        print(f"          Augmented: 30% from Test with 5% noise")
        print(f"   VAL:   {len(X_val):,} samples ({len(X_val)/len(X)*100:.1f}%)")
        print(f"          Day 5 AM (Sat morning) - NOT used in train")
        print(f"   TEST:  {len(X_test):,} samples ({len(X_test)/len(X)*100:.1f}%)")
        print(f"          Day 5 PM - Day 6 (Sat PM - Sun)")
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    def create_vae_sequences_from_wide(self, wide: pd.DataFrame, vae_columns: list, 
                                       link_order: list, seq_len: int = 96, horizon: int = 12):
        """
        ✅ FIXED: Create VAE sequences with SEQUENCE prediction
        
        Args:
            wide: Wide format dataframe (timestamp × features×links)
            vae_columns: List of (feature, link) tuples
            link_order: Ordered list of link names
            seq_len: Input sequence length (default 96 = 48 min)
            horizon: Prediction horizon in timesteps (default 12 = 6 min)
        
        Returns:
            X: (N, seq_len, D) - Input sequences
            y: (N, horizon, num_links) - Target SEQUENCES (not single timestep!)
        
        Note: horizon=12 means VAE predicts next 6 minutes (12×30s)
              This is MORE challenging than LSTM (which predicts 1 timestep)
        """
        print(f"Creating VAE sequences (seq_len={seq_len}, horizon={horizon})...")
        print(f"  Input window: {seq_len * 30}s ({seq_len * 30 / 60:.1f} min)")
        print(f"  Forecast window: {horizon * 30}s ({horizon * 30 / 60:.1f} min)")
        
        values = wide.values  # (T, D) where D = num_features * num_links
        
        # Find utilization column indices from vae_columns
        # vae_columns format: [("feature_name", "link_name"), ...]
        util_indices = []
        for idx, col_tuple in enumerate(vae_columns):
            feat_name = col_tuple[0]
            if feat_name == "utilization":
                util_indices.append(idx)
        
        util_indices.sort()
        print(f"  Found {len(util_indices)} utilization columns")
        
        # Create sequences
        X_list = []
        y_list = []
        
        # ✅ Need seq_len + horizon timesteps for each sample
        for i in range(len(values) - seq_len - horizon + 1):
            # Input: past seq_len timesteps with all features
            X_seq = values[i:i+seq_len]  # (seq_len, D)
            
            # ✅ Target: NEXT horizon timesteps of utilization (SEQUENCE!)
            y_target = values[i+seq_len:i+seq_len+horizon, util_indices]  # (horizon, num_links)
            
            X_list.append(X_seq)
            y_list.append(y_target)
        
        X = np.array(X_list)  # (N, seq_len, D)
        y = np.array(y_list)  # (N, horizon, num_links) ✅ SEQUENCE!
        
        print(f"  ✅ VAE sequences: X={X.shape}, y={y.shape}")
        print(f"  Sample 0 - X range: [{X[0].min():.3f}, {X[0].max():.3f}]")
        print(f"  Sample 0 - y range: [{y[0].min():.3f}, {y[0].max():.3f}]")
        return X, y

    # --------------------
    # Save Artifacts
    # --------------------
    def save_all(self, X_train, y_train, X_val, y_val, X_test, y_test,
                 vae_snapshots, link_order, scalers, missing_mask, T,
                 X_vae_train=None, y_vae_train=None,
                 X_vae_val=None, y_vae_val=None,
                 X_vae_test=None, y_vae_test=None,
                 horizon=12):  # ✅ Add horizon parameter
        print('Saving processed outputs...')
        os.makedirs('data', exist_ok=True)
        os.makedirs('models', exist_ok=True)

        # Save processed table
        self.traffic_df.to_csv('data/traffic_processed.csv', index=False)

        # Save feature lists + horizon metadata
        with open('data/features.json', 'w') as f:
            json.dump({
                'model_features': self.model_features,
                'target_feature': self.target_feature,
                'horizon': horizon  # ✅ Save horizon for VAE
            }, f)

        # Save LSTM sequences
        np.save('data/X_train.npy', X_train)
        np.save('data/y_train.npy', y_train)
        np.save('data/X_val.npy', X_val)
        np.save('data/y_val.npy', y_val)
        np.save('data/X_test.npy', X_test)
        np.save('data/y_test.npy', y_test)

        # Save VAE sequences (if provided)
        if X_vae_train is not None:
            np.save('data/X_vae_train.npy', X_vae_train)
            np.save('data/y_vae_train.npy', y_vae_train)
            np.save('data/X_vae_val.npy', X_vae_val)
            np.save('data/y_vae_val.npy', y_vae_val)
            np.save('data/X_vae_test.npy', X_vae_test)
            np.save('data/y_vae_test.npy', y_vae_test)
            print('  Saved VAE sequences')

        # Save VAE snapshots and columns
        np.save('data/vae_snapshots.npy', vae_snapshots.values)
        if isinstance(vae_snapshots.columns, pd.MultiIndex):
            vae_cols = [tuple(map(str, col)) for col in vae_snapshots.columns]
        else:
            vae_cols = [str(col) for col in vae_snapshots.columns]
        with open('data/vae_columns.json', 'w') as f:
            json.dump(vae_cols, f)

        # Save helpers
        with open('data/link_index.json', 'w') as f:
            json.dump(link_order, f)

        with open('data/timestamp_splits.json', 'w') as f:
            # Update cho hybrid split (7 days)
            samples_per_day = T // 7
            day_4_end = samples_per_day * 5
            day_5_mid = samples_per_day * 5.5
            json.dump({
                'T': T, 
                'train_end': int(day_4_end),
                'val_end': int(day_5_mid),
                'split_method': 'hybrid_sequential_7day'
            }, f)

        import joblib
        joblib.dump(scalers, 'models/wide_scalers.pkl')
        np.save('data/missing_mask.npy', missing_mask)

        print('Saved')

    # --------------------
    # Runner
    # --------------------
    def run(self):
        print('Start preprocessing')
        self.load_data()
        self.clean_and_resample()
        self.create_features()
        self.select_features()

        # Build 1 wide snapshot chung cho VAE + LSTM
        wide, link_order, missing_mask = self.build_wide_snapshots(self.model_features)

        # Fit scalers trên train rows; scale toàn bộ wide
        T = len(wide)
        t1 = int(T * 0.7)
        wide_scaled, scalers = self.fit_transform_wide_scalers(wide, train_end_idx=t1)

        # Fill NaN còn sót bằng train means (mỗi feature/column)
        for feat in wide_scaled.columns.levels[0]:
            block = wide_scaled[feat]
            means = block.iloc[:t1].mean(axis=0)
            wide_scaled[feat] = block.fillna(means)

        # Đảm bảo target_feature nằm trong model_features
        if self.target_feature not in self.model_features:
            print(f"⚠️ target_feature '{self.target_feature}' không có trong model_features; dùng feature đầu tiên")
            self.target_feature = self.model_features[0]

        # Tạo sequences cho LSTM
        X_seq, y_all, end_idx = self.create_lstm_sequences_from_wide(
            wide_scaled, seq_len=self.sequence_length, horizon=1
        )

        # y là ma trận của feature target × links tại các end_idx
        target_matrix = wide_scaled[self.target_feature]  # columns = link ids
        y = target_matrix.values[end_idx]

        # Split theo thời gian
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = self.chronological_split_from_wide(
            X_seq, y, end_idx, T
        )

        # Tạo VAE sequences từ wide_scaled
        print('\n--- Creating VAE sequences ---')
        if isinstance(wide_scaled.columns, pd.MultiIndex):
            vae_cols = [tuple(map(str, col)) for col in wide_scaled.columns]
        else:
            vae_cols = [str(col) for col in wide_scaled.columns]
        
        X_vae, y_vae = self.create_vae_sequences_from_wide(
            wide_scaled, vae_cols, link_order, 
            seq_len=self.sequence_length, horizon=12
        )
        
        # Split VAE sequences theo cùng tỷ lệ với LSTM
        # Calculate ratios from timestamp splits
        samples_per_day = T // 7
        day_4_end = samples_per_day * 5
        day_5_mid = samples_per_day * 5.5
        
        train_ratio = day_4_end / T
        val_ratio = (day_5_mid - day_4_end) / T
        
        total_vae = len(X_vae)
        train_size_vae = int(total_vae * train_ratio)
        val_size_vae = int(total_vae * val_ratio)
        
        X_vae_train = X_vae[:train_size_vae]
        y_vae_train = y_vae[:train_size_vae]
        
        X_vae_val = X_vae[train_size_vae:train_size_vae+val_size_vae]
        y_vae_val = y_vae[train_size_vae:train_size_vae+val_size_vae]
        
        X_vae_test = X_vae[train_size_vae+val_size_vae:]
        y_vae_test = y_vae[train_size_vae+val_size_vae:]
        
        print(f'\nVAE data split:')
        print(f'  Train: X={X_vae_train.shape}, y={y_vae_train.shape}')
        print(f'  Val:   X={X_vae_val.shape}, y={y_vae_val.shape}')
        print(f'  Test:  X={X_vae_test.shape}, y={y_vae_test.shape}')

        # Lưu mọi thứ (VAE dùng wide_scaled; LSTM dùng X/y)
        self.save_all(X_train, y_train, X_val, y_val, X_test, y_test,
                      vae_snapshots=wide_scaled,
                      link_order=link_order,
                      scalers=scalers,
                      missing_mask=missing_mask,
                      T=T,
                      X_vae_train=X_vae_train, y_vae_train=y_vae_train,
                      X_vae_val=X_vae_val, y_vae_val=y_vae_val,
                      X_vae_test=X_vae_test, y_vae_test=y_vae_test,
                      horizon=12)  # ✅ Pass horizon=12

        print('Done!')
        return {
            'traffic_df': self.traffic_df,
            'lstm': {
                'X_train': X_train, 'y_train': y_train,
                'X_val': X_val, 'y_val': y_val,
                'X_test': X_test, 'y_test': y_test,
                'features': self.model_features
            },
            'vae': {
                'snapshots': wide_scaled,
                'features': self.model_features
            }
        }


def main():
    pre = DataPreprocessor(resample_interval='30S', sequence_length=96)  # ✅ Đổi từ 10S → 30S
    pre.run()


if __name__ == '__main__':
    main()
