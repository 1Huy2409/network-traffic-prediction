import os
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')


class DataPreprocessor:
    def __init__(self, resample_interval: str = '10S', sequence_length: int = 24):
        self.resample_interval = resample_interval
        self.sequence_length = sequence_length
        self.lstm_scaler = None
        self.vae_scaler = None
        self.lstm_features = []
        self.vae_features = []
        self.target_feature = 'utilization'

        # DataFrames
        self.nodes_df = None
        self.topology_df = None
        self.traffic_df = None
        
    def load_data(self):
        print('📊 Loading raw data...')
        # function to read csv file
        def safe_read_csv(path: str) -> pd.DataFrame:
            try:
                return pd.read_csv(path)
            except PermissionError:
                # Windows file lock fallback: copy to a temp file then read
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
        print(f"✅ Nodes: {self.nodes_df.shape}")
        print(f"✅ Topology: {self.topology_df.shape}")
        print(f"✅ Traffic: {self.traffic_df.shape}")
        
    def clean_and_resample(self):
        print('🧹 Cleaning + ⏱️ Resampling to 10s...')

        # Ensure timestamp dtype
        self.traffic_df['timestamp'] = pd.to_datetime(self.traffic_df['timestamp'])

        # Early required-columns check to fail fast
        required = ['timestamp', 'link_id', 'bytes_sent', 'capacity_bps']
        missing = [c for c in required if c not in self.traffic_df.columns]
        assert not missing, f"Thiếu cột bắt buộc cho resample/feature: {missing}"
        
        # Sort first
        self.traffic_df = self.traffic_df.sort_values(['link_id', 'timestamp']).reset_index(drop=True)
        
        # Forward-fill basic missing textual/meta columns before resample (numeric handled in agg)
        self.traffic_df[['source_layer', 'destination_layer']] = self.traffic_df[['source_layer', 'destination_layer']].ffill()

        # Custom resample per link with per-column aggregation
        interval = self.resample_interval
        window_seconds = int(pd.to_timedelta(interval).total_seconds()) # độ dài của interval resample tính bằng giây

        def resample_group(g: pd.DataFrame) -> pd.DataFrame: # nhận vào 1 data frame và trả về 1 data frame
            g = g.set_index('timestamp').sort_index()
            idx = g.resample(interval).asfreq().index
            out = pd.DataFrame(index=idx)

            # SUM
            if 'bytes_sent' in g.columns:
                out['bytes_sent'] = g['bytes_sent'].resample(interval).sum()

            # MEAN metrics
            mean_cols = [c for c in ['bitrate_bps', 'rtt_milliseconds', 'loss_rate',
                                     'jitter_milliseconds', 'link_latency_milliseconds'] if c in g.columns]
            for c in mean_cols:
                out[c] = g[c].resample(interval).mean()

            # LAST/FFILL capacity
            if 'capacity_bps' in g.columns:
                out['capacity_bps'] = g['capacity_bps'].resample(interval).last().ffill()

            # Carry meta columns
            if 'source_layer' in g.columns:
                out['source_layer'] = g['source_layer'].resample(interval).last().ffill()
            if 'destination_layer' in g.columns:
                out['destination_layer'] = g['destination_layer'].resample(interval).last().ffill()

            # Re-attach link_id
            out['link_id'] = g['link_id'].iloc[0] if 'link_id' in g.columns else None
            out = out.reset_index().rename(columns={'index': 'timestamp'})
            return out

        self.traffic_df = (
            self.traffic_df
                .groupby('link_id', group_keys=False)
                .apply(resample_group)
                .sort_values(['link_id', 'timestamp'])
                .drop_duplicates()
                .reset_index(drop=True)
        )

        print(f"✅ Resampled traffic: {self.traffic_df.shape}")
        
    def create_features(self):
        print('🔧 Creating features...')
        df = self.traffic_df

        # Time features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

        # Derived metrics
        if 'capacity_bps' in df.columns and 'bytes_sent' in df.columns:
            # utilization = (8 * bytes_sent / window_seconds) / capacity_bps
            window_seconds = int(pd.to_timedelta(self.resample_interval).total_seconds())
            df['utilization'] = ((8.0 * df['bytes_sent'] / max(window_seconds, 1)) / df['capacity_bps']).clip(0, 1)
        else:
            df['utilization'] = 0.0

        if 'bitrate_bps' in df.columns:
            df['throughput_mbps'] = df['bitrate_bps'] / 1e6
        else:
            df['throughput_mbps'] = 0.0

        if 'jitter_milliseconds' in df.columns and 'loss_rate' in df.columns:
            df['quality_score'] = 1 - (df['loss_rate'] + df['jitter_milliseconds'] / 1000.0) # đánh giá mức độ ổn định của bằng thông
            df['quality_score'] = df['quality_score'].clip(0, 1)
        else:
            df['quality_score'] = 1.0

        df['efficiency'] = df['utilization'] * df['quality_score']

        self.traffic_df = df
        print('✅ Features created')
        
    def select_features(self):
        print('🎯 Selecting features (auto-detect numeric + exist)...')
        
        numeric_features = self.traffic_df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Desired sets based on your spec
        desired_lstm = [
            'utilization', 'bitrate_bps', 'throughput_mbps', 'quality_score', 'hour', 'is_weekend'
        ]
        desired_vae = [
            'utilization', 'bitrate_bps', 'throughput_mbps', 'quality_score',
            'loss_rate', 'hour', 'is_weekend', 'efficiency' 
        ]

        # Keep only available numeric
        self.lstm_features = [c for c in desired_lstm if c in numeric_features]
        self.vae_features = [c for c in desired_vae if c in numeric_features]
        
        print(f"✅ LSTM features ({len(self.lstm_features)}): {self.lstm_features}")
        print(f"✅ VAE features ({len(self.vae_features)}): {self.vae_features}")
        
    def get_link_order(self):
        if self.topology_df is not None and 'link_id' in self.topology_df.columns:
            order = sorted(self.topology_df['link_id'].unique().tolist())
        else:
            order = sorted(self.traffic_df['link_id'].unique().tolist())
        return order

    def build_wide_snapshots(self):
        print('🧱 Building timestamp×link snapshots...')
        # Exclude time-only features from pivot to avoid duplication per link
        features = [f for f in self.vae_features if f not in ['hour', 'is_weekend']]
        link_order = self.get_link_order()
        wide = (
            self.traffic_df
                .set_index(['timestamp', 'link_id'])[features]
                .unstack('link_id')
                .sort_index()
        )
        # Reindex link order for stable columns
        wide = wide.reindex(columns=pd.MultiIndex.from_product([features, link_order]), fill_value=np.nan)

        # Keep a missing mask before filling
        missing_mask = wide.isna().values

        # Conservative fill: limited interpolate + limited ffill/bfill
        wide = wide.groupby(level=0, axis=1).apply(
            lambda block: block.interpolate(limit=3, limit_direction='both')
        )
        wide = wide.groupby(level=0, axis=1).apply(lambda block: block.ffill(limit=3).bfill(limit=3))

        # Normalize columns to exactly (feature, link) after groupby-apply
        if isinstance(wide.columns, pd.MultiIndex) and wide.columns.nlevels > 2:
            cols = wide.columns
            wide.columns = pd.MultiIndex.from_tuples(list(zip(cols.get_level_values(-2), cols.get_level_values(-1))))

        print(f"✅ Wide shape: {wide.shape}")
        return wide, link_order, missing_mask

    def fit_transform_wide_scalers(self, wide: pd.DataFrame, train_end_idx: int):
        print('📏 Scaling wide snapshots (train-only fit, per feature across links)...')
        from sklearn.preprocessing import MinMaxScaler
        scalers = {}
        scaled = wide.copy()
        features = scaled.columns.levels[0].tolist()
        for feat in features:
            cols = scaled[feat].columns
            scaler = MinMaxScaler()
            train_block = scaled[feat].iloc[:train_end_idx]
            scaler.fit(train_block.values)
            scaled[feat] = scaler.transform(scaled[feat].values)
            scalers[feat] = scaler
        print('✅ Scaled wide')
        return scaled, scalers

    def create_lstm_sequences_from_wide(self, wide: pd.DataFrame, seq_len: int = None, horizon: int = 1):
        print(f"🔄 Creating LSTM sequences from timestamp×link snapshots (len={seq_len or self.sequence_length}, horizon={horizon})...")
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
        print(f"✅ LSTM sequences (wide): {X.shape}, targets: {y.shape}")
        return X, y, end_idx

    def chronological_split_from_wide(self, X: np.ndarray, y: np.ndarray, end_idx: np.ndarray, total_T: int, train_ratio=0.7, val_ratio=0.15):
        print('✂️ Chronological split train/val/test (no shuffle)...')
        t1 = int(total_T * train_ratio)
        t2 = int(total_T * (train_ratio + val_ratio))
        train_mask = end_idx < t1
        val_mask = (end_idx >= t1) & (end_idx < t2)
        test_mask = end_idx >= t2
        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        print(f"✅ Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
        
    def prepare_vae_snapshots(self):
        # Deprecated in favor of build_wide_snapshots
        return self.build_wide_snapshots()[0]

    def save_all(self, X_train, y_train, X_val, y_val, X_test, y_test, vae_snapshots):
        print('💾 Saving processed outputs...')
        os.makedirs('data', exist_ok=True)
        os.makedirs('models', exist_ok=True)
        
        # Save processed table
        self.traffic_df.to_csv('data/traffic_processed.csv', index=False)
        
        # Save feature lists
        with open('data/features.json', 'w') as f:
            json.dump({'lstm_features': self.lstm_features, 'vae_features': self.vae_features}, f)
        
        # Save LSTM sequences
        np.save('data/X_train.npy', X_train)
        np.save('data/y_train.npy', y_train)
        np.save('data/X_val.npy', X_val)
        np.save('data/y_val.npy', y_val)
        np.save('data/X_test.npy', X_test)
        np.save('data/y_test.npy', y_test)
        
        # Save VAE snapshots and columns
        np.save('data/vae_snapshots.npy', vae_snapshots.values)
        if isinstance(vae_snapshots.columns, pd.MultiIndex):
            vae_cols = [tuple(map(str, col)) for col in vae_snapshots.columns]
        else:
            vae_cols = [str(col) for col in vae_snapshots.columns]
        with open('data/vae_columns.json', 'w') as f:
            json.dump(vae_cols, f)

        print('✅ Saved')

    def run(self):
        print('🚀 Start preprocessing')
        self.load_data()
        self.clean_and_resample()
        self.create_features()
        self.select_features()
        # Build snapshot wide matrix and enforce link order
        wide, link_order, missing_mask = self.build_wide_snapshots()
        T = len(wide)
        t1 = int(T * 0.7)
        t2 = int(T * 0.85)
        # Fit scalers on train rows only, scale entire wide once
        wide_scaled, scalers = self.fit_transform_wide_scalers(wide, train_end_idx=t1)
        # Expose scalers for reuse; keep only wide scalers
        self.vae_scaler = scalers
        self.lstm_scaler = None
        # Fill any remaining NaNs with train means (per feature/column)
        for feat in wide_scaled.columns.levels[0]:
            block = wide_scaled[feat]
            means = block.iloc[:t1].mean(axis=0)
            block = block.fillna(means)
            wide_scaled[feat] = block
        # Create sequences from wide for LSTM; target = selected feature across links
        if self.target_feature not in self.vae_features:
            print(f"⚠️ target_feature '{self.target_feature}' not in features; falling back to first feature")
            self.target_feature = self.vae_features[0]
        X_seq, y_all, end_idx = self.create_lstm_sequences_from_wide(wide_scaled, seq_len=self.sequence_length, horizon=1)
        # Extract target matrix by taking the feature slice (columns = link ids)
        target_matrix = wide_scaled[self.target_feature]
        # Build y using same end_idx (rows correspond to time indices)
        y = target_matrix.values[end_idx]
        # Chronological split
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = self.chronological_split_from_wide(X_seq, y, end_idx, T)
        # Save artifacts and snapshots (scaled)
        self.save_all(X_train, y_train, X_val, y_val, X_test, y_test, wide_scaled)
        # Save reproducibility helpers
        os.makedirs('data', exist_ok=True)
        with open('data/link_index.json', 'w') as f:
            json.dump(link_order, f)
        with open('data/timestamp_splits.json', 'w') as f:
            json.dump({'T': T, 'train_end': t1, 'val_end': t2}, f)
        import joblib
        joblib.dump(scalers, 'models/wide_scalers.pkl')
        np.save('data/missing_mask.npy', missing_mask)

        print('🎯 Done!')
        return {
            'traffic_df': self.traffic_df,
            'lstm': {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test,
                'features': self.lstm_features
            },
            'vae': {
                'snapshots': wide_scaled,
                'features': self.vae_features
            }
        }


def main():
    pre = DataPreprocessor(resample_interval='10S', sequence_length=24)
    pre.run()
    

if __name__ == '__main__':
    main()


