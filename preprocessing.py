import os
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings('ignore')


class DataPreprocessor:
    def __init__(self, resample_interval: str = '10S', sequence_length: int = 24):
        self.resample_interval = resample_interval
        self.sequence_length = sequence_length

        # scalers
        self.scalers = None

        # feature lists
        self.model_features = []       # ✅ bộ feature chung cho cả VAE & LSTM
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
        print('Cleaning + Resampling to 10s...')
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
            idx = g.resample(interval).asfreq().index
            out = pd.DataFrame(index=idx)

            # SUM over window
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

            # Carry meta
            for c in ['source_layer', 'destination_layer']:
                if c in g.columns:
                    out[c] = g[c].resample(interval).last().ffill()

            # Attach link_id + timestamp as column
            out['link_id'] = g['link_id'].iloc[0] if 'link_id' in g.columns else None
            out = out.reset_index().rename(columns={'index': 'timestamp'})
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

        # Time features (có thể bổ sung sin/cos sau)
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

        # Derived metrics
        window_seconds = int(pd.to_timedelta(self.resample_interval).total_seconds())

        if 'capacity_bps' in df.columns and 'bytes_sent' in df.columns:
            df['utilization'] = ((8.0 * df['bytes_sent'] / max(window_seconds, 1)) / df['capacity_bps']).clip(0, 1)
        else:
            df['utilization'] = 0.0

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
        print('Features created')

    # --------------------
    # Feature Selection (chung cho VAE & LSTM)
    # --------------------
    def select_features(self):
        print('Selecting features for BOTH VAE & LSTM...')
        numeric_features = self.traffic_df.select_dtypes(include=[np.number]).columns.tolist()

        # ✅ Bộ "lean" tránh trùng lặp, phù hợp cả VAE và LSTM
        desired = [
            'utilization',
            'bitrate_bps',
            'loss_rate',
            'jitter_milliseconds',
            'rtt_milliseconds',
            'capacity_bps'
        ]
        # Tự loại cột thiếu
        self.model_features = [c for c in desired if c in numeric_features]

        # (tuỳ chọn) thêm time sin/cos (nếu muốn học chu kỳ)
        # hour_sin = sin(2π*hour/24), etc. — bỏ qua để patch gọn.

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
        use_feats = [f for f in features if f not in ['hour', 'is_weekend']]
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
        print('Chronological split train/val/test (no shuffle)...')
        t1 = int(total_T * train_ratio)
        t2 = int(total_T * (train_ratio + val_ratio))
        train_mask = end_idx < t1
        val_mask = (end_idx >= t1) & (end_idx < t2)
        test_mask = end_idx >= t2
        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        print(f"Split -> Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    # --------------------
    # Save Artifacts
    # --------------------
    def save_all(self, X_train, y_train, X_val, y_val, X_test, y_test,
                 vae_snapshots, link_order, scalers, missing_mask, T):
        print('Saving processed outputs...')
        os.makedirs('data', exist_ok=True)
        os.makedirs('models', exist_ok=True)

        # Save processed table
        self.traffic_df.to_csv('data/traffic_processed.csv', index=False)

        # Save feature lists
        with open('data/features.json', 'w') as f:
            json.dump({'model_features': self.model_features,
                       'target_feature': self.target_feature}, f)

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

        # Save helpers
        with open('data/link_index.json', 'w') as f:
            json.dump(link_order, f)

        with open('data/timestamp_splits.json', 'w') as f:
            t1 = int(T * 0.7); t2 = int(T * 0.85)
            json.dump({'T': T, 'train_end': t1, 'val_end': t2}, f)

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

        # Fit scalers on train rows only; scale entire wide
        T = len(wide)
        t1 = int(T * 0.7)
        wide_scaled, scalers = self.fit_transform_wide_scalers(wide, train_end_idx=t1)

        # Fill any remaining NaNs with train means (per feature/column)
        for feat in wide_scaled.columns.levels[0]:
            block = wide_scaled[feat]
            means = block.iloc[:t1].mean(axis=0)
            wide_scaled[feat] = block.fillna(means)

        # Đảm bảo target_feature xuất hiện trong bộ feature chung
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

        # Lưu mọi thứ (VAE dùng wide_scaled; LSTM dùng X/y)
        self.save_all(X_train, y_train, X_val, y_val, X_test, y_test,
                      vae_snapshots=wide_scaled,
                      link_order=link_order,
                      scalers=scalers,
                      missing_mask=missing_mask,
                      T=T)

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
    pre = DataPreprocessor(resample_interval='10S', sequence_length=24)
    pre.run()


if __name__ == '__main__':
    main()
