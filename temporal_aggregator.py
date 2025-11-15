"""
Temporal Aggregator - Upsample 1s snapshot to 30s window
=========================================================
Generate synthetic 30-second window from single simulator record
to match training data format (30s aggregation)

Author: PBL4 Team
Date: 2025-11-09
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Set seed for reproducibility
REPRODUCIBILITY_SEED = 42
np.random.seed(REPRODUCIBILITY_SEED)


class TemporalAggregator:
    """
    Upsample 1-second simulator snapshot to 30-second aggregated window
    
    Method:
    1. Take 1 record from simulator
    2. Generate 29 synthetic records with realistic noise
    3. Aggregate to match training format
    
    Why this works:
    - Training uses 30s mean/sum aggregation
    - Single snapshot is noisy → need smoothing
    - Synthetic samples preserve statistical properties
    """
    
    def __init__(self, window_size=30):
        """
        Args:
            window_size: Number of seconds to aggregate (default: 30)
        """
        self.window_size = window_size
        print(f"✅ TemporalAggregator initialized (window: {window_size}s)")
    
    def upsample_record(self, record):
        """
        Generate 30 records from 1 simulator snapshot
        
        Args:
            record: dict with single record from traffic_data.csv
                {
                    'timestamp': '2025-11-08 22:05:45',
                    'utilization': 0.21,
                    'bitrate_bps': 12672699,
                    'loss_rate': 0.00163,
                    'jitter_milliseconds': 26.44,
                    'rtt_milliseconds': 41.61,
                    ...
                }
        
        Returns:
            list of 30 dicts (synthetic 30-second window)
        """
        # Base timestamp
        base_time = pd.to_datetime(record['timestamp'])
        
        # Extract base values
        base_util = record['utilization']
        base_bitrate = record['bitrate_bps']
        base_loss = record['loss_rate']
        base_jitter = record['jitter_milliseconds']
        base_rtt = record['rtt_milliseconds']
        
        # Noise parameters - MATCH TRAINING DATA VARIANCE!
        # Training: util σ=0.066, bitrate σ=0.08, etc.
        util_noise_std = 0.06      # ✅ 6% std (was 2% → match training!)
        bitrate_noise_std = 0.08   # ✅ 8% std (was 3%)
        loss_noise_std = 0.0005    # Small for loss (keep same)
        jitter_noise_std = 5.0     # ✅ 5ms std (was 2ms)
        rtt_noise_std = 8.0        # ✅ 8ms std (was 3ms)
        
        synthetic_records = []
        
        for i in range(self.window_size):
            # Timestamp: base - (29-i) seconds
            # So timestamps go: [base-29s, base-28s, ..., base-1s, base]
            ts = base_time - timedelta(seconds=(self.window_size - 1 - i))
            
            # Add realistic noise with temporal correlation
            # Use sine wave for smooth variation
            phase = 2 * np.pi * i / self.window_size
            trend_factor = 1.0 + 0.05 * np.sin(phase)
            
            # Generate noisy values
            util_noisy = base_util * trend_factor + np.random.normal(0, util_noise_std)
            util_noisy = np.clip(util_noisy, 0.01, 0.98)
            
            bitrate_noisy = base_bitrate * trend_factor + np.random.normal(0, bitrate_noise_std * base_bitrate)
            bitrate_noisy = max(bitrate_noisy, 0)
            
            loss_noisy = base_loss + np.random.normal(0, loss_noise_std)
            loss_noisy = np.clip(loss_noisy, 0, 0.3)
            
            jitter_noisy = base_jitter + np.random.normal(0, jitter_noise_std)
            jitter_noisy = np.clip(jitter_noisy, 2, 200)
            
            rtt_noisy = base_rtt + np.random.normal(0, rtt_noise_std)
            rtt_noisy = max(rtt_noisy, record.get('link_latency_milliseconds', 20) * 0.8)
            
            # Create synthetic record
            synthetic = record.copy()
            synthetic.update({
                'timestamp': ts.strftime('%Y-%m-%d %H:%M:%S'),
                'utilization': util_noisy,
                'bitrate_bps': bitrate_noisy,
                'loss_rate': loss_noisy,
                'jitter_milliseconds': jitter_noisy,
                'rtt_milliseconds': rtt_noisy,
            })
            
            synthetic_records.append(synthetic)
        
        return synthetic_records
    
    def aggregate_window(self, records):
        """
        Aggregate 30 records to single window (match training format)
        
        Args:
            records: list of 30 dicts
        
        Returns:
            dict with aggregated values (same format as training)
        """
        df = pd.DataFrame(records) # create a dataframe for easy aggregation
        
        # Bytes: SUM (total bytes in 30s) - MUST calculate first
        bytes_sent_sum = df['bytes_sent'].sum() if 'bytes_sent' in df.columns else 0
        
        # Capacity from last record
        capacity_bps = records[-1].get('capacity_bps', 0)
        # This matches preprocessing.py line 172:
        # utilization = ((8.0 * bytes_sent / window_seconds) / capacity_bps)
        window_seconds = self.window_size  # 30 seconds
        if capacity_bps > 0 and bytes_sent_sum > 0:
            utilization_calculated = ((8.0 * bytes_sent_sum / window_seconds) / capacity_bps).clip(0, 1)
        else:
            # Fallback to mean if calculation fails
            utilization_calculated = df['utilization'].mean() if 'utilization' in df.columns else 0.0
        
        # Aggregation rules (match preprocessing.py):
        aggregated = {
            # Use LAST timestamp as representative
            'timestamp': records[-1]['timestamp'],
            
            'utilization': utilization_calculated,
            
            # Bitrate: MEAN (it's a rate!)
            'bitrate_bps': df['bitrate_bps'].mean(),
            
            # Loss: MEAN
            'loss_rate': df['loss_rate'].mean(),
            
            # Jitter: MEAN
            'jitter_milliseconds': df['jitter_milliseconds'].mean(),
            
            # RTT: MEAN
            'rtt_milliseconds': df['rtt_milliseconds'].mean(),
            
            # Bytes: SUM (total bytes in 30s)
            'bytes_sent': bytes_sent_sum,
            
            # Other fields: copy from last record
            'capacity_bps': capacity_bps,
            'source_layer': records[-1].get('source_layer', ''),
            'destination_layer': records[-1].get('destination_layer', ''),
            'link_id': records[-1].get('link_id', ''),
            'hour': records[-1].get('hour', 0),
            'day_of_week': records[-1].get('day_of_week', 0),
            'is_weekend': records[-1].get('is_weekend', 0),
            'hour_sin': records[-1].get('hour_sin', 0),
            'hour_cos': records[-1].get('hour_cos', 0),
            'day_sin': records[-1].get('day_sin', 0),
            'day_cos': records[-1].get('day_cos', 0),
            'throughput_mbps': df['bitrate_bps'].mean() / 1e6,
            'quality_score': records[-1].get('quality_score', 0.9),
            'efficiency': records[-1].get('efficiency', 0.8),
            'link_latency_milliseconds': df['rtt_milliseconds'].mean() * 0.9,  # Approximate
        }
        
        return aggregated
    
    def process_simulator_record(self, record):
        """
        End-to-end: 1 record → 30 synthetic → 1 aggregated
        
        Args:
            record: Single record from simulator
        
        Returns:
            Aggregated record matching training format
        """
        # Step 1: Generate 30 synthetic records
        synthetic_records = self.upsample_record(record)
        
        # Step 2: Aggregate to match training format
        aggregated = self.aggregate_window(synthetic_records)
        
        return aggregated


# ============================================
# DEMO
# ============================================
def demo():
    """Test temporal aggregation"""
    print("=" * 70)
    print("🧪 Temporal Aggregator Demo")
    print("=" * 70)
    
    aggregator = TemporalAggregator(window_size=30)
    
    # Simulate a single record from simulator
    simulator_record = {
        'timestamp': '2025-11-08 22:05:45',
        'bytes_sent': 3565061.5,
        'bitrate_bps': 12672699.85,
        'utilization': 0.2112,
        'loss_rate': 0.00163,
        'jitter_milliseconds': 26.44,
        'rtt_milliseconds': 41.61,
        'capacity_bps': 60000000,
        'source_layer': 'air',
        'destination_layer': 'ground',
        'link_id': 'LINK_AIR_GROUND_01',
        'hour': 22,
        'day_of_week': 5,
        'is_weekend': 1,
        'hour_sin': -0.5,
        'hour_cos': 0.866,
        'day_sin': -0.975,
        'day_cos': -0.223,
        'throughput_mbps': 12.67,
        'quality_score': 0.72,
        'efficiency': 0.152,
        'link_latency_milliseconds': 41.57,
    }
    
    print(f"\n📥 INPUT (1-second snapshot from simulator):")
    print(f"   Timestamp:   {simulator_record['timestamp']}")
    print(f"   Utilization: {simulator_record['utilization']:.4f} ({simulator_record['utilization']*100:.2f}%)")
    print(f"   Bitrate:     {simulator_record['bitrate_bps']/1e6:.2f} Mbps")
    print(f"   Loss:        {simulator_record['loss_rate']:.6f}")
    print(f"   Jitter:      {simulator_record['jitter_milliseconds']:.2f} ms")
    
    # Process
    print(f"\n🔄 Processing: Upsampling to 30-second window...")
    aggregated = aggregator.process_simulator_record(simulator_record)
    
    print(f"\n📤 OUTPUT (30-second aggregated window):")
    print(f"   Timestamp:   {aggregated['timestamp']} (last in window)")
    print(f"   Utilization: {aggregated['utilization']:.4f} ({aggregated['utilization']*100:.2f}%)")
    print(f"   Bitrate:     {aggregated['bitrate_bps']/1e6:.2f} Mbps")
    print(f"   Loss:        {aggregated['loss_rate']:.6f}")
    print(f"   Jitter:      {aggregated['jitter_milliseconds']:.2f} ms")
    print(f"   Bytes (sum): {aggregated['bytes_sent']:,.0f} bytes")
    
    print(f"\n✅ Benefits:")
    print(f"   - Smoothed values (less noise)")
    print(f"   - Match training format (30s aggregation)")
    print(f"   - Ready for model inference")
    print(f"   - No waiting for 30 seconds!")
    
    print(f"\n📊 Comparison:")
    print(f"   {'Metric':<20} {'Simulator (1s)':<20} {'Aggregated (30s)':<20}")
    print(f"   {'-'*60}")
    print(f"   {'Utilization':<20} {simulator_record['utilization']:<20.4f} {aggregated['utilization']:<20.4f}")
    print(f"   {'Bitrate (Mbps)':<20} {simulator_record['bitrate_bps']/1e6:<20.2f} {aggregated['bitrate_bps']/1e6:<20.2f}")
    print(f"   {'Loss rate':<20} {simulator_record['loss_rate']:<20.6f} {aggregated['loss_rate']:<20.6f}")
    
    print("\n" + "=" * 70)


if __name__ == '__main__':
    demo()
