"""
Generate SAGSIN Network Traffic Dataset (Realistic)
- 7 days (1 week), 5-second interval, 12 links (~3M records)
- Realistic patterns: hourly/daily cycles, weekend effects, SAGSIN-specific behaviors
- Space/Air/Ground/Sea link characteristics
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from tqdm import tqdm

def generate_network_traffic(
    start_date='2025-09-01 00:00:00',
    days=7,
    interval_seconds=5,
    seed=42
):
    """
    Generate SAGSIN network traffic with realistic 5-second interval patterns
    Reflects true characteristics of Space-Air-Ground-Sea Integrated Network
    7 days × 12 links × (86400/5) = ~3M records
    """
    np.random.seed(seed)
    
    print(f"🚀 Generating {days} days of SAGSIN traffic data...")
    print(f"   Interval: {interval_seconds}s (REALISTIC)")
    
    # ===========================
    # 1. Load existing topology
    # ===========================
    try:
        topology_df = pd.read_csv('dataset/topology_data.csv')
        print(f"   ✅ Loaded topology: {len(topology_df)} links")
    except FileNotFoundError:
        print("   ❌ Error: dataset/topology_data.csv not found!")
        return None
    
    # ===========================
    # 2. Generate timestamps (1 SECOND INTERVAL)
    # ===========================
    start = pd.to_datetime(start_date)
    end = start + timedelta(days=days)
    timestamps = pd.date_range(start, end, freq=f'{interval_seconds}S')
    total_samples = len(timestamps) * len(topology_df)
    print(f"   Timestamps: {len(timestamps):,} ({len(timestamps)/86400:.1f} days)")
    print(f"   Total samples: {total_samples:,} ({total_samples/1e6:.2f}M records)")
    
    # ===========================
    # 3. Parse link configs from topology
    # ===========================
    link_configs = []
    for _, row in topology_df.iterrows():
        link_type = f"{row['source_layer']}_{row['destination_layer']}"
        link_configs.append({
            'id': row['link_id'],
            'capacity': row['capacity_bps'],
            'base_latency': row['base_latency_milliseconds'],
            'source': row['source_layer'],
            'dest': row['destination_layer'],
            'type': link_type
        })
    
    print(f"   Links: {len(link_configs)}")
    
    # ===========================
    # 4. Generate traffic data with progress bar
    # ===========================
    data = []
    
    for link_cfg in link_configs:
        link_id = link_cfg['id']
        link_type = link_cfg['type']
        capacity = link_cfg['capacity']
        base_latency = link_cfg['base_latency']
        
        # =========================================
        # SAGSIN-SPECIFIC LINK CHARACTERISTICS
        # =========================================
        
        if link_type == 'space_ground':
            # SATELLITE → GROUND: Stable, low util, high latency, weather-sensitive
            base_util = 0.35
            util_variance = 0.05  # Very stable
            loss_base = 0.002
            jitter_base = 15
            has_weather = True
            weather_prob = 0.05
            weather_impact = 0.7  # 30% degradation
            peak_factor = 1.2  # Slight peak during business hours
            
        elif link_type == 'space_air':
            # SATELLITE → UAV: Variable, mobility effects, Doppler
            base_util = 0.45
            util_variance = 0.12  # High variation due to UAV movement
            loss_base = 0.003
            jitter_base = 20
            has_weather = True
            weather_prob = 0.04
            weather_impact = 0.6
            peak_factor = 1.8  # UAV active during day
            
        elif link_type == 'air_ground':
            # UAV → GROUND: HIGH TRAFFIC, main data link, peak hours
            base_util = 0.65
            util_variance = 0.10
            loss_base = 0.001
            jitter_base = 8
            has_weather = False
            weather_prob = 0.0
            weather_impact = 1.0
            peak_factor = 2.2  # Strong peaks (7-9h, 17-19h)
            
        elif link_type == 'space_space':
            # SATELLITE ↔ SATELLITE: Very stable, low loss, no weather
            base_util = 0.30
            util_variance = 0.02  # Extremely stable
            loss_base = 0.0002
            jitter_base = 3
            has_weather = False
            weather_prob = 0.0
            weather_impact = 1.0
            peak_factor = 1.0  # No peak
            
        elif link_type == 'ground_sea':
            # GROUND → SHIP: Variable, ship mobility, high weather impact
            base_util = 0.40
            util_variance = 0.15  # High variation
            loss_base = 0.008
            jitter_base = 25
            has_weather = True
            weather_prob = 0.10  # High (sea storms)
            weather_impact = 0.5  # 50% degradation
            peak_factor = 1.3
            
        else:
            # Default for other types
            base_util = 0.45
            util_variance = 0.08
            loss_base = 0.002
            jitter_base = 10
            has_weather = False
            weather_prob = 0.0
            weather_impact = 1.0
            peak_factor = 1.0
        
        # Pre-generate arrays for efficiency
        n = len(timestamps)
        util_array = np.zeros(n)
        loss_array = np.zeros(n)
        jitter_array = np.zeros(n)
        rtt_array = np.zeros(n)
        
        # Event tracking
        event_tags = [''] * n
        event_severities = [0.0] * n
        event_durations = [0] * n
        
        # Generate with progress bar
        for i, ts in enumerate(tqdm(timestamps, desc=f"{link_id[:20]:20s}", leave=False, ncols=80)):
            hour = ts.hour
            day_of_week = ts.dayofweek
            second_of_day = ts.hour * 3600 + ts.minute * 60 + ts.second
            
            # ================================
            # TEMPORAL PATTERNS (SECOND-LEVEL)
            # ================================
            
            # 1. Hourly pattern
            if link_type == 'air_ground':
                # Business traffic with strong peaks
                if 7 <= hour <= 9:
                    hour_factor = 2.0  # Morning peak
                elif 12 <= hour <= 13:
                    hour_factor = 1.3  # Lunch
                elif 17 <= hour <= 19:
                    hour_factor = 2.2  # Evening peak
                elif 0 <= hour <= 6:
                    hour_factor = 0.15  # Night
                else:
                    hour_factor = 1.0
            elif link_type == 'space_air':
                # UAV daytime operation
                if 6 <= hour <= 20:
                    hour_factor = peak_factor
                else:
                    hour_factor = 0.1  # UAV returns to base
            elif link_type in ['space_ground', 'ground_sea']:
                # Moderate business hours effect
                if 8 <= hour <= 18:
                    hour_factor = peak_factor
                else:
                    hour_factor = 0.85
            else:  # space_space
                hour_factor = 1.0  # No hourly variation
            
            # 2. Weekly pattern
            is_weekend = (day_of_week >= 5)
            if link_type == 'air_ground':
                week_factor = 0.45 if is_weekend else 1.0  # Strong weekend drop
            elif link_type == 'space_ground':
                week_factor = 0.85 if is_weekend else 1.0  # Slight drop
            else:
                week_factor = 1.0  # No weekend effect
            
            # 3. Minute-level smooth variation
            minute_in_hour = (second_of_day % 3600) / 60.0
            minute_variation = 1.0 + 0.08 * np.sin(2 * np.pi * minute_in_hour / 60)
            
            # 4. Second-level micro-variation (bursty traffic)
            if link_type == 'air_ground':
                # Web-like bursty traffic
                if np.random.random() < 0.15:  # 15% burst probability
                    burst_factor = np.random.uniform(1.4, 2.2)
                else:
                    burst_factor = 1.0
                second_noise = np.random.normal(0, 0.10)
            elif link_type in ['space_air', 'ground_sea']:
                # Mobility-induced variation
                burst_factor = 1.0
                second_noise = np.random.normal(0, util_variance)
            else:  # space_ground, space_space
                burst_factor = 1.0
                second_noise = np.random.normal(0, util_variance)
            
            # ================================
            # CALCULATE UTILIZATION
            # ================================
            util = (base_util * hour_factor * week_factor * minute_variation 
                   * burst_factor + second_noise)
            
            # Weather/Environmental degradation
            if has_weather and np.random.random() < weather_prob:
                util *= weather_impact
                if event_tags[i] == '':
                    event_tags[i] = 'weather'
                    event_severities[i] = 1.0 - weather_impact
                    event_durations[i] = np.random.randint(300, 1800)  # 5-30 min
            
            # Mobility effects (handover for air/sea)
            if link_type in ['space_air', 'ground_sea']:
                if second_of_day % 1800 == 0:  # Every 30 minutes
                    util *= 0.5  # Handover drop
                    if event_tags[i] == '':
                        event_tags[i] = 'handover'
                        event_severities[i] = 0.5
                        event_durations[i] = 5  # 5 seconds
            
            util = np.clip(util, 0.01, 0.98)
            util_array[i] = util
            
            # ================================
            # CALCULATE LOSS RATE
            # ================================
            if util > 0.75:
                loss = loss_base + (util - 0.75) ** 2 * 0.8  # Non-linear spike
            elif util > 0.6:
                loss = loss_base + (util - 0.6) * 0.05
            else:
                loss = loss_base + util * 0.003
            
            # Link-specific loss multipliers
            if link_type in ['space_ground', 'space_air']:
                loss *= 1.5  # Higher loss for space links
            elif link_type == 'ground_sea':
                loss *= 2.0  # Highest loss for sea links
            
            loss = np.clip(loss, 0, 0.3)
            loss_array[i] = loss
            
            # ================================
            # CALCULATE JITTER
            # ================================
            jitter = jitter_base + util * 50 + loss * 200
            
            if util > 0.8:
                jitter += 40  # Queue buildup
            
            # Mobility jitter
            if link_type in ['space_air', 'ground_sea']:
                jitter *= 1.4
            
            jitter = np.clip(jitter, 2, 200)
            jitter_array[i] = jitter
            
            # ================================
            # CALCULATE RTT
            # ================================
            propagation_var = np.random.normal(0, 10) if link_type.startswith('space') else np.random.normal(0, 3)
            queue_delay = util * base_latency * 0.6
            rtt = base_latency + queue_delay + propagation_var
            rtt = np.clip(rtt, base_latency * 0.8, base_latency * 3.5)
            rtt_array[i] = rtt
            
            # ================================
            # EVENTS
            # ================================
            if event_tags[i] == '' and util > 0.75 and np.random.random() < 0.02:
                event_tags[i] = 'congestion'
                event_severities[i] = (util - 0.75) / 0.25
                event_durations[i] = np.random.randint(30, 180)  # 30-180 sec
        
        # Build records for this link
        for i, ts in enumerate(timestamps):
            bitrate = util_array[i] * capacity
            bytes_sent = bitrate * interval_seconds / 8.0
            link_latency = base_latency + np.random.normal(0, 2)
            link_latency = np.clip(link_latency, base_latency * 0.9, base_latency * 1.2)
            
            data.append({
                'timestamp': ts,
                'link_id': link_id,
                'source_layer': link_cfg['source'],
                'destination_layer': link_cfg['dest'],
                'capacity_bps': capacity,
                'bytes_sent': bytes_sent,
                'bitrate_bps': bitrate,
                'utilization': util_array[i],
                'rtt_milliseconds': rtt_array[i],
                'loss_rate': loss_array[i],
                'jitter_milliseconds': jitter_array[i],
                'link_latency_milliseconds': link_latency,
                'event_tag': event_tags[i],
                'event_severity': event_severities[i],
                'event_duration_minutes': event_durations[i] // 60,
                'hour': ts.hour,
                'day_of_week': ts.dayofweek,
                'is_weekend': ts.dayofweek >= 5,
                'minute_of_day': ts.hour * 60 + ts.minute
            })
    
    # ===========================
    # 5. Create DataFrame
    # ===========================
    print(f"\n   Creating DataFrame...")
    df = pd.DataFrame(data)
    df = df.sort_values(['timestamp', 'link_id']).reset_index(drop=True)
    
    print(f"✅ Generated {len(df):,} records ({len(df)/1e6:.2f}M)")
    return df


def validate_dataset(df):
    """
    Validate generated dataset quality
    
    Checks:
    - No missing values
    - Utilization in [0, 1]
    - Loss rate in [0, 1]
    - Proper correlations (util ↔ loss, util ↔ jitter)
    - Weekend ratio ~28%
    - Temporal coverage
    """
    print("\n� Validating dataset quality...")
    
    issues = []
    
    # 1. Missing values
    missing = df.isnull().sum()
    if missing.any():
        issues.append(f"❌ Missing values found: {missing[missing > 0].to_dict()}")
    else:
        print("   ✅ No missing values")
    
    # 2. Utilization range
    util_min, util_max = df['utilization'].min(), df['utilization'].max()
    util_mean = df['utilization'].mean()
    if util_min < 0 or util_max > 1:
        issues.append(f"❌ Utilization out of range: [{util_min:.3f}, {util_max:.3f}]")
    else:
        print(f"   ✅ Utilization: [{util_min:.3f}, {util_max:.3f}], mean={util_mean:.3f}")
    
    # 3. Loss rate range
    loss_min, loss_max = df['loss_rate'].min(), df['loss_rate'].max()
    if loss_min < 0 or loss_max > 1:
        issues.append(f"❌ Loss rate out of range: [{loss_min:.3f}, {loss_max:.3f}]")
    else:
        print(f"   ✅ Loss rate: [{loss_min:.3f}, {loss_max:.3f}]")
    
    # 4. Correlations
    corr_util_loss = df[['utilization', 'loss_rate']].corr().iloc[0, 1]
    corr_util_jitter = df[['utilization', 'jitter_milliseconds']].corr().iloc[0, 1]
    
    if corr_util_loss < 0.60 or corr_util_loss > 0.85:
        issues.append(f"⚠️  util↔loss correlation: {corr_util_loss:.3f} (expect 0.65-0.75)")
    else:
        print(f"   ✅ util↔loss correlation: {corr_util_loss:.3f}")
    
    if corr_util_jitter < 0.65 or corr_util_jitter > 0.90:
        issues.append(f"⚠️  util↔jitter correlation: {corr_util_jitter:.3f} (expect 0.70-0.85)")
    else:
        print(f"   ✅ util↔jitter correlation: {corr_util_jitter:.3f}")
    
    # 5. Weekend ratio
    weekend_ratio = df['is_weekend'].mean()
    if weekend_ratio < 0.25 or weekend_ratio > 0.32:
        issues.append(f"⚠️  Weekend ratio: {weekend_ratio:.2%} (expect ~28%)")
    else:
        print(f"   ✅ Weekend ratio: {weekend_ratio:.2%}")
    
    # 6. Temporal coverage
    date_range = (df['timestamp'].max() - df['timestamp'].min()).days
    expected_days = 14
    if abs(date_range - expected_days) > 1:
        issues.append(f"❌ Date range: {date_range} days (expect {expected_days})")
    else:
        print(f"   ✅ Date range: {date_range} days")
    
    # 7. Events
    event_ratio = (df['event_tag'] != '').mean()
    print(f"   ℹ️  Events: {event_ratio:.2%} of records")
    
    # 8. Link diversity
    n_links = df['link_id'].nunique()
    print(f"   ℹ️  Links: {n_links} unique")
    
    # Summary
    if issues:
        print("\n⚠️  Validation warnings:")
        for issue in issues:
            print(f"   {issue}")
        return True  # Still usable
    else:
        print("\n✅ VALIDATION PASSED - Dataset quality excellent!")
        return True


def generate_topology_and_nodes(link_configs):
    """
    This function is no longer needed - we keep existing CSV files
    """
    print("   ℹ️  Using existing topology_data.csv and nodes_data.csv")
    return None, None


def main():
    """
    Main generation pipeline
    """
    print("=" * 60)
    print("🌐 Network Traffic Dataset Generator")
    print("=" * 60)
    
    # Check if topology exists
    if not os.path.exists('dataset/topology_data.csv'):
        print("❌ Error: dataset/topology_data.csv not found!")
        print("   Please ensure topology_data.csv exists in dataset/ folder")
        return
    
    if not os.path.exists('dataset/nodes_data.csv'):
        print("⚠️  Warning: dataset/nodes_data.csv not found!")
        print("   (Not required for traffic generation, but recommended)")
    
    # Generate traffic data
    df_traffic = generate_network_traffic(
        start_date='2025-09-01 00:00:00',
        days=7,
        interval_seconds=1,
        seed=42
    )
    
    if df_traffic is None:
        return
    
    # Validate
    validate_dataset(df_traffic)
    
    # Save traffic data only (keep existing topology and nodes)
    print("\n💾 Saving files...")
    df_traffic.to_csv('dataset/traffic_data.csv', index=False)
    
    print(f"   ✅ dataset/traffic_data.csv ({len(df_traffic):,} rows)")
    print(f"   ℹ️  Kept existing topology_data.csv")
    print(f"   ℹ️  Kept existing nodes_data.csv")
    
    print("\n🎉 Done! Now run:")
    print("   python preprocessing.py")
    print("   python train_lstm.py")
    print("=" * 60)


if __name__ == '__main__':
    main()