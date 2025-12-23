import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from prometheus_api_client import PrometheusConnect
import joblib
import os
import datetime
import sys

# Import config to connect to real DB
try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from config import PROMETHEUS_URL, PROM_USER, PROM_PASSWORD
except ImportError:
    PROMETHEUS_URL = "http://localhost:9090"
    PROM_USER = None
    PROM_PASSWORD = None

BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'anomaly_model.pkl')

def ensure_directory():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

def fetch_real_data(days=14):
    print(f"🔌 Connecting to Prometheus at {PROMETHEUS_URL}...")
    
    headers = {}
    if PROM_USER and PROM_PASSWORD:
        import base64
        auth = base64.b64encode(f"{PROM_USER}:{PROM_PASSWORD}".encode()).decode()
        headers = {"Authorization": f"Basic {auth}"}

    prom = PrometheusConnect(url=PROMETHEUS_URL, headers=headers, disable_ssl=True)
    
    start_time = datetime.datetime.now() - datetime.timedelta(days=days)
    end_time = datetime.datetime.now()
    step = '1h'

    print(f"📥 Fetching last {days} days of metrics...")

    try:
        # Helper to get DF with 'hostname' column
        def get_df_with_host(query, col_name):
            data = prom.custom_query_range(query=query, start_time=start_time, end_time=end_time, step=step)
            if not data: return pd.DataFrame()
            
            records = []
            for series in data:
                host = series['metric'].get('hostname', 'unknown')
                for val in series['values']:
                    records.append({
                        'timestamp': pd.to_datetime(val[0], unit='s'),
                        'hostname': host,
                        col_name: float(val[1])
                    })
            
            return pd.DataFrame(records)

        # 1. Fetch Speed Metrics with Hostnames (Critical for Normalization)
        # We need raw data per probe to calculate baselines
        df_lan_down = get_df_with_host('avg_over_time(LAN_EXTERNAL_SPEEDTEST{type="Download"}[1h])', 'lan_down')
        df_lan_up = get_df_with_host('avg_over_time(LAN_EXTERNAL_SPEEDTEST{type="Upload"}[1h])', 'lan_up')
        df_wlan_down = get_df_with_host('avg_over_time(WLAN_EXTERNAL_SPEEDTEST{type="Download"}[1h])', 'wlan_down')
        df_wlan_up = get_df_with_host('avg_over_time(WLAN_EXTERNAL_SPEEDTEST{type="Upload"}[1h])', 'wlan_up')
        
        # Latency/DNS (No normalization needed, but fetched for model context)
        df_wlan_ping = get_df_with_host('avg_over_time(GENERAL_info{wlan_google_response_time!=""}[1h])', 'wlan_ping')
        # Note: If GENERAL_info stores value in label, custom_query_range is tricky. 
        # Assuming for training we rely on speed primarily for this normalization logic fix.
        # Or assuming you have corrected metrics.
        # For simplicity in this 'Option A' implementation, I'll focus on SPEED normalization.
        
        # Merge all speed dataframes on timestamp + hostname
        dfs = [df_lan_down, df_lan_up, df_wlan_down, df_wlan_up]
        df = dfs[0]
        for d in dfs[1:]:
            if not d.empty:
                df = pd.merge(df, d, on=['timestamp', 'hostname'], how='outer')
        
        df = df.fillna(0)

        # --- CALCULATE BASELINES (Per Probe) ---
        print("📊 Calculating baselines per probe...")
        # We define "Baseline" as the 95th percentile speed (Max Capacity) seen over 2 weeks
        baselines = {}
        probes = df['hostname'].unique()
        
        for probe in probes:
            probe_data = df[df['hostname'] == probe]
            baselines[probe] = {
                'lan_down_max': probe_data['lan_down'].quantile(0.95) or 1, # Avoid div/0
                'lan_up_max': probe_data['lan_up'].quantile(0.95) or 1,
                'wlan_down_max': probe_data['wlan_down'].quantile(0.95) or 1,
                'wlan_up_max': probe_data['wlan_up'].quantile(0.95) or 1
            }

        # --- NORMALIZE DATA ---
        # Convert absolute Mbps to % of Max Capacity (0.0 to 1.0)
        print("⚖️ Normalizing data for AI training...")
        
        def normalize(row):
            host = row['hostname']
            if host in baselines:
                base = baselines[host]
                row['lan_down'] = row['lan_down'] / base['lan_down_max']
                row['lan_up'] = row['lan_up'] / base['lan_up_max']
                row['wlan_down'] = row['wlan_down'] / base['wlan_down_max']
                row['wlan_up'] = row['wlan_up'] / base['wlan_up_max']
            return row

        df_normalized = df.apply(normalize, axis=1)
        
        # Add Time Features
        df_normalized['hour'] = df_normalized['timestamp'].dt.hour
        df_normalized['is_weekend'] = df_normalized['timestamp'].dt.dayofweek.isin([5, 6]).astype(int)
        
        # Select Features for Model
        feature_cols = ['hour', 'is_weekend', 'lan_down', 'lan_up', 'wlan_down', 'wlan_up']
        # Add dummy ping columns if needed or just train on speed + time
        # For simplicity, we train mainly on speed/time here as requested
        
        return df_normalized[feature_cols], baselines

    except Exception as e:
        print(f"⚠️ Error fetching real data: {e}")
        return generate_enhanced_synthetic_data()

def generate_enhanced_synthetic_data(n_samples=5000):
    """
    Generates NORMALIZED synthetic data (0.0 - 1.0 scale).
    """
    print("🧪 Generating synthetic normalized data...")
    
    hours = np.random.randint(0, 24, n_samples)
    is_weekend = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    
    # Generate percentages (0.0 to 1.0) instead of Mbps
    lan_downs = []
    
    for h, w in zip(hours, is_weekend):
        # Busy times = Lower % of max speed
        if w == 0 and 9 <= h <= 17:
            base_pct = 0.4 # 40% capacity
        else:
            base_pct = 0.9 # 90% capacity
            
        lan_downs.append(max(0, min(1, np.random.normal(base_pct, 0.1))))

    df = pd.DataFrame({
        'hour': hours,
        'is_weekend': is_weekend,
        'lan_down': lan_downs,
        'lan_up': lan_downs, # Simulating symmetric
        'wlan_down': [x * 0.8 for x in lan_downs],
        'wlan_up': [x * 0.8 for x in lan_downs],
        # Add dummy ping if model expects it
        'lan_ping': np.random.normal(20, 5, n_samples),
        'wlan_ping': np.random.normal(30, 5, n_samples),
        'lan_dns': np.random.normal(30, 5, n_samples),
        'wlan_dns': np.random.normal(35, 5, n_samples),
    })
    
    # Mock baselines for synthetic
    baselines = {"default": {"lan_down_max": 1000, "lan_up_max": 1000, "wlan_down_max": 500, "wlan_up_max": 500}}
    
    return df, baselines

def train_model():
    ensure_directory()
    df, baselines = fetch_real_data()
    
    print("🧠 Training AI Model on Normalized Data...")
    
    # Ensure all required columns exist
    required_cols = ['hour', 'is_weekend', 'lan_down', 'lan_up', 'wlan_down', 'wlan_up', 
                     'lan_ping', 'wlan_ping', 'lan_dns', 'wlan_dns']
    for c in required_cols:
        if c not in df.columns: df[c] = 0.5 if 'down' in c or 'up' in c else 20 # Defaults
            
    model = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
    model.fit(df[required_cols])
    
    # Save Model AND Probe Baselines
    model_package = {
        'model': model,
        'baselines': baselines, # <--- Crucial: Analyzer needs this to normalize input
        'features': required_cols
    }
    
    joblib.dump(model_package, MODEL_PATH)
    print(f"✅ Model & Baselines saved to: {MODEL_PATH}")

if __name__ == "__main__":
    train_model()