import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from prometheus_api_client import PrometheusConnect
import joblib
import os
import datetime
import sys
import time

# --- IMPORT SETTINGS MANAGER ---
try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from config import PROMETHEUS_URL, PROM_USER, PROM_PASSWORD
    import settings_manager
except ImportError:
    PROMETHEUS_URL = "http://localhost:9090"
    PROM_USER = None
    PROM_PASSWORD = None
    # Mock if missing
    class MockSettings:
        def get_settings(self): return {"anomaly_contamination": 0.01, "anomaly_estimators": 200}
    settings_manager = MockSettings()

BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, 'models')
ANOMALY_MODEL_PATH = os.path.join(MODEL_DIR, 'anomaly_model.pkl')

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
        def get_df_with_host(query, col_name):
            data = prom.custom_query_range(query=query, start_time=start_time, end_time=end_time, step=step)
            if not data: return pd.DataFrame()
            records = []
            for series in data:
                host = series['metric'].get('hostname', 'unknown')
                for val in series['values']:
                    records.append({ 'timestamp': pd.to_datetime(val[0], unit='s'), 'hostname': host, col_name: float(val[1]) })
            return pd.DataFrame(records)

        # Fetch Data
        df_lan_down = get_df_with_host('avg_over_time(LAN_EXTERNAL_SPEEDTEST{type="Download"}[1h])', 'lan_down')
        df_lan_up = get_df_with_host('avg_over_time(LAN_EXTERNAL_SPEEDTEST{type="Upload"}[1h])', 'lan_up')
        df_wlan_down = get_df_with_host('avg_over_time(WLAN_EXTERNAL_SPEEDTEST{type="Download"}[1h])', 'wlan_down')
        df_wlan_up = get_df_with_host('avg_over_time(WLAN_EXTERNAL_SPEEDTEST{type="Upload"}[1h])', 'wlan_up')
        df_lan_ping = get_df_with_host('avg_over_time(LAN_PING{metrics="avgRTT", type="EXTERNAL"}[1h])', 'lan_ping')
        df_wlan_ping = get_df_with_host('avg_over_time(WLAN_PING{metrics="avgRTT", type="EXTERNAL"}[1h])', 'wlan_ping')

        dfs = [df_lan_down, df_lan_up, df_wlan_down, df_wlan_up, df_lan_ping, df_wlan_ping]
        df = dfs[0]
        for d in dfs[1:]:
            if not d.empty: df = pd.merge(df, d, on=['timestamp', 'hostname'], how='outer')
        df = df.fillna(0)

        # Baselines
        baselines = {}
        probes = df['hostname'].unique()
        for probe in probes:
            probe_data = df[df['hostname'] == probe]
            baselines[probe] = {
                'lan_down_max': probe_data['lan_down'].quantile(0.95) or 1,
                'lan_up_max': probe_data['lan_up'].quantile(0.95) or 1,
                'wlan_down_max': probe_data['wlan_down'].quantile(0.95) or 1,
                'wlan_up_max': probe_data['wlan_up'].quantile(0.95) or 1,
                'lan_ping_max': probe_data['lan_ping'].quantile(0.95) or 100,
                'wlan_ping_max': probe_data['wlan_ping'].quantile(0.95) or 100
            }

        def normalize(row):
            host = row['hostname']
            if host in baselines:
                base = baselines[host]
                row['lan_down'] = row['lan_down'] / base['lan_down_max']
                row['lan_up'] = row['lan_up'] / base['lan_up_max']
                row['wlan_down'] = row['wlan_down'] / base['wlan_down_max']
                row['wlan_up'] = row['wlan_up'] / base['wlan_up_max']
                row['lan_ping'] = row['lan_ping'] / base['lan_ping_max']
                row['wlan_ping'] = row['wlan_ping'] / base['wlan_ping_max']
            return row

        df_normalized = df.apply(normalize, axis=1)
        
        df_normalized['hour'] = df_normalized['timestamp'].dt.hour
        df_normalized['day_of_week'] = df_normalized['timestamp'].dt.dayofweek
        df_normalized['is_weekend'] = df_normalized['day_of_week'].isin([5, 6]).astype(int)
        
        for c in ['lan_dns', 'wlan_dns']:
            if c not in df_normalized.columns: df_normalized[c] = 0

        if len(df) > 50:
            print(f"✅ Fetched {len(df)} real data points.")
            return df_normalized, baselines
            
    except Exception as e:
        print(f"⚠️ Error: {e}")
        return generate_enhanced_synthetic_data()

    return generate_enhanced_synthetic_data()

def generate_enhanced_synthetic_data(n_samples=5000):
    start = datetime.datetime.now() - datetime.timedelta(hours=n_samples)
    timestamps = [start + datetime.timedelta(hours=i) for i in range(n_samples)]
    df = pd.DataFrame({'timestamp': timestamps})
    
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    vals = [max(0, min(1, 0.5 + np.random.normal(0, 0.1))) for _ in range(n_samples)]
    df['lan_down'] = vals
    df['lan_up'] = vals
    df['wlan_down'] = vals
    df['wlan_up'] = vals
    df['lan_ping'] = [0.2 + (v*0.1) for v in vals]
    df['wlan_ping'] = vals
    df['lan_dns'] = 0.2
    df['wlan_dns'] = 0.2
    
    baselines = {"default": {"lan_down_max": 1000}}
    return df, baselines

def train_models():
    ensure_directory()
    print("=== 🚀 Starting AI Training Pipeline ===")
    
    # 1. Load User Config
    config = settings_manager.get_settings()
    contamination = config.get('anomaly_contamination', 0.01)
    n_estimators = int(config.get('anomaly_estimators', 200))
    print(f"⚙️ Config: Sensitivity={contamination}, Complexity={n_estimators}")

    # 2. Get Data
    df, baselines = fetch_real_data()
    if df.empty: 
        print("❌ No data available.")
        return

    # 3. Train Anomaly Detection
    print("\n🧠 Training Anomaly Detection...")
    anomaly_cols = ['hour', 'is_weekend', 'lan_down', 'lan_up', 'wlan_down', 'wlan_up', 'lan_ping', 'wlan_ping', 'lan_dns', 'wlan_dns']
    for c in anomaly_cols: 
        if c not in df.columns: df[c] = 0
            
    model = IsolationForest(n_estimators=n_estimators, contamination=contamination, random_state=42)
    model.fit(df[anomaly_cols])
    
    stats = df[anomaly_cols].describe().loc[['mean', 'std']].to_dict()
    joblib.dump({'model': model, 'baselines': baselines, 'features': anomaly_cols, 'stats': stats}, ANOMALY_MODEL_PATH)
    print(f"✅ Anomaly Model saved.")

    # Note: Forecast training block removed as per request
    print("\n=== ✨ Training Complete ===")

if __name__ == "__main__":
    train_models()