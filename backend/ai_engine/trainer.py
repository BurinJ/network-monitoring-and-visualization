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

# Setup directories
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'anomaly_model.pkl')

def ensure_directory():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

def fetch_real_data(days=14):
    """
    Fetches real historical data from Prometheus to train the model.
    """
    print(f"🔌 Connecting to Prometheus at {PROMETHEUS_URL}...")
    
    headers = {}
    if PROM_USER and PROM_PASSWORD:
        import base64
        auth = base64.b64encode(f"{PROM_USER}:{PROM_PASSWORD}".encode()).decode()
        headers = {"Authorization": f"Basic {auth}"}

    prom = PrometheusConnect(url=PROMETHEUS_URL, headers=headers, disable_ssl=True)
    
    # Range: Last N days
    start_time = datetime.datetime.now() - datetime.timedelta(days=days)
    end_time = datetime.datetime.now()
    step = '1h' # Hourly resolution is good for patterns

    print(f"📥 Fetching last {days} days of metrics...")

    try:
        # We need to fetch metrics individually and merge them by timestamp
        # 1. WLAN Latency
        # Note: We use avg_over_time to smooth out spikes
        wlan_ping_data = prom.custom_query_range(
            query='avg(avg_over_time(GENERAL_info{wlan_google_response_time!=""}[1h]))', 
            start_time=start_time, end_time=end_time, step=step
        )
        
        # 2. LAN Latency
        lan_ping_data = prom.custom_query_range(
            query='avg(avg_over_time(GENERAL_info{lan_google_response_time!=""}[1h]))', 
            start_time=start_time, end_time=end_time, step=step
        )

        # Helper to convert Prometheus result to DataFrame
        def to_df(prom_data, col_name):
            if not prom_data: return pd.DataFrame()
            # prom_data[0]['values'] is a list of [timestamp, value]
            df = pd.DataFrame(prom_data[0]['values'], columns=['timestamp', col_name])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            df[col_name] = pd.to_numeric(df[col_name])
            return df.set_index('timestamp')

        df_wlan = to_df(wlan_ping_data, 'wlan_ping')
        df_lan = to_df(lan_ping_data, 'lan_ping')
        
        # Merge DataFrames
        df = df_wlan.join(df_lan, how='outer').fillna(0)
        
        # Add Time Features for Seasonality
        df['hour'] = df.index.hour
        df['is_weekend'] = df.index.dayofweek.isin([5, 6]).astype(int)
        
        # If we have enough data points, return it. Otherwise fall back to synthetic.
        if len(df) > 100:
            print(f"✅ Fetched {len(df)} real data points.")
            # Add missing columns with defaults if your query didn't cover them
            if 'lan_dns' not in df.columns: df['lan_dns'] = df['lan_ping'] * 0.8
            if 'wlan_dns' not in df.columns: df['wlan_dns'] = df['wlan_ping'] * 1.2
            
            # Ensure column order matches analyzer expectation
            # ['hour', 'is_weekend', 'lan_ping', 'wlan_ping', 'lan_dns', 'wlan_dns']
            return df[['hour', 'is_weekend', 'lan_ping', 'wlan_ping', 'lan_dns', 'wlan_dns']]
            
    except Exception as e:
        print(f"⚠️ Error fetching real data: {e}")
        print("   Falling back to synthetic data...")

    return generate_enhanced_synthetic_data()

def generate_enhanced_synthetic_data(n_samples=5000):
    """
    Generates training data that simulates University Seasonality.
    """
    print("🧪 Generating time-aware training data...")
    
    # Create time series
    hours = np.random.randint(0, 24, n_samples) # 0 to 23
    is_weekend = np.random.choice([0, 1], n_samples, p=[0.7, 0.3]) # 0=Weekday, 1=Weekend
    
    lan_pings = []
    wlan_pings = []
    
    for h, w in zip(hours, is_weekend):
        # Logic: Traffic is higher (slower ping) during day (9-17) on weekdays
        base_ping = 20
        if w == 0 and 9 <= h <= 17:
            base_ping += 40 # Busy classes
        elif w == 0 and 18 <= h <= 22:
            base_ping += 20 # Dorm usage
        else:
            base_ping += 5 # Night/Weekend
            
        # Add noise
        lan_pings.append(max(1, np.random.normal(base_ping * 0.5, 5)))
        wlan_pings.append(max(1, np.random.normal(base_ping, 10)))

    df = pd.DataFrame({
        'hour': hours,
        'is_weekend': is_weekend,
        'lan_ping': lan_pings,
        'wlan_ping': wlan_pings,
        'lan_dns': [p * 0.8 for p in lan_pings], # Correlation
        'wlan_dns': [p * 1.2 for p in wlan_pings],
    })
    
    print(f"   - Generated {n_samples} samples with seasonality patterns.")
    return df

def train_model():
    ensure_directory()
    
    # 1. Get Data (Enhanced)
    df = fetch_real_data()
    
    # 2. Train Isolation Forest
    # We include 'hour' and 'is_weekend' so the model learns that 
    # high ping at 2PM is normal, but high ping at 4AM is an anomaly.
    print("🧠 Training Seasonality-Aware AI Model...")
    model = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
    model.fit(df)
    
    # 3. Save
    joblib.dump(model, MODEL_PATH)
    print(f"✅ Model saved to: {MODEL_PATH}")

if __name__ == "__main__":
    train_model()