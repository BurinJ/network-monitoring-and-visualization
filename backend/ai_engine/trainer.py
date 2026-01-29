import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest # , RandomForestRegressor
from prometheus_api_client import PrometheusConnect
import joblib
import os
import datetime
import sys
# import time

# --- PyTorch Imports for LSTM (EXPERIMENTAL) ---
'''
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("⚠️ PyTorch not found. LSTM training will be skipped.")
'''
# Configuration
try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from config import PROMETHEUS_URL, PROM_USER, PROM_PASSWORD
except ImportError:
    PROMETHEUS_URL = "http://localhost:9090"
    PROM_USER = None
    PROM_PASSWORD = None

BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# Model Paths
ANOMALY_MODEL_PATH = os.path.join(MODEL_DIR, 'anomaly_model.pkl')
# RF_LAN_PATH = os.path.join(MODEL_DIR, 'rf_lan.pkl')
# RF_WLAN_PATH = os.path.join(MODEL_DIR, 'rf_wlan.pkl')
# LSTM_LAN_PATH = os.path.join(MODEL_DIR, 'lstm_lan.pth')
# LSTM_WLAN_PATH = os.path.join(MODEL_DIR, 'lstm_wlan.pth')

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

    print(f"📥 Fetching last {days} days of INDIVIDUAL metrics...")
    
    try:
        def get_df_with_host(query, col_name):
            # We use `sum by (hostname)` (or just the metric if it has hostname label) 
            # to ensure we get individual series, not global avg.
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

        # 1. Fetch Individual Series (to calculate per-probe baselines)
        # Note: Removing 'avg()' wrapper to get per-instance data
        df_lan_down = get_df_with_host('avg_over_time(LAN_EXTERNAL_SPEEDTEST{type="Download"}[1h])', 'lan_down')
        df_lan_up = get_df_with_host('avg_over_time(LAN_EXTERNAL_SPEEDTEST{type="Upload"}[1h])', 'lan_up')
        df_wlan_down = get_df_with_host('avg_over_time(WLAN_EXTERNAL_SPEEDTEST{type="Download"}[1h])', 'wlan_down')
        df_wlan_up = get_df_with_host('avg_over_time(WLAN_EXTERNAL_SPEEDTEST{type="Upload"}[1h])', 'wlan_up')
        df_lan_ping = get_df_with_host('avg_over_time(LAN_PING{metrics="avgRTT", type="EXTERNAL"}[1h])', 'lan_ping')
        df_wlan_ping = get_df_with_host('avg_over_time(WLAN_PING{metrics="avgRTT", type="EXTERNAL"}[1h])', 'wlan_ping')

        print("   ...Processing and merging data frames...")
        # We need to merge carefully. Outer join on timestamp+hostname.
        dfs = [df_lan_down, df_lan_up, df_wlan_down, df_wlan_up, df_lan_ping, df_wlan_ping]
        df = dfs[0]
        for d in dfs[1:]:
            if not d.empty:
                df = pd.merge(df, d, on=['timestamp', 'hostname'], how='outer')
        
        df = df.fillna(0)

        # --- CALCULATE PER-PROBE BASELINES ---
        print("   ...Calculating baselines...")
        baselines = {}
        probes = df['hostname'].unique()
        for probe in probes:
            probe_data = df[df['hostname'] == probe]
            # Use 95th percentile as capacity
            baselines[probe] = {
                'lan_down_max': probe_data['lan_down'].quantile(0.95) or 1,
                'lan_up_max': probe_data['lan_up'].quantile(0.95) or 1,
                'wlan_down_max': probe_data['wlan_down'].quantile(0.95) or 1,
                'wlan_up_max': probe_data['wlan_up'].quantile(0.95) or 1,
                # For ping, we can use a standard threshold or max observed
                'lan_ping_max': 100, # Standard threshold for normalization
                'wlan_ping_max': 200
            }

        # --- NORMALIZE (For Anomaly Detection) ---
        # This allows the AI to compare "Percent Load" across different networks
        def normalize(row):
            host = row['hostname']
            if host in baselines:
                base = baselines[host]
                row['lan_down'] = row['lan_down'] / base['lan_down_max']
                row['lan_up'] = row['lan_up'] / base['lan_up_max']
                row['wlan_down'] = row['wlan_down'] / base['wlan_down_max']
                row['wlan_up'] = row['wlan_up'] / base['wlan_up_max']
                # Ping is usually absolute, but we can normalize to a "Bad" threshold
                row['lan_ping'] = row['lan_ping'] / base['lan_ping_max']
                row['wlan_ping'] = row['wlan_ping'] / base['wlan_ping_max']
            return row

        df_normalized = df.copy().apply(normalize, axis=1)
        
        # Add Time Features (For RF Forecast)
        df_normalized['hour'] = df_normalized['timestamp'].dt.hour
        df_normalized['day_of_week'] = df_normalized['timestamp'].dt.dayofweek
        df_normalized['is_weekend'] = df_normalized['day_of_week'].isin([5, 6]).astype(int)

        # Fill any remaining missing columns needed for model
        for c in ['lan_dns', 'wlan_dns']: df_normalized[c] = 0

        # --- AGGREGATE FOR GLOBAL FORECAST ---
        # Calculate global average download at each timestamp
        # This dataset will be used for training the Forecast Models
        df_global = df.groupby('timestamp')[['lan_down', 'wlan_down']].mean().reset_index()
        # Add time features to global
        df_global['hour'] = df_global['timestamp'].dt.hour
        df_global['day_of_week'] = df_global['timestamp'].dt.dayofweek
        df_global['is_weekend'] = df_global['day_of_week'].isin([5, 6]).astype(int)

        if len(df) > 50:
            print(f"✅ Fetched {len(df)} individual data points.")
            return df_normalized, df_global, baselines
            
    except Exception as e:
        print(f"⚠️ Error fetching real data: {e}")
        return generate_synthetic_data()

    return generate_synthetic_data()

def generate_synthetic_data(n_samples=5000):
    # Generates synthetic data if DB fails
    start = datetime.datetime.now() - datetime.timedelta(hours=n_samples)
    timestamps = [start + datetime.timedelta(hours=i) for i in range(n_samples)]
    
    # 1. Create Global Trend
    df_global = pd.DataFrame({'timestamp': timestamps})
    df_global['hour'] = df_global['timestamp'].dt.hour
    df_global['day_of_week'] = df_global['timestamp'].dt.dayofweek
    df_global['is_weekend'] = df_global['day_of_week'].isin([5, 6]).astype(int)
    
    # Global Load Pattern (Capacity dips during day)
    global_lan = []
    global_wlan = []
    for i, row in df_global.iterrows():
        h = row['hour']
        usage = 0.3 * np.exp(-((h - 14)**2) / 20) # Peak usage at 14:00
        global_lan.append(1000 * (1.0 - usage + np.random.normal(0, 0.01)))
        global_wlan.append(500 * (1.0 - (usage*1.5) + np.random.normal(0, 0.05)))
        
    df_global['lan_down'] = global_lan
    df_global['wlan_down'] = global_wlan
    
    # 2. Create Individual (Normalized) Data
    # Just reusing global pattern normalized 0-1 for anomaly training
    df_indiv = df_global.copy()
    df_indiv['lan_down'] /= 1000
    df_indiv['wlan_down'] /= 500
    df_indiv['lan_up'] = df_indiv['lan_down'] * 0.9
    df_indiv['wlan_up'] = df_indiv['wlan_down'] * 0.8
    df_indiv['lan_ping'] = 0.1
    df_indiv['wlan_ping'] = 0.2
    df_indiv['lan_dns'] = 0
    df_indiv['wlan_dns'] = 0
    
    baselines = {"default": {"lan_down_max": 1000}}
    return df_indiv, df_global, baselines

# --- LSTM SINGLE METRIC MODEL (EXPERIMENTAL) ---
'''
if PYTORCH_AVAILABLE:
    class LSTMSingle(nn.Module):
        def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
            super(LSTMSingle, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            out, _ = self.lstm(x, (h0, c0))
            out = self.fc(out[:, -1, :])
            return out
    
    def get_device():
        if torch.cuda.is_available(): return torch.device('cuda'), "CUDA"
        return torch.device('cpu'), "CPU"

    def train_single_lstm(df, col_name, save_path, seq_length=24, epochs=5):
        device, dev_name = get_device()
        print(f"\n🧠 Training LSTM for {col_name} on {dev_name}...")
        
        # Data: Single column of the GLOBAL trend
        data = df[col_name].values.astype(np.float32)
        # Normalize Global Trend 0-1
        max_val = np.max(data)
        if max_val > 0: data = data / max_val
        
        xs, ys = [], []
        for i in range(len(data) - seq_length):
            x = data[i:(i + seq_length)].reshape(seq_length, 1) # [24, 1]
            y = data[i + seq_length]
            xs.append(x)
            ys.append(y)
            
        if not xs: return
        
        X_tensor = torch.from_numpy(np.array(xs))
        y_tensor = torch.from_numpy(np.array(ys).reshape(-1, 1))
        
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        model = LSTMSingle(input_size=1).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        model.train()
        for epoch in range(epochs):
            for inputs, targets in loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
        
        # Save Dictionary with Metadata
        torch.save({
            'model_state': model.state_dict(),
            'max_val': float(max_val) # Save global max for denormalization
        }, save_path)
        print(f"✅ Saved to {save_path}")
'''

def train_models():
    ensure_directory()
    print("=== 🚀 Starting AI Training Pipeline ===")
    
    # 1. Fetch Data
    df_indiv, df_global, baselines = fetch_real_data()
    if df_indiv.empty: 
        print("❌ No data available.")
        return

    # 2. Train Anomaly Detection (On Normalized Individual Data)
    print("\n🧠 Training Anomaly Detector (Individual)...")
    anomaly_cols = ['lan_down', 'lan_up', 'wlan_down', 'wlan_up', 'lan_ping', 'wlan_ping', 'lan_dns', 'wlan_dns']
    model = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
    model.fit(df_indiv[anomaly_cols])
    
    # Save Model + Per-Probe Baselines
    joblib.dump({'model': model, 'baselines': baselines, 'features': anomaly_cols}, ANOMALY_MODEL_PATH)
    print(f"✅ Anomaly Model saved.")

    # 3. Train Global Forecasts (On Global Aggregated Data)
    # print("\n🔮 Training Global Forecasts (Random Forest)...")
    # X_time = df_global[['hour', 'day_of_week', 'is_weekend']]
    
    # rf_lan = RandomForestRegressor(n_estimators=100)
    # rf_lan.fit(X_time, df_global['lan_down'])
    # joblib.dump(rf_lan, RF_LAN_PATH)
    
    # rf_wlan = RandomForestRegressor(n_estimators=100)
    # rf_wlan.fit(X_time, df_global['wlan_down'])
    # joblib.dump(rf_wlan, RF_WLAN_PATH)
    # print(f"✅ RF Models saved.")

    # 4. Train Global LSTMs
    # if PYTORCH_AVAILABLE:
    #    train_single_lstm(df_global, 'lan_down', LSTM_LAN_PATH)
    #    train_single_lstm(df_global, 'wlan_down', LSTM_WLAN_PATH)

if __name__ == "__main__":
    train_models()