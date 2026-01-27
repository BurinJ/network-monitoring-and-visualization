import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from prometheus_api_client import PrometheusConnect
import joblib
import os
import datetime
import sys
import time

# --- PyTorch Imports for LSTM ---
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("⚠️ PyTorch not found. LSTM training will be skipped.")

# Configuration imports
try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from config import PROMETHEUS_URL, PROM_USER, PROM_PASSWORD
except ImportError:
    PROMETHEUS_URL = "http://localhost:9090"
    PROM_USER = None
    PROM_PASSWORD = None

BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, 'models')
ANOMALY_MODEL_PATH = os.path.join(MODEL_DIR, 'anomaly_model.pkl')
FORECAST_MODEL_PATH = os.path.join(MODEL_DIR, 'forecast_model.pkl')
LSTM_MODEL_PATH = os.path.join(MODEL_DIR, 'lstm_forecast_model.pth')

def ensure_directory():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

def fetch_real_data(days=14):
    print(f"🔌 Connecting to Prometheus at {PROMETHEUS_URL}...")
    t_start = time.time()
    
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
                    records.append({
                        'timestamp': pd.to_datetime(val[0], unit='s'),
                        'hostname': host,
                        col_name: float(val[1])
                    })
            return pd.DataFrame(records)

        # Fetch Data
        df_lan_down = get_df_with_host('avg_over_time(LAN_EXTERNAL_SPEEDTEST{type="Download"}[1h])', 'lan_down')
        df_lan_up = get_df_with_host('avg_over_time(LAN_EXTERNAL_SPEEDTEST{type="Upload"}[1h])', 'lan_up')
        df_wlan_down = get_df_with_host('avg_over_time(WLAN_EXTERNAL_SPEEDTEST{type="Download"}[1h])', 'wlan_down')
        df_wlan_up = get_df_with_host('avg_over_time(WLAN_EXTERNAL_SPEEDTEST{type="Upload"}[1h])', 'wlan_up')
        
        # IMPORTANT: Fetch PING for multi-output training
        df_lan_ping = get_df_with_host('avg_over_time(LAN_PING{metrics="avgRTT", type="EXTERNAL"}[1h])', 'lan_ping')

        print("   ...Processing and merging data frames...")
        dfs = [df_lan_down, df_lan_up, df_wlan_down, df_wlan_up, df_lan_ping]
        df = dfs[0]
        for d in dfs[1:]:
            if not d.empty:
                df = pd.merge(df, d, on=['timestamp', 'hostname'], how='outer')
        
        df = df.fillna(0)

        # --- CALCULATE BASELINES ---
        baselines = {}
        probes = df['hostname'].unique()
        for probe in probes:
            probe_data = df[df['hostname'] == probe]
            baselines[probe] = {
                'lan_down_max': probe_data['lan_down'].quantile(0.95) or 1,
                'lan_up_max': probe_data['lan_up'].quantile(0.95) or 1,
                'wlan_down_max': probe_data['wlan_down'].quantile(0.95) or 1,
                'wlan_up_max': probe_data['wlan_up'].quantile(0.95) or 1,
                'lan_ping_max': probe_data['lan_ping'].quantile(0.95) or 100
            }

        # --- NORMALIZE & FEATURE ENGINEERING ---
        def normalize(row):
            host = row['hostname']
            if host in baselines:
                base = baselines[host]
                row['lan_down'] = row['lan_down'] / base['lan_down_max']
                row['lan_up'] = row['lan_up'] / base['lan_up_max']
                row['wlan_down'] = row['wlan_down'] / base['wlan_down_max']
                row['wlan_up'] = row['wlan_up'] / base['wlan_up_max']
                row['lan_ping'] = row['lan_ping'] / base['lan_ping_max']
            return row

        df_normalized = df.apply(normalize, axis=1)
        
        # Time Features
        df_normalized['hour'] = df_normalized['timestamp'].dt.hour
        df_normalized['day_of_week'] = df_normalized['timestamp'].dt.dayofweek
        df_normalized['is_weekend'] = df_normalized['day_of_week'].isin([5, 6]).astype(int)
        
        # Add dummy ping/dns for anomaly model compatibility if missing
        for c in ['lan_ping', 'wlan_ping', 'lan_dns', 'wlan_dns']:
            if c not in df_normalized.columns: df_normalized[c] = 0

        if len(df) > 50:
            print(f"✅ Fetched {len(df)} real data points in {time.time() - t_start:.2f}s.")
            return df_normalized, baselines
            
    except Exception as e:
        print(f"⚠️ Error fetching real data: {e}")
        return generate_enhanced_synthetic_data()

    return generate_enhanced_synthetic_data()

def generate_enhanced_synthetic_data(n_samples=5000):
    # (Simplified for brevity, ensures 2D target works)
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

if PYTORCH_AVAILABLE:
    class LSTMUniversal(nn.Module):
        def __init__(self, input_size=6, hidden_size=64, num_layers=2, output_size=1):
            super(LSTMUniversal, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            # Increased dropout to 0.3 to reduce overfitting/jitter
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            out, _ = self.lstm(x, (h0, c0))
            out = self.fc(out[:, -1, :])
            return out

    def train_lstm_model(df, seq_length=24, epochs=150, batch_size=32): # Increased epochs
        # Select device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\n🧠 Training Context-Aware LSTM Model (Device: {device})...")
        
        dataset_list = []
        metrics_config = [
            ('lan_down', 0, 0), ('lan_up', 0, 0),
            ('wlan_down', 1, 0), ('ping', 0, 1) # 'ping' mapped to lan_ping generic
        ]
        
        # Prepare context-aware dataset... (Same as before)
        # Using simplified loop for robustness
        for col, is_wlan, is_ping in metrics_config:
            if col == 'ping': col = 'lan_ping'
            if col not in df.columns: continue
            
            sub_df = df[[col, 'hour', 'day_of_week', 'is_weekend']].copy()
            sub_df['is_wlan'] = is_wlan
            sub_df['is_ping'] = is_ping
            
            # Normalize Value using 95th percentile to handle outliers better than max()
            # This avoids squashing normal data if one huge spike exists
            max_val = sub_df[col].quantile(0.95)
            if max_val > 0:
                sub_df[col] = sub_df[col] / max_val
                # Cap at 1.0 (or slightly higher if needed, but 1.0 is standard for 0-1 scaling)
                sub_df[col] = sub_df[col].clip(upper=1.0)
            
            # Normalize time
            sub_df['hour'] = sub_df['hour'] / 23.0
            sub_df['day_of_week'] = sub_df['day_of_week'] / 6.0
            
            dataset_list.append(sub_df.values)

        if not dataset_list: return
        full_data = np.vstack(dataset_list).astype(np.float32)
        
        xs, ys = [], []
        for i in range(len(full_data) - seq_length):
            x = full_data[i:(i + seq_length)]
            y = full_data[i + seq_length, 0] 
            xs.append(x)
            ys.append(y)
        
        X_tensor = torch.from_numpy(np.array(xs))
        y_tensor = torch.from_numpy(np.array(ys).reshape(-1, 1))
        
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Move model to GPU
        model = LSTMUniversal(input_size=6, output_size=1).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # Slightly lower LR for stability
        
        model.train()
        t_start = time.time()
        for epoch in range(epochs):
            epoch_loss = 0
            for inputs, targets in loader:
                # Move batches to GPU
                inputs, targets = inputs.to(device), targets.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() # Accumulate loss
            print(f"   [Epoch {epoch+1}/{epochs}] Loss: {epoch_loss/len(loader):.5f}")
        
        # Move model back to CPU for saving (ensures compatibility with CPU-only inference envs)
        model.to('cpu')
        torch.save(model.state_dict(), LSTM_MODEL_PATH)
        print(f"✅ LSTM Model saved to: {LSTM_MODEL_PATH}")
        print(f"   -> Training took {time.time() - t_start:.2f}s")

def train_models():
    ensure_directory()
    df, baselines = fetch_real_data()
    if df.empty: return

    # 1. Anomaly
    print("\n🧠 Training Anomaly Detection...")
    anomaly_cols = ['hour', 'is_weekend', 'lan_down', 'lan_up', 'wlan_down', 'wlan_up', 'lan_ping', 'wlan_ping', 'lan_dns', 'wlan_dns']
    model = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
    model.fit(df[anomaly_cols])
    joblib.dump({'model': model, 'baselines': baselines, 'features': anomaly_cols, 'stats': {}}, ANOMALY_MODEL_PATH)

    # 2. Forecast (RF) - MULTI-OUTPUT FIX
    print("\n🔮 Training Forecast (RF)...")
    X = df[['hour', 'day_of_week', 'is_weekend']]
    y = df[['lan_down', 'lan_ping']] # Target has 2 columns now
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X, y)
    joblib.dump(rf_model, FORECAST_MODEL_PATH)
    print(f"✅ RF Model Saved.")

    # 3. LSTM
    if PYTORCH_AVAILABLE: train_lstm_model(df)

if __name__ == "__main__":
    train_models()