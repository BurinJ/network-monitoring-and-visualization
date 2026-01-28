import joblib
import pandas as pd
import os
import datetime
import numpy as np

try:
    import torch
    import torch.nn as nn
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
ANOMALY_PATH = os.path.join(MODEL_DIR, 'anomaly_model.pkl')
FORECAST_PATH = os.path.join(MODEL_DIR, 'forecast_model.pkl')
LSTM_PATH = os.path.join(MODEL_DIR, 'lstm_forecast_model.pth')

ANOMALY_MODEL = None
FORECAST_MODEL = None
LSTM_MODEL = None
STATS = None
BASELINES = {}
FEATURES = []
DEPARTMENTS = {}

if PYTORCH_AVAILABLE:
    class LSTMUniversal(nn.Module):
        def __init__(self, input_size=3, hidden_size=64, num_layers=2, output_size=1):
            super(LSTMUniversal, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
            out, _ = self.lstm(x, (h0, c0))
            out = self.fc(out[:, -1, :])
            return out

def load_models():
    global ANOMALY_MODEL, FORECAST_MODEL, LSTM_MODEL, STATS, BASELINES, FEATURES, DEPARTMENTS
    try:
        # Load Anomaly
        if os.path.exists(ANOMALY_PATH):
            data = joblib.load(ANOMALY_PATH)
            if isinstance(data, dict):
                ANOMALY_MODEL = data['model']
                FEATURES = data.get('features', [])
                STATS = data.get('stats', {})
                BASELINES = data.get('baselines', {})
                DEPARTMENTS = data.get('departments', {})
            else: ANOMALY_MODEL = data

        # Load Forecast (Restored)
        if os.path.exists(FORECAST_PATH):
            FORECAST_MODEL = joblib.load(FORECAST_PATH)
            print("✅ Random Forest Model loaded.")

        # Load LSTM
        if PYTORCH_AVAILABLE and os.path.exists(LSTM_PATH):
            model = LSTMUniversal(input_size=3, output_size=1)
            try:
                model.load_state_dict(torch.load(LSTM_PATH, map_location=torch.device('cpu')))
                model.eval()
                LSTM_MODEL = model
                print("✅ Universal LSTM Model loaded.")
            except Exception as e: print(f"❌ LSTM Load Error: {e}")

    except Exception as e:
        print(f"❌ Error loading models: {e}")

load_models()

def detect_anomaly(metrics, probe_id="default"):
    # (Same as before)
    if not ANOMALY_MODEL: return {"is_anomaly": False, "score": 0, "desc": "AI Not Initialized", "department": "Unknown"}
    baseline = BASELINES.get(probe_id, BASELINES.get("default", {'lan_down_max': 1000, 'wlan_down_max': 500}))
    department = DEPARTMENTS.get(probe_id, "General")
    def safe_div(n, d): return n / d if d else 0

    norm_lan_down = safe_div(metrics.get('lan_down', 0), baseline.get('lan_down_max', 1))
    norm_lan_up = safe_div(metrics.get('lan_up', 0), baseline.get('lan_up_max', 1))
    norm_wlan_down = safe_div(metrics.get('wlan_down', 0), baseline.get('wlan_down_max', 1))
    norm_wlan_up = safe_div(metrics.get('wlan_up', 0), baseline.get('wlan_up_max', 1))

    # NO TIME FEATURES for Anomaly
    input_data = {
        'lan_down': norm_lan_down, 'lan_up': norm_lan_up,
        'wlan_down': norm_wlan_down, 'wlan_up': norm_wlan_up,
        'lan_ping': metrics.get('lan_ping', 0), 'wlan_ping': metrics.get('wlan_ping', 0),
        'lan_dns': metrics.get('lan_dns', 0), 'wlan_dns': metrics.get('wlan_dns', 0),
    }

    features = pd.DataFrame([input_data])
    if FEATURES:
        for f in FEATURES: 
            if f not in features.columns: features[f] = 0
        features = features[FEATURES]
    try:
        prediction = ANOMALY_MODEL.predict(features)[0]
        score = ANOMALY_MODEL.decision_function(features)[0] 
        if prediction == -1:
            desc = "Traffic deviation"
            if norm_wlan_down < 0.05: desc = "Abnormal speed drop"
            elif input_data['lan_dns'] > 100: desc = "Latency anomaly"
            return {"is_anomaly": True, "score": round(float(score), 4), "desc": desc, "department": department}
        return {"is_anomaly": False, "score": round(float(score), 4), "desc": "Normal", "department": department}
    except: return {"is_anomaly": False, "score": 0, "desc": "Error", "department": "Unknown"}

def predict_future_traffic(recent_history=None, is_wlan=False, is_ping=False):
    forecasts = []
    now = datetime.datetime.now()
    start_time = (now + datetime.timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    
    # 1. Random Forest (Restored)
    rf_preds = []
    if FORECAST_MODEL:
        future_data = []
        for i in range(24):
            future_time = start_time + datetime.timedelta(hours=i)
            future_data.append({
                'hour': future_time.hour,
                'day_of_week': future_time.weekday(),
                'is_weekend': 1 if future_time.weekday() >= 5 else 0
            })
        try:
            # Predict
            rf_raw = FORECAST_MODEL.predict(pd.DataFrame(future_data))
            # Normalize 0-100%
            rf_preds = [max(0, min(100, int(val * 100))) for val in rf_raw]
        except: 
            rf_preds = [50] * 24

    # 2. LSTM Prediction
    lstm_preds = []
    seq_len = 24
    
    if LSTM_MODEL and recent_history and len(recent_history) >= seq_len:
        try:
            history_vals = recent_history[-seq_len:]
            sequence_inputs = []
            
            flag_wlan = 1.0 if is_wlan else 0.0
            flag_ping = 1.0 if is_ping else 0.0
            
            for val in history_vals:
                sequence_inputs.append([val, flag_wlan, flag_ping])

            curr_seq = torch.tensor(sequence_inputs, dtype=torch.float32).view(1, seq_len, 3)
            
            with torch.no_grad():
                for _ in range(24):
                    pred = LSTM_MODEL(curr_seq)
                    val = max(0.0, min(1.5, pred.item()))
                    lstm_preds.append(int(val * 100))
                    
                    new_pt = torch.tensor([[[val, flag_wlan, flag_ping]]], dtype=torch.float32)
                    curr_seq = torch.cat((curr_seq[:, 1:, :], new_pt), dim=1)
        except Exception as e:
            print(f"LSTM Error: {e}")

    # 3. Combine
    for i in range(24):
        time_label = (start_time + datetime.timedelta(hours=i)).strftime("%H:00")
        
        rf_val = rf_preds[i] if i < len(rf_preds) else 50
        lstm_val = lstm_preds[i] if i < len(lstm_preds) else rf_val
        
        forecasts.append({
            "time": time_label,
            "rf_load": rf_val,
            "lstm_load": lstm_val
        })
            
    return forecasts