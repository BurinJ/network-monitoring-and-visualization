import joblib
import pandas as pd
import os
import datetime
import numpy as np

# --- PyTorch Setup ---
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
BASELINES = {}
ANOMALY_FEATURES = []

# --- LSTM Class Definition (Must match trainer.py) ---
if PYTORCH_AVAILABLE:
    class LSTMForecaster(nn.Module):
        def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
            super(LSTMForecaster, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            out, _ = self.lstm(x, (h0, c0))
            out = self.fc(out[:, -1, :])
            return out

def load_models():
    global ANOMALY_MODEL, FORECAST_MODEL, LSTM_MODEL, BASELINES, ANOMALY_FEATURES
    try:
        # Load Anomaly Model
        if os.path.exists(ANOMALY_PATH):
            data = joblib.load(ANOMALY_PATH)
            if isinstance(data, dict):
                ANOMALY_MODEL = data['model']
                BASELINES = data['baselines']
                ANOMALY_FEATURES = data['features']
            else:
                print("⚠️ Legacy anomaly model.")
        
        # Load Forecast Model (Random Forest)
        if os.path.exists(FORECAST_PATH):
            FORECAST_MODEL = joblib.load(FORECAST_PATH)
            print("✅ Random Forest Model loaded.")

        # Load LSTM Model
        if PYTORCH_AVAILABLE and os.path.exists(LSTM_PATH):
            model = LSTMForecaster()
            try:
                model.load_state_dict(torch.load(LSTM_PATH))
                model.eval() # Set to evaluation mode
                LSTM_MODEL = model
                print("✅ LSTM Model loaded.")
            except Exception as e:
                print(f"❌ Failed to load LSTM weights: {e}")

    except Exception as e:
        print(f"❌ Error loading models: {e}")

load_models()

def detect_anomaly(metrics, probe_id="default"):
    if not ANOMALY_MODEL: return {"is_anomaly": False, "score": 0, "desc": "AI Not Ready"}
    
    baseline = BASELINES.get(probe_id, BASELINES.get("default", {'lan_down_max': 1000, 'wlan_down_max': 500}))
    def safe_div(n, d): return n / d if d else 0
    
    input_data = {
        'hour': datetime.datetime.now().hour,
        'is_weekend': 1 if datetime.datetime.now().weekday() >= 5 else 0,
        'lan_down': safe_div(metrics.get('lan_down', 0), baseline.get('lan_down_max', 1)),
        'lan_up': safe_div(metrics.get('lan_up', 0), baseline.get('lan_up_max', 1)),
        'wlan_down': safe_div(metrics.get('wlan_down', 0), baseline.get('wlan_down_max', 1)),
        'wlan_up': safe_div(metrics.get('wlan_up', 0), baseline.get('wlan_up_max', 1)),
        'lan_ping': metrics.get('lan_ping', 0),
        'wlan_ping': metrics.get('wlan_ping', 0),
        'lan_dns': metrics.get('lan_dns', 0),
        'wlan_dns': metrics.get('wlan_dns', 0),
    }

    features = pd.DataFrame([input_data])
    if ANOMALY_FEATURES:
        valid_feats = {k: v for k, v in input_data.items() if k in ANOMALY_FEATURES}
        features = pd.DataFrame([valid_feats])
        features = features.reindex(columns=ANOMALY_FEATURES, fill_value=0)

    try:
        pred = ANOMALY_MODEL.predict(features)[0]
        score = ANOMALY_MODEL.decision_function(features)[0]
        if pred == -1:
            desc = "Traffic deviation"
            if input_data['wlan_down'] < 0.05: desc = "Abnormal speed drop"
            elif input_data['lan_dns'] > 100: desc = "Latency anomaly"
            return {"is_anomaly": True, "score": round(float(score), 4), "desc": desc}
        return {"is_anomaly": False, "score": round(float(score), 4), "desc": "Normal"}
    except:
        return {"is_anomaly": False, "score": 0, "desc": "Error"}

def predict_future_traffic(recent_history=None):
    """
    Generates 24-hour forecasts using RF and LSTM.
    Args:
        recent_history (list): Last 24 hours of normalized traffic data (0.0 - 1.0)
    """
    forecasts = []
    now = datetime.datetime.now()
    
    # --- 1. Random Forest Prediction (Time-based) ---
    rf_preds = []
    if FORECAST_MODEL:
        future_data = []
        for i in range(24):
            future_time = now + datetime.timedelta(hours=i)
            future_data.append({
                'hour': future_time.hour,
                'day_of_week': future_time.weekday(),
                'is_weekend': 1 if future_time.weekday() >= 5 else 0
            })
        try:
            rf_raw = FORECAST_MODEL.predict(pd.DataFrame(future_data))
            rf_preds = [max(0, min(100, int(val * 100))) for val in rf_raw]
        except: pass
    
    # --- 2. LSTM Prediction (Sequence-based) ---
    lstm_preds = []
    if LSTM_MODEL and recent_history and len(recent_history) >= 24:
        try:
            # Take last 24 points
            seq = recent_history[-24:] 
            # Normalize inputs to 0-1 range if not already (assuming input is Mbps, we need %)
            # Actually, fetcher should pass normalized data. Let's assume input is 0-1 floats.
            
            curr_seq = torch.tensor(seq, dtype=torch.float32).view(1, 24, 1)
            
            with torch.no_grad():
                for _ in range(24):
                    pred = LSTM_MODEL(curr_seq)
                    val = pred.item()
                    # Clamp 0-1
                    val = max(0.0, min(1.0, val))
                    lstm_preds.append(int(val * 100))
                    
                    # Recursive step
                    new_pt = torch.tensor([[[val]]], dtype=torch.float32)
                    curr_seq = torch.cat((curr_seq[:, 1:, :], new_pt), dim=1)
        except Exception as e:
            print(f"LSTM Error: {e}")

    # Combine results
    for i in range(24):
        time_label = (now + datetime.timedelta(hours=i)).strftime("%H:00")
        
        # Defaults if models missing
        rf_val = rf_preds[i] if i < len(rf_preds) else 0
        lstm_val = lstm_preds[i] if i < len(lstm_preds) else None
        
        entry = {
            "time": time_label,
            "rf_load": rf_val,
            "lstm_load": lstm_val if lstm_val is not None else rf_val # Fallback to RF if LSTM fails
        }
        forecasts.append(entry)
            
    return forecasts