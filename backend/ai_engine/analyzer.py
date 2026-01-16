import joblib
import pandas as pd
import os
import datetime
import numpy as np

# --- PyTorch Setup for LSTM ---
try:
    import torch
    import torch.nn as nn
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

# Define path to the saved models
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
    global ANOMALY_MODEL, FORECAST_MODEL, LSTM_MODEL, STATS, BASELINES, FEATURES
    try:
        # 1. Load Anomaly Model & Stats
        if os.path.exists(ANOMALY_PATH):
            data = joblib.load(ANOMALY_PATH)
            # Handle different versions of the model file
            if isinstance(data, dict) and 'model' in data:
                ANOMALY_MODEL = data['model']
                FEATURES = data.get('features', [])
                
                # Try to load stats/baselines from dictionary
                STATS = data.get('stats', {})
                BASELINES = data.get('baselines', {})
                
                print(f"✅ AI Model loaded from {ANOMALY_PATH}")
            else:
                ANOMALY_MODEL = data
                STATS = {}
                BASELINES = {}
                FEATURES = []
                print("⚠️ Loaded legacy model format.")
        else:
            print("⚠️ Anomaly Model not found.")

        # 2. Load Random Forest Forecast Model
        if os.path.exists(FORECAST_PATH):
            FORECAST_MODEL = joblib.load(FORECAST_PATH)
            print("✅ Random Forest Model loaded.")

        # 3. Load LSTM Model
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

# Load models immediately
load_models()

def detect_anomaly(metrics, probe_id="default"):
    """
    Analyzes metrics using Isolation Forest + Statistical Deviation Check.
    Output: { is_anomaly: bool, score: float, desc: str }
    """
    if not ANOMALY_MODEL:
        return {"is_anomaly": False, "score": 0, "desc": "AI Not Initialized"}

    # 1. Normalize Speed Data (Convert Mbps -> % Utilization)
    baseline = BASELINES.get(probe_id, BASELINES.get("default", {
        'lan_down_max': 1000, 'lan_up_max': 1000, 
        'wlan_down_max': 500, 'wlan_up_max': 500
    }))
    
    def safe_div(n, d): return n / d if d else 0

    norm_lan_down = safe_div(metrics.get('lan_down', 0), baseline.get('lan_down_max', 1))
    norm_lan_up = safe_div(metrics.get('lan_up', 0), baseline.get('lan_up_max', 1))
    norm_wlan_down = safe_div(metrics.get('wlan_down', 0), baseline.get('wlan_down_max', 1))
    norm_wlan_up = safe_div(metrics.get('wlan_up', 0), baseline.get('wlan_up_max', 1))

    # 2. Prepare Features
    now = datetime.datetime.now()
    current_hour = now.hour
    is_weekend = 1 if now.weekday() >= 5 else 0

    input_data = {
        'hour': current_hour,
        'is_weekend': is_weekend,
        'lan_down': norm_lan_down,
        'lan_up': norm_lan_up,
        'wlan_down': norm_wlan_down,
        'wlan_up': norm_wlan_up,
        'lan_ping': metrics.get('lan_ping', 0),
        'wlan_ping': metrics.get('wlan_ping', 0),
        'lan_dns': metrics.get('lan_dns', 0),
        'wlan_dns': metrics.get('wlan_dns', 0),
    }

    # Ensure feature order matches training
    features = pd.DataFrame([input_data])
    if FEATURES:
        # Fill missing features with 0 and order them
        for f in FEATURES:
            if f not in features.columns: features[f] = 0
        features = features[FEATURES]

    try:
        # 2. Predict
        prediction = ANOMALY_MODEL.predict(features)[0] # -1 = Anomaly
        score = ANOMALY_MODEL.decision_function(features)[0] 

        if prediction == -1:
            culprits = []
            
            # Variables to track the "Most Suspicious" metric if no critical errors found
            max_deviation_feat = None
            max_deviation_score = 0
            
            # 3. Root Cause Analysis (Statistical Check)
            if STATS:
                for feat in FEATURES:
                    if feat in ['hour', 'is_weekend']: continue
                    
                    val = input_data.get(feat, 0)
                    # Check if stats exist for this feature
                    if feat not in STATS: continue
                    
                    mean = STATS[feat]['mean']
                    std = STATS[feat]['std']
                    
                    if std == 0: std = 1 
                    
                    z_score = (val - mean) / std
                    
                    # Track the metric with the absolute highest deviation from normal
                    if abs(z_score) > abs(max_deviation_score):
                        max_deviation_score = z_score
                        max_deviation_feat = feat

                    # --- CRITICAL THRESHOLD CHECKS ---
                    if 'ping' in feat or 'dns' in feat:
                        if z_score > 3: 
                            culprits.append(f"Critical {feat} ({val:.1f}ms)")
                    elif 'down' in feat or 'up' in feat:
                        if z_score < -2 and val < 0.01: 
                            culprits.append(f"Critical Speed Drop on {feat}")

            # 4. Fallback: If no CRITICAL issues, explain the DEVIATION
            if not culprits and max_deviation_feat:
                feat_clean = max_deviation_feat.replace('_', ' ').title()
                if 'Ping' in feat_clean or 'Dns' in feat_clean:
                    if max_deviation_score > 0:
                        culprits.append(f"Elevated {feat_clean} (Z={max_deviation_score:.1f})")
                    else:
                        culprits.append(f"Unusually Low {feat_clean}")
                elif 'Down' in feat_clean or 'Up' in feat_clean:
                    if max_deviation_score < 0:
                         culprits.append(f"Lower than avg {feat_clean} (Z={max_deviation_score:.1f})")
                    else:
                         culprits.append(f"High Traffic Surge on {feat_clean}")
            
            # 5. Last Resort Fallback
            if not culprits:
                culprits.append("Complex Pattern Deviation")

            # 6. Formatting Output
            utc_now = datetime.datetime.utcnow()
            local_time = utc_now + datetime.timedelta(hours=7)
            ts_str = local_time.strftime("%H:%M:%S")
            
            reason_str = ", ".join(culprits)
            desc = f"Anomaly: {reason_str} at {ts_str} (ICT) (Isolation Tree)"

            return {
                "is_anomaly": True,
                "score": round(float(score), 4),
                "desc": desc
            }
        
        return {
            "is_anomaly": False, 
            "score": round(float(score), 4), 
            "desc": "Normal Behavior"
        }

    except Exception as e:
        print(f"Prediction Error: {e}")
        return {"is_anomaly": False, "score": 0, "desc": "Analysis Error"}

def predict_future_traffic(recent_history=None):
    """
    Generates 24-hour forecasts using Random Forest and LSTM.
    
    Args:
        recent_history (list): List of normalized load values (0.0 - 1.0) 
                               from the last 24 hours. Required for LSTM.
    
    Returns:
        list: List of dicts { time, rf_load, lstm_load }
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
            # Predict using dataframe (same format as training)
            rf_raw = FORECAST_MODEL.predict(pd.DataFrame(future_data))
            # Convert 0.0-1.0 to 0-100 integer
            rf_preds = [max(0, min(100, int(val * 100))) for val in rf_raw]
        except Exception as e:
            print(f"RF Forecast Error: {e}")
    
    # --- 2. LSTM Prediction (Sequence-based) ---
    lstm_preds = []
    if LSTM_MODEL and recent_history and len(recent_history) >= 24:
        try:
            # Take last 24 points
            seq = recent_history[-24:] 
            # Prepare tensor [batch, seq_len, features]
            curr_seq = torch.tensor(seq, dtype=torch.float32).view(1, 24, 1)
            
            with torch.no_grad():
                for _ in range(24):
                    # Predict next step
                    pred = LSTM_MODEL(curr_seq)
                    val = pred.item()
                    
                    # Clamp and store
                    val_clamped = max(0.0, min(1.0, val))
                    lstm_preds.append(int(val_clamped * 100))
                    
                    # Update sequence: remove oldest, add prediction
                    new_pt = torch.tensor([[[val_clamped]]], dtype=torch.float32)
                    curr_seq = torch.cat((curr_seq[:, 1:, :], new_pt), dim=1)
        except Exception as e:
            print(f"LSTM Forecast Error: {e}")

    # --- 3. Combine Results ---
    for i in range(24):
        time_label = (now + datetime.timedelta(hours=i)).strftime("%H:00")
        
        # Get RF value or default to 0
        rf_val = rf_preds[i] if i < len(rf_preds) else 0
        
        # Get LSTM value or None
        lstm_val = lstm_preds[i] if i < len(lstm_preds) else None
        
        # If LSTM failed or wasn't run, fallback to RF value
        if lstm_val is None:
            lstm_val = rf_val
        
        entry = {
            "time": time_label,
            "rf_load": rf_val,
            "lstm_load": lstm_val
        }
        forecasts.append(entry)
            
    return forecasts