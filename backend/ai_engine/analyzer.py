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
RF_LAN_PATH = os.path.join(MODEL_DIR, 'rf_lan.pkl')
RF_WLAN_PATH = os.path.join(MODEL_DIR, 'rf_wlan.pkl')
LSTM_LAN_PATH = os.path.join(MODEL_DIR, 'lstm_lan.pth')
LSTM_WLAN_PATH = os.path.join(MODEL_DIR, 'lstm_wlan.pth')

ANOMALY_MODEL = None
RF_LAN = None
RF_WLAN = None
LSTM_LAN = None
LSTM_WLAN = None
BASELINES = {}
FEATURES = []
DEPARTMENTS = {}
# Global max values for forecast denormalization
LAN_MAX = 1.0
WLAN_MAX = 1.0

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

def load_models():
    global ANOMALY_MODEL, RF_LAN, RF_WLAN, LSTM_LAN, LSTM_WLAN, BASELINES, FEATURES, LAN_MAX, WLAN_MAX
    try:
        # 1. Load Anomaly
        if os.path.exists(ANOMALY_PATH):
            data = joblib.load(ANOMALY_PATH)
            if isinstance(data, dict) and 'model' in data:
                ANOMALY_MODEL = data['model']
                BASELINES = data.get('baselines', {})
                FEATURES = data.get('features', [])
            else:
                print("⚠️ Unexpected anomaly model format.")
        
        # 2. Load RF
        if os.path.exists(RF_LAN_PATH): RF_LAN = joblib.load(RF_LAN_PATH)
        if os.path.exists(RF_WLAN_PATH): RF_WLAN = joblib.load(RF_WLAN_PATH)

        # 3. Load LSTM
        if PYTORCH_AVAILABLE:
            def load_lstm(path):
                if not os.path.exists(path): return None, 1.0
                try:
                    checkpoint = torch.load(path, map_location=torch.device('cpu'))
                    model = LSTMSingle(input_size=1)
                    model.load_state_dict(checkpoint['model_state'])
                    model.eval()
                    return model, checkpoint.get('max_val', 1.0)
                except Exception as e:
                    print(f"❌ LSTM Load Error {path}: {e}")
                    return None, 1.0

            LSTM_LAN, LAN_MAX = load_lstm(LSTM_LAN_PATH)
            LSTM_WLAN, WLAN_MAX = load_lstm(LSTM_WLAN_PATH)
            
        print("✅ Models loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading models: {e}")

load_models()
def detect_anomaly(metrics, probe_id="default"):
    if not ANOMALY_MODEL: return {"is_anomaly": False, "score": 0, "desc": "AI Not Initialized", "department": "Unknown"}
    baseline = BASELINES.get(probe_id, BASELINES.get("default", {'lan_down_max': 1000, 'wlan_down_max': 500}))
    department = DEPARTMENTS.get(probe_id, "General")
    def safe_div(n, d): return n / d if d else 0

    norm_lan_down = safe_div(metrics.get('lan_down', 0), baseline.get('lan_down_max', 1))
    norm_lan_up = safe_div(metrics.get('lan_up', 0), baseline.get('lan_up_max', 1))
    norm_wlan_down = safe_div(metrics.get('wlan_down', 0), baseline.get('wlan_down_max', 1))
    norm_wlan_up = safe_div(metrics.get('wlan_up', 0), baseline.get('wlan_up_max', 1))

    now = datetime.datetime.now()
    input_data = {
        'hour': now.hour, 'is_weekend': 1 if now.weekday() >= 5 else 0,
        'lan_down': norm_lan_down, 'lan_up': norm_lan_up, 'wlan_down': norm_wlan_down, 'wlan_up': norm_wlan_up,
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
            causes = []

            # --- Expanded Root Cause Analysis (Structured List) ---
            
            # 1. Check for Congestion (Low Speed + High Ping)
            if norm_lan_down < 0.4 and metrics.get('lan_ping', 0) > 50:
                causes.append({
                    "title": "LAN Congestion",
                    "detail": f"High Latency ({int(metrics['lan_ping'])}ms) correlating with Low Throughput ({int(norm_lan_down*100)}%). Pipe is likely saturated."
                })
            if norm_wlan_down < 0.4 and metrics.get('wlan_ping', 0) > 100:
                causes.append({
                    "title": "WLAN Congestion",
                    "detail": f"High Wi-Fi Latency ({int(metrics['wlan_ping'])}ms) with Low Throughput ({int(norm_wlan_down*100)}%). Access point overload likely."
                })

            # 2. Check for Pure Latency Issues (Good Speed + High Ping)
            if norm_lan_down > 0.7 and metrics.get('lan_ping', 0) > 50:
                causes.append({
                    "title": "High LAN Latency",
                    "detail": f"Ping is high ({int(metrics['lan_ping'])}ms) despite healthy bandwidth. Check routing or physical distance."
                })
            if norm_wlan_down > 0.7 and metrics.get('wlan_ping', 0) > 100:
                causes.append({
                    "title": "High WLAN Latency",
                    "detail": f"Ping is high ({int(metrics['wlan_ping'])}ms) despite healthy bandwidth. Possible interference or signal noise."
                })

            # 3. Check for Speed Drops
            if norm_lan_down < 0.05: 
                causes.append({
                    "title": "Critical LAN Speed Drop",
                    "detail": "Download speed near zero (< 5% capacity). Potential interface negotiation error or cable fault."
                })
            elif norm_lan_down < 0.3 and not any("LAN Congestion" in c['title'] for c in causes): 
                causes.append({
                    "title": "Low LAN Throughput",
                    "detail": f"Download speed ({int(norm_lan_down*100)}%) is significantly below baseline average."
                })
                
            if norm_wlan_down < 0.05: 
                causes.append({
                    "title": "Critical WLAN Speed Drop",
                    "detail": "Wi-Fi download speed near zero (< 5% capacity). Signal loss or AP failure."
                })
            elif norm_wlan_down < 0.3 and not any("WLAN Congestion" in c['title'] for c in causes): 
                causes.append({
                    "title": "Low WLAN Throughput",
                    "detail": f"Wi-Fi speed ({int(norm_wlan_down*100)}%) is abnormally low compared to baseline."
                })

            # 4. Check Upload Failures (Detailed Criteria)
            if norm_lan_up < 0.05:
                val = int(metrics.get('lan_up', 0))
                if norm_lan_down > 0.3:
                    causes.append({
                        "title": "LAN Upload Failure (Asymmetric)",
                        "detail": f"Upload is critical ({val} Mbps < 5%) while Download is healthy. Potential Duplex mismatch, uplink saturation, or firewall blocking."
                    })
                else:
                    causes.append({
                        "title": "Critical LAN Upload Drop",
                        "detail": f"Upload speed is near zero ({val} Mbps). Check physical cabling (pairs 3 & 6)."
                    })
            elif norm_lan_up < 0.3:
                 causes.append({
                    "title": "Low LAN Upload Speed",
                    "detail": f"Upload speed ({int(norm_lan_up*100)}%) is significantly below baseline. Possible uplink congestion or shaping."
                })
            
            if norm_wlan_up < 0.05:
                val = int(metrics.get('wlan_up', 0))
                if norm_wlan_down > 0.3:
                    causes.append({
                        "title": "WLAN Upload Failure (Asymmetric)",
                        "detail": f"Upload is critical ({val} Mbps < 5%) while Download is healthy. Likely 'Near-Far' problem: AP can reach client, but client signal too weak to reach AP."
                    })
                else:
                     causes.append({
                        "title": "Critical WLAN Upload Drop",
                        "detail": f"Upload speed is near zero ({val} Mbps). Severe interference on uplink channel or client driver issue."
                    })
            elif norm_wlan_up < 0.3:
                 causes.append({
                    "title": "Low WLAN Upload Speed",
                    "detail": f"Upload speed ({int(norm_wlan_up*100)}%) is lower than expected. Common with high client density or distant clients."
                })

            # 5. Fallback Latency Check
            is_lan_flagged = any("LAN" in c['title'] for c in causes)
            if not is_lan_flagged and metrics.get('lan_ping', 0) > 100:
                 causes.append({
                     "title": "Abnormal LAN Ping",
                     "detail": f"Ping ({int(metrics['lan_ping'])}ms) is significantly higher than historical average."
                 })
            
            is_wlan_flagged = any("WLAN" in c['title'] for c in causes)
            if not is_wlan_flagged and metrics.get('wlan_ping', 0) > 200:
                 causes.append({
                     "title": "Abnormal WLAN Ping",
                     "detail": f"Ping ({int(metrics['wlan_ping'])}ms) is significantly higher than historical average."
                 })
            
            # 6. Check DNS
            if metrics.get('lan_dns', 0) > 100: 
                causes.append({
                    "title": "Slow LAN DNS",
                    "detail": f"DNS Resolution took {int(metrics['lan_dns'])}ms (High)."
                })
            if metrics.get('wlan_dns', 0) > 150:
                causes.append({
                    "title": "Slow WLAN DNS",
                    "detail": f"DNS Resolution took {int(metrics['wlan_dns'])}ms (High)."
                })

            # Fallback
            if not causes:
                causes.append({
                    "title": "Pattern Deviation",
                    "detail": "Complex multi-variable anomaly detected by Isolation Forest. No single metric crossed critical thresholds."
                })

            # Create summary string for legacy consumers
            desc = ", ".join([c['title'] for c in causes])

            return {
                "is_anomaly": True, 
                "score": round(float(score), 4), 
                "desc": desc, 
                "causes": causes,
                "department": department
            }
            
        return {"is_anomaly": False, "score": round(float(score), 4), "desc": "Normal", "department": department, "causes": []}
    except Exception as e: return {"is_anomaly": False, "score": 0, "desc": f"Error: {str(e)}", "department": "Unknown", "causes": []}

# EXPERIMENTAL FORECAST
'''
def predict_metric_trend(model_rf, model_lstm, history, max_val):
    """
    Predicts global trend using Univariate LSTM.
    history: List of raw values (float)
    """
    forecasts = []
    now = datetime.datetime.now()
    start_time = (now + datetime.timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    
    # 1. RF Prediction
    rf_preds = []
    if model_rf:
        future_data = []
        for i in range(24):
            t = start_time + datetime.timedelta(hours=i)
            future_data.append({'hour': t.hour, 'day_of_week': t.weekday(), 'is_weekend': 1 if t.weekday()>=5 else 0})
        try:
            rf_preds = model_rf.predict(pd.DataFrame(future_data)).tolist()
        except: pass

    # 2. LSTM Prediction
    lstm_preds = []
    seq_len = 24
    if model_lstm and len(history) >= seq_len:
        try:
            # Normalize sequence using GLOBAL MAX (saved in model file)
            norm_hist = [h / max_val for h in history[-seq_len:]]
            curr_seq = torch.tensor(norm_hist, dtype=torch.float32).view(1, seq_len, 1)
            
            with torch.no_grad():
                for _ in range(24):
                    pred = model_lstm(curr_seq)
                    val = max(0.0, pred.item())
                    lstm_preds.append(val * max_val) # Denormalize
                    
                    new_pt = torch.tensor([[[val]]], dtype=torch.float32)
                    curr_seq = torch.cat((curr_seq[:, 1:, :], new_pt), dim=1)
        except Exception as e: print(e)

    # 3. Combine
    result = []
    for i in range(24):
        rf_val = rf_preds[i] if i < len(rf_preds) else 0
        lstm_val = lstm_preds[i] if i < len(lstm_preds) else rf_val
        result.append({
            "time": (start_time + datetime.timedelta(hours=i)).strftime("%H:00"),
            "rf": round(rf_val, 1),
            "lstm": round(lstm_val, 1)
        })
    return result

def predict_global_trends(lan_history, wlan_history):
    lan_forecast = predict_metric_trend(RF_LAN, LSTM_LAN, lan_history, LAN_MAX)
    wlan_forecast = predict_metric_trend(RF_WLAN, LSTM_WLAN, wlan_history, WLAN_MAX)
    return { "lan": lan_forecast, "wlan": wlan_forecast }
'''