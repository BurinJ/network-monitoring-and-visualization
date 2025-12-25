import joblib
import pandas as pd
import os
import datetime
import numpy as np

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
ANOMALY_PATH = os.path.join(MODEL_DIR, 'anomaly_model.pkl')
FORECAST_PATH = os.path.join(MODEL_DIR, 'forecast_model.pkl')

ANOMALY_MODEL = None
FORECAST_MODEL = None
BASELINES = {}
ANOMALY_FEATURES = []

def load_models():
    global ANOMALY_MODEL, FORECAST_MODEL, BASELINES, ANOMALY_FEATURES
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
        
        # Load Forecast Model
        if os.path.exists(FORECAST_PATH):
            FORECAST_MODEL = joblib.load(FORECAST_PATH)
            print("✅ All AI Models loaded.")
        else:
            print("⚠️ Models missing. Run 'python ai_engine/trainer.py'")

    except Exception as e:
        print(f"❌ Error loading models: {e}")

load_models()

# ... (detect_anomaly function remains the same as previous step) ...
def detect_anomaly(metrics, probe_id="default"):
    if not ANOMALY_MODEL: return {"is_anomaly": False, "score": 0, "desc": "AI Not Ready"}
    
    # 1. Normalize
    baseline = BASELINES.get(probe_id, BASELINES.get("default", {'lan_down_max': 1000, 'wlan_down_max': 500}))
    
    # Safe division
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
    # Filter only trained features
    if ANOMALY_FEATURES:
        valid_feats = {k: v for k, v in input_data.items() if k in ANOMALY_FEATURES}
        features = pd.DataFrame([valid_feats])
        # Ensure column order matches
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

def predict_future_traffic():
    """
    Generates a 24-hour load forecast.
    Returns: List of { time: 'HH:00', load: int (0-100) }
    """
    if not FORECAST_MODEL:
        return []

    future_data = []
    now = datetime.datetime.now()
    
    # Generate next 24 hours features
    for i in range(24):
        future_time = now + datetime.timedelta(hours=i)
        future_data.append({
            'hour': future_time.hour,
            'day_of_week': future_time.weekday(),
            'is_weekend': 1 if future_time.weekday() >= 5 else 0
        })
    
    df_future = pd.DataFrame(future_data)
    
    try:
        # Predict normalized load (0.0 - 1.0)
        predictions = FORECAST_MODEL.predict(df_future)
        
        result = []
        for i, pred in enumerate(predictions):
            time_label = (now + datetime.timedelta(hours=i)).strftime("%H:00")
            # Convert 0-1 to 0-100%
            load_pct = max(0, min(100, int(pred * 100)))
            result.append({"time": time_label, "predicted_load": load_pct})
            
        return result
    except Exception as e:
        print(f"Forecast Error: {e}")
        return []