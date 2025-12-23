import joblib
import pandas as pd
import os
import datetime
import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'anomaly_model.pkl')

MODEL = None
BASELINES = {}
FEATURES = []

def load_model():
    global MODEL, BASELINES, FEATURES
    try:
        if os.path.exists(MODEL_PATH):
            data = joblib.load(MODEL_PATH)
            if isinstance(data, dict) and 'model' in data:
                MODEL = data['model']
                BASELINES = data['baselines'] # Load the max speeds map
                FEATURES = data['features']
                print(f"✅ AI Model & Baselines loaded from {MODEL_PATH}")
            else:
                print("⚠️ Legacy model format. Please retrain.")
        else:
            print("⚠️ Model not found.")
    except Exception as e:
        print(f"❌ Error loading model: {e}")

load_model()

def detect_anomaly(metrics, probe_id="default"):
    """
    Analyzes metrics using Normalized Scaling.
    metrics: dict of raw values (Mbps, ms)
    probe_id: hostname (needed to lookup specific baseline)
    """
    if not MODEL:
        return {"is_anomaly": False, "score": 0, "desc": "AI Not Initialized"}

    # 1. Normalize Speed Data (Convert Mbps -> % Utilization)
    # If probe unknown, use default or raw (fallback)
    baseline = BASELINES.get(probe_id, BASELINES.get("default", {
        'lan_down_max': 1000, 'lan_up_max': 1000, 
        'wlan_down_max': 500, 'wlan_up_max': 500
    }))
    
    # Normalize (Value / Max = % Capacity)
    # This solves the 100Mbps vs 500Mbps issue. Both 90Mbps/100 and 450Mbps/500 become 0.9 (90% capacity).
    norm_lan_down = metrics.get('lan_down', 0) / baseline.get('lan_down_max', 1)
    norm_lan_up = metrics.get('lan_up', 0) / baseline.get('lan_up_max', 1)
    norm_wlan_down = metrics.get('wlan_down', 0) / baseline.get('wlan_down_max', 1)
    norm_wlan_up = metrics.get('wlan_up', 0) / baseline.get('wlan_up_max', 1)

    # 2. Prepare Features
    now = datetime.datetime.now()
    input_data = {
        'hour': now.hour,
        'is_weekend': 1 if now.weekday() >= 5 else 0,
        'lan_down': norm_lan_down,
        'lan_up': norm_lan_up,
        'wlan_down': norm_wlan_down,
        'wlan_up': norm_wlan_up,
        # Latency isn't normalized usually, as 200ms is bad regardless of network size
        'lan_ping': metrics.get('lan_ping', 0),
        'wlan_ping': metrics.get('wlan_ping', 0),
        'lan_dns': metrics.get('lan_dns', 0),
        'wlan_dns': metrics.get('wlan_dns', 0),
    }

    # Order columns
    features = pd.DataFrame([input_data])
    if FEATURES:
        # Fill missing features with 0
        for f in FEATURES:
            if f not in features.columns: features[f] = 0
        features = features[FEATURES]

    try:
        prediction = MODEL.predict(features)[0] 
        score = MODEL.decision_function(features)[0] 

        if prediction == -1:
            desc = "Traffic deviation"
            # Smart Diagnostics on Normalized Data
            # If normalized speed is < 5% of capacity, it's a drop
            if norm_wlan_down < 0.05: 
                desc = "Abnormal speed drop (<< 5% capacity)"
            elif input_data['lan_dns'] > 100:
                desc = "Latency anomaly (DNS)"
            
            return {
                "is_anomaly": True,
                "score": round(float(score), 4),
                "desc": desc
            }
        
        return {"is_anomaly": False, "score": round(float(score), 4), "desc": "Normal"}

    except Exception as e:
        print(f"Prediction Error: {e}")
        return {"is_anomaly": False, "score": 0, "desc": "Error"}