import joblib
import pandas as pd
import os
import numpy as np

# Load Model logic
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'anomaly_model.pkl')
MODEL = None

def load_model():
    global MODEL
    try:
        if os.path.exists(MODEL_PATH):
            MODEL = joblib.load(MODEL_PATH)
            print("✅ AI Model loaded successfully.")
        else:
            print("⚠️ Model not found. Run 'python ai_engine/trainer.py' first.")
    except Exception as e:
        print(f"❌ Error loading model: {e}")

# Load immediately on import
load_model()

def detect_anomaly(metrics):
    """
    Input: Dictionary with keys ['lan_ping', 'wlan_ping', 'lan_dns', 'wlan_dns']
    Output: { is_anomaly: bool, score: float, description: str }
    """
    if not MODEL:
        return {"is_anomaly": False, "score": 0, "desc": "AI Not Ready"}

    # 1. Format input as DataFrame (must match training features)
    features = pd.DataFrame([{
        'lan_ping': metrics.get('lan_ping', 0),
        'wlan_ping': metrics.get('wlan_ping', 0),
        'lan_dns': metrics.get('lan_dns', 0),
        'wlan_dns': metrics.get('wlan_dns', 0),
    }])

    try:
        # 2. Predict
        # Returns -1 for Anomaly, 1 for Normal
        prediction = MODEL.predict(features)[0]
        
        # decision_function returns specific anomaly score (lower = more anomalous)
        score = MODEL.decision_function(features)[0] 

        if prediction == -1:
            return {
                "is_anomaly": True,
                "score": round(score, 4),
                "desc": "Unusual Latency Pattern Detected"
            }
        
        return {"is_anomaly": False, "score": round(score, 4), "desc": "Normal Behavior"}

    except Exception as e:
        print(f"Prediction Error: {e}")
        return {"is_anomaly": False, "score": 0, "desc": "Error"}