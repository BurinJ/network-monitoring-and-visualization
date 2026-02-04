import joblib
import pandas as pd
import os
import datetime
import numpy as np

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
ANOMALY_PATH = os.path.join(MODEL_DIR, 'anomaly_model.pkl')

ANOMALY_MODEL = None
BASELINES = {}
FEATURES = []
DEPARTMENTS = {}
# Global max values for forecast denormalization
LAN_MAX = 1.0
WLAN_MAX = 1.0

def load_models():
    global ANOMALY_MODEL, BASELINES, FEATURES, LAN_MAX, WLAN_MAX
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
            
        print("✅ Models loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading models: {e}")

load_models()

def detect_anomaly(metrics, probe_id="default", timestamp=None):
    """
    Detects anomalies for a single probe based on its specific metrics.
    
    Args:
        metrics (dict): The raw metric values (ping, speed, etc.)
        probe_id (str): The ID to look up baselines.
        timestamp (datetime): Optional. The specific time the data was recorded. 
                              Used to correctly set 'hour' and 'is_weekend' features for historical analysis.
    """
    # FIX: Ensure 'causes' is returned even if AI is not ready
    if not ANOMALY_MODEL: 
        return {
            "is_anomaly": False, 
            "score": 0, 
            "desc": "AI Not Initialized", 
            "department": "Unknown", 
            "causes": [] 
        }
    
    baseline = BASELINES.get(probe_id, BASELINES.get("default", {'lan_down_max': 1000, 'wlan_down_max': 500}))
    department = DEPARTMENTS.get(probe_id, "General")
    
    def safe_div(n, d): return n / d if d else 0

    # 1. Normalize Inputs
    norm_lan_down = safe_div(metrics.get('lan_down', 0), baseline.get('lan_down_max', 1))
    norm_lan_up = safe_div(metrics.get('lan_up', 0), baseline.get('lan_up_max', 1))
    norm_wlan_down = safe_div(metrics.get('wlan_down', 0), baseline.get('wlan_down_max', 1))
    norm_wlan_up = safe_div(metrics.get('wlan_up', 0), baseline.get('wlan_up_max', 1))

    # 2. Determine Time Context
    # Use the provided timestamp (for history graphs) or current time (for live status)
    target_time = timestamp if timestamp else datetime.datetime.now()

    input_data = {
        'hour': target_time.hour, 
        'is_weekend': 1 if target_time.weekday() >= 5 else 0,
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
    
    except Exception as e: 
        # FIX: Ensure causes is returned on error too
        return {"is_anomaly": False, "score": 0, "desc": f"Error: {str(e)}", "department": "Unknown", "causes": []}