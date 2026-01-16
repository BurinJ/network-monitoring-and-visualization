# ... imports ...
import time
import math
import sys
import os
import base64
import datetime 
import random
from prometheus_api_client import PrometheusConnect

# --- IMPORT HISTORY MANAGER ---
try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    import history
except ImportError:
    # Fallback if history.py not found during dev
    class MockHistory:
        def log_alert(self, *args): pass
    history = MockHistory()

# --- IMPORT AI ANALYZER ---
try:
    from .analyzer import detect_anomaly, predict_future_traffic
except ImportError:
    def detect_anomaly(m): return {"is_anomaly": False, "score": 0, "desc": "AI Module Missing"}

# --- IMPORT CONFIGURATION ---
try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from config import PROMETHEUS_URL, HOSTNAME_MAPPING, PROM_PASSWORD, PROM_USER
except ImportError:
    print("⚠️ Config file not found. Using defaults.")
    PROMETHEUS_URL = "http://localhost:9090"
    HOSTNAME_MAPPING = {}

# --- CONNECTION SETUP ---
try:
    auth_str = f"{PROM_USER}:{PROM_PASSWORD}"
    b64_auth = base64.b64encode(auth_str.encode()).decode()
    headers = {"Authorization": f"Basic {b64_auth}"}
    prom = PrometheusConnect(url=PROMETHEUS_URL, headers=headers, disable_ssl=True)
    prom.custom_query(query="up")
    CONNECTED = True
    print(f"✅ Connected to Prometheus at {PROMETHEUS_URL}")
except Exception:
    CONNECTED = False
    print(f"❌ Prometheus not found at {PROMETHEUS_URL}. Switching to Mock Data Mode.")

def safe_get_value(query, default=0):
    if not CONNECTED: return default
    try:
        result = prom.custom_query(query=query)
        if result and len(result) > 0:
            return float(result[0]['value'][1])
    except Exception as e:
        print(f"Query Error ({query}): {e}")
    return default

def get_label_value(metric_name, label_name, hostname):
    if not CONNECTED: return 0
    query = f'{metric_name}{{hostname="{hostname}"}}'
    try:
        result = prom.custom_query(query=query)
        if result and len(result) > 0:
            val_str = result[0]['metric'].get(label_name, "0")
            return float(val_str)
    except Exception:
        return 0
    return 0

def get_all_labels(metric_name, hostname):
    if not CONNECTED: return {}
    query = f'{metric_name}{{hostname="{hostname}"}}'
    try:
        result = prom.custom_query(query=query)
        if result and len(result) > 0:
            return result[0]['metric']
    except Exception:
        return {}
    return {}

# --- HELPER: Fetch Range Data (History) ---
def get_metric_history(query, hours=24, step='1h'):
    if not CONNECTED: 
        # Generate Mock History
        now = datetime.datetime.now()
        data = []
        for i in range(hours):
            t = now - datetime.timedelta(hours=hours-i)
            # Create a fake sine wave + noise pattern
            val = 25 + (math.sin(i) * 10) + random.randint(-5, 5) # Default ping-like values
            if "SPEEDTEST" in query:
                 val = 400 + (math.sin(i) * 100) + random.randint(-50, 50)
            data.append([t.timestamp(), max(0, val)])
        return data

    try:
        start_time = datetime.datetime.now() - datetime.timedelta(hours=hours)
        end_time = datetime.datetime.now()
        
        result = prom.custom_query_range(
            query=query,
            start_time=start_time,
            end_time=end_time,
            step=step
        )
        if result and len(result) > 0:
            return [[float(x[0]), float(x[1])] for x in result[0]['values']]
    except Exception as e:
        print(f"History Query Error: {e}")
    return []

# --- ALERT CACHE (Prevent Spamming DB) ---
# Format: { "probe_id_error_type": timestamp_last_logged }
ALERT_COOLDOWN = {} 

def check_and_log(probe_id, level, category, message, details):
    """
    Logs an alert only if it hasn't been logged in the last 15 minutes.
    """
    key = f"{probe_id}_{category}"
    now = time.time()
    last_time = ALERT_COOLDOWN.get(key, 0)
    
    # 15 minutes cooldown = 900 seconds
    if (now - last_time) > 900:
        history.log_alert(probe_id, level, category, message, details)
        ALERT_COOLDOWN[key] = now
        
# --- PAGE 1: NETWORK STATUS LOGIC ---
def fetch_network_status():
    if not CONNECTED: return _mock_pulse_data()

    timestamp_map = {}
    try:
        ts_results = prom.custom_query(query='timestamp')
        if ts_results:
            for item in ts_results:
                host = item['metric'].get('hostname')
                if host:
                    try: timestamp_map[host] = float(item['value'][1])
                    except: pass
    except Exception: pass

    query = 'GENERAL_info'
    try:
        results = prom.custom_query(query=query)
        if not results: return _mock_pulse_data()

        lan_probes = []
        wlan_probes = []
        current_time = time.time()

        for item in results:
            raw_hostname = item['metric'].get('hostname', 'Unknown Device')
            display_name = HOSTNAME_MAPPING.get(raw_hostname, raw_hostname)
            metric_labels = item['metric']
            
            probe_ts = timestamp_map.get(raw_hostname, 0)
            is_stale = False
            stale_label = "Offline"
            if probe_ts > 0 and (current_time - probe_ts) > 3600:
                is_stale = True
                check_and_log(display_name, "Critical", "Stale", "Probe Offline", f"Last seen > 1h ago")

            error_label = metric_labels.get('error', 'None')
            if error_label != "None":
                 check_and_log(display_name, "Critical", "Self-Report", "Hardware/Software Error", error_label)

            lan_curl_err = metric_labels.get('lan_google_curl', 'True') == 'False'
            wlan_curl_err = metric_labels.get('wlan_google_curl', 'True') == 'False'

            # LAN
            try:
                lan_latency = float(metric_labels.get('lan_google_response_time', '0'))
                lan_dns = float(metric_labels.get('lan_dns_response_time', '0'))
            except ValueError: lan_latency, lan_dns = 0.0, 0.0

            lan_status = "Active"
            lan_color = "green"
            if is_stale:
                lan_status = stale_label
                lan_color = "red"
            elif error_label == "CURL-ERR" and lan_curl_err:
                lan_status = "Curl Error"
                lan_color = "red"
            elif lan_latency == 0 or lan_latency > 2000:
                lan_status = "Down"
                lan_color = "red"
                check_and_log(display_name, "Critical", "LAN Connectivity", "LAN Unreachable", "Ping Timeout")
            elif lan_dns > 500:
                lan_status = "DNS Failure"
                lan_color = "red"
                check_and_log(display_name, "Critical", "LAN DNS", "DNS Resolution Failed", f"Time: {lan_dns}ms")
            elif lan_dns > 100:
                lan_status = "Slow DNS"
                lan_color = "orange"
                check_and_log(display_name, "Warning", "LAN DNS", "Slow DNS Response", f"Time: {lan_dns}ms")
            elif lan_latency > 100:
                lan_status = "Laggy"
                lan_color = "orange"
                check_and_log(display_name, "Warning", "LAN Latency", "High Latency", f"Ping: {lan_latency}ms")

            lan_probes.append({
                "name": display_name,
                "id": raw_hostname,
                "type": "LAN",
                "status": lan_status,
                "color": lan_color,
                "latency": round(lan_latency, 2), 
                "dns": round(lan_dns, 2),
                "lat": float(metric_labels.get('latitude', '0')),
                "lng": float(metric_labels.get('longitude', '0'))
            })

            # WLAN
            try:
                wlan_latency = float(metric_labels.get('wlan_google_response_time', '0'))
                wlan_dns = float(metric_labels.get('wlan_dns_response_time', '0'))
            except ValueError: wlan_latency, wlan_dns = 0.0, 0.0

            wlan_status = "Active"
            wlan_color = "green"
            if is_stale:
                wlan_status = stale_label
                wlan_color = "red"
            elif error_label == "CURL&DNS&WLAN-ERR":
                wlan_status = "WLAN Error"
                wlan_color = "red"
            elif error_label == "CURL-ERR" and wlan_curl_err:
                wlan_status = "Curl Error"
                wlan_color = "red"
            elif wlan_latency == 0 or wlan_latency > 2000:
                wlan_status = "Down"
                wlan_color = "red"
                check_and_log(display_name, "Critical", "WLAN Connectivity", "Wi-Fi Unreachable", "Ping Timeout")
            elif wlan_dns > 500:
                wlan_status = "DNS Failure"
                wlan_color = "red"
                check_and_log(display_name, "Critical", "WLAN DNS", "DNS Resolution Failed", f"Time: {wlan_dns}ms")
            elif wlan_dns > 100:
                wlan_status = "Slow DNS"
                wlan_color = "orange"
                check_and_log(display_name, "Warning", "WLAN DNS", "Slow DNS Response", f"Time: {wlan_dns}ms")
            elif wlan_latency > 200:
                wlan_status = "Laggy"
                wlan_color = "orange"
                check_and_log(display_name, "Warning", "WLAN Latency", "High Latency", f"Ping: {wlan_latency}ms")

            wlan_probes.append({
                "name": display_name,
                "id": raw_hostname,
                "type": "WLAN",
                "status": wlan_status,
                "color": wlan_color,
                "latency": round(wlan_latency, 2), 
                "dns": round(wlan_dns, 2),
                "lat": float(metric_labels.get('latitude', '0')),
                "lng": float(metric_labels.get('longitude', '0'))
            })

            # --- AI CHECK (Run periodically here too?) ---
            # To avoid performance hit, maybe only run simple checks here.
            # Or run AI detection:
            try:
                # We need speed for AI, but this loop only has labels.
                # Just skipping heavy AI here for loop performance. 
                # Ideally, a separate thread handles detailed analysis.
                pass 
            except: pass

        return {"lan": lan_probes, "wlan": wlan_probes}
    except Exception as e:
        print(f"❌ DEBUG: Exception: {e}")
        return _mock_pulse_data()

# --- PAGE 2: COMMAND CENTER LOGIC ---
def fetch_command_center():
    data = fetch_network_status()
    all_interfaces = data['lan'] + data['wlan']
    
    down_count = sum(1 for p in all_interfaces if p['color'] == 'red')
    avg_bw = safe_get_value('avg(LAN_EXTERNAL_SPEEDTEST{type="Download"})', default=850.5)
    
    total_dns = sum(p.get('dns', 0) for p in all_interfaces)
    count = len(all_interfaces)
    avg_dns = total_dns / count if count > 0 else 24
    satisfaction = max(0, min(100, 100 - (avg_dns / 5)))

    issues = []
    for p in all_interfaces:
        if p['color'] != 'green':
            issues.append({
                "location": f"{p['name']} ({p['type']})",
                "issue": p['status'],
                "severity": "High" if p['color'] == 'red' else "Medium"
            })

    # Map logic
    map_markers = {}
    for p in all_interfaces:
        if p.get('lat') == 0 or p.get('lng') == 0: continue
        key = p['name']
        if key not in map_markers:
            map_markers[key] = {
                "name": p['name'],
                "lat": p['lat'],
                "lng": p['lng'],
                "status": "Healthy", 
                "color": "green"
            }
        
        if p['color'] == 'red':
            map_markers[key]['status'] = "Critical"
            map_markers[key]['color'] = "red"
        elif p['color'] == 'orange' and map_markers[key]['color'] != 'red':
            map_markers[key]['status'] = "Warning"
            map_markers[key]['color'] = "orange"

    return {
        "alerts": down_count,
        "bandwidth": round(avg_bw, 1),
        "satisfaction": int(satisfaction),
        "total_probes": len(all_interfaces),
        "active_probes": len(all_interfaces) - down_count,
        "priority_issues": issues,
        "map_markers": list(map_markers.values())
    }

# --- PAGE 3: INSPECTOR LOGIC (UPDATED WITH AI) ---
def fetch_inspector_data(probe_id, duration='24h'):
    clean_id = probe_id
    if clean_id.endswith(" (LAN)"): clean_id = clean_id[:-6]
    elif clean_id.endswith(" (WLAN)"): clean_id = clean_id[:-7]

    raw_hostname = next((k for k, v in HOSTNAME_MAPPING.items() if v == clean_id), clean_id)

    # Determine History Config based on duration
    if duration == '1h':
        hist_hours = 1
        hist_step = '2m' 
    elif duration == '1w':
        hist_hours = 168 
        hist_step = '6h' 
    else: # 24h
        hist_hours = 24
        hist_step = '1h'

    all_labels = get_all_labels('GENERAL_info', raw_hostname)
    wlan_v4 = all_labels.get('wlan_ipv4', 'None')
    wlan_v6 = all_labels.get('wlan_ipv6', 'None')
    has_wlan = True 
    lan_v4 = all_labels.get('lan_ipv4', 'None')
    lan_v6 = all_labels.get('lan_ipv6', 'None')

    probe_ts = safe_get_value(f'timestamp{{hostname="{raw_hostname}"}}', default=0)
    current_time = time.time()
    is_stale = False
    stale_label = "Offline"
    if probe_ts > 0:
        diff = current_time - probe_ts
        if diff > 3600:
            is_stale = True
            days = int(diff // 86400)
            hours = int((diff % 86400) // 3600)
            minutes = int((diff % 3600) // 60)
            parts = []
            if days > 0: parts.append(f"{days}d")
            if hours > 0: parts.append(f"{hours}h")
            parts.append(f"{minutes}m")
            stale_label = f"Offline (~{' '.join(parts)})"

    error_label = all_labels.get('error', 'None')
    wlan_curl_err = all_labels.get('wlan_google_curl', 'True') == 'False'
    lan_curl_err = all_labels.get('lan_google_curl', 'True') == 'False'

    # WLAN Logic
    try:
        wlan_dns = float(all_labels.get('wlan_dns_response_time', '0'))
        wlan_ping = float(all_labels.get('wlan_google_response_time', '0'))
    except ValueError:
        wlan_dns, wlan_ping = 0.0, 0.0
    
    wlan_status = "Active"
    wlan_color = "green"
    if is_stale:
        wlan_status = stale_label
        wlan_color = "red"
    elif error_label == "CURL&DNS&WLAN-ERR":
        wlan_status = "WLAN Error"
        wlan_color = "red"
    elif error_label == "CURL-ERR" and wlan_curl_err:
        wlan_status = "Curl Error"
        wlan_color = "red"
    elif wlan_ping == 0 or wlan_ping > 2000:
        wlan_status = "Down"
        wlan_color = "red"
    elif wlan_dns > 500 or (wlan_dns == 0 and wlan_ping > 0):
        wlan_status = "DNS Failure"
        wlan_color = "red"
    elif wlan_dns > 100:
        wlan_status = "Slow DNS"
        wlan_color = "orange"
    elif wlan_ping > 200:
        wlan_status = "Laggy"
        wlan_color = "orange"

    # LAN Logic
    try:
        lan_dns = float(all_labels.get('lan_dns_response_time', '0'))
        lan_ping = float(all_labels.get('lan_google_response_time', '0'))
    except ValueError:
        lan_dns, lan_ping = 0.0, 0.0
    
    lan_status = "Active"
    lan_color = "green"
    if is_stale:
        lan_status = stale_label
        lan_color = "red"
    elif error_label == "CURL-ERR" and lan_curl_err:
        lan_status = "Curl Error"
        lan_color = "red"
    elif lan_ping == 0 or lan_ping > 2000:
        lan_status = "Down"
        lan_color = "red"
    elif lan_dns > 500 or (lan_dns == 0 and lan_ping > 0):
        lan_status = "DNS Failure"
        lan_color = "red"
    elif lan_dns > 100:
        lan_status = "Slow DNS"
        lan_color = "orange"
    elif lan_ping > 100:
        lan_status = "Laggy"
        lan_color = "orange"

    # --- FETCH SPEED HISTORY ---
    lan_hist_ext_down = get_metric_history(f'LAN_EXTERNAL_SPEEDTEST{{hostname="{raw_hostname}", type="Download"}}', hours=hist_hours, step=hist_step)
    lan_hist_ext_up   = get_metric_history(f'LAN_EXTERNAL_SPEEDTEST{{hostname="{raw_hostname}", type="Upload"}}', hours=hist_hours, step=hist_step)
    lan_hist_int_down = get_metric_history(f'LAN_INTERNAL_SPEEDTEST{{hostname="{raw_hostname}", type="Download"}}', hours=hist_hours, step=hist_step)
    lan_hist_int_up   = get_metric_history(f'LAN_INTERNAL_SPEEDTEST{{hostname="{raw_hostname}", type="Upload"}}', hours=hist_hours, step=hist_step)
    
    wlan_hist_ext_down = []
    wlan_hist_ext_up   = []
    wlan_hist_int_down = []
    wlan_hist_int_up   = []
    
    if has_wlan:
        wlan_hist_ext_down = get_metric_history(f'WLAN_EXTERNAL_SPEEDTEST{{hostname="{raw_hostname}", type="Download"}}', hours=hist_hours, step=hist_step)
        wlan_hist_ext_up   = get_metric_history(f'WLAN_EXTERNAL_SPEEDTEST{{hostname="{raw_hostname}", type="Upload"}}', hours=hist_hours, step=hist_step)
        wlan_hist_int_down = get_metric_history(f'WLAN_INTERNAL_SPEEDTEST{{hostname="{raw_hostname}", type="Download"}}', hours=hist_hours, step=hist_step)
        wlan_hist_int_up   = get_metric_history(f'WLAN_INTERNAL_SPEEDTEST{{hostname="{raw_hostname}", type="Upload"}}', hours=hist_hours, step=hist_step)

    # --- FETCH PING HISTORY ---
    lan_hist_ping = get_metric_history(f'LAN_PING{{hostname="{raw_hostname}"}}', hours=hist_hours, step=hist_step)
    wlan_hist_ping = []
    if has_wlan:
        wlan_hist_ping = get_metric_history(f'WLAN_PING{{hostname="{raw_hostname}"}}', hours=hist_hours, step=hist_step)

    def calc_avg(hist):
        if not hist: return 0
        vals = [h[1] for h in hist]
        return sum(vals) / len(vals)

    def get_smart_cap(*hists):
        max_avg = 0
        for h in hists:
            a = calc_avg(h)
            if a > max_avg: max_avg = a
        if max_avg == 0: return 1000
        # Round to nearest 100
        return max(100, int(round(max_avg / 100.0) * 100))

    lan_speed_cap = get_smart_cap(lan_hist_ext_down, lan_hist_ext_up, lan_hist_int_down, lan_hist_int_up)
    wlan_speed_cap = get_smart_cap(wlan_hist_ext_down, wlan_hist_ext_up, wlan_hist_int_down, wlan_hist_int_up)

    # Current detailed speeds
    lan_ext_down = safe_get_value(f'last_over_time(LAN_EXTERNAL_SPEEDTEST{{hostname="{raw_hostname}", type="Download"}}[1h])', default=0)
    lan_ext_up   = safe_get_value(f'last_over_time(LAN_EXTERNAL_SPEEDTEST{{hostname="{raw_hostname}", type="Upload"}}[1h])', default=0)
    lan_int_down = safe_get_value(f'last_over_time(LAN_INTERNAL_SPEEDTEST{{hostname="{raw_hostname}", type="Download"}}[1h])', default=0)
    lan_int_up   = safe_get_value(f'last_over_time(LAN_INTERNAL_SPEEDTEST{{hostname="{raw_hostname}", type="Upload"}}[1h])', default=0)

    wlan_ext_down = safe_get_value(f'last_over_time(WLAN_EXTERNAL_SPEEDTEST{{hostname="{raw_hostname}", type="Download"}}[1h])', default=0)
    wlan_ext_up   = safe_get_value(f'last_over_time(WLAN_EXTERNAL_SPEEDTEST{{hostname="{raw_hostname}", type="Upload"}}[1h])', default=0)
    wlan_int_down = safe_get_value(f'last_over_time(WLAN_INTERNAL_SPEEDTEST{{hostname="{raw_hostname}", type="Download"}}[1h])', default=0)
    wlan_int_up   = safe_get_value(f'last_over_time(WLAN_INTERNAL_SPEEDTEST{{hostname="{raw_hostname}", type="Upload"}}[1h])', default=0)

    # --- AI ANOMALY CHECK (INTEGRATED) ---
    ai_result = detect_anomaly({
        'lan_ping': lan_ping,
        'wlan_ping': wlan_ping,
        'lan_dns': lan_dns,
        'wlan_dns': wlan_dns,
        'lan_down': lan_ext_down,
        'lan_up': lan_ext_up,
        'wlan_down': wlan_ext_down,
        'wlan_up': wlan_ext_up
    }, probe_id=raw_hostname)

    diagnoses = []

    if ai_result['is_anomaly']:
        diagnoses.append({
            "status": "Warning",
            "title": "Anomaly Detected By AI",
            "desc": f"{ai_result['desc']} (Isolation Tree)"
        })

    if is_stale:
         diagnoses.append({"status": "Critical", "title": "Probe Offline", "desc": f"Probe unresponsive for > 1 hour."})
    
    if error_label != "None":
         diagnoses.append({"status": "Critical", "title": "Hardware/Software Error", "desc": f"Probe reporting error: {error_label}"})

    if lan_ping == 0 and (not has_wlan or wlan_ping == 0):
         diagnoses.append({"status": "Critical", "title": "Network Unreachable", "desc": "Interfaces unresponsive."})
    
    if has_wlan and wlan_dns > 500:
         diagnoses.append({"status": "Critical", "title": "WLAN DNS Failure", "desc": "Wi-Fi cannot resolve domain names."})
    
    if lan_dns > 500:
         diagnoses.append({"status": "Critical", "title": "LAN DNS Failure", "desc": "Ethernet DNS resolution failed."})
    
    if has_wlan and wlan_ping > 200:
         diagnoses.append({"status": "Warning", "title": "Wi-Fi Congestion", "desc": "High latency on Wi-Fi interface."})

    if not diagnoses:
        diagnoses.append({"status": "Healthy", "title": "Normal Operation", "desc": "No significant anomalies detected."})

    data = {
        "wlan": {
            "status": wlan_status,
            "color": wlan_color,
            "dns": round(wlan_dns, 2),
            "ping": round(wlan_ping, 2),
            "ipv4": wlan_v4 if wlan_v4 != "None" else None,
            "ipv6": wlan_v6 if wlan_v6 != "None" else None,
            "speed": {
                "external": {"down": round(wlan_ext_down, 2), "up": round(wlan_ext_up, 2)},
                "internal": {"down": round(wlan_int_down, 2), "up": round(wlan_int_up, 2)}
            },
            "history": {
                "external": {"down": wlan_hist_ext_down, "up": wlan_hist_ext_up},
                "internal": {"down": wlan_hist_int_down, "up": wlan_hist_int_up},
                "ping": wlan_hist_ping
            },
            "average": {
                "external": {"down": round(calc_avg(wlan_hist_ext_down), 2), "up": round(calc_avg(wlan_hist_ext_up), 2)},
                "internal": {"down": round(calc_avg(wlan_hist_int_down), 2), "up": round(calc_avg(wlan_hist_int_up), 2)},
                "ping": round(calc_avg(wlan_hist_ping), 2)
            },
            "speed_cap": wlan_speed_cap
        },
        "lan": {
            "status": lan_status,
            "color": lan_color,
            "dns": round(lan_dns, 2),
            "ping": round(lan_ping, 2),
            "ipv4": lan_v4 if lan_v4 != "None" else None,
            "ipv6": lan_v6 if lan_v6 != "None" else None,
            "speed": {
                "external": {"down": round(lan_ext_down, 2), "up": round(lan_ext_up, 2)},
                "internal": {"down": round(lan_int_down, 2), "up": round(lan_int_up, 2)}
            },
            "history": {
                "external": {"down": lan_hist_ext_down, "up": lan_hist_ext_up},
                "internal": {"down": lan_hist_int_down, "up": lan_hist_int_up},
                "ping": lan_hist_ping
            },
            "average": {
                "external": {"down": round(calc_avg(lan_hist_ext_down), 2), "up": round(calc_avg(lan_hist_ext_up), 2)},
                "internal": {"down": round(calc_avg(lan_hist_int_down), 2), "up": round(calc_avg(lan_hist_int_up), 2)},
                "ping": round(calc_avg(lan_hist_ping), 2)
            },
            "speed_cap": lan_speed_cap
        }
    }

    return {"metrics": data, "ai_diagnoses": diagnoses, "has_wlan": has_wlan} # Changed key to ai_diagnoses (plural)

# --- PAGE 4: TRENDS LOGIC ---
def fetch_trends_data():
    data = fetch_network_status()
    probes = data.get('wlan', [])
    
    # 1. Fetch Aggregate History for LSTM context (Last 24h)
    # We use LAN download as the main indicator for total network load
    raw_history = get_metric_history('avg(LAN_EXTERNAL_SPEEDTEST{type="Download"})', hours=24, step='1h')
    
    # 2. Normalize History (0-1) for LSTM
    # We need a global baseline. Let's assume 1000Mbps is the max backbone speed for aggregation.
    norm_history = []
    if raw_history:
        norm_history = [val[1] / 1000.0 for val in raw_history]
        
    # 3. Get Forecasts
    forecast = predict_future_traffic(norm_history)
    
    # 4. Fallback if empty
    if not forecast:
        now = datetime.datetime.now()
        for i in range(24):
            forecast.append({
                "time": (now + datetime.timedelta(hours=i)).strftime("%H:00"),
                "rf_load": random.randint(40, 90),
                "lstm_load": random.randint(30, 80)
            })
            
    return {"heatmap": probes, "forecast": forecast}

# --- MOCK DATA ---
def _mock_pulse_data():
    return {
        "lan": [
            {"name": "Library", "type": "LAN", "status": "Active", "color": "green", "latency": 5.42, "dns": 12.5},
            {"name": "Dorm A", "type": "LAN", "status": "Active", "color": "green", "latency": 8.1, "dns": 14.2},
        ],
        "wlan": [
            {"name": "Library", "type": "WLAN", "status": "Laggy", "color": "orange", "latency": 250.3, "dns": 45.1},
            {"name": "Science Hall", "type": "WLAN", "status": "Down", "color": "red", "latency": 0.0, "dns": 0.0},
        ]
    }