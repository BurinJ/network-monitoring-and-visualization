# ... imports ...
import time
import math
import sys
import os
import base64
import datetime 
import random
import json
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

# --- IMPORT SETTINGS MANAGER ---
try:
    import settings_manager
except ImportError:
    # Fallback class if module missing
    class MockSettings:
        def get_settings(self):
            return {
                "lan_ping_threshold": 100,
                "wlan_ping_threshold": 200,
                "dns_threshold": 100,
                "offline_timeout_mins": 60
            }
    settings_manager = MockSettings()

# --- IMPORT AI ANALYZER ---
try:
    from .analyzer import detect_anomaly #, predict_global_trends
except ImportError:
    def detect_anomaly(m, probe_id=None): return {"is_anomaly": False, "score": 0, "desc": "AI Module Missing"}
    # def predict_global_trends(l, w): return {"lan": [], "wlan": []}

# --- IMPORT CONFIGURATION & HISTORY ---
try:
    from config import PROMETHEUS_URL, HOSTNAME_MAPPING, PROM_PASSWORD, PROM_USER
except ImportError:
    print("⚠️ Config or History module not found. Using defaults.")
    PROMETHEUS_URL = "http://localhost:9090"
    HOSTNAME_MAPPING = {}

# --- MAPPING FILES ---
# Using absolute paths to ensure reliability
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MAPPING_FILE = os.path.join(BASE_DIR, '../probe_mappings.json')
DEPT_MAPPING_FILE = os.path.join(BASE_DIR, '../probe_departments.json')

DEPARTMENT_MAPPING = {}

def load_mappings():
    global HOSTNAME_MAPPING, DEPARTMENT_MAPPING
    
    # 1. Load Name Overrides
    if os.path.exists(MAPPING_FILE):
        try:
            with open(MAPPING_FILE, 'r') as f:
                saved_names = json.load(f)
                HOSTNAME_MAPPING.update(saved_names)
                print(f"✅ Loaded {len(saved_names)} probe names from {MAPPING_FILE}")
        except Exception as e:
            print(f"⚠️ Error loading names JSON: {e}")
    
    # 2. Load Department Mappings
    if os.path.exists(DEPT_MAPPING_FILE):
        try:
            with open(DEPT_MAPPING_FILE, 'r') as f:
                saved_depts = json.load(f)
                DEPARTMENT_MAPPING.update(saved_depts)
                print(f"✅ Loaded {len(saved_depts)} departments from {DEPT_MAPPING_FILE}")
        except Exception as e:
            print(f"⚠️ Error loading departments JSON: {e}")
    else:
        print(f"ℹ️ No department mapping file found at {DEPT_MAPPING_FILE}")

def save_mapping(raw_host, friendly_name):
    global HOSTNAME_MAPPING
    HOSTNAME_MAPPING[raw_host] = friendly_name
    try:
        with open(MAPPING_FILE, 'w') as f:
            json.dump(HOSTNAME_MAPPING, f, indent=2)
        print(f"💾 Saved Name: {raw_host} -> {friendly_name}")
        return True
    except Exception as e:
        print(f"❌ Error saving name mapping: {e}")
        return False

def save_department_mapping(raw_host, department):
    global DEPARTMENT_MAPPING
    DEPARTMENT_MAPPING[raw_host] = department
    try:
        with open(DEPT_MAPPING_FILE, 'w') as f:
            json.dump(DEPARTMENT_MAPPING, f, indent=2)
        print(f"💾 Saved Dept: {raw_host} -> {department}")
        return True
    except Exception as e:
        print(f"❌ Error saving department mapping: {e}")
        return False

# Initialize mappings immediately
load_mappings()


# --- IMPORT CONFIGURATION ---
# Load other config vars
try:
    from config import PROMETHEUS_URL, PROM_PASSWORD, PROM_USER
except ImportError:
    PROMETHEUS_URL = "http://localhost:9090"
    PROM_USER = None
    PROM_PASSWORD = None

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
        now = datetime.datetime.now()
        data = []
        for i in range(hours):
            t = now - datetime.timedelta(hours=hours-i)
            val = 25 + (math.sin(i) * 10) + random.randint(-5, 5) 
            if "SPEEDTEST" in query:
                 val = 400 + (math.sin(i) * 100) + random.randint(-50, 50)
            elif "INTERNAL" in query and "PING" in query:
                 val = 2 + random.random()
            data.append([t.timestamp(), max(0, val)])
        return data

    try:
        start_time = datetime.datetime.now() - datetime.timedelta(hours=hours)
        end_time = datetime.datetime.now()
        result = prom.custom_query_range(query=query, start_time=start_time, end_time=end_time, step=step)
        if result and len(result) > 0:
            return [[float(x[0]), float(x[1])] for x in result[0]['values']]
    except Exception as e:
        print(f"History Query Error: {e}")
    return []

# --- HELPER: Bulk Fetch Latest Values for Map ---
def get_metric_map(query):
    mapping = {}
    if not CONNECTED: return mapping
    try:
        results = prom.custom_query(query=query)
        for item in results:
            host = item['metric'].get('hostname')
            if host:
                try: mapping[host] = float(item['value'][1])
                except: pass
    except: pass
    return mapping

# --- ALERT CACHE ---
ALERT_COOLDOWN = {} 

def check_and_log(probe_id, level, category, message, details):
    key = f"{probe_id}_{category}"
    now = time.time()
    last_time = ALERT_COOLDOWN.get(key, 0)
    if (now - last_time) > 900:
        if history: history.log_alert(probe_id, level, category, message, details)
        ALERT_COOLDOWN[key] = now
        
# --- PAGE 1: NETWORK STATUS LOGIC ---
def fetch_network_status():
    if not CONNECTED: return _mock_pulse_data()
    
    config = settings_manager.get_settings()
    LAN_PING_THRESH = config.get('lan_ping_threshold', 100)
    WLAN_PING_THRESH = config.get('wlan_ping_threshold', 200)
    DNS_THRESH = config.get('dns_threshold', 100)
    TIMEOUT_SEC = config.get('offline_timeout_mins', 60) * 60

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

    lan_down_map = get_metric_map('last_over_time(LAN_EXTERNAL_SPEEDTEST{type="Download"}[1h])')
    lan_up_map = get_metric_map('last_over_time(LAN_EXTERNAL_SPEEDTEST{type="Upload"}[1h])')
    wlan_down_map = get_metric_map('last_over_time(WLAN_EXTERNAL_SPEEDTEST{type="Download"}[1h])')
    wlan_up_map = get_metric_map('last_over_time(WLAN_EXTERNAL_SPEEDTEST{type="Upload"}[1h])')
    lan_ping_map = get_metric_map('last_over_time(LAN_PING{metrics="avgRTT", type="EXTERNAL"}[1h])')
    wlan_ping_map = get_metric_map('last_over_time(WLAN_PING{metrics="avgRTT", type="EXTERNAL"}[1h])')

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
            department = DEPARTMENT_MAPPING.get(raw_hostname, "Undefined")
            metric_labels = item['metric']
            
            probe_ts = timestamp_map.get(raw_hostname, 0)
            is_stale = False
            stale_label = "Offline"
            if probe_ts > 0:
                diff = current_time - probe_ts
                if diff > TIMEOUT_SEC:
                    is_stale = True
                    days = int(diff // 86400)
                    hours = int((diff % 86400) // 3600)
                    minutes = int((diff % 3600) // 60)
                    stale_label = f"Offline (~{' '.join([f'{days}d' if days else '', f'{hours}h' if hours else '', f'{minutes}m']).strip()})"

            error_label = metric_labels.get('error', 'None')
            lan_curl_err = metric_labels.get('lan_google_curl', 'True') == 'False'
            wlan_curl_err = metric_labels.get('wlan_google_curl', 'True') == 'False'

            # 1. GATHER ALL METRICS FIRST
            try: lan_dns = float(metric_labels.get('lan_dns_response_time', '0'))
            except: lan_dns = 0.0
            lan_down = lan_down_map.get(raw_hostname, 0)
            lan_up = lan_up_map.get(raw_hostname, 0)
            lan_latency = lan_ping_map.get(raw_hostname, 0)

            try: wlan_dns = float(metric_labels.get('wlan_dns_response_time', '0'))
            except: wlan_dns = 0.0
            wlan_down = wlan_down_map.get(raw_hostname, 0)
            wlan_up = wlan_up_map.get(raw_hostname, 0)
            wlan_latency = wlan_ping_map.get(raw_hostname, 0)

            # 2. RUN AI CHECK ONCE WITH FULL CONTEXT
            ai_res = detect_anomaly({
                'lan_ping': lan_latency, 'lan_dns': lan_dns, 
                'lan_down': lan_down, 'lan_up': lan_up,
                'wlan_ping': wlan_latency, 'wlan_dns': wlan_dns,
                'wlan_down': wlan_down, 'wlan_up': wlan_up
            }, probe_id=raw_hostname)

            # 3. APPLY STATUS FOR LAN
            lan_status, lan_color = "Active", "green"
            if is_stale: lan_status, lan_color = stale_label, "red"
            elif error_label == "CURL-ERR" and lan_curl_err: lan_status, lan_color = "Curl Error", "red"
            elif lan_latency == 0 or lan_latency > 2000: lan_status, lan_color = "Down", "red"
            elif lan_dns > 500: lan_status, lan_color = "DNS Failure", "red"
            elif lan_dns > DNS_THRESH: lan_status, lan_color = "Slow DNS", "orange"
            elif lan_latency > LAN_PING_THRESH: lan_status, lan_color = "Laggy", "orange"
            elif ai_res['is_anomaly']: # Use shared AI result
                lan_status, lan_color = "AI Warning", "orange"
                check_and_log(display_name, "Warning", "AI Anomaly", ai_res['desc'], {"interface": "LAN"})

            if lan_color == 'red': check_and_log(display_name, "Critical", "Connection Failure", lan_status, {"interface": "LAN"})

            lan_probes.append({
                "name": display_name, "id": raw_hostname, "department": department, "type": "LAN", "status": lan_status, "color": lan_color,
                "latency": round(lan_latency, 2), "dns": round(lan_dns, 2), "down": round(lan_down, 1), "up": round(lan_up, 1),
                "lat": float(metric_labels.get('latitude', '0')), "lng": float(metric_labels.get('longitude', '0'))
            })

            # 4. APPLY STATUS FOR WLAN
            wlan_status, wlan_color = "Active", "green"
            if is_stale: wlan_status, wlan_color = stale_label, "red"
            elif error_label == "CURL&DNS&WLAN-ERR": wlan_status, wlan_color = "WLAN Error", "red"
            elif error_label == "CURL-ERR" and wlan_curl_err: wlan_status, wlan_color = "Curl Error", "red"
            elif wlan_latency == 0 or wlan_latency > 2000: wlan_status, wlan_color = "Down", "red"
            elif wlan_dns > 500: wlan_status, wlan_color = "DNS Failure", "red"
            elif wlan_dns > DNS_THRESH: wlan_status, wlan_color = "Slow DNS", "orange"
            elif wlan_latency > WLAN_PING_THRESH: wlan_status, wlan_color = "Laggy", "orange"
            elif ai_res['is_anomaly']: # Use shared AI result
                wlan_status, wlan_color = "AI Warning", "orange"
                check_and_log(display_name, "Warning", "AI Anomaly", ai_res['desc'], {"interface": "WLAN"})

            if wlan_color == 'red': check_and_log(display_name, "Critical", "Connection Failure", wlan_status, {"interface": "WLAN"})

            wlan_probes.append({
                "name": display_name, "id": raw_hostname, "department": department, "type": "WLAN", "status": wlan_status, "color": wlan_color,
                "latency": round(wlan_latency, 2), "dns": round(wlan_dns, 2), "down": round(wlan_down, 1), "up": round(wlan_up, 1),
                "lat": float(metric_labels.get('latitude', '0')), "lng": float(metric_labels.get('longitude', '0'))
            })

        return {"lan": lan_probes, "wlan": wlan_probes}
    except Exception as e:
        print(f"❌ DEBUG: Exception: {e}")
        return _mock_pulse_data()
    
# --- PAGE 2: COMMAND CENTER LOGIC ---
def fetch_command_center():
    data = fetch_network_status()
    all_interfaces = data.get('lan', []) + data.get('wlan', [])
    down_count = sum(1 for p in all_interfaces if p['color'] == 'red')
    total_dl = sum(p['down'] for p in all_interfaces)
    count = len(all_interfaces)
    avg_bw = total_dl / count if count > 0 else 0
    total_dns = sum(p.get('dns', 0) for p in all_interfaces)
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

        # Update severity if needed (Red > Orange > Green)
        existing = map_markers[key]
        if p['color'] == 'red':
            existing['status'] = "Critical"
            existing['color'] = "red"
        elif p['color'] == 'orange' and existing['color'] != 'red':
            existing['status'] = "Warning"
            existing['color'] = "orange"
    return {
        "alerts": down_count,
        "bandwidth": round(avg_bw, 1),
        "satisfaction": int(satisfaction),
        "total_probes": len(all_interfaces),
        "active_probes": len(all_interfaces) - down_count,
        "priority_issues": issues,
        "map_markers": list(map_markers.values())
    }

# --- PAGE 3: INSPECTOR LOGIC ---
def fetch_inspector_data(probe_id, duration='24h'):
    # LOAD DYNAMIC SETTINGS
    config = settings_manager.get_settings()
    LAN_PING_THRESH = config.get('lan_ping_threshold', 100)
    WLAN_PING_THRESH = config.get('wlan_ping_threshold', 200)
    DNS_THRESH = config.get('dns_threshold', 100)
    TIMEOUT_SEC = config.get('offline_timeout_mins', 60) * 60

    clean_id = probe_id
    if clean_id.endswith(" (LAN)"): clean_id = clean_id[:-6]
    elif clean_id.endswith(" (WLAN)"): clean_id = clean_id[:-7]
    raw_hostname = next((k for k, v in HOSTNAME_MAPPING.items() if v == clean_id), clean_id)

    if duration == '1h': hist_hours, hist_step = 1, '2m' 
    elif duration == '1w': hist_hours, hist_step = 168, '6h' 
    else: hist_hours, hist_step = 24, '1h'

    all_labels = get_all_labels('GENERAL_info', raw_hostname)
    wlan_v4, wlan_v6 = all_labels.get('wlan_ipv4', 'None'), all_labels.get('wlan_ipv6', 'None')
    has_wlan = True 
    lan_v4, lan_v6 = all_labels.get('lan_ipv4', 'None'), all_labels.get('lan_ipv6', 'None')

    probe_ts = safe_get_value(f'timestamp{{hostname="{raw_hostname}"}}', default=0)
    current_time = time.time()
    is_stale, stale_label = False, "Offline"
    if probe_ts > 0 and (current_time - probe_ts) > TIMEOUT_SEC:
        is_stale = True
        days = int((current_time - probe_ts) // 86400)
        hours = int(((current_time - probe_ts) % 86400) // 3600)
        minutes = int(((current_time - probe_ts) % 3600) // 60)
        stale_label = f"Offline (~{days}d {hours}h {minutes}m)"

    error_label = all_labels.get('error', 'None')
    wlan_curl_err, lan_curl_err = all_labels.get('wlan_google_curl', 'True') == 'False', all_labels.get('lan_google_curl', 'True') == 'False'

    try: wlan_dns = float(all_labels.get('wlan_dns_response_time', '0'))
    except: wlan_dns = 0.0
    wlan_ping = safe_get_value(f'last_over_time(WLAN_PING{{hostname="{raw_hostname}", metrics="avgRTT", type="EXTERNAL"}}[1h])', default=0) if has_wlan else 0.0
    wlan_status, wlan_color = "Active", "green"
    if is_stale: wlan_status, wlan_color = stale_label, "red"
    elif error_label == "CURL&DNS&WLAN-ERR": wlan_status, wlan_color = "WLAN Error", "red"
    elif error_label == "CURL-ERR" and wlan_curl_err: wlan_status, wlan_color = "Curl Error", "red"
    elif wlan_ping == 0 or wlan_ping > 2000: wlan_status, wlan_color = "Down", "red"
    elif wlan_dns > 500: wlan_status, wlan_color = "DNS Failure", "red"
    elif wlan_dns > DNS_THRESH: wlan_status, wlan_color = "Slow DNS", "orange"
    elif wlan_ping > WLAN_PING_THRESH: wlan_status, wlan_color = "Laggy", "orange"

    try: lan_dns = float(all_labels.get('lan_dns_response_time', '0'))
    except: lan_dns = 0.0
    lan_ping = safe_get_value(f'last_over_time(LAN_PING{{hostname="{raw_hostname}", metrics="avgRTT", type="EXTERNAL"}}[1h])', default=0)
    lan_status, lan_color = "Active", "green"
    if is_stale: lan_status, lan_color = stale_label, "red"
    elif error_label == "CURL-ERR" and lan_curl_err: lan_status, lan_color = "Curl Error", "red"
    elif lan_ping == 0 or lan_ping > 2000: lan_status, lan_color = "Down", "red"
    elif lan_dns > 500: lan_status, lan_color = "DNS Failure", "red"
    elif lan_dns > DNS_THRESH: lan_status, lan_color = "Slow DNS", "orange"
    elif lan_ping > LAN_PING_THRESH: lan_status, lan_color = "Laggy", "orange"

    lan_hist_ext_down = get_metric_history(f'LAN_EXTERNAL_SPEEDTEST{{hostname="{raw_hostname}", type="Download"}}', hours=hist_hours, step=hist_step)
    lan_hist_ext_up = get_metric_history(f'LAN_EXTERNAL_SPEEDTEST{{hostname="{raw_hostname}", type="Upload"}}', hours=hist_hours, step=hist_step)
    lan_hist_int_down = get_metric_history(f'LAN_INTERNAL_SPEEDTEST{{hostname="{raw_hostname}", type="Download"}}', hours=hist_hours, step=hist_step)
    lan_hist_int_up = get_metric_history(f'LAN_INTERNAL_SPEEDTEST{{hostname="{raw_hostname}", type="Upload"}}', hours=hist_hours, step=hist_step)
    wlan_hist_ext_down = get_metric_history(f'WLAN_EXTERNAL_SPEEDTEST{{hostname="{raw_hostname}", type="Download"}}', hours=hist_hours, step=hist_step) if has_wlan else []
    wlan_hist_ext_up = get_metric_history(f'WLAN_EXTERNAL_SPEEDTEST{{hostname="{raw_hostname}", type="Upload"}}', hours=hist_hours, step=hist_step) if has_wlan else []
    wlan_hist_int_down = get_metric_history(f'WLAN_INTERNAL_SPEEDTEST{{hostname="{raw_hostname}", type="Download"}}', hours=hist_hours, step=hist_step) if has_wlan else []
    wlan_hist_int_up = get_metric_history(f'WLAN_INTERNAL_SPEEDTEST{{hostname="{raw_hostname}", type="Upload"}}', hours=hist_hours, step=hist_step) if has_wlan else []
    lan_hist_ping_ext = get_metric_history(f'LAN_PING{{hostname="{raw_hostname}", type="EXTERNAL", metrics="avgRTT"}}', hours=hist_hours, step=hist_step)
    lan_hist_ping_int = get_metric_history(f'LAN_PING{{hostname="{raw_hostname}", type="INTERNAL", metrics="avgRTT"}}', hours=hist_hours, step=hist_step)
    wlan_hist_ping_ext = get_metric_history(f'WLAN_PING{{hostname="{raw_hostname}", type="EXTERNAL", metrics="avgRTT"}}', hours=hist_hours, step=hist_step) if has_wlan else []
    wlan_hist_ping_int = get_metric_history(f'WLAN_PING{{hostname="{raw_hostname}", type="INTERNAL", metrics="avgRTT"}}', hours=hist_hours, step=hist_step) if has_wlan else []

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
        return max(100, int(round(max_avg / 100.0) * 100))

    lan_speed_cap = get_smart_cap(lan_hist_ext_down, lan_hist_ext_up, lan_hist_int_down, lan_hist_int_up)
    wlan_speed_cap = get_smart_cap(wlan_hist_ext_down, wlan_hist_ext_up, wlan_hist_int_down, wlan_hist_int_up)

    lan_ext_down = safe_get_value(f'last_over_time(LAN_EXTERNAL_SPEEDTEST{{hostname="{raw_hostname}", type="Download"}}[1h])', default=0)
    lan_ext_up = safe_get_value(f'last_over_time(LAN_EXTERNAL_SPEEDTEST{{hostname="{raw_hostname}", type="Upload"}}[1h])', default=0)
    lan_int_down = safe_get_value(f'last_over_time(LAN_INTERNAL_SPEEDTEST{{hostname="{raw_hostname}", type="Download"}}[1h])', default=0)
    lan_int_up = safe_get_value(f'last_over_time(LAN_INTERNAL_SPEEDTEST{{hostname="{raw_hostname}", type="Upload"}}[1h])', default=0)
    wlan_ext_down = safe_get_value(f'last_over_time(WLAN_EXTERNAL_SPEEDTEST{{hostname="{raw_hostname}", type="Download"}}[1h])', default=0)
    wlan_ext_up = safe_get_value(f'last_over_time(WLAN_EXTERNAL_SPEEDTEST{{hostname="{raw_hostname}", type="Upload"}}[1h])', default=0)
    wlan_int_down = safe_get_value(f'last_over_time(WLAN_INTERNAL_SPEEDTEST{{hostname="{raw_hostname}", type="Download"}}[1h])', default=0)
    wlan_int_up = safe_get_value(f'last_over_time(WLAN_INTERNAL_SPEEDTEST{{hostname="{raw_hostname}", type="Upload"}}[1h])', default=0)

    ai_result = detect_anomaly({
        'lan_ping': lan_ping, 'wlan_ping': wlan_ping, 'lan_dns': lan_dns, 'wlan_dns': wlan_dns,
        'lan_down': lan_ext_down, 'lan_up': lan_ext_up, 'wlan_down': wlan_ext_down, 'wlan_up': wlan_ext_up
    }, probe_id=raw_hostname)

    diagnoses = []
    if ai_result['is_anomaly']: 
        diagnoses.append({
            "status": "Warning", 
            "title": "Anomaly Detected By AI", 
            "desc": f"{ai_result['desc']} (Isolation Tree)",
            "causes": ai_result.get('causes', []) # Pass the detailed causes list
        })
        
    if is_stale: diagnoses.append({"status": "Critical", "title": "Probe Offline", "desc": f"Probe unresponsive for > {TIMEOUT_SEC // 60} minutes."})
    if error_label != "None": diagnoses.append({"status": "Critical", "title": "Hardware/Software Error", "desc": f"Probe reporting error: {error_label}"})
    if lan_ping == 0 and (not has_wlan or wlan_ping == 0): diagnoses.append({"status": "Critical", "title": "Network Unreachable", "desc": "Interfaces unresponsive."})
    if has_wlan and wlan_dns > 500: diagnoses.append({"status": "Critical", "title": "WLAN DNS Failure", "desc": "Wi-Fi cannot resolve domain names."})
    if lan_dns > 500: diagnoses.append({"status": "Critical", "title": "LAN DNS Failure", "desc": "Ethernet DNS resolution failed."})
    if has_wlan and wlan_ping > WLAN_PING_THRESH: diagnoses.append({"status": "Warning", "title": "Wi-Fi Congestion", "desc": "High latency on Wi-Fi interface."})
    if lan_ping > LAN_PING_THRESH: diagnoses.append({"status": "Warning", "title": "LAN Latency", "desc": f"High latency on Ethernet interface. (LAN ping = {lan_ping}ms > {LAN_PING_THRESH}ms)"})
    
    if not diagnoses: diagnoses.append({"status": "Healthy", "title": "Normal Operation", "desc": "No significant anomalies detected."})

    data = {
        "wlan": {
            "status": wlan_status, "color": wlan_color, "dns": round(wlan_dns, 2), "ping": round(wlan_ping, 2),
            "ipv4": wlan_v4 if wlan_v4 != "None" else None, "ipv6": wlan_v6 if wlan_v6 != "None" else None,
            "speed": { "external": {"down": round(wlan_ext_down, 2), "up": round(wlan_ext_up, 2)}, "internal": {"down": round(wlan_int_down, 2), "up": round(wlan_int_up, 2)} },
            "history": { "external": {"down": wlan_hist_ext_down, "up": wlan_hist_ext_up}, "internal": {"down": wlan_hist_int_down, "up": wlan_hist_int_up}, "ping": {"external": wlan_hist_ping_ext, "internal": wlan_hist_ping_int} },
            "average": { "external": {"down": round(calc_avg(wlan_hist_ext_down), 2), "up": round(calc_avg(wlan_hist_ext_up), 2)}, "internal": {"down": round(calc_avg(wlan_hist_int_down), 2), "up": round(calc_avg(wlan_hist_int_up), 2)}, "ping": {"external": round(calc_avg(wlan_hist_ping_ext), 2), "internal": round(calc_avg(wlan_hist_ping_int), 2)} },
            "speed_cap": wlan_speed_cap
        },
        "lan": {
            "status": lan_status, "color": lan_color, "dns": round(lan_dns, 2), "ping": round(lan_ping, 2),
            "ipv4": lan_v4 if lan_v4 != "None" else None, "ipv6": lan_v6 if lan_v6 != "None" else None,
            "speed": { "external": {"down": round(lan_ext_down, 2), "up": round(lan_ext_up, 2)}, "internal": {"down": round(lan_int_down, 2), "up": round(lan_int_up, 2)} },
            "history": { "external": {"down": lan_hist_ext_down, "up": lan_hist_ext_up}, "internal": {"down": lan_hist_int_down, "up": lan_hist_int_up}, "ping": {"external": lan_hist_ping_ext, "internal": lan_hist_ping_int} },
            "average": { "external": {"down": round(calc_avg(lan_hist_ext_down), 2), "up": round(calc_avg(lan_hist_ext_up), 2)}, "internal": {"down": round(calc_avg(lan_hist_int_down), 2), "up": round(calc_avg(lan_hist_int_up), 2)}, "ping": {"external": round(calc_avg(lan_hist_ping_ext), 2), "internal": round(calc_avg(lan_hist_ping_int), 2)} },
            "speed_cap": lan_speed_cap
        }
    }
    return {"metrics": data, "ai_diagnoses": diagnoses, "has_wlan": has_wlan}

# --- EXPERIMENTAL FORECAST PAGE ---
'''
def fetch_trends_data():
    # 1. Fetch Global History (48h for LSTM context)
    lan_hist = get_metric_history('avg(LAN_EXTERNAL_SPEEDTEST{type="Download"})', hours=48, step='1h')
    wlan_hist = get_metric_history('avg(WLAN_EXTERNAL_SPEEDTEST{type="Download"})', hours=48, step='1h')
    
    lan_vals = [x[1] for x in lan_hist] if lan_hist else []
    wlan_vals = [x[1] for x in wlan_hist] if wlan_hist else []

    # 2. Get Forecasts
    # predict_global_trends returns raw Mbps values: {"lan": [{"rf_load": 800, ...}], ...}
    forecasts = predict_global_trends(lan_vals, wlan_vals)
    
    # 3. Normalize to Percentages (0-100) for UI Scaling
    # The Trends page graph calculates Y as (value / 100).
    if forecasts.get('lan'):
        for item in forecasts['lan']:
            # Normalize against 1000Mbps capacity
            item['rf_load'] = min(100, round((item.get('rf_load', 0) / 1000.0) * 100, 1))
            item['lstm_load'] = min(100, round((item.get('lstm_load', 0) / 1000.0) * 100, 1))
            
    if forecasts.get('wlan'):
        for item in forecasts['wlan']:
            # Normalize against 500Mbps capacity
            item['rf_load'] = min(100, round((item.get('rf_load', 0) / 500.0) * 100, 1))
            item['lstm_load'] = min(100, round((item.get('lstm_load', 0) / 500.0) * 100, 1))

    # 4. Check Validity & Trigger Fallback
    def is_data_empty(data_list):
        if not data_list: return True
        total = sum([item.get('rf_load', 0) + item.get('lstm_load', 0) for item in data_list])
        return total == 0

    if is_data_empty(forecasts.get('lan')) or is_data_empty(forecasts.get('wlan')):
        now = datetime.datetime.now()
        dummy_lan = []
        dummy_wlan = []
        for i in range(24):
            t = (now + datetime.timedelta(hours=i)).strftime("%H:00")
            # FIXED: Dummy values are now within 0-100 range
            dummy_lan.append({
                "time": t, 
                "rf_load": 85 + random.randint(-2, 2), 
                "lstm_load": 82 + random.randint(-5, 5)
            })
            dummy_wlan.append({
                "time": t, 
                "rf_load": 30 + random.randint(-5, 5), 
                "lstm_load": 34 + random.randint(-8, 8)
            })
        forecasts = {"lan": dummy_lan, "wlan": dummy_wlan}

    # Fetch heatmap data for the UI
    status_data = fetch_network_status()
    
    return {
        "heatmap": status_data.get('wlan', []), 
        "lan": forecasts['lan'],
        "wlan": forecasts['wlan']
    }
'''
    
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