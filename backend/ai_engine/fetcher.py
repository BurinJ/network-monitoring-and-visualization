import time
import math
import sys
import os
import base64
import datetime 
import random
import json
from prometheus_api_client import PrometheusConnect

# --- IMPORT SETTINGS MANAGER ---
try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    import settings_manager
except ImportError:
    class MockSettings:
        def get_settings(self): return {}
    settings_manager = MockSettings()

# --- IMPORT AI ANALYZER ---
try:
    from .analyzer import detect_anomaly
except ImportError:
    def detect_anomaly(m, probe_id=None, timestamp=None): return {"is_anomaly": False, "score": 0, "desc": "AI Module Missing", "causes": []}

# --- IMPORT CONFIGURATION & HISTORY ---
try:
    from config import PROMETHEUS_URL, HOSTNAME_MAPPING, PROM_PASSWORD, PROM_USER
    import history
except ImportError:
    print("⚠️ Config or History module not found. Using defaults.")
    PROMETHEUS_URL = "http://localhost:9090"
    HOSTNAME_MAPPING = {}
    history = None

# --- MAPPING FILES ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MAPPING_FILE = os.path.join(BASE_DIR, '../probe_mappings.json')
DEPT_MAPPING_FILE = os.path.join(BASE_DIR, '../probe_departments.json')
DEPARTMENT_MAPPING = {}
HOSTNAME_MAPPING = {}

def load_mappings():
    global HOSTNAME_MAPPING, DEPARTMENT_MAPPING
    if os.path.exists(MAPPING_FILE):
        try:
            with open(MAPPING_FILE, 'r') as f: HOSTNAME_MAPPING.update(json.load(f))
        except: pass
    if os.path.exists(DEPT_MAPPING_FILE):
        try:
            with open(DEPT_MAPPING_FILE, 'r') as f: DEPARTMENT_MAPPING = json.load(f)
        except: pass

def save_mapping(raw_host, friendly_name):
    global HOSTNAME_MAPPING
    HOSTNAME_MAPPING[raw_host] = friendly_name
    try:
        with open(MAPPING_FILE, 'w') as f: json.dump(HOSTNAME_MAPPING, f, indent=2)
        return True
    except: return False

def save_department_mapping(raw_host, department):
    global DEPARTMENT_MAPPING
    DEPARTMENT_MAPPING[raw_host] = department
    try:
        with open(DEPT_MAPPING_FILE, 'w') as f: json.dump(DEPARTMENT_MAPPING, f, indent=2)
        return True
    except: return False

load_mappings()

# --- CONNECTION SETUP ---
try:
    auth_str = f"{PROM_USER}:{PROM_PASSWORD}"
    b64_auth = base64.b64encode(auth_str.encode()).decode()
    headers = {"Authorization": f"Basic {b64_auth}"}
    prom = PrometheusConnect(url=PROMETHEUS_URL, headers=headers, disable_ssl=True)
    prom.custom_query(query="up")
    CONNECTED = True
except:
    CONNECTED = False

def safe_get_value(query, default=0):
    if not CONNECTED: return default
    try:
        result = prom.custom_query(query=query)
        if result and len(result) > 0:
            return float(result[0]['value'][1])
    except: pass
    return default

def get_label_value(metric_name, label_name, hostname): return 0
def get_all_labels(metric_name, hostname): return {}

def get_metric_history(query, hours=24, step='1h'):
    if not CONNECTED: 
        now = datetime.datetime.now()
        data = []
        for i in range(hours):
            data.append([(now - datetime.timedelta(hours=hours-i)).timestamp(), random.randint(10, 100)])
        return data
    try:
        start_time = datetime.datetime.now() - datetime.timedelta(hours=hours)
        end_time = datetime.datetime.now()
        result = prom.custom_query_range(query=query, start_time=start_time, end_time=end_time, step=step)
        if result and len(result) > 0:
            return [[float(x[0]), float(x[1])] for x in result[0]['values']]
    except: pass
    return []

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
    except: pass

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
                    parts = []
                    if days > 0: parts.append(f"{days}d")
                    if hours > 0: parts.append(f"{hours}h")
                    parts.append(f"{minutes}m")
                    stale_label = f"Offline (~{' '.join(parts)})"

            error_label = metric_labels.get('error', 'None')
            lan_curl_err = metric_labels.get('lan_google_curl', 'True') == 'False'
            wlan_curl_err = metric_labels.get('wlan_google_curl', 'True') == 'False'

            # --- GATHER ALL METRICS ---
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

            # --- SINGLE AI CHECK (HOLISTIC) ---
            ai_res = detect_anomaly({
                'lan_ping': lan_latency, 'lan_dns': lan_dns, 
                'lan_down': lan_down, 'lan_up': lan_up,
                'wlan_ping': wlan_latency, 'wlan_dns': wlan_dns,
                'wlan_down': wlan_down, 'wlan_up': wlan_up
            }, probe_id=raw_hostname)

            # --- PROCESS LAN STATUS ---
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
            elif lan_dns > 500:
                lan_status = "DNS Failure"
                lan_color = "red"
            elif lan_dns > DNS_THRESH:
                lan_status = "Slow DNS"
                lan_color = "orange"
            elif lan_latency > LAN_PING_THRESH:
                lan_status = "Laggy"
                lan_color = "orange"
            elif ai_res['is_anomaly']:
                # If everything else is green but AI flags anomaly, check causes
                # Or just flag it. The inspector logic is smarter about attribution.
                lan_status = "AI Warning"
                lan_color = "orange"
                check_and_log(display_name, "Warning", "AI Anomaly", ai_res['desc'], {"interface": "LAN"})

            if lan_color == 'red':
                check_and_log(display_name, "Critical", "Connection Failure", lan_status, {"interface": "LAN"})

            lan_probes.append({
                "name": display_name, "id": raw_hostname, "department": department, "type": "LAN", "status": lan_status, "color": lan_color,
                "latency": round(lan_latency, 2), "dns": round(lan_dns, 2), "down": round(lan_down, 1), "up": round(lan_up, 1),
                "lat": float(metric_labels.get('latitude', '0')), "lng": float(metric_labels.get('longitude', '0'))
            })

            # --- PROCESS WLAN STATUS ---
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
            elif wlan_dns > 500:
                wlan_status = "DNS Failure"
                wlan_color = "red"
            elif wlan_dns > DNS_THRESH:
                wlan_status = "Slow DNS"
                wlan_color = "orange"
            elif wlan_latency > WLAN_PING_THRESH:
                wlan_status = "Laggy"
                wlan_color = "orange"
            elif ai_res['is_anomaly']:
                 wlan_status = "AI Warning"
                 wlan_color = "orange"
                 check_and_log(display_name, "Warning", "AI Anomaly", ai_res['desc'], {"interface": "WLAN"})

            if wlan_color == 'red':
                check_and_log(display_name, "Critical", "Connection Failure", wlan_status, {"interface": "WLAN"})

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
    issues = []
    for p in all_interfaces:
        if p['color'] != 'green':
            issues.append({"location": f"{p['name']} ({p['type']})", "issue": p['status'], "severity": "High" if p['color'] == 'red' else "Medium"})
    map_markers = []
    for p in all_interfaces:
        if p.get('lat') == 0 or p.get('lng') == 0: continue
        key = p['name']
        existing = next((m for m in map_markers if m['name'] == key), None)
        if not existing:
            existing = { "name": p['name'], "lat": p['lat'], "lng": p['lng'], "status": "Healthy", "color": "green" }
            map_markers.append(existing)
        if p['color'] == 'red': existing['status'] = "Critical"; existing['color'] = "red"
        elif p['color'] == 'orange' and existing['color'] != 'red': existing['status'] = "Warning"; existing['color'] = "orange"
    return { "alerts": down_count, "bandwidth": round(avg_bw, 1), "satisfaction": 95, "total_probes": len(all_interfaces), "active_probes": len(all_interfaces) - down_count, "priority_issues": issues, "map_markers": list(map_markers.values()) }

# --- PAGE 3: INSPECTOR LOGIC ---
def fetch_inspector_data(probe_id, duration='24h'):
    config = settings_manager.get_settings()
    LAN_PING_THRESH = config.get('lan_ping_threshold', 100)
    WLAN_PING_THRESH = config.get('wlan_ping_threshold', 200)
    DNS_THRESH = config.get('dns_threshold', 100)
    TIMEOUT_SEC = config.get('offline_timeout_mins', 60) * 60

    clean_id = probe_id.replace(" (LAN)", "").replace(" (WLAN)", "")
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
    wlan_ping = safe_get_value(f'last_over_time(WLAN_PING{{hostname="{raw_hostname}", metrics="avgRTT", type="EXTERNAL"}}[1h])', default=0)
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

    # --- HISTORICAL ANOMALY DETECTION LOOP ---
    historical_anomalies = []
    ref_history = lan_hist_ext_down if lan_hist_ext_down else []
    if ref_history:
        for i in range(len(ref_history)):
            ts = ref_history[i][0]
            ts_dt = datetime.datetime.fromtimestamp(ts)
            def val_at(hist, idx): return hist[idx][1] if hist and idx < len(hist) else 0
            snapshot = {
                'lan_down': val_at(lan_hist_ext_down, i), 'lan_up': val_at(lan_hist_ext_up, i),
                'wlan_down': val_at(wlan_hist_ext_down, i), 'wlan_up': val_at(wlan_hist_ext_up, i),
                'lan_ping': val_at(lan_hist_ping_ext, i), 'wlan_ping': val_at(wlan_hist_ping_ext, i),
                'lan_dns': 0, 'wlan_dns': 0 
            }
            res = detect_anomaly(snapshot, probe_id=raw_hostname, timestamp=ts_dt)
            if res['is_anomaly']: historical_anomalies.append(ts)

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
            "causes": ai_result.get('causes', [])
        })

    if is_stale:
        diagnoses.append({
            "status": "Critical", 
            "title": "Probe Offline", 
            "desc": f"Probe unresponsive for > {TIMEOUT_SEC // 60} minutes.",
            "causes": [{"title": "Timeout", "detail": f"Last seen: {datetime.datetime.fromtimestamp(probe_ts)}"}]
        })
    if error_label != "None":
        diagnoses.append({
            "status": "Critical", 
            "title": "Hardware/Software Error", 
            "desc": f"Probe reporting error: {error_label}",
            "causes": [{"title": "Error Code", "detail": error_label}]
        })
    if lan_ping == 0 and (not has_wlan or wlan_ping == 0):
        diagnoses.append({
            "status": "Critical", 
            "title": "Network Unreachable", 
            "desc": "Interfaces unresponsive.",
            "causes": [{"title": "Connectivity Loss", "detail": "Ping check failed for all interfaces."}]
        })
    if has_wlan and wlan_dns > 500:
        diagnoses.append({
            "status": "Critical", 
            "title": "WLAN DNS Failure", 
            "desc": "Wi-Fi cannot resolve domain names.",
            "causes": [{"title": "High DNS Latency", "detail": f"{wlan_dns}ms > 500ms"}]
        })
    if lan_dns > 500:
        diagnoses.append({
            "status": "Critical", 
            "title": "LAN DNS Failure", 
            "desc": "Ethernet DNS resolution failed.",
            "causes": [{"title": "High DNS Latency", "detail": f"{lan_dns}ms > 500ms"}]
        })
    if has_wlan and wlan_ping > WLAN_PING_THRESH:
        diagnoses.append({
            "status": "Warning", 
            "title": "Wi-Fi Congestion", 
            "desc": "High latency on Wi-Fi interface.",
            "causes": [{"title": "High Ping", "detail": f"{wlan_ping}ms > {WLAN_PING_THRESH}ms"}]
        })
    if lan_ping > LAN_PING_THRESH:
        diagnoses.append({
            "status": "Warning", 
            "title": "LAN Latency", 
            "desc": "High latency on Ethernet interface.",
            "causes": [{"title": "High Ping", "detail": f"{lan_ping}ms > {LAN_PING_THRESH}ms"}]
        })

    if not diagnoses:
        diagnoses.append({"status": "Healthy", "title": "Normal Operation", "desc": "No significant anomalies detected."})

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
        },
        "anomalies": historical_anomalies
    }
    return {"metrics": data, "ai_diagnoses": diagnoses, "has_wlan": has_wlan}

# --- PAGE 4: TRENDS LOGIC ---
def fetch_trends_data():
    return {"heatmap": [], "forecast": []}

def _mock_pulse_data():
    return { "lan": [], "wlan": [] }