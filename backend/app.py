from flask import Flask, jsonify, request
from flask_cors import CORS
from ai_engine.fetcher import fetch_network_status, fetch_command_center, fetch_inspector_data, save_mapping, save_department_mapping
from ai_engine.trainer import train_models
from ai_engine import analyzer
import threading
import time
import history
import settings_manager

app = Flask(__name__)
CORS(app) 

# --- BACKGROUND MONITOR ---
def background_monitor():
    print("⏰ Background Monitor Started")
    while True:
        try:
            fetch_network_status()
        except Exception as e:
            print(f"Background Monitor Error: {e}")
        time.sleep(60)

monitor_thread = threading.Thread(target=background_monitor, daemon=True)
monitor_thread.start()

@app.route('/')
def home():
    return jsonify({"status": "University Monitor API is Running", "version": "1.0"})

@app.route('/api/network-status', methods=['GET'])
def get_pulse():
    data = fetch_network_status()
    all_interfaces = data.get('lan', []) + data.get('wlan', [])
    overall = "Operational"
    if any(p['color'] == 'red' for p in all_interfaces):
        overall = "Partial Outage"
    return jsonify({"overall": overall, "buildings": data})

@app.route('/api/command-center', methods=['GET'])
def get_command_center():
    data = fetch_command_center()
    return jsonify(data)

@app.route('/api/inspector/<probe_id>', methods=['GET'])
def get_inspector(probe_id):
    duration = request.args.get('duration', '24h')
    data = fetch_inspector_data(probe_id, duration)
    return jsonify(data)

# EXPERIMENTAL
'''
@app.route('/api/trends', methods=['GET'])
def get_trends():
    data = fetch_trends_data()
    return jsonify(data)
'''

@app.route('/api/probes', methods=['GET'])
def get_probe_list():
    data = fetch_network_status()
    # Return friendly names for dropdowns
    names = sorted(list(set(
        [p['name'] for p in data.get('lan', [])] + 
        [p['name'] for p in data.get('wlan', [])]
    )))
    return jsonify(names)

@app.route('/api/settings/probes', methods=['GET'])
def get_settings_probes():
    data = fetch_network_status()
    # print(data.get('lan'))
    all_p = data.get('lan', []) + data.get('wlan', [])
    # Unique by ID to list all raw probes for renaming
    unique_probes = {p['id']: {'name': p['name'], 'department': p.get('department', 'Undefined')} for p in all_p}
    # print(unique_probes.items())
    return jsonify([{"id": k, "name": v['name'], "department": v['department']} for k,v in unique_probes.items()])

@app.route('/api/settings/probe', methods=['POST'])
def update_probe_name():
    data = request.json
    raw_id = data.get('id')
    new_name = data.get('name')
    new_dept = data.get('department')
    
    success = True
    if new_name:
        if not save_mapping(raw_id, new_name): success = False
    if new_dept:
        if not save_department_mapping(raw_id, new_dept): success = False
        
    if success:
        return jsonify({"status": "success"})
    return jsonify({"status": "error"}), 500

@app.route('/api/settings/thresholds', methods=['GET'])
def get_thresholds():
    return jsonify(settings_manager.get_settings())

@app.route('/api/settings/thresholds', methods=['POST'])
def update_thresholds():
    if settings_manager.update_settings(request.json):
        return jsonify({"status": "success"})
    return jsonify({"status": "error"}), 500

@app.route('/api/alerts/history', methods=['GET'])
def get_alert_history():
    limit = request.args.get('limit', 100)
    alerts = history.get_recent_alerts(int(limit))
    return jsonify(alerts)

@app.route('/api/admin/train', methods=['POST'])
def trigger_training():
    try:
        print("🔄 Manual training triggered via API...")
        train_models()
        analyzer.load_models()
        return jsonify({"status": "success", "message": "Models retrained and reloaded successfully."})
    except Exception as e:
        print(f"❌ Training Error: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500

if __name__ == "__main__":
    print("🚀 Starting AI Backend Server on http://localhost:5000")
    app.run(debug=True, port=5000)