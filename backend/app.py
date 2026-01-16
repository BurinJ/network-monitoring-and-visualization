from flask import Flask, jsonify, request
from flask_cors import CORS
from ai_engine.fetcher import fetch_network_status, fetch_command_center, fetch_inspector_data, fetch_trends_data
from ai_engine.trainer import train_models
from ai_engine import analyzer
import history, time, threading

app = Flask(__name__)
CORS(app) 

# --- BACKGROUND MONITOR ---
def background_monitor():
    """Runs every 60 seconds to log alerts to DB."""
    print("<T> Background Monitor Started")
    while True:
        try:
            # This function now has internal logging logic
            fetch_network_status()
        except Exception as e:
            print(f"Background Monitor Error: {e}")
        time.sleep(60)

# Start thread
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
    # Get duration from query param, default to '24h'
    duration = request.args.get('duration', '24h')
    data = fetch_inspector_data(probe_id, duration)
    return jsonify(data)

@app.route('/api/trends', methods=['GET'])
def get_trends():
    data = fetch_trends_data()
    return jsonify(data)

@app.route('/api/probes', methods=['GET'])
def get_probe_list():
    data = fetch_network_status()
    names = sorted(list(set(
        [p['name'] for p in data.get('lan', [])] + 
        [p['name'] for p in data.get('wlan', [])]
    )))
    return jsonify(names)

@app.route('/api/alerts/history', methods=['GET'])
def get_alert_history():
    limit = request.args.get('limit', 100)
    alerts = history.get_recent_alerts(int(limit))
    return jsonify(alerts)

@app.route('/api/admin/train', methods=['POST'])
def trigger_training():
    try:
        print("🔄 Manual training triggered via API...")
        # 1. Run the training process
        train_models()
        
        # 2. Reload the models in the analyzer so changes take effect immediately
        analyzer.load_models()
        
        return jsonify({"status": "success", "message": "Models retrained and reloaded successfully."})
    except Exception as e:
        print(f"❌ Training Error: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting AI Backend Server on http://localhost:5000")
    app.run(debug=True, port=5000)