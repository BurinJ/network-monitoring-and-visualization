from flask import Flask, jsonify
from flask_cors import CORS
from ai_engine.fetcher import fetch_network_status, fetch_command_center, fetch_inspector_data, fetch_trends_data

app = Flask(__name__)
CORS(app) 

@app.route('/')
def home():
    return jsonify({"status": "University Monitor API is Running", "version": "1.0"})

@app.route('/api/network-status', methods=['GET'])
def get_pulse():
    data = fetch_network_status()
    
    # Check overall status across both LAN and WLAN lists
    all_interfaces = data.get('lan', []) + data.get('wlan', [])
    overall = "Operational"
    if any(p['color'] == 'red' for p in all_interfaces):
        overall = "Partial Outage"
    
    # Returns { "overall": "...", "buildings": { "lan": [...], "wlan": [...] } }
    return jsonify({"overall": overall, "buildings": data})

@app.route('/api/command-center', methods=['GET'])
def get_command_center():
    data = fetch_command_center()
    return jsonify(data)

@app.route('/api/inspector/<probe_id>', methods=['GET'])
def get_inspector(probe_id):
    data = fetch_inspector_data(probe_id)
    return jsonify(data)

@app.route('/api/trends', methods=['GET'])
def get_trends():
    data = fetch_trends_data()
    return jsonify(data)

@app.route('/api/probes', methods=['GET'])
def get_probe_list():
    data = fetch_network_status()
    # Combine names from both lists, unique values only
    names = sorted(list(set(
        [p['name'] for p in data.get('lan', [])] + 
        [p['name'] for p in data.get('wlan', [])]
    )))
    return jsonify(names)

if __name__ == '__main__':
    print("🚀 Starting AI Backend Server on http://localhost:5000")
    app.run(debug=True, port=5000)