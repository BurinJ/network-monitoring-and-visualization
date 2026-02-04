import json
import os

# File to store dynamic settings
SETTINGS_FILE = os.path.join(os.path.dirname(__file__), 'app_settings.json')

# Default values if file doesn't exist
DEFAULTS = {
    "lan_ping_threshold": 100,      # ms
    "wlan_ping_threshold": 200,     # ms
    "dns_threshold": 100,           # ms
    "offline_timeout_mins": 60,     # minutes
    "speed_drop_threshold": 10,      # Mbps (Absolute low limit)
    # Anomaly Detection Config
    "anomaly_contamination": 0.01,  # 1% (Sensitivity)
    "anomaly_estimators": 200       # Tree count (Complexity)
}

def get_settings():
    """Load settings from file or return defaults."""
    if not os.path.exists(SETTINGS_FILE):
        return DEFAULTS.copy()
    try:
        with open(SETTINGS_FILE, 'r') as f:
            data = json.load(f)
            # Merge with defaults to ensure all keys exist
            return {**DEFAULTS, **data}
    except:
        return DEFAULTS.copy()

def update_settings(new_data):
    """Update settings and save to file."""
    current = get_settings()
    # Update only valid keys
    for key in DEFAULTS.keys():
        if key in new_data:
            # Ensure numbers are stored as numbers
            try:
                current[key] = float(new_data[key])
            except ValueError:
                pass
    
    try:
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(current, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving settings: {e}")
        return False