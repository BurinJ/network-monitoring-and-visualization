import sqlite3
import datetime
import os
import json

DB_PATH = os.path.join(os.path.dirname(__file__), 'alerts.db')

def init_db():
    """Creates the alerts table if it doesn't exist."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            probe_id TEXT,
            level TEXT,
            category TEXT,
            message TEXT,
            details TEXT
        )
    ''')
    conn.commit()
    conn.close()

def log_alert(probe_id, level, category, message, details=""):
    """
    Saves an alert to the database.
    timestamp: UTC ISO string
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # FIX: Use UTC to match frontend 'Z' assumption
    now = datetime.datetime.utcnow().isoformat()
    
    # Store complex details as JSON string if needed
    if isinstance(details, dict):
        details = json.dumps(details)
        
    c.execute('''
        INSERT INTO alerts (timestamp, probe_id, level, category, message, details)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (now, probe_id, level, category, message, details))
    
    conn.commit()
    conn.close()

def get_recent_alerts(limit=100):
    """Fetches the most recent alerts."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row # Allow access by column name
    c = conn.cursor()
    
    c.execute('SELECT * FROM alerts ORDER BY id DESC LIMIT ?', (limit,))
    rows = c.fetchall()
    
    conn.close()
    
    return [dict(row) for row in rows]

# Initialize on module load
init_db()