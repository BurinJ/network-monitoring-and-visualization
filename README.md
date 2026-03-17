# KU Net - Network Monitoring & AI Diagnostics

KU Net is a full-stack monitoring system designed to track university network health (LAN/WLAN) in real-time. It integrates with Prometheus telemetry and utilizes an Unsupervised Machine Learning model (Isolation Forest) to detect contextual anomalies, significantly reducing "Notification Fatigue" caused by standard static thresholds.

# Tech Stack

Frontend: React.js, Vite, Tailwind CSS, Recharts, Leaflet Maps

Backend: Python, Flask, Waitress (Production Server)

AI/ML: Scikit-Learn (Isolation Forest), Pandas, NumPy

Data Source: Prometheus Time-Series Database

# Prerequisites

Before running or deploying the application, ensure your environment has the following installed:

**Node.js (v16 or higher)** - For building the React frontend.

**Python (v3.8 or higher)** - For running the AI backend.

**Prometheus (Optional for UI testing, Required for live data)** - The backend expects a Prometheus server to pull telemetry from.

# Local Development Setup (Hot-Reloading)

If you want to edit the code and see changes in real-time, run the frontend and backend separately.

## 1. Start the Python Backend (Flask)

The backend runs on port 5000 and provides the API and AI models.
```
cd backend

# Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate  # On Mac/Linux use: source venv/bin/activate

# Install dependencies
pip install -r pre.txt

# Run the Flask development server
python app.py
```

## 2. Start the React Frontend (Vite)

The frontend runs on port 5173. It is configured (frontend/src/config.js) to automatically route API requests to http://localhost:5000/api when running in dev mode.
```
cd frontend

# Install Node dependencies
npm install

# Start the Vite development server
npm run dev
```

You can now access the development UI at http://localhost:5173.

# Production Deployment (Windows VM)

To deploy the application as a unified production build (where Python serves the compiled React app), we use the provided automated batch script.

Step-by-Step Deployment

Open Command Prompt or PowerShell.

Navigate to the root directory of the project.

Run the deployment script:

deploy.bat


What deploy.bat does automatically:

Installs Frontend Dependencies: Runs npm install in the /frontend folder.

Builds the React App: Runs npm run build to compile the optimized static frontend.

Moves the Build: Takes the frontend/dist folder and moves it to backend/static_frontend.

Prepares the Python Environment: Creates a virtual environment (if missing) and installs requirements (including Waitress).

Starts the Server: Launches a robust production WSGI server (Waitress) on 0.0.0.0:5000.

Accessing the Live Dashboard

Once deploy.bat finishes, the server will keep running in that terminal window.

From the Server itself: Open http://localhost:5000

From Outside (via VPN or LAN): Open http://<SERVER_IP_ADDRESS>:5000 (e.g., http://192.168.1.50:5000)

(Note: The frontend is configured to use relative API paths in production, so accessing it via the external IP will seamlessly route API calls back to the same IP).

# Configuration Files

If you need to tweak system behaviors, check these files:

frontend/src/config.js: Handles the API URL switching between Development (Localhost) and Production (Relative IP).

backend/config.py: Contains the connection details for your Prometheus server (PROMETHEUS_URL, PROM_USER, PROM_PASSWORD).

backend/settings_manager.py: Manages the dynamic thresholds (Ping/Speed/DNS limits) adjustable via the UI.

backend/probe_mappings.json: Stores the human-readable names assigned to raw MAC/IP addresses.

# AI Model Training

The Isolation Forest model is pre-configured to train on recent Prometheus history.
To manually trigger a retraining of the AI model on the latest telemetry data:

Open the UI.

Go to the Admin Settings.

Click "Force Model Retraining" (This calls the /api/admin/train endpoint).
Alternatively, the backend can be configured to retrain automatically via a cron job or background thread in backend/ai_engine/trainer.py.