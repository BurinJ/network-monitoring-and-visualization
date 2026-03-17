@echo off
echo ==========================================
echo      KU NET - WINDOWS DEPLOYMENT
echo ==========================================

echo [1/4] Installing Frontend Dependencies...
cd frontend
call npm install
if %errorlevel% neq 0 exit /b %errorlevel%

echo [2/4] Building React Frontend...
call npm run build
if %errorlevel% neq 0 exit /b %errorlevel%
cd ..

echo [3/4] Cleaning Old Server Files...
if exist "backend\static_client" (
    echo Removing old static files...
    rmdir /s /q "backend\static_client"
)

echo Moving New Build Files...
move "frontend\dist" "backend\static_client"

echo [4/4] Starting Production Server (Waitress)...
cd backend

:: 1. Check/Create Virtual Environment
if not exist "venv" (
    echo Creating Python virtual environment...
    py -m venv venv
)

:: 2. Activate Python Env
call venv\Scripts\activate

:: 3. Install Dependencies (Ensures waitress is present)
echo Installing Backend Dependencies...
py -m pip install -r pre.txt
py -m pip install waitress

:: 4. Run Waitress (Using python directly to avoid path issues)
echo Server is running on http://0.0.0.0:5000
echo (Keep this window open)
python -c "from waitress import serve; from app import app; print('Starting Waitress...'); serve(app, host='0.0.0.0', port=5000)"

pause