@echo off
setlocal
set PORT=7860

echo [1/3] Closing old processes...
wsl -d Ubuntu-24.04 -e bash -c "pkill -f main.py" >nul 2>&1

echo [2/3] Starting SAM 3D Pose Analyzer...
:: CD to app directory and run main.py via conda
:: Using bash -ic to ensure conda is initialized from .bashrc
start /b wsl -d Ubuntu-24.04 -e bash -ic "cd $(wslpath '%~dp0')/app && conda run --no-capture-output -n sam_3d_body python main.py"

echo [3/3] Opening browser at http://127.0.0.1:%PORT%...
timeout /t 8 >nul
start http://127.0.0.1:%PORT%

echo.
echo ==============================================
echo  SAM 3D is now starting.
echo  Please wait for the web page to load.
echo ==============================================
timeout /t 3 >nul
exit
