@echo off
cd /d "%~dp0"

for /f "tokens=1,2 delims==" %%A in (.env) do set %%A=%%B

echo Starting GeoNest at http://localhost:8000 ...
echo Press Ctrl+C to stop.
echo.

"C:\Users\Sraharjo\AppData\Local\Python\bin\python.exe" -m uvicorn main:app --host 127.0.0.1 --port 8000
pause
