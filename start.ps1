# GeoNest local development starter
# Run this script to start the server on your machine

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# Load .env file
$envFile = Join-Path $ScriptDir ".env"
if (Test-Path $envFile) {
    Get-Content $envFile | ForEach-Object {
        if ($_ -match '^\s*([^#][^=]+)=(.*)$') {
            [System.Environment]::SetEnvironmentVariable($Matches[1].Trim(), $Matches[2].Trim(), 'Process')
        }
    }
    Write-Host "Loaded .env"
}

Write-Host "Starting GeoNest at http://localhost:8000 ..."
Write-Host "Press Ctrl+C to stop."
Write-Host ""

$PYTHON = "C:\Users\Sraharjo\AppData\Local\Python\bin\python.exe"
Set-Location $ScriptDir
& $PYTHON -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
