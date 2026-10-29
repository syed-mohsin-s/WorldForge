Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  WorldForge Studio 2.0 Launch" -ForegroundColor Cyan
Write-Host "  MuJoCo 3.17 Physics + Baseten meta-models/muse-spark-open-1026" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

$root = $PSScriptRoot
$python = "$root\.venv\Scripts\python.exe"

Write-Host "[1/2] Launching FastAPI backend on http://localhost:8000 ..." -ForegroundColor Green
$backend = Start-Process -FilePath $python -ArgumentList "-m uvicorn src.api.server:app --host 0.0.0.0 --port 8000" -WorkingDirectory $root -PassThru

Start-Sleep -Seconds 2

Write-Host "[2/2] Launching Vite React frontend on http://localhost:5173 ..." -ForegroundColor Green
$frontend = Start-Process -FilePath "npm.cmd" -ArgumentList "run dev" -WorkingDirectory "$root\frontend" -PassThru

Start-Sleep -Seconds 2

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  WorldForge Studio is LIVE!" -ForegroundColor Yellow
Write-Host "  Frontend:    http://localhost:5173" -ForegroundColor Green
Write-Host "  Backend API: http://localhost:8000/docs" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Press Ctrl+C or close the terminal to stop."

Wait-Process -Id $backend.Id, $frontend.Id

