<#!
Launch the complete Framework web application without Internet access.
The browser talks only to loopback services (127.0.0.1); no external API is used.
#>
[CmdletBinding()]
param([switch]$Development)

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root ".venv_web\Scripts\python.exe"
$frontend = Join-Path $root "frontend"
if (-not (Test-Path -LiteralPath $python)) { throw "Missing .venv_web. Create it with Python 3.11 and install requirements.txt." }
if (-not (Test-Path -LiteralPath (Join-Path $frontend "node_modules"))) { throw "Missing frontend/node_modules. Run npm ci in frontend once while dependencies are available." }
if (-not (Test-Path -LiteralPath (Join-Path $root "models\qwen.gguf"))) { throw "Missing models/qwen.gguf. Configure GENERATOR_MODEL_PATH in .env." }

$apiArgs = "-m uvicorn backend.main:app --host 127.0.0.1 --port 8000"
if (-not $Development -and -not (Test-Path -LiteralPath (Join-Path $frontend ".next\BUILD_ID"))) {
    Write-Host "Creating the local production frontend build…"
    Push-Location $frontend
    try { & npm.cmd run build; if ($LASTEXITCODE -ne 0) { throw "Next.js build failed." } }
    finally { Pop-Location }
}
$uiArgs = if ($Development) { "run dev -- --hostname 127.0.0.1 --port 3000" } else { "run start -- --hostname 127.0.0.1 --port 3000" }
Start-Process -FilePath $python -ArgumentList $apiArgs -WorkingDirectory $root -WindowStyle Hidden
Start-Process -FilePath "npm.cmd" -ArgumentList $uiArgs -WorkingDirectory $frontend -WindowStyle Hidden
Write-Host "Framework is starting locally at http://127.0.0.1:3000"
Write-Host "Both services are bound to loopback only; no external API or cloud inference is used."
