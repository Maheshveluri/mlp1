$ErrorActionPreference = "Stop"

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$app = Join-Path $PSScriptRoot "app.py"
$configDirectory = Join-Path $PSScriptRoot ".streamlit"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Project virtual environment not found. Create it and install requirements first."
}
if (-not (Test-Path -LiteralPath $app)) {
    throw "app.py was not found in the project folder."
}

New-Item -ItemType Directory -Force -Path $configDirectory | Out-Null
$env:STREAMLIT_CONFIG_DIR = $configDirectory
& $python -m streamlit run $app --server.headless true --browser.gatherUsageStats false
