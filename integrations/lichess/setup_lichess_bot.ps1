param(
    [string]$Destination = "$HOME\repos\lichess-bot"
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "git was not found on PATH. Install Git or clone lichess-bot manually."
}

if (-not (Test-Path $Destination)) {
    git clone https://github.com/lichess-bot-devs/lichess-bot.git $Destination
}

Push-Location $Destination
try {
    if (-not (Test-Path "venv")) {
        py -m venv venv
    }
    & ".\venv\Scripts\python.exe" -m pip install --upgrade pip
    & ".\venv\Scripts\python.exe" -m pip install -r requirements.txt

    if (-not (Test-Path "config.yml")) {
        Copy-Item "config.yml.default" "config.yml"
        Write-Host "Created $Destination\config.yml from the current official default."
    }

    Write-Host "lichess-bot installation is ready."
    Write-Host "Next: edit config.yml using D:\atomic_chess\integrations\lichess\config.atomic.snippet.yml"
    Write-Host "Do not upgrade the Lichess account until the local UCI smoke test passes."
}
finally {
    Pop-Location
}
