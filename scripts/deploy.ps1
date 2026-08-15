<#
deploy.ps1 — Развёртывание проекта и пошаговый запуск моделей (Windows / PowerShell)

Порядок работы:
  1. Проверка версии Python (обязательно 3.12)
  2. Создание виртуального окружения venv
  3. Установка зависимостей из requirements.txt
  4. Пошаговый запуск моделей: следующая модель стартует только после
     появления её файла результата в results/ — чтобы не создавать
     избыточную нагрузку на компьютер (одновременно работает одна модель)

Использование:
  .\deploy.ps1                        # полное развёртывание + запуск всех моделей
  .\deploy.ps1 -SetupOnly             # только шаги 1-3, без запуска моделей
  .\deploy.ps1 -Models baseline_model,improved_model   # только выбранные модели
  .\deploy.ps1 -Force                 # переобучить даже те модели, где результат уже есть
#>

param(
    [switch]$SetupOnly,
    [switch]$Force,
    [string[]]$Models = @(
        "baseline_model",
        "improved_model",
        "fully_conn_model",
        "convNet_model"
    )
)

$ErrorActionPreference = "Stop"
$ResultDir = "results"

# 1. Проверка версии Python (первая команда — обязательно 3.12)
$pythonVersion = (python --version 2>&1)
Write-Host "[1/4] Проверка Python: $pythonVersion"
if ($LASTEXITCODE -ne 0) {
    Write-Host "Ошибка: Python не найден. Установите Python 3.12 с python.org" -ForegroundColor Red
    exit 1
}
if ($pythonVersion -notmatch '3\.12(\.\d+)?') {
    Write-Host "Ошибка: требуется Python 3.12, а установлен: $pythonVersion" -ForegroundColor Red
    exit 1
}

#  2. Создание виртуального окружения
if (-not (Test-Path "venv")) {
    Write-Host "[2/4] Создание виртуального окружения venv..."
    python -m venv venv
    if ($LASTEXITCODE -ne 0) { Write-Host "Ошибка создания venv" -ForegroundColor Red; exit 1 }
} else {
    Write-Host "[2/4] Виртуальное окружение venv уже существует — пропускаю создание"
}
$venvPython = Join-Path $PWD "venv\Scripts\python.exe"

# 3. Установка зависимостей
Write-Host "[3/4] Установка зависимостей из requirements.txt..."
& $venvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { Write-Host "Ошибка обновления pip" -ForegroundColor Red; exit 1 }
& $venvPython -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { Write-Host "Ошибка установки зависимостей" -ForegroundColor Red; exit 1 }

# 4. Пошаговый запуск моделей
if ($SetupOnly) {
    Write-Host "SetupOnly: окружение готово, запуск моделей пропущен." -ForegroundColor Green
    exit 0
}

Write-Host "[4/4] Запуск моделей по одной (ожидание результата перед следующей)..."
foreach ($model in $Models) {
    $resultFile = Join-Path $ResultDir "${model}_results.json"

    if ((Test-Path $resultFile) -and -not $Force) {
        Write-Host "  Пропускаю $model — результат уже есть: $resultFile" -ForegroundColor Yellow
        continue
    }

    Write-Host "  Запуск модели: $model ..." -ForegroundColor Cyan
    & $venvPython -m "models.$model"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  Ошибка при обучении $model — остановка" -ForegroundColor Red
        exit 1
    }

    # Ждём появления файла результата (лимит 20 минут на модель)
    $deadline = (Get-Date).AddMinutes(20)
    while (-not (Test-Path $resultFile)) {
        if ((Get-Date) -gt $deadline) {
            Write-Host "  Таймаут: результат для $model не появился в results/" -ForegroundColor Red
            exit 1
        }
        Start-Sleep -Seconds 5
    }
    Write-Host "  Готово: $model -> $resultFile" -ForegroundColor Green
}

Write-Host "Все модели завершены. Результаты в папке results/" -ForegroundColor Green
