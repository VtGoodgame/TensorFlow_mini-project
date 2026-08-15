#!/usr/bin/env bash
# deploy.sh — Развёртывание проекта и пошаговый запуск моделей (Linux / macOS / Git Bash на Windows)
#
# Порядок работы:
#   1. Проверка версии Python (обязательно 3.12)
#   2. Создание виртуального окружения venv
#   3. Установка зависимостей из requirements.txt
#   4. Пошаговый запуск моделей: следующая модель стартует только после
#      появления её файла результата в results/ — чтобы не создавать
#      избыточную нагрузку на компьютер (одновременно работает одна модель)
#
# Использование:
#   ./deploy.sh                               # полное развёртывание + все модели
#   ./deploy.sh --setup-only                  # только шаги 1-3
#   ./deploy.sh --models baseline_model improved_model
#   ./deploy.sh --force                       # переобучить даже там, где результат уже есть

set -euo pipefail

RESULT_DIR="results"
MODELS=(baseline_model improved_model fully_conn_model convNet_model)
FORCE=0
SETUP_ONLY=0

# ---- Разбор аргументов ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --setup-only) SETUP_ONLY=1; shift ;;
        --force)      FORCE=1; shift ;;
        --models)     shift; MODELS=(); while [[ $# -gt 0 && "$1" != --* ]]; do MODELS+=("$1"); shift; done ;;
        -h|--help)    sed -n '1,20p' "$0"; exit 0 ;;
        *) echo "Неизвестный аргумент: $1" >&2; exit 1 ;;
    esac
done

# 1. Проверка версии Python (первая команда — обязательно 3.12)
if command -v python3 >/dev/null 2>&1; then
    PY=python3
else
    PY=python
fi

echo "[1/4] Проверка Python"
PY_VERSION=$($PY --version 2>&1) || { echo "Ошибка: Python не найден. Установите Python 3.12" >&2; exit 1; }
echo "  $PY_VERSION"
case "$PY_VERSION" in
    *3.12*) ;;
    *) echo "Ошибка: требуется Python 3.12, а установлен: $PY_VERSION" >&2; exit 1 ;;
esac

# 2. Создание виртуального окружения
if [ ! -d "venv" ]; then
    echo "[2/4] Создание виртуального окружения venv..."
    $PY -m venv venv
else
    echo "[2/4] Виртуальное окружение venv уже существует — пропускаю создание"
fi

# venv/python: Windows (Git Bash) и Unix отличаются расположением
if [ -f "venv/Scripts/python.exe" ]; then
    VENV_PY="venv/Scripts/python.exe"
else
    VENV_PY="venv/bin/python"
fi

#  3. Установка зависимостей
echo "[3/4] Установка зависимостей из requirements.txt..."
"$VENV_PY" -m pip install --upgrade pip
"$VENV_PY" -m pip install -r requirements.txt

# 4. Пошаговый запуск моделей
if [ "$SETUP_ONLY" = "1" ]; then
    echo "SetupOnly: окружение готово, запуск моделей пропущен."
    exit 0
fi

echo "[4/4] Запуск моделей по одной (ожидание результата перед следующей)..."
for model in "${MODELS[@]}"; do
    result="$RESULT_DIR/${model}_results.json"

    if [ -f "$result" ] && [ "$FORCE" = "0" ]; then
        echo "  Пропускаю $model — результат уже есть: $result"
        continue
    fi

    echo "  Запуск модели: $model ..."
    "$VENV_PY" -m "models.$model" || { echo "  Ошибка при обучении $model — остановка" >&2; exit 1; }

    # Ждём появления файла результата (лимит 20 минут на модель)
    deadline=$(( $(date +%s) + 1200 ))
    until [ -f "$result" ]; do
        now=$(date +%s)
        if [ "$now" -gt "$deadline" ]; then
            echo "  Таймаут: результат для $model не появился в results/" >&2
            exit 1
        fi
        sleep 5
    done
    echo "  Готово: $model -> $result"
done

echo "Все модели завершены. Результаты в папке $RESULT_DIR/"
