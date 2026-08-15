# Makefile — запуск развёртывания и моделей (Windows / Linux / macOS / Git Bash)
#
# Примеры:
#   make help     показать справку
#   make check    проверить версию Python (обязательно 3.12)
#   make venv     создать виртуальное окружение и установить зависимости
#   make deploy   развёртывание + пошаговый запуск всех моделей
#   make clean    удалить venv и кэши Python
#
# Результат выполнения команд:
#   make check  — проверяет только версию Python (3.12). При несоответствии
#                 прерывается с ошибкой; при успехе печатает версию.
#   make venv   — создаёт виртуальное окружение venv, обновляет pip и
#                 устанавливает зависимости из requirements.txt.
#   make deploy — вызывает скрипт развёртывания текущей ОС:
#                   Windows -> powershell -File deploy.ps1
#                   Unix    -> bash deploy.sh
#                 Скрипт проверяет версию Python, создаёт venv, ставит
#                 зависимости и запускает модели по одной, ожидая файл
#                 результата каждой модели в results/.
#   make clean  — удаляет venv и кэши Python (__pycache__).
#
# Все проверки выполняются непосредственно в Makefile — внешних скриптов
# для проверки окружения не требуется.

PYTHON ?= python

ifeq ($(OS),Windows_NT)
  VENV_PY := venv/Scripts/python.exe
  RUN_CMD := powershell -NoProfile -ExecutionPolicy Bypass -File scripts/deploy.ps1
else
  VENV_PY := venv/bin/python
  PYTHON  := python3
  RUN_CMD := bash scripts/deploy.sh
endif

# Python для проверок: если venv уже создан — проверяем его
ifeq ($(wildcard $(VENV_PY)),)
  CHECK_PY := $(PYTHON)
else
  CHECK_PY := $(VENV_PY)
endif

.PHONY: help deploy venv check clean

help:
	@echo "make help     - show this help information"
	@echo "make check    - check Python version (3.12 required)"
	@echo "make venv     - create a virtual environment and install dependencies"
	@echo "make deploy   - deployment + step-by-step model launch (Windows/Unix)"
	@echo "make clean    - delete venv and cache Python"

check:
	@$(CHECK_PY) -c "import sys; assert sys.version_info[:2] == (3, 12), 'Ошибка: требуется Python 3.12, установлен: ' + sys.version.split()[0]; print('Python: ' + sys.version.split()[0])"

venv:
	$(PYTHON) -m venv venv
	$(VENV_PY) -m pip install --upgrade pip
	$(VENV_PY) -m pip install -r requirements.txt
	@echo "Virtual environment is ready: $(VENV_PY)"

deploy:
	$(RUN_CMD)

clean:
	$(PYTHON) -c "import shutil; [shutil.rmtree(p, ignore_errors=True) for p in ['venv', '__pycache__', 'data/__pycache__', 'models/__pycache__', 'handlers/__pycache__']]"
