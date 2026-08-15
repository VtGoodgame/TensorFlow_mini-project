# TensorFlow mini-project

Учебный мини-проект по машинному обучению: от реализации нейросети с нуля на NumPy до обучения свёрточных моделей на реальных датасетах.

## Содержание

- [Описание](#описание)
- [Быстрый старт](#быстрый-старт)
  - [Windows](#windows)
  - [Linux / macOS / Git Bash](#linux--macos--git-bash)
- [Архитектура проекта](#архитектура-проекта)
- [Модели](#модели)
- [Сравнение результатов обучения](#сравнение-результатов-обучения)
  - [Iris (Keras)](#iris-keras)
  - [MNIST (PyTorch)](#mnist-pytorch)
- [Makefile](#makefile)
- [Развёртывание](#развёртывание)
- [Запуск вручную](#запуск-вручную)
- [Формат результатов](#формат-результатов)
- [Зависимости](#зависимости)

## Описание

Проект знакомит с ключевыми этапами ML-пайплайна:

- **Hands-on реализация** прямой и обратной связей, функции потерь и обновления весов с нуля на NumPy;
- **Классификация ирисов** (3 класса) полносвязными сетями на TensorFlow/Keras;
- **Распознавание рукописных цифр MNIST** (10 классов) полносвязной и свёрточной сетями на PyTorch;
- **Автоматическое сохранение метрик** результатов обучения в JSON для сравнения.

## Быстрый старт

### Windows

1. Установите Python 3.12 ([python.org](https://www.python.org/downloads/)); при установке отметьте пункт **Add python.exe to PATH**.
2. Из корня проекта (`TensorFlow_mini-project/`) выполните:

```powershell
make check    # проверка версии Python (3.12)
make venv     # создание виртуального окружения и установка зависимостей
make deploy   # развёртывание + пошаговый запуск всех моделей
```

Если `make` не установлен, то же самое делает PowerShell-скрипт напрямую:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/deploy.ps1
```

### Linux / macOS / Git Bash

1. Установите Python 3.12 (Ubuntu/Debian: `sudo apt install python3.12`; macOS: `brew install python@3.12`).
2. Из корня проекта (`TensorFlow_mini-project/`) выполните:

```bash
make check
make venv
make deploy
```

Если `make` не установлен — используйте bash-скрипт:

```bash
chmod +x scripts/deploy.sh
./scripts/deploy.sh
```

> В Git Bash под Windows команды `make` работают так же, как на Unix.

## Архитектура проекта

```
TensorFlow_mini-project/
│
├── data/                        # Загрузка и подготовка датасетов
│   ├── __init__.py
│   ├── iris.py                  # Iris (4 признака, 3 класса) + train/test split + нормализация
│   └── mnist.py                 # MNIST 784 (нормировка /255, разделение на train/test)
│
├── models/                      # Определение и обучение моделей
│   ├── __init__.py
│   ├── baseline_model.py        # Iris + Keras: полносвязная сеть (базовая)
│   ├── improved_model.py        # Iris + Keras: полносвязная сеть с Dropout
│   ├── fully_conn_model.py      # MNIST + PyTorch: полносвязная сеть 784→256→128→64→10
│   └── convNet_model.py         # MNIST + PyTorch: свёрточная сеть CNN
│
├── handlers/                    # Вспомогательные обработчики
│   ├── save_handler.py          # Сохранение результатов обучения модели в JSON
│   └── training_result.py       # Единая схема результатов обучения (TrainingResult)
│
├── scripts/                     # Скрипты развёртывания
│   ├── deploy.ps1               # Развёртывание + пошаговый запуск моделей (Windows)
│   └── deploy.sh                # Развёртывание + пошаговый запуск моделей (Linux/macOS/Git Bash)
│
├── results/                     # Генерируемые результаты обучения (JSON, в git не хранится)
├── Makefile                     # Единая точка входа: make help / check / venv / deploy
├── requirements.txt             # Зависимости проекта
└── README.md
```

## Модели

| Модель | Фреймворк | Датасет | Архитектура | Params | Test accuracy |
|--------|-----------|---------|-------------|--------|---------------|
| `baseline_model` | Keras | Iris | 4 → 8 → 3 (ReLU, Softmax) | 67 | 0.967 |
| `improved_model` | Keras | Iris | 4 → 16 → 8 → 3 + Dropout | 243 | 1.000 |
| `fully_conn_model` | PyTorch | MNIST | 784 → 256 → 128 → 64 → 10 | 242 762 | 0.980 |
| `convNet_model` | PyTorch | MNIST | Conv2D 32 → 64 + FC 128 → 10 | 421 642 | 0.992 |

## Сравнение результатов обучения

Данные взяты из файлов `results/<model_name>_results.json`, названия файлов соответствуют названиям моделей.

> **О воспроизводимости.** Результаты обучения недетерминированы: в моделях не зафиксированы seed'ы случайной инициализации, поэтому из-за случайных начальных весов, перемешивания данных (shuffle) и Dropout метрики плавают от запуска к запуску. Особенно заметно это на маленьком датасете Iris (тестовая выборка — всего 30 объектов): test accuracy может отличаться на несколько сотых. Числа в таблицах соответствуют единичному прогону (дата и время обучения — в `results/<model_name>_results.json`), поэтому при повторном обучении небольшие отклонения — это норма, а не баг.

### Iris (Keras)

| Модель | Train acc | Train loss | Val acc | Val loss | Test acc | Test loss | Params |
|--------|-----------|------------|---------|----------|----------|-----------|--------|
| `baseline_model` | 0.900 | 0.331 | 0.917 | 0.343 | 0.967 | 0.279 | 67 |
| `improved_model` | 0.975 | 0.160 | 1.000 | 0.141 | 1.000 | 0.132 | 243 |

Вывод: `improved_model` (больше нейронов + Dropout) заметно обходит `baseline_model` по всем метрикам и показывает лучшую обобщающую способность.

### MNIST (PyTorch)

| Модель | Train acc | Train loss | Test acc | Test loss | Params | Эпохи |
|--------|-----------|------------|----------|-----------|--------|-------|
| `fully_conn_model` | 0.996 | 0.012 | 0.980 | 0.084 | 242 762 | 10 |
| `convNet_model` | 0.998 | 0.005 | 0.992 | 0.030 | 421 642 | 10 |

Вывод: обе сети уверенно распознают цифры, но свёрточная (`convNet_model`) обходит полносвязную (`fully_conn_model`) примерно на 1.2 п.п. при большем числе параметров.

> Примечание: модели обучались на разных датасетах (Iris — 3 класса, MNIST — 10 классов), поэтому напрямую сравнивать точность между таблицами некорректно — сравнение уместно только внутри каждого датасета.

## Makefile

Единая точка входа — `make` автоматически выбирает нужный скрипт по платформе (Windows → `scripts/deploy.ps1`, Unix → `scripts/deploy.sh`).

```bash
make help      # справка по командам
make check     # проверить версию Python (3.12)
make venv      # создать виртуальное окружение и установить зависимости
make deploy    # развёртывание + пошаговый запуск всех моделей
make clean     # удалить venv и кэши Python
```

`make check` проверяет только версию Python (требуется 3.12) и завершается с ошибкой при несоответствии. Проверка выполняется непосредственно в Makefile — отдельного скрипта не требуется.

## Развёртывание

Скрипты `scripts/deploy.ps1` (Windows) и `scripts/deploy.sh` (Linux / macOS / Git Bash) автоматически:

1. **Проверяют версию Python** — требуется 3.12;
2. **Создают виртуальное окружение** `venv`;
3. **Устанавливают зависимости** из `requirements.txt`;
4. **Запускают модели по одной**: следующая модель стартует только после появления файла результата в `results/` — одна модель за раз, без избыточной нагрузки на компьютер. Модели, результат которых уже сохранён, пропускаются.

```powershell
# Windows
.\scripts\deploy.ps1                        # развёртывание + все модели
.\scripts\deploy.ps1 -SetupOnly             # только шаги 1–3, без запуска моделей
.\scripts\deploy.ps1 -Models baseline_model,improved_model
.\scripts\deploy.ps1 -Force                 # переобучить даже те модели, где результат уже есть
```

```bash
# Linux / macOS / Git Bash
chmod +x scripts/deploy.sh
./scripts/deploy.sh                         # развёртывание + все модели
./scripts/deploy.sh --setup-only            # только шаги 1–3
./scripts/deploy.sh --models baseline_model improved_model
./scripts/deploy.sh --force
```

## Запуск вручную

> Выполняется из корня проекта. Формат `python -m models.<файл>` — без расширения `.py` (модуль, а не путь к файлу).

```bash
# Обучение моделей на Iris (Keras) — быстро, секунды
python -m models.baseline_model
python -m models.improved_model

# Обучение моделей на MNIST (PyTorch) — долго: при первом запуске MNIST скачивается (~120 МБ)
python -m models.fully_conn_model
python -m models.convNet_model
```

Каждая модель пишет свой результат в `results/<model_name>_results.json` по единой схеме — см. [Формат результатов](#формат-результатов).

## Формат результатов

Все модели сохраняют метрики по единой схеме (`handlers/training_result.py` → класс `TrainingResult`), набор ключей одинаков для всех моделей. Неприменимые для конкретного фреймворка поля записываются как `null`.

| Ключ | Описание |
|------|----------|
| `model_name` | Имя модели |
| `framework` | Фреймворк: `keras` или `pytorch` |
| `timestamp` | Дата и время обучения |
| `train_accuracy` / `train_loss` | Метрики на обучающей выборке |
| `test_accuracy` / `test_loss` | Метрики на тестовой выборке |
| `val_accuracy` / `val_loss` | Метрики на валидации (Keras) |
| `total_params` | Число обучаемых параметров |
| `epochs` | Число эпох (PyTorch) |
| `optimizer` | Оптимизатор (PyTorch) |
| `learning_rate` | Скорость обучения (PyTorch) |

Файлы результатов лежат в `results/`, директория в git не хранится (`.gitignore`).

## Зависимости

Указаны в `requirements.txt`:

- `tensorflow` — включает в себя Keras
- `torch` (CPU-сборка)
- `numpy`
- `scikit-learn`
