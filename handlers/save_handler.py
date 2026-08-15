"""Сохранение результатов обучения модели в JSON."""

import json
import os

from handlers.training_result import TrainingResult


def save_results(
    result: TrainingResult,
    results_dir: str = "results"
):
    """
    Args:
        result (TrainingResult): результаты обучения модели.
        results_dir (str, optional): директория для сохранения результатов. Defaults to 'results'.
    """
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f"{result.model_name}_results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=4, ensure_ascii=False)
    print(f"Сохранено: {result.model_name}_results.json")
