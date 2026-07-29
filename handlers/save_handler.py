import os
import json
from  keras import Model

def save_results(
    results: dict,
    model_name: str,
    results_dir: str = 'results'
    ):
    """_summary_

    Args:
        results (dict): _description_
        model_name (str): _description_
        results_dir (str, optional): _description_. Defaults to 'results'.
    """
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, f"{model_name}_results.json"), "w") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"✅ Сохранено: {results_dir}/{model_name}_results.json")

def save_model(
    model: Model,
    model_name: str,
    results_dir: str = 'results'
    ):
    """_summary_

    Args:
        model (Model): _description_
        model_name (str): _description_
        results_dir (str, optional): _description_. Defaults to 'results'.
    """
    os.makedirs(results_dir, exist_ok=True)
    model.save(os.path.join(results_dir, f"{model_name}.keras"))
    print(f"✅ Сохранено: {results_dir}/{model_name}.keras")