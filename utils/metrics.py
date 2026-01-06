import json
import pandas as pd

METRICS_FILE = "models/model_metrics.json"


def load_model_metrics():
    """
    Load saved model performance metrics
    Returns a Pandas DataFrame
    """

    with open(METRICS_FILE, "r") as f:
        data = json.load(f)

    rows = []
    for model_name, metrics in data.items():
        rows.append({
            "Model": model_name,
            "Accuracy": metrics["accuracy"],
            "Inference Time (sec)": metrics["inference_time"],
            "Macro F1": metrics["macro_f1"],
            "Weighted F1": metrics["weighted_f1"]
        })

    return pd.DataFrame(rows)


def load_classwise_report(model_name):
    """
    Load class-wise precision/recall/F1 for a model
    """

    with open(METRICS_FILE, "r") as f:
        data = json.load(f)

    return pd.DataFrame(data[model_name]["class_report"]).T
