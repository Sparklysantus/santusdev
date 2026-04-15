import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

from src.config import TARGET_COLUMN


def evaluate_model(model, test_df: pd.DataFrame, feature_cols: list[str]) -> dict[str, float]:
    X_test = test_df[feature_cols]
    y_test = test_df[TARGET_COLUMN]

    y_prob = model.predict_proba(X_test)
    y_pred = model.predict(X_test)
    classes = model.named_steps["clf"].classes_

    metrics = {
        "log_loss": float(log_loss(y_test, y_prob, labels=classes)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "brier_score_multiclass": float(_multiclass_brier(y_test, y_prob, classes)),
    }
    return metrics


def _multiclass_brier(y_true: pd.Series, y_prob: np.ndarray, classes: np.ndarray) -> float:
    y_one_hot = np.zeros_like(y_prob)
    class_to_idx = {label: idx for idx, label in enumerate(classes)}

    for i, label in enumerate(y_true):
        y_one_hot[i, class_to_idx[label]] = 1.0

    return np.mean(np.sum((y_prob - y_one_hot) ** 2, axis=1))
