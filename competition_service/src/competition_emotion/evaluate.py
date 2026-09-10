from __future__ import annotations

import numpy as np


def _validate_labels(labels: tuple[str, ...]) -> None:
    if not labels:
        raise ValueError("labels must not be empty")
    if any(not isinstance(label, str) or not label.strip() for label in labels):
        raise ValueError("labels must contain nonblank strings")
    if len(set(labels)) != len(labels):
        raise ValueError("labels must be unique")


def _validate_numeric_matrix(matrix: np.ndarray, name: str) -> None:
    if not np.issubdtype(matrix.dtype, np.number):
        raise ValueError(f"{name} must be numeric")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must contain only finite values")


def metric_report(
    y_true: np.ndarray, scores: np.ndarray, labels: tuple[str, ...]
) -> dict:
    """Report Top-1 metrics, keeping multilabel rows out of strict scoring."""
    if y_true.ndim != 2 or scores.ndim != 2 or y_true.shape != scores.shape:
        raise ValueError("y_true and scores must be two-dimensional arrays with matching shapes")
    _validate_labels(labels)
    if y_true.shape[1] != len(labels):
        raise ValueError("labels must match the number of score columns")
    _validate_numeric_matrix(y_true, "y_true")
    _validate_numeric_matrix(scores, "scores")
    if not np.all((y_true == 0) | (y_true == 1)):
        raise ValueError("y_true must be binary multi-hot values in {0, 1}")

    predicted = np.argmax(scores, axis=1)
    positive = y_true > 0
    row_indices = np.arange(y_true.shape[0])
    any_positive = positive[row_indices, predicted]
    any_positive_top1_accuracy = float(any_positive.mean()) if len(y_true) else 0.0

    singleton_mask = positive.sum(axis=1) == 1
    per_label_recall = {label: 0.0 for label in labels}
    confusion = np.zeros((len(labels), len(labels)), dtype=int)
    if not singleton_mask.any():
        return {
            "any_positive_top1_accuracy": any_positive_top1_accuracy,
            "strict_top1_accuracy": None,
            "macro_recall": 0.0,
            "per_label_recall": per_label_recall,
            "confusion_matrix": confusion.tolist(),
        }

    singleton_true = np.argmax(y_true[singleton_mask], axis=1)
    singleton_predicted = predicted[singleton_mask]
    strict_top1_accuracy = float((singleton_true == singleton_predicted).mean())
    for actual, predicted_label in zip(singleton_true, singleton_predicted, strict=True):
        confusion[actual, predicted_label] += 1

    recall_values = []
    for index, label in enumerate(labels):
        actual_label = singleton_true == index
        if actual_label.any():
            recall = float((singleton_predicted[actual_label] == index).mean())
        else:
            recall = 0.0
        per_label_recall[label] = recall
        recall_values.append(recall)

    return {
        "any_positive_top1_accuracy": any_positive_top1_accuracy,
        "strict_top1_accuracy": strict_top1_accuracy,
        "macro_recall": float(np.mean(recall_values)),
        "per_label_recall": per_label_recall,
        "confusion_matrix": confusion.tolist(),
    }
