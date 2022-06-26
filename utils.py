
from datasets import load_metric
from sklearn.metrics import classification_report
import numpy as np

metric = load_metric("seqeval")


def compute_metrics(p, label_list, return_logits=True):
    """Compute metric in traning phase

    Args:
        p (tuple): (predictions, labels) in idx type
        label_list (list[str]): label name list in str type

    Returns:
        dict: dictionary of metric
    """
    predictions, labels = p
    if return_logits:
        predictions = np.argmax(predictions, axis=2)

    y_pred = [p for prediction, label in zip(predictions, labels) for (
        p, l) in zip(prediction, label) if l not in (-100, 20)]
    y_true = [l for prediction, label in zip(predictions, labels) for (
        p, l) in zip(prediction, label) if l not in (-100, 20)]

    # report = classification_report(y_true, y_pred, labels=range(
    #     len(label_list) - 1), target_names=label_list[:-1], zero_division=0.0)
    # print(report)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(
        predictions=true_predictions, references=true_labels)

    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }
