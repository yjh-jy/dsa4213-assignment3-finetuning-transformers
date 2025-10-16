# src/evaluate_utils.py
from sklearn.metrics import f1_score, accuracy_score
import numpy as np

def compute_metrics_fn(eval_pred):
    """
    eval_pred: (predictions, labels) from Trainer
    Return dictionary with accuracy and macro-f1 named "accuracy" and "f1"
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "f1": f1}
