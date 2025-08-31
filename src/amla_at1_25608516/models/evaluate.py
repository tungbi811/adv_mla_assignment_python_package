# evaluate.py

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

def evaluate_model(model, X_val, y_val, labels=["Not Drafted", "Drafted"]):
    """
    Evaluate a trained model on a validation dataset.

    Parameters
    ----------
    model : sklearn estimator
        Trained model (e.g., grid_search.best_estimator_).
    X_val : array-like
        Validation features.
    y_val : array-like
        Validation labels.
    labels : list, optional
        Class labels for confusion matrix display.

    Returns
    -------
    dict
        Dictionary with evaluation metrics.
    """

    # Predictions
    y_val_proba = model.predict_proba(X_val)[:, 1]
    y_val_pred = model.predict(X_val)

    # Metrics
    accuracy = accuracy_score(y_val, y_val_pred)
    precision = precision_score(y_val, y_val_pred)
    recall = recall_score(y_val, y_val_pred)
    f1 = f1_score(y_val, y_val_pred)
    roc_auc = roc_auc_score(y_val, y_val_proba)
    report = classification_report(y_val, y_val_pred)

    # Print results
    print("==== Evaluation Results ====")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    print("\nClassification Report:\n", report)

    # Confusion matrix
    cm = confusion_matrix(y_val, y_val_pred, labels=[0,1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues, values_format="d")
    plt.title("Confusion Matrix")
    plt.show()

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "report": report
    }
