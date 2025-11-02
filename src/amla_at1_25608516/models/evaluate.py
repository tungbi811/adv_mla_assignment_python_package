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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_regression_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    # Print results
    print("==== Evaluation Results (Regression) ====")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }


def evaluate_classification_model(model, X_val, y_val, labels=["Not Drafted", "Drafted"]):
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

def print_classifier_scores(y_preds, y_actuals, set_name=None):
    """Print the Accuracy and F1 score for the provided data.
    The value of the 'average' parameter for F1 score will be determined according to the number of distinct values of the target variable: 'binary' for bianry classification' or 'weighted' for multi-classs classification

    Parameters
    ----------
    y_preds : Numpy Array
        Predicted target
    y_actuals : Numpy Array
        Actual target
    set_name : str
        Name of the set to be printed
    Returns
    -------
    """
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import f1_score
    import pandas as pd

    average = 'weighted' if pd.Series(y_actuals).nunique() > 2 else 'binary'

    print(f"Accuracy {set_name}: {accuracy_score(y_actuals, y_preds)}")
    print(f"F1 {set_name}: {f1_score(y_actuals, y_preds, average=average)}")


def assess_classifier_set(model, features, target, set_name=''):
    """Save the predictions from a trained model on a given set and print its accuracy and F1 scores

    Parameters
    ----------
    model: sklearn.base.BaseEstimator
        Trained Sklearn model with set hyperparameters
    features : Numpy Array
        Features
    target : Numpy Array
        Target variable
    set_name : str
        Name of the set to be printed

    Returns
    -------
    """
    preds = model.predict(features)
    print_classifier_scores(y_preds=preds, y_actuals=target, set_name=set_name)