# test_evaluate.py

import pytest
import numpy as np
from evaluate import evaluate_predictions

def test_evaluate_predictions_returns_metrics():
    # Example true and predicted labels
    y_true = np.array([0, 1, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 0, 0, 1, 1])

    # Run evaluation
    results = evaluate_predictions(y_true, y_pred)

    # Check that results is a dictionary
    assert isinstance(results, dict)

    # Required keys
    for key in ["accuracy", "precision", "recall", "f1", "report"]:
        assert key in results

    # Types of returned values
    assert isinstance(results["accuracy"], float)
    assert isinstance(results["precision"], float)
    assert isinstance(results["recall"], float)
    assert isinstance(results["f1"], float)
    assert isinstance(results["report"], str)

    # Values within valid range
    assert 0.0 <= results["accuracy"] <= 1.0
    assert 0.0 <= results["precision"] <= 1.0
    assert 0.0 <= results["recall"] <= 1.0
    assert 0.0 <= results["f1"] <= 1.0
