# test_evaluate.py

import pytest
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from amla_at1_25608516.models.evaluate import evaluate_model

def test_evaluate_model_runs_and_returns_metrics():
    # Create a simple synthetic dataset
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    
    # Train a simple model
    model = LogisticRegression()
    model.fit(X, y)
    
    # Evaluate model
    results = evaluate_model(model, X, y)
    
    # Check that results is a dictionary
    assert isinstance(results, dict)
    
    # Check required keys are present
    for key in ["accuracy", "precision", "recall", "f1", "report"]:
        assert key in results
    
    # Check metric values are floats (except report which is str)
    assert isinstance(results["accuracy"], float)
    assert isinstance(results["precision"], float)
    assert isinstance(results["recall"], float)
    assert isinstance(results["f1"], float)
    assert isinstance(results["report"], str)
    
    # Sanity check: accuracy should be between 0 and 1
    assert 0.0 <= results["accuracy"] <= 1.0
