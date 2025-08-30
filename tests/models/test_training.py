import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from scipy.stats import randint, uniform
from amla_at1_25608516.models.training import grid_search, random_search, load_model, save_model


@pytest.fixture
def iris_data():
    data = load_iris()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    return X, y


def test_grid_search_basic(iris_data, capsys):
    X, y = iris_data

    model = LogisticRegression(max_iter=200)
    param_grid = {
        "C": [0.1, 1.0],
        "solver": ["liblinear", "lbfgs"]
    }

    # Use train/test split for quick test
    X_train, X_val = X.iloc[:100], X.iloc[100:]
    y_train, y_val = y.iloc[:100], y.iloc[100:]

    results = grid_search(
        model,
        param_grid,
        X_train, y_train,
        X_val, y_val,
        scoring="accuracy"
    )

    # Should return all combinations
    assert len(results) == len(param_grid["C"]) * len(param_grid["solver"])
    # Best score should be >= worst score
    assert results[0]["score"] >= results[-1]["score"]

    # Check printing happened
    captured = capsys.readouterr()
    assert "[GridSearch]" in captured.out

def test_random_search_with_list(iris_data):
    X, y = iris_data
    model = LogisticRegression(max_iter=200, solver="liblinear")
    param_space = {
        "C": [0.01, 0.1, 1.0],
        "solver": ["liblinear"]
    }
    results = random_search(model, param_space, X.iloc[:100], y.iloc[:100],
                            X.iloc[100:], y.iloc[100:], n_iter=3)
    assert len(results) == 3
    assert results[0]["score"] >= results[-1]["score"]


def test_random_search_with_distributions(iris_data):
    X, y = iris_data
    model = LogisticRegression(max_iter=200, solver="liblinear")
    param_distributions = {
        "C": uniform(0.01, 10.0),  # continuous
        "solver": ["liblinear"]
    }
    results = random_search(model, param_distributions, X.iloc[:100], y.iloc[:100],
                            X.iloc[100:], y.iloc[100:], n_iter=3)
    assert len(results) == 3
    assert all("params" in r and "score" in r for r in results)
@pytest.fixture
def iris_model(tmp_path):
    """Train a simple LogisticRegression model on iris dataset."""
    data = load_iris()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)

    model = LogisticRegression(max_iter=200).fit(X, y)
    model_path = tmp_path / "logreg.pkl"
    return model, model_path

def test_save_and_load_model(iris_model):
    model, model_path = iris_model

    # Save model
    save_model(model, model_path)
    assert model_path.exists()

    # Load model
    loaded = load_model(model_path)

    # Both should predict the same outputs
    sample = [[5.1, 3.5, 1.4, 0.2]]
    orig_pred = model.predict(sample)
    loaded_pred = loaded.predict(sample)

    assert (orig_pred == loaded_pred).all()


def test_load_model_missing_file(tmp_path):
    fake_path = tmp_path / "missing.pkl"

    with pytest.raises(FileNotFoundError):
        _ = load_model(fake_path)