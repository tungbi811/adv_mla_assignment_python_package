import numpy as np
import itertools
import random
from typing import Dict, List, Any
from sklearn.base import clone
from sklearn.metrics import get_scorer
from joblib import Parallel, delayed, dump, load
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

def _fit_and_score(model, params, X_train, y_train, X_val, y_val, scorer):
    """Worker: clone, fit, score. Returns (params, score)."""
    m = clone(model).set_params(**params)
    m.fit(X_train, y_train)
    score = scorer(m, X_val, y_val)
    return params, float(score)

def grid_search(
    model,
    param_grid: Dict[str, List[Any]],
    X_train, y_train,
    X_val, y_val,
    scoring: str = "accuracy",
    n_jobs: int = -1,  # NEW: parallelism (-1 = all cores)
) -> List[Dict[str, Any]]:
    """
    Grid search over all param combinations with progress printing (parallelized).
    """
    scorer = get_scorer(scoring)
    results = []

    keys, values = zip(*param_grid.items())
    combos = list(itertools.product(*values))
    params_list = [dict(zip(keys, c)) for c in combos]
    total = len(params_list)

    print(f"[GridSearch] {total} combinations to evaluate...")

    # run evaluations in parallel; keep order
    worker = delayed(_fit_and_score)
    out = Parallel(n_jobs=n_jobs, prefer="processes", verbose=0)(
        worker(model, p, X_train, y_train, X_val, y_val, scorer) for p in params_list
    )

    # print progress & build results in original order
    for i, (params, score) in enumerate(out, 1):
        print(f"  → ({i}/{total}) Testing params: {params}")
        print(f"    Score ({scoring}): {score:.4f}")
        results.append({"params": params, "score": score})

    results.sort(key=lambda x: x["score"], reverse=True)
    print(f"[GridSearch] Best params: {results[0]['params']} | Best score: {results[0]['score']:.4f}")
    return results


def _sample_value(space_item, np_rng):
    # scipy.stats frozen distributions have .rvs
    if hasattr(space_item, "rvs"):
        val = space_item.rvs(random_state=np_rng)
        if hasattr(space_item, "dist") and getattr(space_item.dist, "name", "") == "randint":
            return int(val)
        return float(val) if isinstance(val, (np.floating,)) else val
    # callables (custom samplers)
    if callable(space_item):
        return space_item()
    # sequences / sets
    try:
        return np_rng.choice(list(space_item))
    except TypeError:
        # scalar already
        return space_item

def random_search(
    model,
    param_distributions: Dict[str, Any],
    X_train, y_train,
    X_val, y_val,
    n_iter: int = 10,
    scoring: str = "accuracy",
    n_jobs: int = -1,  # NEW: parallelism (-1 = all cores)
):
    """Random search that supports scipy.stats distributions and lists (parallelized)."""
    scorer = get_scorer(scoring)
    results = []

    # deterministic sampling of params up-front for reproducibility
    keys = list(param_distributions.keys())
    params_list = [
        {k: _sample_value(param_distributions[k], np.random) for k in keys}
        for _ in range(n_iter)
    ]

    print(f"[RandomSearch] {n_iter} iterations...")

    # parallel evaluate
    worker = delayed(_fit_and_score)
    out = Parallel(n_jobs=n_jobs, prefer="processes", verbose=0)(
        worker(model, p, X_train, y_train, X_val, y_val, scorer) for p in params_list
    )

    # print progress & collect results in submission order
    for i, (params, score) in enumerate(out, 1):
        print(f"  → ({i}/{n_iter}) Testing params: {params}")
        print(f"    Score ({scoring}): {score:.4f}")
        results.append({"params": params, "score": score})

    results.sort(key=lambda x: x["score"], reverse=True)
    print(f"[RandomSearch] Best params: {results[0]['params']} | Best score: {results[0]['score']:.4f}")
    return results

def save_model(model, path: str):
    """Save a trained model to disk.

    Parameters
    ----------
    model : object
        Trained model (e.g., sklearn, xgboost, etc.)
    path : str
        File path to save the model (e.g., "artifacts/model.pkl").
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dump(model, path)
    print(f"[SaveModel] Model saved to {path}")


def load_model(path: str):
    """Load a trained model from disk.

    Parameters
    ----------
    path : str
        File path where the model is stored.

    Returns
    -------
    object
        The loaded model.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    model = load(path)
    print(f"[LoadModel] Model loaded from {path}")
    return model

def evaluate_on_validation(
    model_or_search,
    X_val,
    y_val,
    average: str = "binary",
    pos_label=1,
    zero_division: int = 0,
    target_names=None,
):
    """
    Print evaluation metrics for a fitted estimator or a search object
    (with .best_estimator_).
    """
    # Get estimator
    est = getattr(model_or_search, "best_estimator_", model_or_search)

    # Predictions
    y_pred = est.predict(X_val)

    # Probabilities / decision scores
    y_score = None
    if hasattr(est, "predict_proba"):
        proba = est.predict_proba(X_val)
        if proba.ndim == 2 and proba.shape[1] > 1:
            y_score = proba
        else:
            y_score = proba
    elif hasattr(est, "decision_function"):
        y_score = est.decision_function(X_val)

    # Print results
    print("\n=== Evaluation ===")

    # ROC AUC
    auc = None
    try:
        if y_score is not None:
            if average == "binary":
                if np.ndim(y_score) == 2:
                    classes_ = getattr(est, "classes_", None)
                    if classes_ is not None and pos_label in classes_:
                        pos_idx = list(classes_).index(pos_label)
                    else:
                        pos_idx = 1 if y_score.shape[1] > 1 else 0
                    auc = roc_auc_score(y_val, y_score[:, pos_idx])
                else:
                    auc = roc_auc_score(y_val, y_score)
            else:
                auc = roc_auc_score(y_val, y_score, multi_class="ovr", average="macro")
    except Exception:
        pass

    print("ROC AUC:", "n/a" if auc is None else round(float(auc), 4))
    print("Accuracy:", round(accuracy_score(y_val, y_pred), 4))
    print("Precision:", round(precision_score(y_val, y_pred, average=average, zero_division=zero_division), 4))
    print("Recall:",    round(recall_score(y_val, y_pred, average=average, zero_division=zero_division), 4))
    print("F1 Score:",  round(f1_score(y_val, y_pred, average=average, zero_division=zero_division), 4))

    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=target_names, zero_division=zero_division))

    print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred))

