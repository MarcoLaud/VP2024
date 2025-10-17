#!/usr/bin/env python3
"""
train_regressors.py

- Loads X.npy, y.npy from --datadir
- Stratified 70/10/20 split on y (seconds) using quantile bins
- Tries multiple regressors, each under two target transforms:
    * identity
    * log (TransformedTargetRegressor with func=log, inverse=exp)
  Regressors:
    - LinearRegression (degree-1)
    - RidgeCV with PolynomialFeatures(degree=2)
    - ElasticNetCV with PolynomialFeatures(degree=2)
    - SVR (RBF) with tuned grid
    - KernelRidge (RBF) with tuned grid
- Selects the model/transform with the best **validation RMSE**
- Refits best on **train+val**, then evaluates on **test**
- Saves:
    best_model.pkl   (final fitted estimator incl. preprocessing & target transform)
    X_test.npy, y_test.npy
    feature_names.json (already from preprocessing)
    metrics.json

Usage:
  python train_regressors.py --datadir "postprocessing data"
"""

import argparse
import json
import os
import pickle
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression, RidgeCV, ElasticNetCV
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.base import clone  # <-- important fix

SEED = 42

def make_strat_bins(y: np.ndarray, bins: int = 5) -> np.ndarray:
    y1d = y.ravel()
    qs = np.linspace(0.0, 1.0, num=bins + 1)
    edges = np.quantile(y1d, qs)
    edges = np.unique(edges)
    if len(edges) <= 2:
        return np.zeros_like(y1d, dtype=int)
    return np.digitize(y1d, edges[1:-1])

def identity(x): return x

def build_candidates():
    """Return a list of (name, estimator, is_search, param_grid_or_None)."""
    # Plain linear (degree-1)
    lin = Pipeline([
        ("scaler", StandardScaler()),
        ("lin", LinearRegression())
    ])

    # Poly(2) + RidgeCV
    ridge2 = Pipeline([
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("scaler", StandardScaler()),
        ("ridge", RidgeCV(alphas=np.logspace(-6, 3, 30), cv=5))
    ])

    # Poly(2) + ElasticNetCV
    enet2 = Pipeline([
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("scaler", StandardScaler()),
        ("enet", ElasticNetCV(l1_ratio=np.linspace(0.1, 0.9, 9),
                              alphas=np.logspace(-6, 1, 20),
                              max_iter=20000,
                              cv=5))
    ])

    # SVR (RBF) tuned
    svr_base = Pipeline([
        ("scaler", StandardScaler()),
        ("svr", SVR(kernel="rbf"))
    ])
    svr_grid = {
        "regressor__svr__C": [1.0, 10.0, 100.0],
        "regressor__svr__gamma": ["scale", 0.1, 0.01],
        "regressor__svr__epsilon": [0.01, 0.05, 0.1],
    }

    # Kernel Ridge (RBF) tuned
    krr_base = Pipeline([
        ("scaler", StandardScaler()),
        ("krr", KernelRidge(kernel="rbf"))
    ])
    krr_grid = {
        "regressor__krr__alpha": np.logspace(-3, 1, 5),
        "regressor__krr__gamma": np.logspace(-2, 1, 6),
    }

    return [
        ("lin", lin, False, None),
        ("ridge2", ridge2, False, None),
        ("enet2", enet2, False, None),
        ("svr_rbf", svr_base, True, svr_grid),
        ("krr_rbf", krr_base, True, krr_grid),
    ]

def wrap_transform(estimator, transform: str):
    if transform == "log":
        return TransformedTargetRegressor(regressor=estimator, func=np.log, inverse_func=np.exp)
    else:
        return TransformedTargetRegressor(regressor=estimator, func=identity, inverse_func=identity)

def main():
    ap = argparse.ArgumentParser(description="Train classical regressors with model selection.")
    ap.add_argument("--datadir", default="postprocessing data", help="Directory with X.npy/y.npy")
    args = ap.parse_args()

    outdir = args.datadir
    os.makedirs(outdir, exist_ok=True)

    # Load arrays
    X = np.load(os.path.join(outdir, "X.npy"))
    y = np.load(os.path.join(outdir, "y.npy")).ravel()  # shape (N,)

    # 70/10/20 stratified split
    y_bins = make_strat_bins(y, bins=5)
    X_trainval, X_test, y_trainval, y_test, bins_trainval, _ = train_test_split(
        X, y, y_bins, test_size=0.20, random_state=SEED, stratify=y_bins
    )
    bins_train = make_strat_bins(y_trainval, bins=max(2, len(np.unique(bins_trainval))))
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.125, random_state=SEED, stratify=bins_train
    )

    candidates = build_candidates()
    transforms = ["identity", "log"]

    best = None  # (val_rmse, name, transform, fitted_estimator)
    results = []

    for name, base_est, is_search, grid in candidates:
        for transform in transforms:
            # Skip log if y has non-positive
            if transform == "log" and (np.any(y_train <= 0) or np.any(y_val <= 0)):
                continue

            est = wrap_transform(base_est, transform)

            if is_search:
                # GridSearch over the *wrapped* estimator (scoring in original target space)
                gs = GridSearchCV(
                    est, grid, cv=5, scoring="neg_mean_squared_error", n_jobs=-1, refit=True
                )
                gs.fit(X_train, y_train)
                fitted = gs.best_estimator_
            else:
                fitted = est.fit(X_train, y_train)

            # Validate
            y_val_pred = fitted.predict(X_val)
            val_rmse = float(np.sqrt(mean_squared_error(y_val, y_val_pred)))
            val_mae = float(mean_absolute_error(y_val, y_val_pred))
            val_r2 = float(r2_score(y_val, y_val_pred))
            results.append({
                "name": name, "transform": transform,
                "val_rmse": val_rmse, "val_mae": val_mae, "val_r2": val_r2
            })

            if (best is None) or (val_rmse < best[0]):
                best = (val_rmse, name, transform, fitted)

    if best is None:
        raise RuntimeError("Model selection failed.")

    _, best_name, best_transform, best_estimator = best
    print(f"Selected: {best_name} with transform={best_transform} (by validation RMSE)")

    # ---- FIX: clone the selected estimator and refit on TRAIN+VAL ----
    X_trval = np.vstack([X_train, X_val])
    y_trval = np.concatenate([y_train, y_val])
    final_est = clone(best_estimator)   # <--- clone avoids bad kwargs like regressor__memory
    final_est.fit(X_trval, y_trval)

    # Evaluate on TEST
    y_test_pred = final_est.predict(X_test)
    test_rmse = float(np.sqrt(mean_squared_error(y_test, y_test_pred)))
    test_mae = float(mean_absolute_error(y_test, y_test_pred))
    test_r2 = float(r2_score(y_test, y_test_pred))

    metrics = {
        "selected_model": best_name,
        "target_transform": best_transform,
        "val_results": results,
        "test_mae_seconds": test_mae,
        "test_rmse_seconds": test_rmse,
        "test_r2": test_r2,
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
    }
    with open(os.path.join(outdir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Persist test arrays (for plotting) and the model
    np.save(os.path.join(outdir, "X_test.npy"), X_test)
    np.save(os.path.join(outdir, "y_test.npy"), y_test)
    with open(os.path.join(outdir, "best_model.pkl"), "wb") as f:
        pickle.dump(final_est, f)

    print("Test metrics:", {"MAE": test_mae, "RMSE": test_rmse, "R2": test_r2})
    print(f"Saved best_model.pkl and metrics.json to: {outdir}")

if __name__ == "__main__":
    main()
