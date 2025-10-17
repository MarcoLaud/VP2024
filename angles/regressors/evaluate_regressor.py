#!/usr/bin/env python3
"""
evaluate_regressor.py

Loads:
  - best_model.pkl (sklearn estimator, possibly with PolynomialFeatures/Scaler + TransformedTargetRegressor)
  - X_test.npy, y_test.npy
  - feature_names.json (if present)

Produces:
  - parity_plot.png (true vs predicted RT)
  - residuals_hist.png
  - residuals_vs_<feature>.png for each feature
Appends test metrics to metrics.json (idempotent).
"""

import argparse
import json
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score

def main():
    ap = argparse.ArgumentParser(description="Evaluate saved classical regressor and plot diagnostics.")
    ap.add_argument("--datadir", default="postprocessing data", help="Directory with artifacts")
    args = ap.parse_args()

    datadir = args.datadir

    with open(os.path.join(datadir, "best_model.pkl"), "rb") as f:
        model = pickle.load(f)

    X_test = np.load(os.path.join(datadir, "X_test.npy"))
    y_test = np.load(os.path.join(datadir, "y_test.npy")).reshape(-1, 1)

    # Feature names
    feat_path = os.path.join(datadir, "feature_names.json")
    if os.path.exists(feat_path):
        with open(feat_path, "r", encoding="utf-8") as f:
            feature_names = json.load(f)
    else:
        feature_names = [f"feat_{i}" for i in range(X_test.shape[1])]

    # Predict
    y_pred = model.predict(X_test).reshape(-1, 1)

    # Metrics
    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))
    r2 = float(r2_score(y_test, y_pred))
    print(f"MAE [s]: {mae:.6f}")
    print(f"RMSE [s]: {rmse:.6f}")
    print(f"R^2: {r2:.6f}")

    # Append to metrics.json
    metrics_path = os.path.join(datadir, "metrics.json")
    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics_all = json.load(f)
    except Exception:
        metrics_all = {}
    metrics_all.update({"eval_mae_seconds": mae, "eval_rmse_seconds": rmse, "eval_r2": r2})
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_all, f, indent=2)

    # Parity plot
    fig = plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.85)
    lo = float(min(y_test.min(), y_pred.min()))
    hi = float(max(y_test.max(), y_pred.max()))
    plt.plot([lo, hi], [lo, hi])
    plt.xlabel("True RT [s]")
    plt.ylabel("Predicted RT [s]")
    plt.title(f"Parity Plot (R²={r2:.3f})")
    plt.tight_layout()
    fig.savefig(os.path.join(datadir, "parity_plot.png"), dpi=200)
    plt.close(fig)

    # Residuals
    residuals = (y_test - y_pred).ravel()

    # Residuals histogram
    fig = plt.figure(figsize=(7, 5))
    plt.hist(residuals, bins=15)
    plt.xlabel("Residual [s] (true − predicted)")
    plt.ylabel("Count")
    plt.title("Residuals Histogram")
    plt.tight_layout()
    fig.savefig(os.path.join(datadir, "residuals_hist.png"), dpi=200)
    plt.close(fig)

    # Residuals vs each feature
    for i, name in enumerate(feature_names):
        fig = plt.figure(figsize=(6, 5))
        plt.scatter(X_test[:, i], residuals, alpha=0.85)
        plt.axhline(0.0)
        plt.xlabel(name)
        plt.ylabel("Residual [s]")
        plt.title(f"Residuals vs {name}")
        plt.tight_layout()
        safe_name = name.lower().replace("/", "_").replace(" ", "_")
        fig.savefig(os.path.join(datadir, f"residuals_vs_{safe_name}.png"), dpi=200)
        plt.close(fig)

    print(f"Saved plots to {datadir}")

if __name__ == "__main__":
    main()
