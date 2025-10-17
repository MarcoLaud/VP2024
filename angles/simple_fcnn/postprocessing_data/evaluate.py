#!/usr/bin/env python3
"""
evaluate.py

Loads a Keras 3 (.keras) model and scalers, runs predictions on saved test split,
and generates:
- Parity plot (true vs predicted RT in seconds)
- Residuals histogram (true - predicted, seconds)
- Residuals vs each input feature (X, Y, Sina, Cosa)

Artifacts expected in --datadir:
- model.keras (or best_model.keras)
- x_scaler.pkl, y_scaler.pkl
- X_test.npy, y_test.npy

Saves figures to --datadir and appends MAE/RMSE/R^2 to metrics.json.
"""

import argparse
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
from sklearn.metrics import mean_absolute_error, r2_score

def choose_model_path(datadir: str, explicit: str | None) -> str:
    if explicit and explicit.strip():
        mp = os.path.join(datadir, explicit)
        if not os.path.exists(mp):
            raise FileNotFoundError(f"Model file not found: {mp}")
        if not mp.endswith(".keras"):
            raise ValueError(f"Expected a .keras model file, got: {mp}")
        return mp

    # Auto-pick .keras model
    for name in ("model.keras", "best_model.keras"):
        cand = os.path.join(datadir, name)
        if os.path.exists(cand):
            return cand

    raise FileNotFoundError(
        "No model file found. Looked for 'model.keras' or 'best_model.keras' in: "
        f"{datadir}"
    )

def main():
    parser = argparse.ArgumentParser(description="Evaluate FCNN model and generate plots (Keras 3 only).")
    parser.add_argument("--datadir", default="postprocessing data", help="Directory with artifacts")
    parser.add_argument("--model", default="", help="Optional model filename inside datadir (must end with .keras)")
    args = parser.parse_args()

    datadir = args.datadir
    os.makedirs(datadir, exist_ok=True)

    # Load scalers and test data
    with open(os.path.join(datadir, "x_scaler.pkl"), "rb") as f:
        x_scaler = pickle.load(f)
    with open(os.path.join(datadir, "y_scaler.pkl"), "rb") as f:
        y_scaler = pickle.load(f)

    X_test = np.load(os.path.join(datadir, "X_test.npy"))
    y_test = np.load(os.path.join(datadir, "y_test.npy"))
    if y_test.ndim == 1:
        y_test = y_test.reshape(-1, 1)

    # Prepare inputs
    X_test_s = x_scaler.transform(X_test)

    # Choose and load model (.keras only)
    model_path = choose_model_path(datadir, args.model)
    print(f"Loading model: {model_path}")
    model = tf.keras.models.load_model(model_path, compile=False)

    # Predict and inverse-transform to seconds
    y_pred_s = model.predict(X_test_s, verbose=0)
    y_pred = y_scaler.inverse_transform(y_pred_s)
    y_true = y_test

    # Metrics
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    r2 = float(r2_score(y_true, y_pred))
    print(f"MAE [s]: {mae:.6f}")
    print(f"RMSE [s]: {rmse:.6f}")
    print(f"R^2: {r2:.6f}")

    # Save/append metrics
    metrics_path = os.path.join(datadir, "metrics.json")
    try:
        with open(metrics_path, "r") as f:
            metrics_all = json.load(f)
    except Exception:
        metrics_all = {}
    metrics_all.update({"eval_mae_seconds": mae, "eval_rmse_seconds": rmse, "eval_r2": r2})
    with open(metrics_path, "w") as f:
        json.dump(metrics_all, f, indent=2)

    # Parity plot
    fig = plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.8)
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    plt.plot([lo, hi], [lo, hi])
    plt.xlabel("True RT [s]")
    plt.ylabel("Predicted RT [s]")
    plt.title(f"Parity Plot (R^2={r2:.3f})")
    plt.tight_layout()
    fig.savefig(os.path.join(datadir, "parity_plot.png"), dpi=200)
    plt.close(fig)

    # Residuals
    residuals = (y_true - y_pred).ravel()

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
    feature_names = ["X", "Y", "Sina", "Cosa"]
    for i, name in enumerate(feature_names):
        fig = plt.figure(figsize=(6, 5))
        plt.scatter(X_test[:, i], residuals, alpha=0.8)
        plt.axhline(0.0)
        plt.xlabel(name)
        plt.ylabel("Residual [s]")
        plt.title(f"Residuals vs {name}")
        plt.tight_layout()
        fig.savefig(os.path.join(datadir, f"residuals_vs_{name}.png"), dpi=200)
        plt.close(fig)

    print(f"Saved plots to {datadir}")

if __name__ == "__main__":
    main()
