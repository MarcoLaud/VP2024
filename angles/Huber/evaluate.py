#!/usr/bin/env python3
"""
evaluate.py

Averages ensemble members model_m*.keras if present; otherwise uses model.keras/best_model.keras.
Generates:
  - Parity plot (true vs predicted RT)
  - Residuals histogram (true - predicted)
  - Residuals vs each input feature (uses feature_names.json if present)
Appends metrics to metrics.json.
"""

import argparse
import glob
import json
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, r2_score

def main():
    ap = argparse.ArgumentParser(description="Evaluate FCNN ensemble and generate plots.")
    ap.add_argument("--datadir", default="postprocessing data", help="Directory with artifacts")
    ap.add_argument("--model", default="", help="Optional explicit .keras model filename inside datadir")
    args = ap.parse_args()

    datadir = args.datadir
    os.makedirs(datadir, exist_ok=True)

    # Load scalers and test arrays
    with open(os.path.join(datadir, "x_scaler.pkl"), "rb") as f:
        x_scaler = pickle.load(f)
    with open(os.path.join(datadir, "y_scaler.pkl"), "rb") as f:
        y_scaler = pickle.load(f)
    X_test = np.load(os.path.join(datadir, "X_test.npy"))
    y_test = np.load(os.path.join(datadir, "y_test.npy"))
    if y_test.ndim == 1:
        y_test = y_test.reshape(-1, 1)

    # Feature names
    feature_names_path = os.path.join(datadir, "feature_names.json")
    if os.path.exists(feature_names_path):
        with open(feature_names_path, "r", encoding="utf-8") as f:
            feature_names = json.load(f)
    else:
        feature_names = [f"feat_{i}" for i in range(X_test.shape[1])]

    # Scale X_test
    X_test_s = x_scaler.transform(X_test)

    # Choose models
    preds = []
    member_paths = sorted(glob.glob(os.path.join(datadir, "model_m*.keras")))
    if member_paths:
        print(f"Found {len(member_paths)} ensemble members; averaging predictions.")
        for p in member_paths:
            mdl = tf.keras.models.load_model(p, compile=False)
            y_pred_s = mdl.predict(X_test_s, verbose=0)
            preds.append(y_scaler.inverse_transform(y_pred_s))
    else:
        if args.model.strip():
            model_path = os.path.join(datadir, args.model)
            if not (model_path.endswith(".keras") and os.path.exists(model_path)):
                raise FileNotFoundError(f"Model not found or not .keras: {model_path}")
        else:
            candidates = [os.path.join(datadir, "model.keras"),
                          os.path.join(datadir, "best_model.keras")]
            model_path = next((p for p in candidates if os.path.exists(p)), "")
            if not model_path:
                raise FileNotFoundError("No model found (looked for model_m*.keras, model.keras, best_model.keras).")
        print(f"Loading single model: {model_path}")
        mdl = tf.keras.models.load_model(model_path, compile=False)
        y_pred_s = mdl.predict(X_test_s, verbose=0)
        preds.append(y_scaler.inverse_transform(y_pred_s))

    # Aggregate predictions
    y_pred = np.mean(np.stack(preds, axis=-1), axis=-1)
    y_true = y_test

    # Metrics
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    r2 = float(r2_score(y_true, y_pred))
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
    plt.scatter(y_true, y_pred, alpha=0.85)
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    plt.plot([lo, hi], [lo, hi])
    plt.xlabel("True RT [s]")
    plt.ylabel("Predicted RT [s]")
    plt.title(f"Parity Plot (R²={r2:.3f})")
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
