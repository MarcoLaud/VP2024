#!/usr/bin/env python3
"""
evaluate.py

- Loads X_test/y_test, scalers, config.json, linear.pkl
- If model_m*.keras present: averages residual-MLP members; otherwise uses model.keras/best_model.keras
- Combines linear baseline + residual predictions in standardized log-space,
  then inverts y scaling and log to report in seconds.
- Generates:
    * Parity plot (true vs predicted RT)
    * Residuals histogram (true - predicted)
    * Residuals vs each input feature (uses feature_names.json if available)
- Appends metrics to metrics.json
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
    ap = argparse.ArgumentParser(description="Evaluate linear+residual-MLP ensemble and generate plots.")
    ap.add_argument("--datadir", default="postprocessing data", help="Directory with artifacts")
    ap.add_argument("--model", default="", help="Optional explicit .keras model filename inside datadir")
    args = ap.parse_args()

    datadir = args.datadir
    os.makedirs(datadir, exist_ok=True)

    # Config & scalers
    cfg_path = os.path.join(datadir, "config.json")
    cfg = {"target_transform": "log"}
    if os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    target_transform = cfg.get("target_transform", "log")

    with open(os.path.join(datadir, "x_scaler.pkl"), "rb") as f:
        x_scaler = pickle.load(f)
    with open(os.path.join(datadir, "y_scaler.pkl"), "rb") as f:
        y_scaler = pickle.load(f)

    # Linear baseline params (in y_log_s space)
    with open(os.path.join(datadir, "linear.pkl"), "rb") as f:
        lin = pickle.load(f)
    coef = np.array(lin["coef"]).reshape(1, -1)  # (1, n_features) or (1, n_features)
    intercept = float(lin["intercept"])

    # Data
    X_test = np.load(os.path.join(datadir, "X_test.npy"))
    y_test = np.load(os.path.join(datadir, "y_test.npy"))
    if y_test.ndim == 1:
        y_test = y_test.reshape(-1, 1)

    # Feature names
    feat_path = os.path.join(datadir, "feature_names.json")
    if os.path.exists(feat_path):
        with open(feat_path, "r", encoding="utf-8") as f:
            feature_names = json.load(f)
    else:
        feature_names = [f"feat_{i}" for i in range(X_test.shape[1])]

    # Scale X
    X_test_s = x_scaler.transform(X_test)

    # Linear baseline prediction in standardized log-space
    # (X_test_s @ coef.T + intercept)
    base_test = (X_test_s @ coef.T) + intercept  # shape (N,1)

    # Residual models
    preds_log_s = []
    member_paths = sorted(glob.glob(os.path.join(datadir, "model_m*.keras")))
    if member_paths:
        print(f"Found {len(member_paths)} residual-MLP members; averaging predictions.")
        for p in member_paths:
            mdl = tf.keras.models.load_model(p, compile=False)
            resid = mdl.predict(X_test_s, verbose=0)  # residual in y_log_s
            preds_log_s.append(base_test + resid)
    else:
        # Fallback to single model (or just linear if missing)
        if args.model.strip():
            model_path = os.path.join(datadir, args.model)
            if not (model_path.endswith(".keras") and os.path.exists(model_path)):
                raise FileNotFoundError(f"Model not found or not .keras: {model_path}")
        else:
            candidates = [os.path.join(datadir, "model.keras"),
                          os.path.join(datadir, "best_model.keras")]
            model_path = next((p for p in candidates if os.path.exists(p)), "")
        if model_path:
            print(f"Loading single residual model: {model_path}")
            mdl = tf.keras.models.load_model(model_path, compile=False)
            resid = mdl.predict(X_test_s, verbose=0)
            preds_log_s.append(base_test + resid)
        else:
            print("No residual model found; using linear baseline only.")
            preds_log_s.append(base_test)

    # Average in standardized log-space, then invert
    y_pred_log_s = np.mean(np.stack(preds_log_s, axis=-1), axis=-1)
    y_pred_log = y_scaler.inverse_transform(y_pred_log_s)

    if target_transform == "log":
        y_pred = np.exp(y_pred_log)
    else:
        y_pred = y_pred_log  # just in case we switch off the transform in future

    y_true = y_test  # seconds

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
