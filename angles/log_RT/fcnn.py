#!/usr/bin/env python3
"""
fcnn.py

- Loads X.npy, y.npy from --datadir
- 70/10/20 split stratified on raw y (seconds)
- Target transform: y_log = log(y). Train in standardized y_log space.
- Linear baseline on (X_train_s -> y_train_log_s), then MLP on residuals.
- 5-model seed ensemble; average final predictions on test.
- Saves to --datadir:
    model_m*.keras        (residual MLPs)
    model.keras           (last residual MLP, convenience)
    linear.pkl            (coeffs/intercept for baseline, in y_log_s space)
    x_scaler.pkl, y_scaler.pkl
    config.json           (records target_transform="log")
    X_test.npy, y_test.npy
    metrics.json
"""

import argparse
import json
import os
import random
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

def make_strat_bins(y: np.ndarray, bins: int = 5) -> np.ndarray:
    """Quantile-based bins for stratification in regression."""
    y1d = y.ravel()
    qs = np.linspace(0.0, 1.0, num=bins + 1)
    edges = np.quantile(y1d, qs)
    edges = np.unique(edges)  # guard against duplicates
    if len(edges) <= 2:
        return np.zeros_like(y1d, dtype=int)
    return np.digitize(y1d, edges[1:-1])

def build_residual_mlp(input_dim: int) -> keras.Model:
    inp = keras.Input(shape=(input_dim,), name="features")
    x = layers.Dense(64, activation="relu")(inp)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1, activation="linear", name="residual")(x)
    model = keras.Model(inp, out, name="residual_mlp")
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
    return model

def main():
    ap = argparse.ArgumentParser(description="Train linear+residual-MLP ensemble for RT prediction.")
    ap.add_argument("--datadir", default="postprocessing data", help="Directory with X.npy/y.npy")
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--n_models", type=int, default=5)
    args = ap.parse_args()

    outdir = args.datadir
    os.makedirs(outdir, exist_ok=True)

    # Load data
    X = np.load(os.path.join(outdir, "X.npy"))
    y = np.load(os.path.join(outdir, "y.npy"))  # seconds
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    n_samples, n_features = X.shape

    # Stratified split on raw y (seconds)
    y_bins = make_strat_bins(y, bins=5)
    X_trainval, X_test, y_trainval, y_test, bins_trainval, _ = train_test_split(
        X, y, y_bins, test_size=0.20, random_state=SEED, stratify=y_bins
    )
    bins_train = make_strat_bins(y_trainval, bins=max(2, len(np.unique(bins_trainval))))
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.125, random_state=SEED, stratify=bins_train
    )

    # Scale X on TRAIN only
    x_scaler = StandardScaler()
    X_train_s = x_scaler.fit_transform(X_train)
    X_val_s   = x_scaler.transform(X_val)
    X_test_s  = x_scaler.transform(X_test)

    # Target transform: log (ensure positivity)
    if np.any(y_train <= 0):
        raise ValueError("Found non-positive RT values; log transform requires y>0.")
    y_train_log = np.log(y_train)
    y_val_log   = np.log(y_val)
    y_test_log  = np.log(y_test)

    # Scale y_log on TRAIN only
    y_scaler = StandardScaler()
    y_train_log_s = y_scaler.fit_transform(y_train_log)
    y_val_log_s   = y_scaler.transform(y_val_log)
    y_test_log_s  = y_scaler.transform(y_test_log)

    # Save scalers and test arrays for evaluation
    with open(os.path.join(outdir, "x_scaler.pkl"), "wb") as f:
        pickle.dump(x_scaler, f)
    with open(os.path.join(outdir, "y_scaler.pkl"), "wb") as f:
        pickle.dump(y_scaler, f)
    np.save(os.path.join(outdir, "X_test.npy"), X_test)
    np.save(os.path.join(outdir, "y_test.npy"), y_test)

    # Save config (so evaluate.py knows to invert log)
    config = {"target_transform": "log"}
    with open(os.path.join(outdir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # ---- Linear baseline on TRAIN (to y_log_s) ----
    lin = LinearRegression()
    lin.fit(X_train_s, y_train_log_s)
    # Save linear params
    linear_params = {"coef": lin.coef_.astype(float), "intercept": float(lin.intercept_)}
    with open(os.path.join(outdir, "linear.pkl"), "wb") as f:
        pickle.dump(linear_params, f)

    # Compute residuals in standardized log-space
    base_train = lin.predict(X_train_s).reshape(-1, 1)
    base_val   = lin.predict(X_val_s).reshape(-1, 1)
    base_test  = lin.predict(X_test_s).reshape(-1, 1)

    y_train_resid = y_train_log_s - base_train
    y_val_resid   = y_val_log_s   - base_val
    # y_test_resid is not used for training; kept for analysis if needed

    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=60, restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=20, min_lr=1e-5, verbose=1),
    ]

    # ---- Train residual-MLP ensemble ----
    all_preds_log_s = []
    for m in range(args.n_models):
        print(f"\n=== Training residual MLP member {m+1}/{args.n_models} ===")
        tf.random.set_seed(SEED + m)
        np.random.seed(SEED + m)
        random.seed(SEED + m)

        model = build_residual_mlp(input_dim=n_features)
        model.fit(
            X_train_s, y_train_resid,
            validation_data=(X_val_s, y_val_resid),
            epochs=args.epochs,
            batch_size=args.batch_size,
            verbose=0,
            callbacks=callbacks
        )
        # Save member and convenience copy
        member_path = os.path.join(outdir, f"model_m{m}.keras")
        model.save(member_path)
        if m == args.n_models - 1:
            model.save(os.path.join(outdir, "model.keras"))

        # Predict residuals on TEST
        resid_test_pred = model.predict(X_test_s, verbose=0)  # in y_log_s space
        # Combine with baseline (still in y_log_s space)
        y_pred_log_s = base_test + resid_test_pred
        all_preds_log_s.append(y_pred_log_s)

    # ---- Ensemble average (y_log_s space) -> invert scaling -> invert log ----
    y_pred_log_s_ens = np.mean(np.stack(all_preds_log_s, axis=-1), axis=-1)
    y_pred_log = y_scaler.inverse_transform(y_pred_log_s_ens)
    y_pred_seconds = np.exp(y_pred_log)  # invert log

    # ---- Metrics on test (seconds) ----
    mae = float(mean_absolute_error(y_test, y_pred_seconds))
    rmse = float(np.sqrt(np.mean((y_test - y_pred_seconds) ** 2)))
    r2 = float(r2_score(y_test, y_pred_seconds))

    metrics = {
        "test_mae_seconds": mae,
        "test_rmse_seconds": rmse,
        "test_r2": r2,
        "n_models": int(args.n_models),
        "n_features": int(n_features),
        "n_samples": int(n_samples),
        "target_transform": "log",
        "model": "linear + residual_mlp (ensemble)",
    }
    with open(os.path.join(outdir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("Test metrics (ensemble):", metrics)
    print(f"Artifacts saved to: {outdir}")

if __name__ == "__main__":
    main()
