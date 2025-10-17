#!/usr/bin/env python3
"""
fcnn.py

- Loads X.npy, y.npy from --datadir
- Stratified 70/10/20 split on y using quantile bins (no leakage)
- Standardizes X and y on TRAIN ONLY (StandardScaler)
- Trains an ensemble (n_models=5) with different seeds; test preds are averaged
- Saves to --datadir (default: "postprocessing data"):
    model_m*.keras, model.keras, x_scaler.pkl, y_scaler.pkl,
    X_test.npy, y_test.npy, metrics.json
"""

import argparse
import json
import os
import random
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

def build_model(input_dim: int) -> keras.Model:
    inp = keras.Input(shape=(input_dim,), name="features")
    x = layers.Dense(64, activation="relu")(inp)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1, activation="linear", name="rt")(x)
    model = keras.Model(inp, out, name="fcnn_rt")
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
    return model

def make_strat_bins(y: np.ndarray, bins: int = 5) -> np.ndarray:
    """Quantile-based bins for stratification in regression."""
    y1d = y.ravel()
    qs = np.linspace(0.0, 1.0, num=bins + 1)
    edges = np.quantile(y1d, qs)
    edges = np.unique(edges)  # guard against duplicates
    if len(edges) <= 2:
        return np.zeros_like(y1d, dtype=int)
    # np.digitize returns 0..(len(edges)-2)
    return np.digitize(y1d, edges[1:-1])

def main():
    ap = argparse.ArgumentParser(description="Train FCNN ensemble for RT prediction.")
    ap.add_argument("--datadir", default="postprocessing data", help="Directory containing X.npy and y.npy")
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--n_models", type=int, default=5)
    args = ap.parse_args()

    outdir = args.datadir
    os.makedirs(outdir, exist_ok=True)

    X = np.load(os.path.join(outdir, "X.npy"))
    y = np.load(os.path.join(outdir, "y.npy"))
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    n_samples, n_features = X.shape
    if n_features < 4:
        raise ValueError(f"Expected at least 4 features, got {n_features}. Did you run preprocessing.py?")

    # Stratified 70/10/20 split via quantile bins
    y_bins = make_strat_bins(y, bins=5)
    X_trainval, X_test, y_trainval, y_test, bins_trainval, _ = train_test_split(
        X, y, y_bins, test_size=0.20, random_state=SEED, stratify=y_bins
    )
    bins_train = make_strat_bins(y_trainval, bins=max(2, len(np.unique(bins_trainval))))
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.125, random_state=SEED, stratify=bins_train
    )

    # Standardize (fit on TRAIN only)
    x_scaler = StandardScaler()
    X_train_s = x_scaler.fit_transform(X_train)
    X_val_s   = x_scaler.transform(X_val)
    X_test_s  = x_scaler.transform(X_test)

    y_scaler = StandardScaler()
    y_train_s = y_scaler.fit_transform(y_train)
    y_val_s   = y_scaler.transform(y_val)
    y_test_s  = y_scaler.transform(y_test)  # for predictions only

    # Save scalers and raw test arrays for evaluation
    with open(os.path.join(outdir, "x_scaler.pkl"), "wb") as f:
        pickle.dump(x_scaler, f)
    with open(os.path.join(outdir, "y_scaler.pkl"), "wb") as f:
        pickle.dump(y_scaler, f)
    np.save(os.path.join(outdir, "X_test.npy"), X_test)
    np.save(os.path.join(outdir, "y_test.npy"), y_test)

    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=15, verbose=1),
    ]

    # Train ensemble
    all_preds = []
    for m in range(args.n_models):
        print(f"\n=== Training ensemble member {m+1}/{args.n_models} ===")
        tf.random.set_seed(SEED + m)
        np.random.seed(SEED + m)
        random.seed(SEED + m)

        model = build_model(input_dim=n_features)
        model.fit(
            X_train_s, y_train_s,
            validation_data=(X_val_s, y_val_s),
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

        # Predict test and invert scaling
        y_pred_test_s = model.predict(X_test_s, verbose=0)
        y_pred_test = y_scaler.inverse_transform(y_pred_test_s)
        all_preds.append(y_pred_test)

    # Ensemble average
    y_pred_ens = np.mean(np.stack(all_preds, axis=-1), axis=-1)

    # Metrics on test (seconds)
    mae = float(mean_absolute_error(y_test, y_pred_ens))
    rmse = float(np.sqrt(np.mean((y_test - y_pred_ens) ** 2)))
    r2 = float(r2_score(y_test, y_pred_ens))

    metrics = {
        "test_mae_seconds": mae,
        "test_rmse_seconds": rmse,
        "test_r2": r2,
        "n_models": int(args.n_models),
        "n_features": int(n_features),
        "n_samples": int(n_samples),
    }
    with open(os.path.join(outdir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("Test metrics (ensemble):", metrics)
    print(f"Artifacts saved to: {outdir}")

if __name__ == "__main__":
    main()
