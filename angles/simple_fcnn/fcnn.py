#!/usr/bin/env python3
"""
fcnn.py

Builds and trains a simple FCNN (Keras/TensorFlow) to predict RT from [X, Y, Sina, Cosa].
- Loads raw X.npy and y.npy produced by preprocessing.py
- Splits data into train/val/test with 70/10/20
- Fits StandardScaler on train only (for both X and y) to avoid leakage
- Trains with early stopping and LR reduction
- Saves artifacts in "postprocessing data":
    model.h5, weights.h5, x_scaler.pkl, y_scaler.pkl,
    X_test.npy, y_test.npy (raw, for evaluation),
    history.csv, metrics.json
"""

import argparse
import json
import os
import random
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

import pickle

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

def build_model(input_dim: int) -> keras.Model:
    inp = keras.Input(shape=(input_dim,), name="features")
    x = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(inp)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(32, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
    out = layers.Dense(1, activation="linear", name="rt_seconds")(x)
    model = keras.Model(inputs=inp, outputs=out, name="fcnn_rt")
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
    return model

def main():
    parser = argparse.ArgumentParser(description="Train FCNN on absorber panel data.")
    parser.add_argument("--datadir", default="postprocessing data", help="Directory containing X.npy and y.npy")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    outdir = args.datadir
    os.makedirs(outdir, exist_ok=True)

    # Load raw arrays
    X = np.load(os.path.join(outdir, "X.npy"))
    y = np.load(os.path.join(outdir, "y.npy"))

    if X.ndim != 2 or X.shape[1] != 4:
        raise ValueError(f"Expected X shape (N,4), got {X.shape}")
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if y.shape[1] != 1:
        raise ValueError(f"Expected y shape (N,1), got {y.shape}")

    # 70/10/20 split: first take test 20%, then val 12.5% of remaining (0.1 overall)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.125, random_state=SEED
    )

    # Fit scalers on train only
    x_scaler = StandardScaler()
    X_train_s = x_scaler.fit_transform(X_train)
    X_val_s = x_scaler.transform(X_val)
    X_test_s = x_scaler.transform(X_test)

    y_scaler = StandardScaler()
    y_train_s = y_scaler.fit_transform(y_train)
    y_val_s = y_scaler.transform(y_val)
    y_test_s = y_scaler.transform(y_test)

    # Save raw test arrays for later evaluation
    np.save(os.path.join(outdir, "X_test.npy"), X_test)
    np.save(os.path.join(outdir, "y_test.npy"), y_test)

    # Save scalers
    with open(os.path.join(outdir, "x_scaler.pkl"), "wb") as f:
        pickle.dump(x_scaler, f)
    with open(os.path.join(outdir, "y_scaler.pkl"), "wb") as f:
        pickle.dump(y_scaler, f)

    # Build model
    model = build_model(input_dim=4)

    ckpt_best_model = os.path.join(outdir, "best_model.keras")

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=50,
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                          patience=15, verbose=1),
        keras.callbacks.ModelCheckpoint(ckpt_best_model, monitor="val_loss",
                                        save_best_only=True, verbose=1),
    ]

    # Train
    history = model.fit(
        X_train_s, y_train_s,
        validation_data=(X_val_s, y_val_s),
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=1,
        callbacks=callbacks
    )

    # Save training history
    pd.DataFrame(history.history).to_csv(os.path.join(outdir, "history.csv"), index=False)

    # Evaluate on test (inverse-transform to seconds for metrics)
    y_pred_test_s = model.predict(X_test_s, verbose=0)
    y_pred_test = y_scaler.inverse_transform(y_pred_test_s)
    y_true_test = y_test

    mae = float(mean_absolute_error(y_true_test, y_pred_test))
    rmse = float(np.sqrt(np.mean((y_true_test - y_pred_test) ** 2)))
    r2 = float(r2_score(y_true_test, y_pred_test))

    metrics = {"test_mae_seconds": mae, "test_rmse_seconds": rmse, "test_r2": r2}
    with open(os.path.join(outdir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("Test metrics:", metrics)

    # --- Final save (native Keras format) ---
    model.save(os.path.join(outdir, "model.keras"))
    print("Saved model.keras")

if __name__ == "__main__":
    main()
