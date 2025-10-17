#!/usr/bin/env python3
"""
preprocessing.py

Loads the dataset text file, parses robustly (commas or semicolons),
extracts features X = [X, Y, Sina, Cosa] and target y = RT, and saves
raw numpy arrays to an output directory (default: "postprocessing data").

No normalization or splitting is performed here to avoid data leakage.
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd

REQUIRED_COLS = ["X", "Y", "Sina", "Cosa", "RT"]

def load_table(path: str) -> pd.DataFrame:
    # Robust read: auto-detect mixed delimiters, strip spaces
    try:
        df = pd.read_csv(path, engine="python", sep=r"[;,]", skip_blank_lines=True)
    except Exception as e:
        print(f"Failed to read file: {e}", file=sys.stderr)
        raise

    # Trim whitespace in column names
    df.columns = [c.strip() for c in df.columns]

    # If first row is header-like but misparsed, try to fix
    # Ensure required columns are present
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        # Try with default sep=None (sniff)
        try:
            df = pd.read_csv(path, engine="python", sep=None)
            df.columns = [c.strip() for c in df.columns]
            missing = [c for c in REQUIRED_COLS if c not in df.columns]
        except Exception:
            pass

    if missing:
        raise ValueError(f"Missing columns: {missing}. Found columns: {list(df.columns)}")

    return df

def main():
    parser = argparse.ArgumentParser(description="Preprocess dataset into X.npy and y.npy without normalization.")
    parser.add_argument("--input", required=True, help="Path to dataset file (txt/csv)")
    parser.add_argument("--outdir", default="postprocessing data", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = load_table(args.input)

    # Select and ensure numeric dtype
    X = df[["X", "Y", "Sina", "Cosa"]].astype(float).to_numpy()
    y = df[["RT"]].astype(float).to_numpy()  # shape (N,1)

    np.save(os.path.join(args.outdir, "X.npy"), X)
    np.save(os.path.join(args.outdir, "y.npy"), y)

    print(f"Saved: {os.path.join(args.outdir, 'X.npy')} with shape {X.shape}")
    print(f"Saved: {os.path.join(args.outdir, 'y.npy')} with shape {y.shape}")

if __name__ == "__main__":
    main()
