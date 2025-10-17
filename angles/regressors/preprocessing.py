#!/usr/bin/env python3
"""
preprocessing.py

Loads your dataset using numpy.genfromtxt with delimiter="," and skip_header=1.
Assumes columns in this order:
  0:X, 1:Y, 2:Sina, 3:Cosa, 4:RT

Builds compact engineered features:
  [X, Y, Sina, Cosa, X*Sina, X*Cosa, Y*Sina, Y*Cosa, R=sqrt(X^2+Y^2)]

Saves to --outdir (default: "postprocessing data"):
  - X.npy  (shape [N, 9])
  - y.npy  (shape [N, 1])  # RT in seconds
  - feature_names.json
"""

import argparse
import json
import os
import sys
import numpy as np

def main():
    ap = argparse.ArgumentParser(description="Prepare X.npy and y.npy with engineered features.")
    ap.add_argument("--input", required=True, help="Path to dataset file (CSV-like, commas, header on row 1)")
    ap.add_argument("--outdir", default="postprocessing data", help="Output directory")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Read numeric data (skip header line)
    data = np.genfromtxt(args.input, delimiter=",", skip_header=1)

    if data.ndim == 1:  # only one data row
        data = data.reshape(1, -1)

    if data.size == 0 or data.shape[1] < 5:
        print(f"Parsing error: got shape {data.shape}. Expected at least 5 columns.", file=sys.stderr)
        sys.exit(1)

    # Drop any rows with NaNs
    mask = np.isfinite(data).all(axis=1)
    data = data[mask]
    if data.shape[0] == 0:
        print("No valid rows after filtering NaNs.", file=sys.stderr)
        sys.exit(1)

    # Columns
    Xc  = data[:, 0].astype(float)  # X
    Yc  = data[:, 1].astype(float)  # Y
    SA  = data[:, 2].astype(float)  # Sina
    CA  = data[:, 3].astype(float)  # Cosa
    RT  = data[:, 4].astype(float)  # RT (seconds)

    # Engineered features (compact set)
    XS = Xc * SA
    XC = Xc * CA
    YS = Yc * SA
    YC = Yc * CA
    R  = np.sqrt(Xc**2 + Yc**2)

    feature_names = ["X", "Y", "Sina", "Cosa", "X_Sina", "X_Cosa", "Y_Sina", "Y_Cosa", "R"]
    X = np.column_stack([Xc, Yc, SA, CA, XS, XC, YS, YC, R])
    y = RT.reshape(-1, 1)

    # Save
    np.save(os.path.join(args.outdir, "X.npy"), X)
    np.save(os.path.join(args.outdir, "y.npy"), y)
    with open(os.path.join(args.outdir, "feature_names.json"), "w", encoding="utf-8") as f:
        json.dump(feature_names, f, indent=2)

    print(f"Saved X.npy (shape {X.shape}), y.npy (shape {y.shape}) to: {args.outdir}")
    print(f"Features: {feature_names}")

if __name__ == "__main__":
    main()
