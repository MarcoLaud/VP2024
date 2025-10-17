#!/usr/bin/env python3
"""
preprocessing.py

Simple parser using numpy.genfromtxt with delimiter="," and skip_header=1.
Assumed column order in the file:
  0:X, 1:Y, 2:Sina, 3:Cosa, 4:RT

Engineered features:
  [X, Y, Sina, Cosa,
   X*Sina, X*Cosa, Y*Sina, Y*Cosa,
   R=sqrt(X^2+Y^2), R2=R^2,
   X^2, Y^2, X*Y,
   ALIGN = X*Cosa + Y*Sina,
   PERP  = -X*Sina + Y*Cosa]

Saves in --outdir (default: "postprocessing data"):
  - X.npy               shape (N, 15)
  - y.npy               shape (N, 1)
  - feature_names.json  names for plotting
"""

import argparse
import json
import os
import sys
import numpy as np

def main():
    ap = argparse.ArgumentParser(description="Preprocess dataset into X.npy/y.npy (engineered features).")
    ap.add_argument("--input", required=True, help="Path to dataset (comma-separated, header row present)")
    ap.add_argument("--outdir", default="postprocessing data", help="Output directory")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Read numeric data (skip header line)
    data = np.genfromtxt(args.input, delimiter=",", skip_header=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.size == 0 or data.shape[1] < 5:
        print(f"Parsing error: got shape {data.shape}. Expected at least 5 columns.", file=sys.stderr)
        sys.exit(1)

    # Drop rows with NaNs
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
    RT  = data[:, 4].astype(float)  # RT [s]

    # Engineered features
    XS = Xc * SA
    XC = Xc * CA
    YS = Yc * SA
    YC = Yc * CA
    R  = np.sqrt(Xc**2 + Yc**2)
    R2 = R**2
    X2 = Xc**2
    Y2 = Yc**2
    XY = Xc * Yc
    ALIGN = Xc * CA + Yc * SA
    PERP  = -Xc * SA + Yc * CA

    feature_names = [
        "X", "Y", "Sina", "Cosa",
        "X_Sina", "X_Cosa", "Y_Sina", "Y_Cosa",
        "R", "R2",
        "X2", "Y2", "XY",
        "ALIGN", "PERP"
    ]
    X = np.column_stack([Xc, Yc, SA, CA, XS, XC, YS, YC, R, R2, X2, Y2, XY, ALIGN, PERP])
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
