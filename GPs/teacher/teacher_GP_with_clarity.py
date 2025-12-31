#!/usr/bin/env python3
"""
GP teacher for EDT bands with:
- deterministic panel orientation: normal aligned with radial line from emitter -> panel center
- enforced wall margin during TRAINING (default 2.0 m)
- uncertainty maps and active-learning suggestions WITHOUT sampling angle
- robust CSV reading (drops repeated header rows inside files)

Inputs:
  input.txt  columns: x,y,sina,cosa,dist_emitter,dist_receiver,dist_wall,dist_corner
  output.txt columns: EDT_62.5Hz,...,EDT_8kHz (may contain extra outputs)

Key point:
  We recompute sina/cosa, dist_* from (x,y) and fixed emitter/receiver positions to enforce consistency.
  The file's sina/cosa/dist_* can be present but are not relied upon for modeling.
"""

from __future__ import annotations

import argparse
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.exceptions import ConvergenceWarning


EDT_COLS = [
    "EDT_62.5Hz", "EDT_125Hz", "EDT_250Hz", "EDT_500Hz",
    "EDT_1kHz", "EDT_2kHz", "EDT_4kHz", "EDT_8kHz",
]


C50_COLS = [
    "C50_62.5Hz", "C50_125Hz", "C50_250Hz", "C50_500Hz",
    "C50_1kHz", "C50_2kHz", "C50_4kHz", "C50_8kHz",
]

C80_COLS = [
    "C80_62.5Hz", "C80_125Hz", "C80_250Hz", "C80_500Hz",
    "C80_1kHz", "C80_2kHz", "C80_4kHz", "C80_8kHz",
]

# "Clarity" outputs (C50 + C80)
CLARITY_COLS = C50_COLS + C80_COLS

# All outputs we expect in output.txt
ALL_OUTPUT_COLS = EDT_COLS + CLARITY_COLS

# Columns expected to exist in input file (for compatibility), but we recompute most of them.
IN_COLS = ["x", "y", "sina", "cosa", "dist_emitter", "dist_receiver", "dist_wall", "dist_corner"]


def try_import_scipy_least_squares():
    try:
        from scipy.optimize import least_squares  # type: ignore
        return least_squares
    except Exception:
        return None


def read_csv_numeric(path: str, required_cols: list[str]) -> pd.DataFrame:
    """
    Read CSV and coerce required_cols to numeric. Drops rows with NaNs in required_cols
    (handles repeated header rows inside file).
    """
    df = pd.read_csv(path)
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Missing column '{c}' in {path}")
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=required_cols).reset_index(drop=True)
    return df


def compute_dist_wall(x: np.ndarray, y: np.ndarray, L: float, l: float) -> np.ndarray:
    return np.minimum.reduce([x, L - x, y, l - y])


def compute_dist_corner(x: np.ndarray, y: np.ndarray, L: float, l: float) -> np.ndarray:
    d1 = np.sqrt((x - 0.0) ** 2 + (y - 0.0) ** 2)
    d2 = np.sqrt((x - L) ** 2 + (y - 0.0) ** 2)
    d3 = np.sqrt((x - 0.0) ** 2 + (y - l) ** 2)
    d4 = np.sqrt((x - L) ** 2 + (y - l) ** 2)
    return np.minimum.reduce([d1, d2, d3, d4])


def radial_angle_from_emitter(x: np.ndarray, y: np.ndarray, emitter_xy: tuple[float, float]) -> np.ndarray:
    ex, ey = emitter_xy
    return np.arctan2(y - ey, x - ex)


def estimate_point_from_distances(
    xy: np.ndarray,
    d: np.ndarray,
    bounds_xy: tuple[tuple[float, float], tuple[float, float]],
    seed: int = 0,
    n_random_starts: int = 25,
) -> tuple[float, float]:
    """
    Estimate fixed point p=(px,py) from observed distances d_i ≈ ||p - xy_i||.
    Uses scipy least_squares when available.
    """
    (xmin, xmax), (ymin, ymax) = bounds_xy
    rng = np.random.default_rng(seed)
    least_squares = try_import_scipy_least_squares()

    def residuals(p: np.ndarray) -> np.ndarray:
        px, py = p
        pred = np.sqrt((xy[:, 0] - px) ** 2 + (xy[:, 1] - py) ** 2)
        return pred - d

    if least_squares is None:
        # fallback random search
        best = None
        best_rmse = np.inf
        for _ in range(100_000):
            px = rng.uniform(xmin, xmax)
            py = rng.uniform(ymin, ymax)
            r = residuals(np.array([px, py]))
            rmse = float(np.sqrt(np.mean(r ** 2)))
            if rmse < best_rmse:
                best_rmse = rmse
                best = (px, py)
        assert best is not None
        return best

    best_x = None
    best_cost = np.inf

    starts = [np.array([(xmin + xmax) / 2, (ymin + ymax) / 2], dtype=float)]
    for _ in range(max(0, n_random_starts - 1)):
        starts.append(np.array([rng.uniform(xmin, xmax), rng.uniform(ymin, ymax)], dtype=float))

    for x0 in starts:
        res = least_squares(
            residuals,
            x0=x0,
            bounds=([xmin, ymin], [xmax, ymax]),
            max_nfev=20_000,
        )
        if float(res.cost) < best_cost:
            best_cost = float(res.cost)
            best_x = res.x

    assert best_x is not None
    return float(best_x[0]), float(best_x[1])


def build_kernel(n_features: int, ls_upper: float, noise_upper: float):
    return (
        ConstantKernel(1.0, (1e-3, 1e3))
        * Matern(
            length_scale=np.ones(n_features),
            length_scale_bounds=(1e-2, ls_upper),
            nu=2.5,
        )
        + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-8, noise_upper))
    )


def train_gps(
    Xs: np.ndarray,
    Ys: np.ndarray,
    kernel,
    n_restarts: int,
    suppress_warnings: bool,
) -> list[GaussianProcessRegressor]:
    gps = []
    for j in range(Ys.shape[1]):
        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=0.0,
            normalize_y=False,
            n_restarts_optimizer=n_restarts,
            random_state=0,
        )
        if suppress_warnings:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                gp.fit(Xs, Ys[:, j])
        else:
            gp.fit(Xs, Ys[:, j])
        gps.append(gp)
    return gps


def cv_report_with_uncertainty(
    X: np.ndarray,
    Y: np.ndarray,
    col_names: list[str],
    n_splits: int,
    kernel,
    n_restarts: int,
    suppress_warnings: bool,
    seed: int = 0,
) -> np.ndarray:
    """
    5-fold CV metrics + uncertainty calibration.
    Returns per-output sigma_scale (clipped) to calibrate predicted std:
      sigma_cal = sigma_pred * sigma_scale[j]
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    n_out = Y.shape[1]

    y_all = [[] for _ in range(n_out)]
    mu_all = [[] for _ in range(n_out)]
    sig_all = [[] for _ in range(n_out)]

    for tr_idx, te_idx in kf.split(X):
        Xtr, Xte = X[tr_idx], X[te_idx]
        Ytr, Yte = Y[tr_idx], Y[te_idx]

        xs = StandardScaler().fit(Xtr)
        ys = StandardScaler().fit(Ytr)

        Xtr_s = xs.transform(Xtr)
        Xte_s = xs.transform(Xte)
        Ytr_s = ys.transform(Ytr)

        gps = train_gps(Xtr_s, Ytr_s, kernel, n_restarts=n_restarts, suppress_warnings=suppress_warnings)

        for j, gp in enumerate(gps):
            mu_s, sig_s = gp.predict(Xte_s, return_std=True)
            mu = mu_s * ys.scale_[j] + ys.mean_[j]
            sig = sig_s * ys.scale_[j]

            y_all[j].append(Yte[:, j])
            mu_all[j].append(mu)
            sig_all[j].append(sig)

    y_all = [np.concatenate(v) for v in y_all]
    mu_all = [np.concatenate(v) for v in mu_all]
    sig_all = [np.concatenate(v) for v in sig_all]

    print("\n=== CV with uncertainty calibration (per output) ===")
    sigma_scale = np.ones(n_out, dtype=float)

    for j, name in enumerate(col_names):
        y = y_all[j]
        mu = mu_all[j]
        sig = np.maximum(sig_all[j], 1e-12)

        rmse = float(np.sqrt(mean_squared_error(y, mu)))
        r2 = float(r2_score(y, mu))

        z = (y - mu) / sig
        z_mean = float(np.mean(z))
        z_std = float(np.std(z))

        cov1 = float(np.mean(np.abs(y - mu) <= 1.0 * sig))
        cov2 = float(np.mean(np.abs(y - mu) <= 2.0 * sig))

        nll = float(np.mean(0.5 * np.log(2 * np.pi * sig**2) + 0.5 * ((y - mu) / sig) ** 2))

        mse = float(np.mean((y - mu) ** 2))
        mean_sig2 = float(np.mean(sig**2))
        scale = float(np.sqrt(mse / (mean_sig2 + 1e-18)))
        scale = float(np.clip(scale, 0.3, 3.0))
        sigma_scale[j] = scale

        print(
            f"{name:12s}  RMSE={rmse:.4f}  R2={r2:+.3f}  "
            f"z_mean={z_mean:+.2f} z_std={z_std:.2f}  "
            f"cov@1σ={cov1:.2f} cov@2σ={cov2:.2f}  "
            f"NLL={nll:.3f}  sigma_scale={scale:.2f}"
        )

    return sigma_scale

def _pick_holdout_indices(n_total: int, holdout_n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n_total)
    rng.shuffle(idx)
    hold_idx = np.sort(idx[:holdout_n])
    tr_idx = np.sort(idx[holdout_n:])
    return tr_idx, hold_idx


def evaluate_holdout_surrogate(
    X: np.ndarray,
    Y: np.ndarray,
    train_xy: np.ndarray,
    kernel,
    outdir: str,
    col_names: list[str],
    holdout_n: int = 0,
    holdout_frac: float = 0.0,
    seed: int = 0,
    n_restarts: int = 7,
    cv_splits: int = 5,
    suppress_warnings: bool = False,
    calibrate_uncertainty: bool = False,
    tag: str = "",
    split_indices: tuple[np.ndarray, np.ndarray] | None = None,
) -> None:
    """
    Hold out N (or frac) points, train on remaining, evaluate on hold-out.
    Saves:
      - outdir/holdout_metrics.csv
      - outdir/holdout_predictions.csv
      - outdir/holdout_parity_<col>.png (one per output)
    """
    suffix = f"_{tag}" if tag else ""
    tag_title = f" ({tag})" if tag else ""

    n_total = X.shape[0]
    if n_total < 10:
        print(f"\nHold-out eval{tag_title} skipped: too few points.")
        return

    if split_indices is not None:
        tr_idx, ho_idx = split_indices
    else:
        if holdout_n <= 0 and holdout_frac > 0:
            holdout_n = int(np.round(n_total * holdout_frac))

        if holdout_n <= 0:
            return

        # keep at least a few points for training
        holdout_n = int(np.clip(holdout_n, 1, max(1, n_total - 5)))
        if holdout_n >= n_total:
            print(f"\nHold-out eval{tag_title} skipped: holdout_n >= n_total.")
            return

        tr_idx, ho_idx = _pick_holdout_indices(n_total=n_total, holdout_n=holdout_n, seed=seed)

    Xtr, Ytr = X[tr_idx], Y[tr_idx]
    Xho, Yho = X[ho_idx], Y[ho_idx]
    xyho = train_xy[ho_idx]

    print(f"\n=== Hold-out evaluation{tag_title} ===")
    print(f"Using {len(Xtr)} train / {len(Xho)} hold-out points (seed={seed})")

    # Optional: uncertainty calibration on TRAIN subset only
    if calibrate_uncertainty:
        sigma_scale = cv_report_with_uncertainty(
            X=Xtr,
            Y=Ytr,
            col_names=col_names,
            n_splits=cv_splits,
            kernel=kernel,
            n_restarts=max(1, n_restarts // 2),
            suppress_warnings=suppress_warnings,
            seed=seed,
        )
    else:
        sigma_scale = np.ones(Y.shape[1], dtype=float)

    # Train model on TRAIN subset
    x_scaler = StandardScaler().fit(Xtr)
    y_scaler = StandardScaler().fit(Ytr)

    Xtr_s = x_scaler.transform(Xtr)
    Ytr_s = y_scaler.transform(Ytr)

    gps = train_gps(Xtr_s, Ytr_s, kernel, n_restarts=n_restarts, suppress_warnings=suppress_warnings)

    # Predict on HOLD-OUT
    Xho_s = x_scaler.transform(Xho)
    n_out = Y.shape[1]
    mu = np.zeros((len(Xho), n_out), dtype=float)
    sig = np.zeros((len(Xho), n_out), dtype=float)

    for j, gp in enumerate(gps):
        mu_s, sig_s = gp.predict(Xho_s, return_std=True)
        mu[:, j] = mu_s * y_scaler.scale_[j] + y_scaler.mean_[j]
        sig[:, j] = np.maximum(sig_s * y_scaler.scale_[j] * sigma_scale[j], 1e-12)

    # Metrics per output
    rows = []
    for j, name in enumerate(col_names):
        y = Yho[:, j]
        yhat = mu[:, j]
        rmse = float(np.sqrt(mean_squared_error(y, yhat)))
        mae = float(np.mean(np.abs(y - yhat)))
        r2 = float(r2_score(y, yhat))

        # If you enabled calibration, these are meaningful; otherwise they're still informative.
        cov1 = float(np.mean(np.abs(y - yhat) <= 1.0 * sig[:, j]))
        cov2 = float(np.mean(np.abs(y - yhat) <= 2.0 * sig[:, j]))
        nll = float(np.mean(0.5 * np.log(2 * np.pi * sig[:, j] ** 2) + 0.5 * ((y - yhat) / sig[:, j]) ** 2))

        rows.append({
            "output": name,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "cov_1sigma": cov1,
            "cov_2sigma": cov2,
            "nll": nll,
        })

    dfm = pd.DataFrame(rows)
    mpath = os.path.join(outdir, f"holdout_metrics{suffix}.csv")
    dfm.to_csv(mpath, index=False)
    print(f"Saved hold-out metrics: {mpath}")

    # Save per-point predictions
    dfp = pd.DataFrame({"x": xyho[:, 0], "y": xyho[:, 1]})
    for j, name in enumerate(col_names):
        dfp[f"{name}_true"] = Yho[:, j]
        dfp[f"{name}_pred"] = mu[:, j]
        dfp[f"{name}_std"] = sig[:, j]
    ppath = os.path.join(outdir, f"holdout_predictions{suffix}.csv")
    dfp.to_csv(ppath, index=False)
    print(f"Saved hold-out predictions: {ppath}")

    # --- Single parity figure with multiple panels (one per output) ---
    n_out = len(col_names)
    ncols = min(3, n_out)  # change to 2/4 if you prefer
    nrows = int(np.ceil(n_out / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 5.2 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for j, name in enumerate(col_names):
        ax = axes[j]
        y = Yho[:, j]
        yhat = mu[:, j]

        lo = float(min(y.min(), yhat.min()))
        hi = float(max(y.max(), yhat.max()))
        pad = 0.05 * (hi - lo + 1e-12)
        lo -= pad
        hi += pad

        rmse = dfm.loc[dfm["output"] == name, "rmse"].values[0]
        r2 = dfm.loc[dfm["output"] == name, "r2"].values[0]

        ax.scatter(y, yhat, s=22, edgecolors="k", linewidths=0.35)
        ax.plot([lo, hi], [lo, hi], linestyle="--")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{name}\nRMSE={rmse:.4f}  R2={r2:+.3f}")
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")

    # hide unused panels
    for k in range(n_out, len(axes)):
        axes[k].axis("off")

    fig.tight_layout()
    out_png = os.path.join(outdir, f"holdout_parity_all{suffix}.png")
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    print(f"Saved hold-out parity figure: {out_png}")


def make_features_from_xy(
    x: np.ndarray,
    y: np.ndarray,
    L: float,
    l: float,
    emitter_xy: tuple[float, float],
    receiver_xy: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Deterministic features for each (x,y):
      - angle a is radial from emitter to panel center
      - sina/cosa derived from a
      - dist_* derived geometrically
    Returns:
      X (n,8), a (n,)
    """
    ex, ey = emitter_xy
    rx, ry = receiver_xy

    a = np.arctan2(y - ey, x - ex)
    sina = np.sin(a)
    cosa = np.cos(a)

    dist_em = np.sqrt((x - ex) ** 2 + (y - ey) ** 2)
    dist_rc = np.sqrt((x - rx) ** 2 + (y - ry) ** 2)
    dist_w = compute_dist_wall(x, y, L, l)
    dist_c = compute_dist_corner(x, y, L, l)

    X = np.column_stack([x, y, sina, cosa, dist_em, dist_rc, dist_w, dist_c])
    return X, a


def compute_stds(
    gps: list[GaussianProcessRegressor],
    X_s: np.ndarray,
    y_scaler: StandardScaler,
    sigma_scale: np.ndarray,
) -> np.ndarray:
    n = X_s.shape[0]
    m = len(gps)
    stds = np.zeros((n, m), dtype=float)
    for j, gp in enumerate(gps):
        _, std_s = gp.predict(X_s, return_std=True)
        std = std_s * y_scaler.scale_[j]
        stds[:, j] = std * sigma_scale[j]
    return stds


def plot_uncertainty_heatmap(
    gps: list[GaussianProcessRegressor],
    x_scaler: StandardScaler,
    y_scaler: StandardScaler,
    sigma_scale: np.ndarray,
    L: float,
    l: float,
    emitter_xy: tuple[float, float],
    receiver_xy: tuple[float, float],
    train_xy: np.ndarray,
    wall_margin: float,
    out_png: str,
    nx: int,
    ny: int,
    band_index: int | None,
) -> None:
    xs = np.linspace(0.0, L, nx)
    ys = np.linspace(0.0, l, ny)
    XX, YY = np.meshgrid(xs, ys)

    xg = XX.ravel()
    yg = YY.ravel()

    dw = compute_dist_wall(xg, yg, L, l)
    valid = dw >= wall_margin

    UNC = np.full(xg.shape[0], np.nan, dtype=float)

    if np.any(valid):
        Xq, _ = make_features_from_xy(xg[valid], yg[valid], L, l, emitter_xy, receiver_xy)
        Xq_s = x_scaler.transform(Xq)
        stds = compute_stds(gps, Xq_s, y_scaler, sigma_scale)

        score = stds.mean(axis=1) if band_index is None else stds[:, band_index]
        UNC[valid] = score

    UNC2 = UNC.reshape(ny, nx)

    plt.figure(figsize=(12, 4.5))
    im = plt.imshow(
        UNC2,
        origin="lower",
        extent=[0, L, 0, l],
        aspect="auto",
    )
    cbar = plt.colorbar(im)
    cbar.set_label("Predictive σ (s)")

    plt.scatter(train_xy[:, 0], train_xy[:, 1], s=18, marker="o",
                edgecolors="k", linewidths=0.4, label="train")

    plt.scatter([emitter_xy[0]], [emitter_xy[1]], marker="*", s=200,
                edgecolors="k", linewidths=0.6, label="emitter")
    plt.scatter([receiver_xy[0]], [receiver_xy[1]], marker="X", s=140,
                edgecolors="k", linewidths=0.6, label="receiver")

    title_band = "mean over bands" if band_index is None else EDT_COLS[band_index]
    plt.title(f"GP uncertainty map | {title_band} | wall_margin={wall_margin:.1f} m (angle is radial)")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def suggest_points(
    gps: list[GaussianProcessRegressor],
    x_scaler: StandardScaler,
    y_scaler: StandardScaler,
    sigma_scale: np.ndarray,
    L: float,
    l: float,
    emitter_xy: tuple[float, float],
    receiver_xy: tuple[float, float],
    wall_margin: float,
    n_pool: int,
    top_k: int,
    minsep: float,
    band_index: int | None,
    seed: int = 0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, L, size=n_pool)
    y = rng.uniform(0.0, l, size=n_pool)

    # enforce wall margin
    dw = compute_dist_wall(x, y, L, l)
    keep = dw >= wall_margin
    x = x[keep]
    y = y[keep]

    Xp, a = make_features_from_xy(x, y, L, l, emitter_xy, receiver_xy)
    Xp_s = x_scaler.transform(Xp)
    stds = compute_stds(gps, Xp_s, y_scaler, sigma_scale)

    score = stds.mean(axis=1) if band_index is None else stds[:, band_index]

    df = pd.DataFrame({
        "x": x,
        "y": y,
        "a_rad": a,
        "sina": np.sin(a),
        "cosa": np.cos(a),
        "dist_emitter": Xp[:, 4],
        "dist_receiver": Xp[:, 5],
        "dist_wall": Xp[:, 6],
        "dist_corner": Xp[:, 7],
        "unc_score": score,
    })
    for j, col in enumerate(EDT_COLS[:stds.shape[1]]):
        df[f"sigma_{col}"] = stds[:, j]

    df = df.sort_values("unc_score", ascending=False).reset_index(drop=True)

    # pick top_k with minimum separation in (x,y)
    picked = []
    picked_xy = []

    for _, row in df.iterrows():
        xy = np.array([row["x"], row["y"]], dtype=float)
        if not picked_xy:
            picked.append(row)
            picked_xy.append(xy)
        else:
            dmin = min(float(np.linalg.norm(xy - p)) for p in picked_xy)
            if dmin >= minsep:
                picked.append(row)
                picked_xy.append(xy)
        if len(picked) >= top_k:
            break

    return pd.DataFrame(picked)

def generate_training_data_from_gp(
    gps: list[GaussianProcessRegressor],
    x_scaler: StandardScaler,
    y_scaler: StandardScaler,
    sigma_scale: np.ndarray,
    L: float,
    l: float,
    emitter_xy: tuple[float, float],
    receiver_xy: tuple[float, float],
    wall_margin: float,
    n_training: int,
    seed: int,
    outdir: str,) -> None:
    """
    Generate synthetic training data for a downstream NN (e.g. Grasshopper/Pug):
      - Sample (x,y) uniformly in the room, enforcing wall_margin
      - Build deterministic features X (same as the GP uses)
      - Use the trained GP teacher to predict labels Y (EDT bands)

    Writes 4 files into outdir:
      - X.npy, X.csv
      - Y.npy, Y.csv
    """
    if n_training <= 0:
        raise ValueError("n_training must be > 0")

    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(seed)

    # --- sample (x,y) with wall-margin constraint ---
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    n_kept = 0

    # Oversample in chunks until we have enough valid points.
    while n_kept < n_training:
        need = n_training - n_kept
        n_draw = int(max(1000, 2 * need))
        x = rng.uniform(0.0, L, size=n_draw)
        y = rng.uniform(0.0, l, size=n_draw)

        dw = compute_dist_wall(x, y, L, l)
        keep = dw >= wall_margin
        x = x[keep]
        y = y[keep]

        if x.size == 0:
            continue

        take = min(x.size, need)
        xs.append(x[:take])
        ys.append(y[:take])
        n_kept += take

    x = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)

    # --- build features and predict labels ---
    X_gen, _ = make_features_from_xy(x, y, L, l, emitter_xy, receiver_xy)
    X_gen_s = x_scaler.transform(X_gen)

    n_out = len(gps)
    Y_gen = np.zeros((n_training, n_out), dtype=float)

    for j, gp in enumerate(gps):
        mu_s = gp.predict(X_gen_s, return_std=False)
        # back-transform to physical units
        Y_gen[:, j] = mu_s * y_scaler.scale_[j] + y_scaler.mean_[j]

    # --- save 4 files ---
    x_npy = os.path.join(outdir, "X.npy")
    y_npy = os.path.join(outdir, "Y.npy")
    x_csv = os.path.join(outdir, "X.csv")
    y_csv = os.path.join(outdir, "Y.csv")

    np.save(x_npy, X_gen.astype(np.float32))
    np.save(y_npy, Y_gen.astype(np.float32))

    pd.DataFrame(X_gen, columns=IN_COLS).to_csv(x_csv, index=False)
    pd.DataFrame(Y_gen, columns=EDT_COLS[:n_out]).to_csv(y_csv, index=False)

    print(f"Saved NN training features: {x_npy} and {x_csv}")
    print(f"Saved NN training labels:   {y_npy} and {y_csv}")

def parse_band(band: str) -> int | None:
    band = band.strip().lower()
    if band in ("mean", "avg", "all"):
        return None
    # allow "2khz", "500hz", "62.5hz", or full column name
    if band.upper() in EDT_COLS:
        return EDT_COLS.index(band.upper())
    mapping = {c.lower().replace("edt_", ""): i for i, c in enumerate(EDT_COLS)}
    key = band.replace(" ", "")
    if key in mapping:
        return mapping[key]
    raise ValueError(f"Unknown band '{band}'. Use 'mean' or one of: {', '.join([c.replace('EDT_','') for c in EDT_COLS])}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="input.txt")
    ap.add_argument("--output", default="output.txt")
    ap.add_argument("--L", type=float, default=23.5)
    ap.add_argument("--l", type=float, default=8.6)

    ap.add_argument(
    "--command",
    choices=["train", "generate_training_data"],
    default="train",
    help="train: run full GP training + maps/suggestions. "
         "generate_training_data: fit the GP teacher and export X/Y datasets for NN training.",)
    ap.add_argument("--n_training", type=int, default=50_000,
                help="Number of (X,Y) samples to generate when --command=generate_training_data.")
    ap.add_argument("--gen_seed", type=int, default=0,
                help="RNG seed used when sampling points for --command=generate_training_data.")

    
    ap.add_argument("--holdout_n", type=int, default=0,
                    help="Hold out N random points (after wall-margin filtering) for surrogate evaluation. 0 disables.")
    ap.add_argument("--holdout_frac", type=float, default=0.0,
                    help="Hold out this fraction of points (0..1). Used only if holdout_n=0 and >0.")
    ap.add_argument("--holdout_seed", type=int, default=0,
                    help="Random seed for hold-out split.")
    ap.add_argument("--holdout_calibrate", action="store_true",
                    help="If set, runs a CV on the training subset to get sigma_scale for uncertainty metrics on hold-out.")
    
    ap.add_argument("--outdir", default="gp_teacher_outputs")
    ap.add_argument("--band", default="mean")
    ap.add_argument("--wall_margin", type=float, default=2.0, help="Enforced for training + maps + suggestions (m)")

    ap.add_argument("--n_restarts", type=int, default=7)
    ap.add_argument("--cv_splits", type=int, default=5)
    ap.add_argument("--suppress_warnings", action="store_true")

    ap.add_argument("--ls_upper", type=float, default=1e5)
    ap.add_argument("--noise_upper", type=float, default=1.0)

    ap.add_argument("--pool", type=int, default=80_000)
    ap.add_argument("--topk", type=int, default=25)
    ap.add_argument("--minsep", type=float, default=0.8)

    ap.add_argument("--grid_nx", type=int, default=100)
    ap.add_argument("--grid_ny", type=int, default=40)

    ap.add_argument("--emitter_xy", type=float, nargs=2, default=None, metavar=("EX", "EY"))
    ap.add_argument("--receiver_xy", type=float, nargs=2, default=None, metavar=("RX", "RY"))

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # Load and clean
    df_in_raw = read_csv_numeric(args.input, ["x", "y", "dist_emitter", "dist_receiver"])
    try:
        df_out = read_csv_numeric(args.output, ALL_OUTPUT_COLS)
        has_clarity = True
    except ValueError as e:
        print(f"\nWARNING: {e}")
        print("Clarity columns not found in output file. Proceeding with EDT only.")
        df_out = read_csv_numeric(args.output, EDT_COLS)
        has_clarity = False

    # align length
    n = min(len(df_in_raw), len(df_out))
    df_in_raw = df_in_raw.iloc[:n].reset_index(drop=True)
    df_out = df_out.iloc[:n].reset_index(drop=True)

    train_xy = df_in_raw[["x", "y"]].to_numpy(dtype=float)
    d_em = df_in_raw["dist_emitter"].to_numpy(dtype=float)
    d_rc = df_in_raw["dist_receiver"].to_numpy(dtype=float)

    # Determine emitter/receiver coords
    if args.emitter_xy is not None and args.receiver_xy is not None:
        emitter_xy = (float(args.emitter_xy[0]), float(args.emitter_xy[1]))
        receiver_xy = (float(args.receiver_xy[0]), float(args.receiver_xy[1]))
    else:
        bounds = ((0.0, args.L), (0.0, args.l))
        emitter_xy = estimate_point_from_distances(train_xy, d_em, bounds, seed=0)
        receiver_xy = estimate_point_from_distances(train_xy, d_rc, bounds, seed=1)

    # Report distance fit
    em_pred = np.sqrt((train_xy[:, 0] - emitter_xy[0]) ** 2 + (train_xy[:, 1] - emitter_xy[1]) ** 2)
    rc_pred = np.sqrt((train_xy[:, 0] - receiver_xy[0]) ** 2 + (train_xy[:, 1] - receiver_xy[1]) ** 2)
    em_rmse = float(np.sqrt(np.mean((em_pred - d_em) ** 2)))
    rc_rmse = float(np.sqrt(np.mean((rc_pred - d_rc) ** 2)))
    print(f"\nEmitter position:  ({emitter_xy[0]:.3f}, {emitter_xy[1]:.3f}) m   dist RMSE={em_rmse:.4f} m")
    print(f"Receiver position: ({receiver_xy[0]:.3f}, {receiver_xy[1]:.3f}) m   dist RMSE={rc_rmse:.4f} m")

    # Build deterministic features from xy (radial orientation)
    X_all, a_all = make_features_from_xy(train_xy[:, 0], train_xy[:, 1], args.L, args.l, emitter_xy, receiver_xy)
    Y_all = df_out[EDT_COLS].to_numpy(dtype=float)
    if has_clarity:
        Y_all_clarity = df_out[CLARITY_COLS].to_numpy(dtype=float)
    else:
        Y_all_clarity = np.zeros((len(df_out), 0), dtype=float)

    # Enforce wall margin for TRAINING
    dw = X_all[:, 6]
    keep = dw >= args.wall_margin
    dropped = int(np.sum(~keep))
    if dropped > 0:
        print(f"\nTraining filter: dropped {dropped}/{len(dw)} points with dist_wall < {args.wall_margin:.1f} m")
    X = X_all[keep]
    Y_edt = Y_all[keep]
    Y_clarity = Y_all_clarity[keep]
    # Keep 'Y' as the default target for the rest of the pipeline (EDT)
    Y = Y_edt
    train_xy_kept = train_xy[keep]

        # Kernel (needed for hold-out eval and CV)
    kernel = build_kernel(n_features=X.shape[1], ls_upper=args.ls_upper, noise_upper=args.noise_upper)

# --- Command: generate NN training data (X,Y) from the GP teacher ---
    if args.command == "generate_training_data":
        # Fit GPs on the full (filtered) dataset. (Means are unaffected by sigma calibration.)
        x_scaler = StandardScaler().fit(X)
        y_scaler = StandardScaler().fit(Y)
        Xs = x_scaler.transform(X)
        Ys = y_scaler.transform(Y)
    
        print("\nTraining GP teacher (for NN data generation)...")
        gps = train_gps(Xs, Ys, kernel, n_restarts=args.n_restarts, suppress_warnings=args.suppress_warnings)
    
        sigma_scale = np.ones(Y.shape[1], dtype=float)
        generate_training_data_from_gp(
            gps=gps,
            x_scaler=x_scaler,
            y_scaler=y_scaler,
            sigma_scale=sigma_scale,
            L=args.L,
            l=args.l,
            emitter_xy=emitter_xy,
            receiver_xy=receiver_xy,
            wall_margin=args.wall_margin,
            n_training=args.n_training,
            seed=args.gen_seed,
            outdir=args.outdir,
        )
        return

    # --- Hold-out evaluation (optional) ---
    # Use the SAME split for EDT and clarity to make the comparison fair.
    split_indices = None
    holdout_n_eff = args.holdout_n
    if holdout_n_eff <= 0 and args.holdout_frac > 0:
        holdout_n_eff = int(np.round(len(X) * args.holdout_frac))
    if holdout_n_eff > 0:
        holdout_n_eff = int(np.clip(holdout_n_eff, 1, max(1, len(X) - 5)))
        split_indices = _pick_holdout_indices(n_total=len(X), holdout_n=holdout_n_eff, seed=args.holdout_seed)

    evaluate_holdout_surrogate(
        X=X,
        Y=Y,
        train_xy=train_xy_kept,
        kernel=kernel,
        outdir=args.outdir,
        col_names=EDT_COLS,
        holdout_n=args.holdout_n,
        holdout_frac=args.holdout_frac,
        seed=args.holdout_seed,
        n_restarts=args.n_restarts,
        cv_splits=args.cv_splits,
        suppress_warnings=args.suppress_warnings,
        calibrate_uncertainty=args.holdout_calibrate,
        tag="",
        split_indices=split_indices,
    )

    # Same hold-out split, but with clarity targets (C50 + C80)
    if has_clarity and Y_clarity.shape[1] > 0:
        evaluate_holdout_surrogate(
            X=X,
            Y=Y_clarity,
            train_xy=train_xy_kept,
            kernel=kernel,
            outdir=args.outdir,
            col_names=CLARITY_COLS,
            holdout_n=args.holdout_n,
            holdout_frac=args.holdout_frac,
            seed=args.holdout_seed,
            n_restarts=args.n_restarts,
            cv_splits=args.cv_splits,
            suppress_warnings=args.suppress_warnings,
            calibrate_uncertainty=args.holdout_calibrate,
            tag="clarity",
            split_indices=split_indices,
        )
    if len(X) < 15:
        print("\nWARNING: very few training points after wall-margin filtering. Consider lowering wall_margin for training.")

    band_index = parse_band(args.band)

    sigma_scale = cv_report_with_uncertainty(
        X=X,
        Y=Y,
        col_names=EDT_COLS,
        n_splits=args.cv_splits,
        kernel=kernel,
        n_restarts=max(1, args.n_restarts // 2),
        suppress_warnings=args.suppress_warnings,
        seed=0,
    )

    # Train full GPs
    x_scaler = StandardScaler().fit(X)
    y_scaler = StandardScaler().fit(Y)

    Xs = x_scaler.transform(X)
    Ys = y_scaler.transform(Y)

    print("\nTraining GPs on full (filtered) dataset...")
    gps = train_gps(Xs, Ys, kernel, n_restarts=args.n_restarts, suppress_warnings=args.suppress_warnings)

    for j, gp in enumerate(gps):
        print(f"  {EDT_COLS[j]:12s} kernel: {gp.kernel_}")

    # Uncertainty map (no angle sampling)
    heat_png = os.path.join(args.outdir, "uncertainty_heatmap.png")
    plot_uncertainty_heatmap(
        gps=gps,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        sigma_scale=sigma_scale,
        L=args.L,
        l=args.l,
        emitter_xy=emitter_xy,
        receiver_xy=receiver_xy,
        train_xy=train_xy_kept,
        wall_margin=args.wall_margin,
        out_png=heat_png,
        nx=args.grid_nx,
        ny=args.grid_ny,
        band_index=band_index,
    )
    print(f"\nSaved uncertainty heatmap: {heat_png}")

    # Suggested next simulations (x,y only; a is derived radially)
    df_sug = suggest_points(
        gps=gps,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        sigma_scale=sigma_scale,
        L=args.L,
        l=args.l,
        emitter_xy=emitter_xy,
        receiver_xy=receiver_xy,
        wall_margin=args.wall_margin,
        n_pool=args.pool,
        top_k=args.topk,
        minsep=args.minsep,
        band_index=band_index,
        seed=0,
    )
    sug_csv = os.path.join(args.outdir, "suggested_next_simulations.csv")
    df_sug.to_csv(sug_csv, index=False)
    print(f"Saved suggested points: {sug_csv}")

    # Scatter plot
    plt.figure(figsize=(10, 4))
    plt.scatter(train_xy_kept[:, 0], train_xy_kept[:, 1], s=18, marker="o",
                edgecolors="k", linewidths=0.4, label="train (kept)")
    plt.scatter(df_sug["x"], df_sug["y"], s=60, marker="^",
                edgecolors="k", linewidths=0.5, label="suggested")
    plt.scatter([emitter_xy[0]], [emitter_xy[1]], marker="*", s=180,
                edgecolors="k", linewidths=0.6, label="emitter")
    plt.scatter([receiver_xy[0]], [receiver_xy[1]], marker="X", s=120,
                edgecolors="k", linewidths=0.6, label="receiver")

    plt.xlim(0, args.L)
    plt.ylim(0, args.l)
    plt.gca().set_aspect("equal", adjustable="box")
    title_band = "mean over bands" if band_index is None else EDT_COLS[band_index]
    plt.title(f"Training points and suggested next simulations | {title_band} | wall_margin={args.wall_margin:.1f} m")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.legend(loc="upper right")
    plt.tight_layout()

    sug_png = os.path.join(args.outdir, "suggestions_scatter.png")
    plt.savefig(sug_png, dpi=200)
    plt.close()
    print(f"Saved suggestions scatter plot: {sug_png}")

    # Metadata
    meta_txt = os.path.join(args.outdir, "meta.txt")
    with open(meta_txt, "w", encoding="utf-8") as f:
        f.write(f"L={args.L}\n")
        f.write(f"l={args.l}\n")
        f.write(f"wall_margin={args.wall_margin}\n")
        f.write(f"emitter_xy={emitter_xy}\n")
        f.write(f"receiver_xy={receiver_xy}\n")
        f.write(f"sigma_scale={sigma_scale.tolist()}\n")
        f.write(f"band={args.band}\n")
        f.write("angle_rule=radial_from_emitter\n")
    print(f"Saved metadata: {meta_txt}")


if __name__ == "__main__":
    main()
