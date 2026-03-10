"""
Generate effectiveness graphs after training.

Uses eval_data/ (separate from training data) for unbiased evaluation.

Produces:
  1. Training vs Validation loss curve
  2. Predicted vs Actual scatter plot  (from eval_data/)
  3. Error distribution histogram       (from eval_data/)
  4. Real-time volume curves             (from eval_data/)

Usage:  python evaluate.py
"""

import json
import glob
import math
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path
from config import Config
from model import SpirometryLSTM


def load_model(cfg, device):
    model = SpirometryLSTM(
        input_size=cfg.input_size,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        fc_size=cfg.fc_size,
        dropout=0.0,
    ).to(device)
    model.load_state_dict(torch.load(cfg.model_save_path, map_location=device))
    model.eval()
    return model


def load_eval_csvs(eval_dir):
    """Load all CSVs from eval_data/, return list of (filename, flow, delta_t)."""
    csv_files = sorted(Path(eval_dir).glob("*.csv"))
    recordings = []
    for f in csv_files:
        df = pd.read_csv(f)
        delta_t = df["delta_t_ms"].values / 1000.0
        flow    = df["processed"].values
        recordings.append((f.name, flow, delta_t))
    return recordings


def evaluate_final_volumes(model, recordings, cfg, device):
    """
    Run model on each eval CSV, return list of (predicted_final_vol, target).
    """
    preds, labels = [], []
    for name, flow_all, delta_t_all in recordings:
        mask    = flow_all > 0.0
        flow    = flow_all[mask]
        delta_t = delta_t_all[mask]

        if len(flow) < 2:
            continue

        features = np.stack([flow, delta_t, flow * delta_t], axis=1)
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
        lengths = torch.tensor([len(features)]).to(device)

        with torch.no_grad():
            volumes = model(x, lengths)         # (1, T, 1)
        final_vol = volumes[0, -1, 0].cpu().item()

        preds.append(final_vol)
        labels.append(cfg.target_volume)

    return preds, labels


# 
#  PLOT FUNCTIONS

def compute_metrics(preds, labels, tolerances=[0.05, 0.1, 0.2, 0.5]):

    preds  = np.array(preds)
    labels = np.array(labels)
    errors = preds - labels
    abs_errors = np.abs(errors)

    n = len(preds)
    mae  = np.mean(abs_errors)
    rmse = np.sqrt(np.mean(errors ** 2))
    mse  = np.mean(errors ** 2)

    # Mean Absolute Percentage Error
    mape = np.mean(abs_errors / np.abs(labels)) * 100.0

    # R² (coefficient of determination)
    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((labels - np.mean(labels)) ** 2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else float("nan")

    # Max error (worst case)
    max_err = np.max(abs_errors)

    # Median absolute error (robust to outliers)
    median_ae = np.median(abs_errors)

    # Bias (positive = overestimating, negative = underestimating)
    bias = np.mean(errors)

    # Within-tolerance accuracy: % of predictions within ±tol of target
    accuracy = {}
    for tol in tolerances:
        pct = np.mean(abs_errors <= tol) * 100.0
        accuracy[tol] = pct

    return {
        "n": n,
        "mae": mae,
        "rmse": rmse,
        "mse": mse,
        "mape": mape,
        "r2": r2,
        "max_error": max_err,
        "median_ae": median_ae,
        "bias": bias,
        "std": np.std(preds),
        "mean_pred": np.mean(preds),
        "accuracy": accuracy,
    }


def print_metrics(metrics, target_volume):
    """Pretty-print all metrics to console."""
    print(f"\n{'═' * 55}")
    print(f"  EVALUATION RESULTS  ({metrics['n']} recordings)")
    print(f"{'═' * 55}")
    print(f"  Target volume:     {target_volume:.2f} L")
    print(f"  Mean prediction:   {metrics['mean_pred']:.4f} L")
    print(f"{'─' * 55}")
    print(f"  MAE  (Mean Abs Error):      {metrics['mae']:.4f} L")
    print(f"  RMSE (Root Mean Sq Error):  {metrics['rmse']:.4f} L")
    print(f"  MSE  (Mean Sq Error):       {metrics['mse']:.6f} L²")
    print(f"  MAPE (Mean Abs % Error):    {metrics['mape']:.2f} %")
    print(f"  Median Abs Error:           {metrics['median_ae']:.4f} L")
    print(f"  Max Error (worst case):     {metrics['max_error']:.4f} L")
    print(f"  R² Score:                   {metrics['r2']:.4f}")
    print(f"{'─' * 55}")
    print(f"  Bias:  {metrics['bias']:+.4f} L", end="")
    if metrics["bias"] > 0:
        print("  (overestimates)")
    elif metrics["bias"] < 0:
        print("  (underestimates)")
    else:
        print("  (no bias)")
    print(f"  Std:   {metrics['std']:.4f} L")
    print(f"{'─' * 55}")
    print(f"  Within-tolerance accuracy:")
    for tol, pct in metrics["accuracy"].items():
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        print(f"    ±{tol:.2f} L:  {bar}  {pct:.1f}%")
    print(f"{'═' * 55}")


def plot_metrics_summary(metrics, save_dir):
    """Save a visual metrics summary card as PNG."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ax = axes[0]
    metric_names = ["MAE", "RMSE", "Median AE", "Max Err", "|Bias|"]
    metric_vals  = [metrics["mae"], metrics["rmse"], metrics["median_ae"],
                    metrics["max_error"], abs(metrics["bias"])]
    colors = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3"]
    bars = ax.bar(metric_names, metric_vals, color=colors, edgecolor="black")
    for bar, val in zip(bars, metric_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Litres")
    ax.set_title("Error Metrics")
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1]
    tols = [f"±{t:.2f}L" for t in metrics["accuracy"]]
    pcts = list(metrics["accuracy"].values())
    bar_colors = ["#d62728" if p < 50 else "#ff7f0e" if p < 80 else "#2ca02c"
                  for p in pcts]
    bars = ax.barh(tols, pcts, color=bar_colors, edgecolor="black")
    for bar, pct in zip(bars, pcts):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{pct:.1f}%", ha="left", va="center", fontsize=10)
    ax.set_xlim(0, 110)
    ax.set_xlabel("% of predictions within tolerance")
    ax.set_title("Accuracy at Different Tolerances")
    ax.grid(True, alpha=0.3, axis="x")

    ax = axes[2]
    ax.axis("off")
    text_lines = [
        f"N recordings:  {metrics['n']}",
        f"",
        f"R² Score:      {metrics['r2']:.4f}",
        f"MAPE:          {metrics['mape']:.2f}%",
        f"",
        f"Mean pred:     {metrics['mean_pred']:.4f} L",
        f"Bias:          {metrics['bias']:+.4f} L",
        f"Std:           {metrics['std']:.4f} L",
    ]
    ax.text(0.1, 0.95, "\n".join(text_lines), transform=ax.transAxes,
            fontsize=12, verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    ax.set_title("Summary")

    fig.suptitle("Model Evaluation Summary", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{save_dir}/metrics_summary.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {save_dir}/metrics_summary.png")


def plot_loss_curve(history, save_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"], label="Train Loss", linewidth=2)
    ax.plot(epochs, history["val_loss"],   label="Val Loss",   linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Training vs Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{save_dir}/loss_curve.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {save_dir}/loss_curve.png")


def plot_pred_vs_actual(preds, labels, save_dir):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(labels, preds, alpha=0.7, edgecolors="k", linewidth=0.5)
    mn = min(min(labels), min(preds))
    mx = max(max(labels), max(preds))
    margin = (mx - mn) * 0.1 + 0.1
    ax.plot([mn - margin, mx + margin], [mn - margin, mx + margin],
            "r--", label="Perfect")
    ax.set_xlabel("Actual Volume (L)")
    ax.set_ylabel("Predicted Volume (L)")
    ax.set_title(f"Predicted vs Actual  |  N={len(preds)} eval recordings")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{save_dir}/pred_vs_actual.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {save_dir}/pred_vs_actual.png")


def plot_error_distribution(preds, labels, save_dir):
    errors = [p - l for p, l in zip(preds, labels)]
    fig, ax = plt.subplots(figsize=(8, 5))
    n_bins = max(5, min(30, len(errors) // 2))
    ax.hist(errors, bins=n_bins, edgecolor="black", alpha=0.75)
    ax.axvline(0, color="red", linestyle="--")
    mae = np.mean(np.abs(errors))
    std = np.std(errors)
    ax.set_xlabel("Error (Predicted − Actual) [L]")
    ax.set_ylabel("Count")
    ax.set_title(f"Error Distribution  |  MAE={mae:.4f} L  |  STD={std:.4f} L  |  N={len(errors)}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{save_dir}/error_distribution.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {save_dir}/error_distribution.png")


def plot_realtime_curves(model, recordings, cfg, device, save_dir):
    """Plot real-time volume curves for ALL eval CSVs, auto-sizing the grid."""
    n_files = len(recordings)
    if n_files == 0:
        print("No eval CSVs to plot.")
        return

    cols = min(4, n_files)
    rows = math.ceil(n_files / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))

    if n_files == 1:
        axes = [axes]
    else:
        axes = np.array(axes).flatten()

    for i, (name, flow_all, delta_t_all) in enumerate(recordings):
        mask    = flow_all > 0.0
        flow    = flow_all[mask]
        delta_t = delta_t_all[mask]

        if len(flow) < 2:
            continue

        features = np.stack([flow, delta_t, flow * delta_t], axis=1)
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
        lengths = torch.tensor([len(features)]).to(device)

        with torch.no_grad():
            volumes = model(x, lengths)
        vols = volumes[0, :, 0].cpu().numpy()

        vols_full = np.zeros(len(flow_all))
        vol_idx, last_vol = 0, 0.0
        for t in range(len(flow_all)):
            if flow_all[t] > 0.0:
                last_vol = vols[vol_idx]
                vol_idx += 1
            vols_full[t] = last_vol

        naive = np.cumsum(flow_all * delta_t_all)

        ax = axes[i]

        ln1 = ax.plot(vols_full, label="LSTM", linewidth=2, color="tab:blue")
        ax.axhline(cfg.target_volume, color="red", linestyle="--",
                    label=f"Target {cfg.target_volume}L")

        y_top = max(cfg.target_volume, np.max(vols_full)) * 1.15
        flow_max = np.max(flow_all) if np.max(flow_all) > 0 else 1.0
        flow_scaled = (flow_all / flow_max) * y_top
        ax.fill_between(range(len(flow_all)), flow_scaled,
                        alpha=0.12, color="gray", label=f"Flow (peak={flow_max:.1f})")
        ax.set_ylim(bottom=0, top=y_top)
        ax.set_ylabel("Volume (L)", fontsize=8)

        ax2 = ax.twinx()
        ln2 = ax2.plot(naive, label=f"Naive ∑(f×dt) = {naive[-1]:.1f}",
                        linewidth=1, alpha=0.7, color="tab:orange")
        ax2.set_ylabel("Raw ∑(flow×dt)", fontsize=8, color="tab:orange")
        ax2.tick_params(axis="y", labelcolor="tab:orange")

        lines = ln1 + ln2 + [ax.get_lines()[1]]  # LSTM + Naive + Target
        labels_leg = [l.get_label() for l in lines]
        # Add flow to legend manually
        labels_leg.append(f"Flow (peak={flow_max:.1f})")
        lines.append(Patch(facecolor="gray", alpha=0.2))
        ax.legend(lines, labels_leg, fontsize=5, loc="upper left")

        ax.set_title(name, fontsize=9)
        ax.set_xlabel("Timestep")
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for j in range(n_files, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"Real-Time Volume: LSTM vs Naive  ({n_files} eval recordings)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{save_dir}/realtime_curves.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {save_dir}/realtime_curves.png")


#  FLOW RATE VS ERROR ANALYSIS

def plot_flowrate_vs_error(model, recordings, cfg, device, save_dir):
    """
    For each eval CSV, compute mean flow rate (proxy for blow speed)
    and final volume error. Scatter plot reveals if the model is
    worse on fast vs slow blows — the core problem being solved.

    Also produces a secondary plot binning recordings into flow-rate
    quartiles and showing error distribution per bin.
    """
    mean_flows, peak_flows, errors, durations, names = [], [], [], [], []

    for name, flow_all, delta_t_all in recordings:
        mask    = flow_all > 0.0
        flow    = flow_all[mask]
        delta_t = delta_t_all[mask]

        if len(flow) < 2:
            continue

        features = np.stack([flow, delta_t, flow * delta_t], axis=1)
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
        lengths = torch.tensor([len(features)]).to(device)

        with torch.no_grad():
            volumes = model(x, lengths)
        final_vol = volumes[0, -1, 0].cpu().item()
        error = final_vol - cfg.target_volume

        mean_flows.append(np.mean(flow))
        peak_flows.append(np.max(flow))
        durations.append(np.sum(delta_t))
        errors.append(error)
        names.append(name)

    mean_flows = np.array(mean_flows)
    peak_flows = np.array(peak_flows)
    durations  = np.array(durations)
    errors     = np.array(errors)
    abs_errors = np.abs(errors)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    sc = ax.scatter(mean_flows, errors, c=abs_errors, cmap="RdYlGn_r",
                    edgecolors="k", linewidth=0.5, s=60)
    ax.axhline(0, color="red", linestyle="--", alpha=0.5)
    ax.set_xlabel("Mean Flow Rate (non-zero rows)")
    ax.set_ylabel("Error (Predicted − Actual) [L]")
    ax.set_title("Mean Flow Rate vs Volume Error")
    ax.grid(True, alpha=0.3)
    plt.colorbar(sc, ax=ax, label="|Error| (L)")

    ax = axes[0, 1]
    sc = ax.scatter(peak_flows, errors, c=abs_errors, cmap="RdYlGn_r",
                    edgecolors="k", linewidth=0.5, s=60)
    ax.axhline(0, color="red", linestyle="--", alpha=0.5)
    ax.set_xlabel("Peak Flow Rate")
    ax.set_ylabel("Error (Predicted − Actual) [L]")
    ax.set_title("Peak Flow Rate vs Volume Error")
    ax.grid(True, alpha=0.3)
    plt.colorbar(sc, ax=ax, label="|Error| (L)")

    ax = axes[1, 0]
    sc = ax.scatter(durations, errors, c=mean_flows, cmap="coolwarm",
                    edgecolors="k", linewidth=0.5, s=60)
    ax.axhline(0, color="red", linestyle="--", alpha=0.5)
    ax.set_xlabel("Blow Duration (s)")
    ax.set_ylabel("Error (Predicted − Actual) [L]")
    ax.set_title("Blow Duration vs Volume Error")
    ax.grid(True, alpha=0.3)
    plt.colorbar(sc, ax=ax, label="Mean Flow Rate")

    ax = axes[1, 1]
    n_bins = min(4, max(2, len(mean_flows) // 3))
    bin_edges = np.percentile(mean_flows, np.linspace(0, 100, n_bins + 1))
    bin_edges[-1] += 0.01  # include max value
    bin_labels = []
    bin_errors = []
    for b in range(n_bins):
        in_bin = (mean_flows >= bin_edges[b]) & (mean_flows < bin_edges[b + 1])
        if np.any(in_bin):
            bin_errors.append(abs_errors[in_bin])
            bin_labels.append(f"{bin_edges[b]:.0f}–{bin_edges[b+1]:.0f}")

    if bin_errors:
        bp = ax.boxplot(bin_errors, labels=bin_labels, patch_artist=True)
        colors_box = plt.cm.coolwarm(np.linspace(0.2, 0.8, len(bin_errors)))
        for patch, color in zip(bp["boxes"], colors_box):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
    ax.set_xlabel("Mean Flow Rate Bin")
    ax.set_ylabel("|Error| (L)")
    ax.set_title("Absolute Error by Flow Rate Group")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Flow Rate vs Prediction Error — Does blow speed affect accuracy?",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{save_dir}/flowrate_vs_error.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {save_dir}/flowrate_vs_error.png")

    print(f"\n{'─' * 55}")
    print(f"  FLOW RATE VS ERROR SUMMARY")
    print(f"{'─' * 55}")
    if len(mean_flows) > 2:
        corr_mean = np.corrcoef(mean_flows, abs_errors)[0, 1]
        corr_peak = np.corrcoef(peak_flows, abs_errors)[0, 1]
        corr_dur  = np.corrcoef(durations, abs_errors)[0, 1]
        print(f"  Correlation (|error| vs mean flow):  {corr_mean:+.3f}")
        print(f"  Correlation (|error| vs peak flow):  {corr_peak:+.3f}")
        print(f"  Correlation (|error| vs duration):   {corr_dur:+.3f}")
        print(f"  → Values near 0 = model handles all speeds equally")
        print(f"  → Values near ±1 = model struggles with fast or slow blows")
    print(f"{'─' * 55}")


#  VOLUME MONOTONICITY CHECK


def plot_monotonicity_check(model, recordings, cfg, device, save_dir):
    """
    Volume should only increase during a blow but it sometimes decrease
    predicts a volume DECREASE and flags them.

    """
    results = []  

    for name, flow_all, delta_t_all in recordings:
        mask    = flow_all > 0.0
        flow    = flow_all[mask]
        delta_t = delta_t_all[mask]

        if len(flow) < 2:
            continue

        features = np.stack([flow, delta_t, flow * delta_t], axis=1)
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
        lengths = torch.tensor([len(features)]).to(device)

        with torch.no_grad():
            volumes = model(x, lengths)
        vols = volumes[0, :len(flow), 0].cpu().numpy()

        diffs = np.diff(vols)
        n_drops = int(np.sum(diffs < 0))
        max_drop = float(np.min(diffs)) if n_drops > 0 else 0.0
        n_steps = len(vols)
        pct_mono = (1.0 - n_drops / max(n_steps - 1, 1)) * 100.0

        results.append((name, n_steps, n_drops, max_drop, pct_mono, vols))

    if not results:
        print("No valid recordings for monotonicity check.")
        return

    names      = [r[0] for r in results]
    n_drops    = [r[2] for r in results]
    max_drops  = [r[3] for r in results]
    pct_monos  = [r[4] for r in results]

    # ── Detail plots for worst offenders ──────────────────
    # Sort by most drops, show top 4 worst
    sorted_results = sorted(results, key=lambda r: r[2], reverse=True)
    worst = [r for r in sorted_results if r[2] > 0][:4]

    if worst:
        n_worst = len(worst)
        fig2, axes2 = plt.subplots(1, n_worst, figsize=(5 * n_worst, 4))
        if n_worst == 1:
            axes2 = [axes2]

        for i, (wname, wn, wd, wmax, wpct, wvols) in enumerate(worst):
            ax = axes2[i]
            steps = range(len(wvols))
            ax.plot(steps, wvols, linewidth=1.5, color="tab:blue")

            # Highlight drops in red
            diffs = np.diff(wvols)
            drop_idx = np.where(diffs < 0)[0]
            for di in drop_idx:
                ax.plot([di, di + 1], [wvols[di], wvols[di + 1]],
                        color="red", linewidth=2.5, alpha=0.8)
                ax.scatter([di + 1], [wvols[di + 1]], color="red", s=20, zorder=5)

            ax.set_title(f"{wname[:20]}\n{wd} drops, worst={wmax:.4f}L", fontsize=9)
            ax.set_xlabel("Timestep (non-zero only)")
            ax.set_ylabel("Predicted Volume (L)")
            ax.grid(True, alpha=0.3)

        fig2.suptitle("Worst Monotonicity Offenders — Red = volume decrease",
                      fontsize=12, fontweight="bold")
        fig2.tight_layout()
        fig2.savefig(f"{save_dir}/monotonicity_worst.png", dpi=150)
        plt.close(fig2)
        print(f"Saved: {save_dir}/monotonicity_worst.png")

    # ── Console summary ───────────────────────────────────
    fully_mono = sum(1 for p in pct_monos if p == 100.0)
    avg_pct    = np.mean(pct_monos)
    total_drops = sum(n_drops)

    print(f"\n{'─' * 55}")
    print(f"  MONOTONICITY CHECK SUMMARY")
    print(f"{'─' * 55}")
    print(f"  Fully monotonic:   {fully_mono}/{len(results)} recordings ({fully_mono/len(results)*100:.0f}%)")
    print(f"  Average monotonic: {avg_pct:.1f}%")
    print(f"  Total drops:       {total_drops} across all recordings")
    if worst:
        print(f"  Worst recording:   {worst[0][0]} ({worst[0][2]} drops, max={worst[0][3]:.4f}L)")
    else:
        print(f"  No volume decreases detected — model is fully monotonic!")
    print(f"{'─' * 55}")

def main():
    cfg = Config()
    save_dir = cfg.results_dir
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    history_path = f"{save_dir}/history.json"
    if os.path.exists(history_path):
        with open(history_path) as f:
            history = json.load(f)
        plot_loss_curve(history, save_dir)
    else:
        print("No history.json found — skipping loss curve.")

    model = load_model(cfg, device)
    recordings = load_eval_csvs(cfg.eval_dir)
    if not recordings:
        print(f"\nNo CSV files found in {cfg.eval_dir}/")
        print("Put your evaluation CSVs there and re-run.")
        return

    print(f"\nEvaluating on {len(recordings)} recordings from {cfg.eval_dir}/")

    preds, labels = evaluate_final_volumes(model, recordings, cfg, device)
    if preds:
        plot_pred_vs_actual(preds, labels, save_dir)
        plot_error_distribution(preds, labels, save_dir)

        metrics = compute_metrics(preds, labels)
        print_metrics(metrics, cfg.target_volume)
        plot_metrics_summary(metrics, save_dir)

        metrics_save = {k: v for k, v in metrics.items() if k != "accuracy"}
        metrics_save["accuracy"] = {str(k): v for k, v in metrics["accuracy"].items()}
        with open(f"{save_dir}/metrics.json", "w") as f:
            json.dump(metrics_save, f, indent=2)
        print(f"Saved: {save_dir}/metrics.json")

    plot_realtime_curves(model, recordings, cfg, device, save_dir)
    plot_flowrate_vs_error(model, recordings, cfg, device, save_dir)
    plot_monotonicity_check(model, recordings, cfg, device, save_dir)

    print(f"\nAll graphs saved to {save_dir}/")


if __name__ == "__main__":
    main()