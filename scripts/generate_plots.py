"""High-resolution figures; every point is derived from executed CSV records."""
import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplconfig"))
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCENARIOS = ["low", "moderate", "high"]
CORE_LABELS = ["Synchronous", "Vanilla async", "Filtered (tau=5)", "Async-MB (B=4, tau=5)"]
COLORS = {label: color for label, color in zip(CORE_LABELS,
          ["#0072B2", "#D55E00", "#009E73", "#CC79A7"])}


def select(frame, **kwargs):
    mask = np.ones(len(frame), dtype=bool)
    for key, value in kwargs.items():
        mask &= frame[key].eq(value).to_numpy()
    return frame.loc[mask]


def curve(ax, frame, x, y, label, color=None):
    """Align seeds with last recorded value, with no extrapolation past a run."""
    runs = [g.sort_values(x).drop_duplicates(x, keep="last") for _, g in frame.groupby("run_id")]
    if not runs:
        return
    common_end = min(g[x].max() for g in runs)
    grid = np.linspace(0, common_end, 350)
    values = []
    for run in runs:
        indices = np.searchsorted(run[x].to_numpy(), grid, side="right") - 1
        values.append(run[y].to_numpy()[np.maximum(indices, 0)])
    values = np.asarray(values)
    mean = values.mean(axis=0)
    line, = ax.step(grid, mean, where="post", label=label, color=color, linewidth=1.8)
    if len(runs) > 1:
        sd = values.std(axis=0, ddof=1)
        ax.fill_between(grid, np.maximum(0, mean - sd), mean + sd, step="post",
                        color=line.get_color(), alpha=0.16, linewidth=0)


def style_axis(ax, title, xlabel, ylabel, loss=False):
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    if loss:
        ax.set_ylim(0, 0.75)  # Full meaningful loss range; no cropped convergence axes.
    ax.grid(alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)


def save(fig, folder, name):
    fig.savefig(folder / f"{name}.png", dpi=250, bbox_inches="tight")
    fig.savefig(folder / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def core_panels(history, folder, x, y, name, title, xlabel, ylabel):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), layout="constrained")
    for ax, scenario in zip(axes, SCENARIOS):
        subset = select(history, workers=8, scenario=scenario)
        for label in CORE_LABELS:
            curve(ax, select(subset, method_label=label), x, y, label, COLORS[label])
        style_axis(ax, f"{scenario.capitalize()} delay", xlabel, ylabel, loss=y.endswith("loss"))
    axes[-1].legend(fontsize=8, loc="best")
    fig.suptitle(title + " (8 workers; shaded bands: ±1 sample SD)", fontsize=13)
    save(fig, folder, name)


def generate_plots(history, events, summary, diagnostics, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
                         "legend.fontsize": 9, "savefig.facecolor": "white"})
    core_panels(history, output_dir, "model_updates", "training_loss", "01_loss_vs_updates",
                "Training loss vs model updates", "Model updates", "Regularized training loss")
    core_panels(history, output_dir, "simulated_time", "training_loss", "02_loss_vs_time",
                "Training progress within equal simulated time budgets", "Simulated wall-clock time", "Regularized training loss")
    core_panels(history, output_dir, "simulated_time", "mean_staleness", "03_staleness_vs_time",
                "Cumulative mean staleness of all completed gradients", "Simulated wall-clock time", "Mean staleness (model updates)")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), layout="constrained")
    for ax, workers in zip(axes, (4, 8, 16)):
        for i, label in enumerate(CORE_LABELS):
            subset = select(summary, workers=workers, method_label=label)
            stats = subset.groupby("scenario").mean_staleness.agg(["mean", "std"]).reindex(SCENARIOS)
            ax.bar(np.arange(3) + (i - 1.5) * 0.19, stats["mean"], width=0.19,
                   yerr=stats["std"].fillna(0), capsize=2, label=label, color=COLORS[label])
        ax.set_xticks(range(3), SCENARIOS)
        ax.set(title=f"{workers} workers", xlabel="Delay scenario", ylabel="Mean arrival staleness (updates)", ylim=(0, None))
        ax.grid(axis="y", alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, fontsize=9)
    fig.suptitle("Mean staleness by method and scenario (error bars: ±1 sample SD)")
    save(fig, output_dir, "04_staleness_by_method")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), layout="constrained")
    for ax, workers in zip(axes, (4, 8, 16)):
        for i, threshold in enumerate((2, 5, 10)):
            stats = select(summary, workers=workers, method="filtered", threshold=threshold).groupby("scenario").discard_percentage.agg(["mean", "std"]).reindex(SCENARIOS)
            ax.bar(np.arange(3) + (i - 1) * 0.25, stats["mean"], width=0.25,
                   yerr=stats["std"].fillna(0), capsize=2, label=f"tau={threshold}")
        ax.set_xticks(range(3), SCENARIOS)
        ax.set(title=f"{workers} workers", xlabel="Delay scenario", ylabel="Discarded / completed gradients (%)", ylim=(0, 100))
        ax.grid(axis="y", alpha=0.2)
    axes[-1].legend()
    fig.suptitle("Cost of stale-gradient filtering (error bars: ±1 sample SD)")
    save(fig, output_dir, "05_discard_percentage")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for ax, label in zip(axes.flat, CORE_LABELS):
        for workers in (4, 8, 16):
            curve(ax, select(history, scenario="high", method_label=label, workers=workers),
                  "simulated_time", "training_loss", f"{workers} workers")
        style_axis(ax, label, "Simulated wall-clock time", "Regularized training loss", loss=True)
        ax.legend()
    fig.suptitle("Worker scaling under high heterogeneity (bands: ±1 sample SD)")
    save(fig, output_dir, "06_worker_scaling")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), layout="constrained")
    for ax, workers in zip(axes, (4, 8, 16)):
        for label in ["Vanilla async", "Filtered (tau=2)", "Filtered (tau=5)", "Filtered (tau=10)"]:
            curve(ax, select(history, workers=workers, scenario="high", method_label=label),
                  "simulated_time", "training_loss", label)
        style_axis(ax, f"{workers} workers", "Simulated wall-clock time", "Regularized training loss", loss=True)
    axes[-1].legend(fontsize=8)
    fig.suptitle("Threshold sensitivity under high heterogeneity (bands: ±1 sample SD)")
    save(fig, output_dir, "07_threshold_comparison")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout="constrained")
    for column, workers in enumerate((4, 8, 16)):
        for scenario in SCENARIOS:
            curve(axes[0, column], select(history, method="async", workers=workers, scenario=scenario),
                  "simulated_time", "training_loss", scenario)
            ages = select(events, method="async", workers=workers, scenario=scenario).staleness
            # Include the maximum even when it falls on an even integer.
            bins = np.arange(-0.5, ages.max() + 2.5, 2)
            counts, _ = np.histogram(ages, bins=bins)
            if counts.sum() != len(ages):
                raise ValueError("Staleness histogram omitted recorded arrivals")
            axes[1, column].hist(ages, bins=bins, density=True, histtype="step", label=scenario, linewidth=1.7)
        style_axis(axes[0, column], f"{workers} workers", "Simulated wall-clock time", "Regularized training loss", loss=True)
        style_axis(axes[1, column], f"Arrival-age distribution: {workers} workers", "Staleness (model updates)", "Density")
    axes[0, -1].legend()
    axes[1, -1].legend()
    fig.suptitle("Vanilla async: heterogeneity changes throughput and the staleness tail")
    save(fig, output_dir, "08_async_heterogeneity")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout="constrained")
    for ax, scenario in zip(axes, ("moderate", "high")):
        for label in CORE_LABELS[1:]:
            curve(ax, select(history, workers=8, scenario=scenario, method_label=label),
                  "simulated_time", "training_loss", label, COLORS[label])
        style_axis(ax, f"{scenario.capitalize()} delay, 8 workers", "Simulated wall-clock time", "Regularized training loss", loss=True)
        ax.legend(fontsize=8)
    fig.suptitle("Simplified fixed-B asynchronous mini-batching (bands: ±1 sample SD)")
    save(fig, output_dir, "09_async_minibatch")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout="constrained")
    for ax, metric, label in zip(axes, ("p95_staleness", "max_staleness"), ("95th percentile staleness", "Maximum staleness")):
        stats = select(summary, workers=8, method="async").groupby("scenario")[metric].agg(["mean", "std"]).reindex(SCENARIOS)
        ax.bar(SCENARIOS, stats["mean"], yerr=stats["std"].fillna(0), capsize=4, color=["#0072B2", "#E69F00", "#D55E00"], label="Vanilla async")
        ax.set(title=label, xlabel="Delay scenario", ylabel="Staleness (model updates)", ylim=(0, None))
        ax.grid(axis="y", alpha=0.2)
        ax.legend()
    fig.suptitle("Tail staleness with 8 workers (error bars: ±1 sample SD across seeds)")
    save(fig, output_dir, "10_tail_staleness")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), layout="constrained")
    for ax, metric, ylabel in zip(axes, ("mean_staleness", "max_staleness", "gradients_computed"),
                                ("Mean staleness (updates)", "Maximum staleness (updates)", "Completed gradients")):
        stats = diagnostics.groupby("slow_fraction")[metric].agg(["mean", "std"])
        ax.errorbar(stats.index * 8, stats["mean"], yerr=stats["std"].fillna(0), marker="o", capsize=4, label="Vanilla async")
        ax.set_xticks([0, 1, 2, 4])
        style_axis(ax, ylabel, "Number of slow workers (out of 8)", ylabel)
        ax.legend()
    fig.suptitle("Controlled straggler-count diagnostic: fixed CV=0.35, slow mean=6 (±1 sample SD)")
    save(fig, output_dir, "11_straggler_count")

    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    for label in CORE_LABELS:
        curve(ax, select(history, workers=8, scenario="high", method_label=label),
              "gradients_computed", "training_loss", label, COLORS[label])
    style_axis(ax, "Compute-efficiency comparison: high delay, 8 workers", "Completed task gradients (including discarded)", "Regularized training loss", loss=True)
    ax.legend()
    save(fig, output_dir, "12_loss_vs_gradient_work")
    print(f"Saved 12 PNG figures and 12 vector PDFs in {output_dir}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--plots-dir", type=Path, default=Path("plots"))
    args = parser.parse_args()
    generate_plots(pd.read_csv(args.results_dir / "raw_results.csv"),
                   pd.read_csv(args.results_dir / "gradient_events.csv.gz"),
                   pd.read_csv(args.results_dir / "summary_results.csv"),
                   pd.read_csv(args.results_dir / "straggler_diagnostics.csv"), args.plots_dir)


if __name__ == "__main__":
    main()
