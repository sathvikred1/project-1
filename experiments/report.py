"""Generate prose from recorded outcomes, including contradicted expectations."""
from pathlib import Path

import pandas as pd

from .run_experiments import format_estimate, write_markdown_table


def paired_comparisons(summary):
    base = summary[summary.method == "async"]
    rows = []
    for label in ("Synchronous", "Filtered (tau=2)", "Filtered (tau=5)",
                  "Filtered (tau=10)", "Async-MB (B=4, tau=5)"):
        other = summary[summary.method_label == label]
        joined = other.merge(base, on=["workers", "scenario", "seed"], suffixes=("_method", "_async"))
        for _, r in joined.iterrows():
            rows.append(dict(method_label=label, workers=r.workers, scenario=r.scenario, seed=r.seed,
                             training_loss_difference=r.training_loss_method - r.training_loss_async,
                             validation_loss_difference=r.validation_loss_method - r.validation_loss_async,
                             validation_accuracy_difference=r.validation_accuracy_method - r.validation_accuracy_async,
                             method_loss=r.training_loss_method, async_loss=r.training_loss_async))
    return pd.DataFrame(rows)


def generate_report(summary, aggregate, diagnostics, manifest, sanity, results_dir):
    results_dir = Path(results_dir)
    paired = paired_comparisons(summary)
    paired.to_csv(results_dir / "paired_comparisons.csv", index=False)
    horizon = manifest["configurations"][0]["time_budget"]
    seeds = manifest["seeds"]
    n = len(seeds)
    focus = aggregate[(aggregate.workers == 8) & (aggregate.scenario == "high")]
    table = pd.DataFrame([{
        "Method": r.method_label,
        "Train loss": format_estimate(r, "training_loss"),
        "Val loss": format_estimate(r, "validation_loss"),
        "Val accuracy": format_estimate(r, "validation_accuracy", 3),
        "Updates": format_estimate(r, "model_updates", 1),
        "Mean / max age": f"{r.mean_staleness_mean:.2f} / {r.max_staleness_mean:.1f}",
        "Discard %": format_estimate(r, "discard_percentage", 1),
    } for _, r in focus.iterrows()])

    lines = [
        "# Week 9: Asynchrony / Delay-Robust Optimization", "",
        "This project is a simplified educational simulation inspired by recent research on asynchronous and delay-robust stochastic optimization. It is not an exact reproduction of the original papers.", "",
        f"Run mode: **{manifest['mode']}**. All numerical observations below come from executed runs in this output directory. Paper 3 is outside the implementation.", "",
        "## A. Experimental setup", "",
        "- Objective: binary logistic regression with L2 coefficient 0.001; intercept unregularized. Losses include the same L2 term on both splits.",
        "- Dataset: `sklearn.datasets.make_classification`, 6,000 samples, 30 features (20 informative, 5 redundant), class separation 1.1, label-flip probability 0.04. Fixed dataset seed 2026.",
        "- Split: stratified 80/20 (4,800 training; 1,200 validation), split seed 2027. Standardization fits training data only. No external download or test-set tuning.",
        "- Every method: zero initialization, constant learning rate 0.15, 32 training examples sampled with replacement per worker gradient. Independent per-worker random streams pair samples and delay draws across methods by seed, worker, and task number.",
        "- Workers: 4, 8, 16. Delay-filter thresholds: inclusive ages 2, 5, 10. Main comparison uses threshold 5, fixed before execution. Async mini-batching averages B=4 accepted task gradients, with inclusive threshold 5.",
        f"- Training/delay seeds: {seeds} ({n} seed{'s' if n != 1 else ''}); {manifest['main_runs']} main configurations plus {manifest['diagnostic_runs']} controlled straggler diagnostics.",
        f"- Stopping: common fixed horizon of {horizon:g} simulated time units, independent of loss and method. Started-but-unfinished jobs and accepted-but-uncommitted buffers are recorded without a free final update.",
        "- Evaluation: initial state, every applied model update, and the horizon. Training/validation losses, accuracy, counts, and staleness are saved. Every gradient arrival is saved separately.",
        "- Dispersion: sample standard deviation across training/delay seeds, with the dataset held fixed. This is neither a confidence interval nor dataset-to-dataset uncertainty. Single-seed quick results have undefined SD.",
        "- Synchronous steps average one task gradient per worker (effective sample batch 32W); vanilla/filtered steps use 32 samples; async mini-batch steps average 128 samples. Equal learning rates do not equalize effective batch size or the number of updates. Time and gradient-work plots show different aspects of this trade-off.", "",
        "Delay laws are independent positive lognormal draws, parameterized by arithmetic mean and coefficient of variation (CV):", "",
        write_markdown_table(pd.DataFrame([{"Scenario": k, "Distribution": v} for k, v in manifest["delays"].items()])), "",
        "The last W/4 worker IDs are slow in moderate/high scenarios; each worker retains its role. The separate diagnostic varies slow-worker counts 0, 1, 2, 4 of 8 at the same high-scenario CV.", "",
        "## B. Preliminary results", "",
        f"High heterogeneity, 8 workers, equal horizon {horizon:g}. Values are means ± sample SD where available. Age means and maxima below summarize all completed arrivals, including discarded arrivals; the max-age column averages each seed's maximum.", "",
        write_markdown_table(table), "",
        "See `summary_results.csv` for individual seeds, `aggregate_results.csv` for all 54 configuration means and SDs, and `summary_results.md` for the complete readable table.", "",
        "## C. Main observations", "",
    ]
    # Questions 1 and 2: throughput, loss, and target crossing, with censoring explicit.
    async_8 = aggregate[(aggregate.method == "async") & (aggregate.workers == 8)].set_index("scenario")
    sync_8 = aggregate[(aggregate.method == "sync") & (aggregate.workers == 8)].set_index("scenario")
    for scenario in ("low", "moderate", "high"):
        a, s = async_8.loc[scenario], sync_8.loc[scenario]
        direction = "lower" if a.training_loss_mean < s.training_loss_mean else "higher or equal"
        lines.append(f"- **Sync vs async, {scenario} delay, 8 workers:** vanilla async ended at loss {a.training_loss_mean:.4f} versus synchronous {s.training_loss_mean:.4f} ({direction} at the same time budget), with {a.model_updates_mean:.1f} versus {s.model_updates_mean:.1f} mean updates. These methods use different effective update batch sizes.")
    target_rows = []
    for label in ("Synchronous", "Vanilla async", "Filtered (tau=5)", "Async-MB (B=4, tau=5)"):
        group = summary[(summary.method_label == label) & (summary.workers == 8) & (summary.scenario == "high")]
        success = group[group.target_reached == 1]
        crossing = "not observed" if success.empty else f"{success.time_to_target.mean():.2f} (successful runs only)"
        target_rows.append({"Method": label, "Logged target crossings": f"{len(success)}/{len(group)}",
                            "Mean first logged time": crossing})
    lines.extend(["", f"Prespecified training-loss target: {manifest['target_training_loss']:.2f}, high delay, 8 workers. Loss is evaluated at every update, so times are actual first-crossing events; no interpolation or invented times for runs without a crossing. Successful-only times must be interpreted together with the crossing counts.", "",
                  write_markdown_table(pd.DataFrame(target_rows)), ""])
    age_text = "; ".join(f"{s}: mean {async_8.loc[s].mean_staleness_mean:.2f}, p95 {async_8.loc[s].p95_staleness_mean:.2f}, max {async_8.loc[s].max_staleness_mean:.1f}" for s in ("low", "moderate", "high"))
    lines.append(f"- **Heterogeneity and vanilla-async staleness, 8 workers:** {age_text}. All three statistics are computed over completed arrivals; p95 and max are then averaged across seeds.")

    # Paired differences reveal actual direction rather than assuming filtering wins.
    for label in ("Filtered (tau=2)", "Filtered (tau=5)", "Filtered (tau=10)", "Async-MB (B=4, tau=5)"):
        group = paired[(paired.method_label == label) & (paired.scenario == "high") & (paired.workers == 8)]
        delta = group.training_loss_difference
        direction = "lower" if delta.mean() < 0 else "higher or equal"
        sd_text = f"{delta.std(ddof=1):.4f}" if n > 1 else "undefined (one seed)"
        lines.append(f"- **{label} vs vanilla async, high delay, 8 workers:** paired final-training-loss difference (method minus vanilla) {delta.mean():+.4f}, sample SD {sd_text}; {direction} mean loss, with lower loss in {int((delta < 0).sum())}/{len(delta)} paired seeds. This is a descriptive comparison, not a significance claim.")
    worker_rows = []
    for workers in (4, 8, 16):
        r = aggregate[(aggregate.method == "async") & (aggregate.scenario == "high") & (aggregate.workers == workers)].iloc[0]
        worker_rows.append({"Workers": workers, "Train loss": format_estimate(r, "training_loss"),
                            "Mean staleness": format_estimate(r, "mean_staleness", 2),
                            "Max staleness": format_estimate(r, "max_staleness", 1),
                            "Completed gradients": format_estimate(r, "gradients_computed", 1)})
    lines.extend(["", "Worker scaling for vanilla async under high heterogeneity:", "", write_markdown_table(pd.DataFrame(worker_rows)), ""])
    ordered_losses = [r["training_loss"] for _, r in summary[(summary.method == "async") & (summary.scenario == "high")].groupby("workers").mean(numeric_only=True).iterrows()]
    scaling = "decreased at each worker-count increase" if all(b < a for a, b in zip(ordered_losses, ordered_losses[1:])) else "did not decrease at every worker-count increase"
    lines.append(f"- Across 4 → 8 → 16 workers, mean final loss {scaling}. This finite-grid result cannot establish that adding workers always helps.")
    thresholds = focus[focus.method == "filtered"].sort_values("threshold")
    lines.append("- **Threshold trade-off, 8 workers, high delay:** " + "; ".join(
        f"tau={r.threshold:g}: loss {r.training_loss_mean:.4f}, discarded {r.discard_percentage_mean:.1f}%" for _, r in thresholds.iterrows()) + ". Thresholds were not adjusted after inspecting outcomes.")
    for scenario in ("moderate", "high"):
        rows = aggregate[(aggregate.workers == 8) & (aggregate.scenario == scenario) & (aggregate.method.isin(["async", "minibatch"]))].set_index("method")
        mb, vanilla = rows.loc["minibatch"], rows.loc["async"]
        lines.append(f"- **Simplified async mini-batch, {scenario} delay, 8 workers:** final loss {mb.training_loss_mean:.4f}, {mb.model_updates_mean:.1f} updates, mean arrival age {mb.mean_staleness_mean:.2f}; vanilla: loss {vanilla.training_loss_mean:.4f}, {vanilla.model_updates_mean:.1f} updates, age {vanilla.mean_staleness_mean:.2f}. Age is measured in model updates, so differing update frequencies change its scale.")

    lines.extend(["", "## D. Unexpected results and sanity checks", "",
        f"Algorithm invariants passed for {sanity['runs_checked']} runs, and both actual-data replay configurations matched exactly. Analytic unit tests separately verify that the slow worker's stored gradient is used, not recomputed on arrival.", ""])
    unsupported = [r for r in sanity["empirical_expectations"] if r["status"] == "NOT_SUPPORTED"]
    if unsupported:
        lines.extend(["The following expected trends were **not supported**, and the observations are retained:", "",
                      write_markdown_table(pd.DataFrame(unsupported)), ""])
    else:
        lines.append("Every empirical expectation in `empirical_checks.csv` was supported on this grid. These are finite-sample observations, not universal guarantees.")
    lines.append("In a closed pool of W continuously busy workers using unfiltered async SGD, mean completed-gradient age can remain near W−1 even when heterogeneity grows. Each applied update ages the other W−1 jobs. The exact identity `sum(completed ages) + sum(outstanding ages) = (W−1) * updates` was checked for every vanilla-async run. Unusually old unfinished jobs at the horizon do not yet enter completed-arrival statistics. Tail age and actual completion delays therefore carry information the mean can conceal. The simulator was audited rather than altered to force a higher mean.")
    diag_stats = diagnostics.groupby("slow_fraction")[["mean_staleness", "max_staleness", "gradients_computed", "mean_computation_delay"]].mean()
    lines.extend(["", "Controlled straggler-count results (mean across seeds):", "",
                  write_markdown_table(pd.DataFrame([{"Slow workers / 8": int(f * 8), "Mean age": f"{r.mean_staleness:.2f}",
                                                     "Max age": f"{r.max_staleness:.1f}", "Completed gradients": f"{r.gradients_computed:.1f}",
                                                     "Mean completed service time": f"{r.mean_computation_delay:.3f}"}
                                                    for f, r in diag_stats.iterrows()])), "",
        "More stragglers need not monotonically increase maximum or mean age: making additional workers slow also reduces the update rate that ages their gradients. Mean service times are deliberately increased by the role assignment; update-based age is an outcome, not a controlled delay parameter.", "",
        "## E. Current limitations", "",
        "- Event-driven simulation, not physical distributed workers: compute happens serially on the host; only completion availability is simulated. No communication cost, contention, server bottleneck, scheduling overhead, worker cancellation, or failures.",
        "- A single synthetic convex logistic-regression dataset and fixed learning rate. These results do not test the papers' nonconvex optimal-time guarantees or establish performance on neural networks.",
        "- Main scenarios change both speed dispersion and mean total throughput. The controlled straggler experiment also changes throughput; a future capacity-matched experiment is needed to isolate variance alone.",
        "- Synchronous updates average W task gradients; async updates use one; simplified mini-batch uses four. The common step size controls one variable but leaves an effective-batch/update-frequency confound. The gradient-work plot helps expose this distinction.",
        "- Fixed thresholds and fixed B replace paper-specific theory, automatic batch adaptation, base-optimizer wrappers, and quantile selection. Our inclusive tau rule corresponds to delta < R with R=tau+1 in the no-cancellation Ringmaster rule, but we do not select its theoretical threshold or claim its theorem.",
        "- Seed variation covers task sampling and simulated delays only. Small seed counts cannot establish statistical significance. Event-weighted ages favor workers that finish often; pending jobs are right-censored.",
        "- Target times are actual first-crossing update events. Plot uncertainty bands align stepwise loss values on a common display grid with no interpolation; averaging can hide individual-run fluctuations.", "",
        "## F. Next experiments", "",
        "1. Add a prespecified common learning-rate sweep applied equally to all methods, with separate tuning and final evaluation seeds; also compare matched effective batch sizes and gradient-work budgets.",
        "2. Match aggregate worker capacity across delay laws, then vary delay variance or straggler prevalence independently. Increase seeds and add other datasets before making broader claims.",
        "3. Explore stress conditions with larger step sizes, nonconvex objectives, time-varying speeds, and server/communication overhead; retain failures and measure tail staleness.",
        "4. If exact reproduction becomes the goal, implement and verify the paper-specific adaptive mini-batching and Ringmaster parameter choices against their pseudocode. Paper 3 remains deferred.", "",
        "## Figures for the progress report", "",
        "- `plots/02_loss_vs_time.png`: primary time-budget comparison; pair with `12_loss_vs_gradient_work.png` to explain differing effective update batches.",
        "- `plots/08_async_heterogeneity.png` and `10_tail_staleness.png`: show what heterogeneity changes and why mean staleness alone is incomplete.",
        "- `plots/07_threshold_comparison.png` together with `05_discard_percentage.png`: show the loss/discard trade-off, including any threshold that underperforms vanilla.",
        "- `plots/06_worker_scaling.png`: worker-count effects for all four main methods; `09_async_minibatch.png` isolates simplified mini-batching.",
        "- `plots/01_loss_vs_updates.png` and `03_staleness_vs_time.png` supply the update-count and cumulative-age views. All figures also have vector PDF copies.", "",
        "## Research references and implementation scope", "",
        "- Attia, Gaash & Koren (ICML 2025), [Faster Stochastic Optimization with Arbitrary Delays via Adaptive Asynchronous Mini-Batching](https://proceedings.mlr.press/v267/attia25a.html). This project borrows asynchronous collection and filtering; B and thresholds are fixed, without the paper's automatic quantile adaptation.",
        "- Maranjyan, Tyurin & Richtárik (ICML 2025), [Ringmaster ASGD: The First Asynchronous SGD with Optimal Time Complexity](https://proceedings.mlr.press/v267/maranjyan25b.html), especially [Algorithm 4](https://arxiv.org/html/2501.16168v2). This project studies a configurable completed-gradient rejection rule, without calculation stops or theoretical parameter selection.", "",
        "Reproducibility: `run_manifest.json` records configurations, package versions, data fingerprint, and source SHA-256 hashes; `sanity_checks.json` records invariant checks and exact replays. All tables and observations above were generated from the accompanying executed CSVs.", "",
    ])
    (results_dir / "week9_summary.md").write_text("\n".join(lines))
    print(f"Generated evidence-based report: {results_dir / 'week9_summary.md'}", flush=True)
