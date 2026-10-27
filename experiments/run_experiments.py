"""Run the same prespecified grid for every method; no per-method tuning."""
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src.config import RunConfig
from src.objective import make_problem
from src.simulation import simulate
from src.validation import audit_result, empirical_checks
from src.workers import DELAY_DESCRIPTION


def configuration_grid(mode, time_budget=None, seeds=None):
    seeds = seeds or ([11] if mode == "quick" else [11, 22, 33])
    horizon = time_budget if time_budget is not None else (40.0 if mode == "quick" else 150.0)
    variants = [("sync", None), ("async", None), ("filtered", 2),
                ("filtered", 5), ("filtered", 10), ("minibatch", 5)]
    return [RunConfig(method, workers=workers, scenario=scenario, seed=seed,
                      threshold=threshold, time_budget=horizon)
            for workers in (4, 8, 16) for scenario in ("low", "moderate", "high")
            for method, threshold in variants for seed in seeds]


def frame_with_metadata(frame, config):
    frame = frame.copy()
    for key, value in config.metadata().items():
        frame[key] = value
    return frame


def aggregate_summary(summary):
    keys = ["method", "method_label", "workers", "scenario", "threshold", "minibatch_gradients"]
    metrics = ["training_loss", "validation_loss", "training_accuracy", "validation_accuracy",
               "model_updates", "simulated_time", "last_update_time", "mean_staleness",
               "max_staleness", "p95_staleness", "mean_accepted_staleness", "gradients_computed",
               "gradients_accepted", "gradients_discarded", "gradients_used", "gradients_pending",
               "gradients_in_flight", "gradients_started", "discard_percentage", "time_to_target",
               "target_reached", "mean_computation_delay", "outstanding_model_age_sum"]
    groups = summary.groupby(keys, dropna=False, sort=False)
    aggregate = groups[metrics].agg(["mean", "std"])
    aggregate.columns = [f"{metric}_{stat}" for metric, stat in aggregate.columns]
    aggregate["seed_count"] = groups.size()
    return aggregate.reset_index()


def write_markdown_table(frame):
    """Keep Markdown export independent of the optional tabulate package."""
    columns = [str(c) for c in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for values in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(v).replace("|", "/") for v in values) + " |")
    return "\n".join(lines)


def format_estimate(row, field, digits=4):
    mean, sd = row[f"{field}_mean"], row[f"{field}_std"]
    if pd.isna(mean):
        return "not reached"
    if pd.isna(sd):
        return f"{mean:.{digits}f} (n=1)"
    return f"{mean:.{digits}f} ± {sd:.{digits}f}"


def readable_summary(aggregate):
    return pd.DataFrame([{
        "Method": r.method_label, "Workers": r.workers, "Delay": r.scenario,
        "Final train loss": format_estimate(r, "training_loss"),
        "Final val accuracy": format_estimate(r, "validation_accuracy", 3),
        "Simulated time": format_estimate(r, "simulated_time", 1),
        "Mean staleness": format_estimate(r, "mean_staleness", 2),
        "Max staleness": format_estimate(r, "max_staleness", 1),
        "Gradients computed": format_estimate(r, "gradients_computed", 1),
        "Accepted": format_estimate(r, "gradients_accepted", 1),
        "Discarded": format_estimate(r, "gradients_discarded", 1),
        "Discard %": format_estimate(r, "discard_percentage", 1),
    } for _, r in aggregate.iterrows()])


def run_suite(mode, output_dir, time_budget=None, seeds=None):
    output_dir = Path(output_dir)
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    configs = configuration_grid(mode, time_budget, seeds)
    target = 0.50  # Fixed before running: no outcome-dependent target selection.
    start = time.perf_counter()
    histories, all_events, summaries, failures = [], [], [], []
    with threadpool_limits(limits=1):
        problem = make_problem()
        first_result = None
        for index, config in enumerate(configs, 1):
            result = simulate(problem, config)
            issues = audit_result(result)
            if issues:
                failures.append(dict(run_id=config.run_id, issues=issues))
            crossed = result.history[result.history.training_loss <= target]
            result.summary["target_training_loss"] = target
            result.summary["time_to_target"] = float(crossed.iloc[0].simulated_time) if len(crossed) else np.nan
            result.summary["target_reached"] = int(len(crossed) > 0)
            histories.append(frame_with_metadata(result.history, config))
            all_events.append(frame_with_metadata(result.events, config))
            summaries.append(result.summary)
            if first_result is None:
                first_result = result
            if index % 6 == 0 or index == len(configs):
                print(f"[{index:3d}/{len(configs)}] {config.workers:2d} workers, {config.scenario:8s}, "
                      f"{config.method_label}: loss={result.summary['training_loss']:.4f}, "
                      f"elapsed={time.perf_counter() - start:.1f}s", flush=True)

        # Replay the actual data and a nontrivial heterogeneous async configuration.
        reproduction = []
        replay_configs = [first_result.config, RunConfig("async", workers=8, scenario="high", time_budget=20)]
        for config in replay_configs:
            a, b = simulate(problem, config), simulate(problem, config)
            identical = (a.history.equals(b.history) and a.events.equals(b.events)
                         and np.array_equal(a.final_parameters, b.final_parameters))
            reproduction.append(dict(run_id=config.run_id, exact_match=identical))
            if not identical:
                failures.append(dict(run_id=config.run_id, issues=["Exact replay failed"]))

        # Controlled sensitivity: same high-delay distribution and 8 workers,
        # varying only the number of slow workers. This is outside the main grid.
        diagnostic_rows = []
        for seed in sorted({c.seed for c in configs}):
            for fraction in (0.0, 0.125, 0.25, 0.5):
                config = RunConfig("async", workers=8, scenario="high", seed=seed,
                                   time_budget=configs[0].time_budget, slow_fraction=fraction)
                result = simulate(problem, config)
                diagnostic_rows.append(result.summary)
                issues = audit_result(result)
                if issues:
                    failures.append(dict(run_id=config.run_id, issues=issues))

    history, events, summary = pd.concat(histories, ignore_index=True), pd.concat(all_events, ignore_index=True), pd.DataFrame(summaries)
    diagnostics = pd.DataFrame(diagnostic_rows)
    aggregate = aggregate_summary(summary)
    # Raw outputs are retained even if an invariant fails, for investigation.
    history.to_csv(results_dir / "raw_results.csv", index=False)
    events.to_csv(results_dir / "gradient_events.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    summary.to_csv(results_dir / "summary_results.csv", index=False)
    aggregate.to_csv(results_dir / "aggregate_results.csv", index=False)
    diagnostics.to_csv(results_dir / "straggler_diagnostics.csv", index=False)
    (results_dir / "summary_results.md").write_text(write_markdown_table(readable_summary(aggregate)) + "\n")
    checks = empirical_checks(summary)
    checks.to_csv(results_dir / "empirical_checks.csv", index=False)
    sanity = dict(algorithm_invariants_passed=not failures, runs_checked=len(configs) + len(diagnostic_rows),
                  failures=failures, reproducibility=reproduction,
                  empirical_expectations=checks.to_dict(orient="records"))
    (results_dir / "sanity_checks.json").write_text(json.dumps(sanity, indent=2) + "\n")
    if failures:
        raise RuntimeError(f"Algorithm sanity checks failed; inspect {results_dir / 'sanity_checks.json'}")

    root = Path(__file__).resolve().parents[1]
    source_paths = sorted([root / "main.py", root / "requirements.txt", root / "requirements-lock.txt",
                           *root.glob("src/*.py"), *root.glob("experiments/*.py"),
                           *root.glob("scripts/*.py"), *root.glob("tests/*.py")])
    source_hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    versions = {name: importlib.metadata.version(name) for name in
                ("numpy", "scipy", "scikit-learn", "pandas", "matplotlib", "threadpoolctl")}
    fingerprint = hashlib.sha256()
    for array in (problem.x_train, problem.y_train, problem.x_validation, problem.y_validation):
        fingerprint.update(array.tobytes())
    manifest = dict(
        mode=mode, dataset=dict(samples=6000, features=30, train_samples=4800, validation_samples=1200,
                               dataset_seed=2026, split_seed=2027, l2=problem.l2,
                               sha256=fingerprint.hexdigest()),
        configurations=[asdict(c) for c in configs], seeds=sorted({c.seed for c in configs}),
        main_runs=len(configs), diagnostic_runs=len(diagnostic_rows),
        target_training_loss=target, delays=DELAY_DESCRIPTION,
        python=platform.python_version(), packages=versions, source_sha256=source_hashes,
        experiment_runtime_seconds=time.perf_counter() - start,
        uncertainty="Sample standard deviation across training/delay seeds; dataset fixed. Single-seed SD undefined.",
    )
    (results_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Sanity checks passed for {sanity['runs_checked']} runs; exact replay passed.", flush=True)
    return history, events, summary, aggregate, diagnostics, manifest, sanity
