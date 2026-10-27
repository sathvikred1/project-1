"""Separate algorithm invariants from scientific hypotheses that may be false."""
import numpy as np
import pandas as pd


def audit_result(result):
    cfg, events, history, summary = result.config, result.events, result.history, result.summary
    issues = []

    def check(condition, message):
        if not condition:
            issues.append(message)

    check(np.isfinite(history.select_dtypes(include="number")).all().all(), "Nonfinite metric")
    check(history.simulated_time.is_monotonic_increasing, "Clock moved backwards")
    check(history.model_updates.is_monotonic_increasing, "Model version moved backwards")
    check(history.iloc[-1].simulated_time == cfg.time_budget, "Unequal time budget")
    check(summary["gradients_computed"] == summary["gradients_accepted"] + summary["gradients_discarded"],
          "Completed gradients do not partition into accepted and discarded")
    check(summary["gradients_accepted"] == summary["gradients_used"] + summary["gradients_pending"],
          "Accepted gradients do not partition into used and pending")
    check(summary["gradients_started"] == summary["gradients_computed"] + summary["gradients_in_flight"],
          "Started/completed/in-flight accounting mismatch")
    check(len(events) == summary["gradients_computed"], "Event count mismatch")
    if events.empty:
        return issues
    check(events.finish_time.is_monotonic_increasing, "Completions out of order")
    check((events.finish_time <= cfg.time_budget).all(), "Arrival beyond horizon")
    check((events.finish_time > events.start_time).all(), "Nonpositive computation delay")
    check((events.staleness == events.arrival_version - events.snapshot_version).all(), "Incorrect staleness")
    check((events.staleness >= 0).all(), "Negative staleness")
    check(events.iloc[0].arrival_version == 0, "Wrong initial version")
    check((events.version_after == events.arrival_version + events.update_applied.astype(int)).all(),
          "Incorrect version increment")
    check(np.array_equal(events.arrival_version.to_numpy()[1:], events.version_after.to_numpy()[:-1]),
          "Version discontinuity")
    check(events.update_applied.sum() == summary["model_updates"], "Update count mismatch")
    for _, worker_events in events.groupby("worker_id"):
        check(np.array_equal(worker_events.task_id, np.arange(1, len(worker_events) + 1)), "Worker task discontinuity")
        if not cfg.method == "sync":
            check(np.allclose(worker_events.start_time.to_numpy()[1:], worker_events.finish_time.to_numpy()[:-1]),
                  "Async worker did not immediately restart")
            check(np.array_equal(worker_events.snapshot_version.to_numpy()[1:], worker_events.version_after.to_numpy()[:-1]),
                  "Async restart used wrong model version")
    if cfg.threshold is None:
        check(events.accepted.all(), "Unfiltered method discarded gradients")
    else:
        check((events.accepted == (events.staleness <= cfg.threshold)).all(), "Threshold rule violated")
    if cfg.method == "sync":
        check((events.staleness == 0).all(), "Synchronous staleness was nonzero")
        for _, group in events.groupby("snapshot_version"):
            check(group.start_time.nunique() == 1, "Sync round did not share start time")
            if group.update_applied.any():
                check(len(group) == cfg.workers, "Sync updated before all workers finished")
                check(group.iloc[-1].update_applied, "Sync update did not wait for slowest worker")
    if cfg.method == "async":
        # Each of U updates ages W-1 other tasks. Their accumulated age is
        # either delivered in completed jobs or still held by outstanding jobs.
        check(events.staleness.sum() + summary["outstanding_model_age_sum"]
              == (cfg.workers - 1) * summary["model_updates"],
              "Closed-worker-pool staleness accounting identity failed")
    batch_size = (cfg.workers if cfg.method == "sync" else
                  cfg.minibatch_gradients if cfg.method == "minibatch" else 1)
    check(summary["gradients_used"] == summary["model_updates"] * batch_size, "Averaging count mismatch")
    check(0 <= summary["gradients_pending"] < batch_size, "Invalid pending buffer size")
    return issues


def empirical_checks(summary):
    """Report expectations as observations, never force a desired conclusion."""
    rows = []
    async_rows = summary[summary.method == "async"]
    for workers, group in async_rows.groupby("workers"):
        by_scenario = group.groupby("scenario")[["mean_staleness", "max_staleness", "p95_staleness"]].mean()
        low, high = by_scenario.loc["low"], by_scenario.loc["high"]
        for metric in by_scenario.columns:
            rows.append(dict(
                check=f"High vs low delay: {metric}, {workers} workers",
                expected="higher under high heterogeneity",
                observed=f"low={low[metric]:.4f}; high={high[metric]:.4f}",
                status="SUPPORTED" if high[metric] > low[metric] else "NOT_SUPPORTED",
            ))
    initial_loss = float(np.log(2))
    improved = int((summary.training_loss < initial_loss).sum())
    rows.append(dict(check="Final training loss below initial loss", expected="all runs",
                     observed=f"{improved}/{len(summary)} runs", status="SUPPORTED" if improved == len(summary) else "NOT_SUPPORTED"))
    filtered = summary[summary.method == "filtered"]
    rejected = int((filtered.gradients_discarded > 0).sum())
    rows.append(dict(check="Filtering is active", expected="some runs discard gradients",
                     observed=f"{rejected}/{len(filtered)} runs discard", status="SUPPORTED" if rejected else "NOT_SUPPORTED"))
    return pd.DataFrame(rows)
