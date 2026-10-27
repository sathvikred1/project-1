"""Serial event queue that models concurrent worker computations exactly."""
from dataclasses import dataclass
import heapq

import numpy as np
import pandas as pd

from .async_minibatch import AsyncMiniBatchSGD
from .async_sgd import AsyncSGD
from .delay_filtered_sgd import DelayFilteredSGD
from .metrics import Metrics
from .synchronous_sgd import SynchronousSGD
from .workers import Worker


@dataclass
class SimulationResult:
    config: object
    history: pd.DataFrame
    events: pd.DataFrame
    final_parameters: np.ndarray
    summary: dict


def simulate(problem, config):
    if config.method == "sync":
        rule = SynchronousSGD(config.workers)
    elif config.method == "async":
        rule = AsyncSGD()
    elif config.method == "filtered":
        rule = DelayFilteredSGD(config.threshold)
    else:
        rule = AsyncMiniBatchSGD(config.minibatch_gradients, config.threshold)

    parameters = np.zeros(problem.dimension)  # Identical initialization for every run.
    version, started = 0, 0
    workers = [Worker(i, config) for i in range(config.workers)]
    queue = []
    metrics = Metrics()

    def schedule(worker, time):
        nonlocal started
        event = worker.start(parameters, version, time, problem, config.sample_batch_size)
        heapq.heappush(queue, event)  # Earliest completion wins; ties use worker ID.
        started += 1

    for worker in workers:
        schedule(worker, 0.0)
    metrics.record(problem, parameters, version, 0.0, started)

    while queue and queue[0].finish_time <= config.time_budget:
        event = heapq.heappop(queue)
        # Only APPLIED model updates increment version; discarded arrivals do not.
        staleness = version - event.model_version
        accepted = rule.accepts(staleness)
        metrics.arrival(event, version, accepted)
        averaged = rule.add(event.gradient) if accepted else None
        updated = averaged is not None
        if updated:
            parameters -= config.learning_rate * averaged
            version += 1
            metrics.used += rule.gradients_per_update
            if not np.all(np.isfinite(parameters)):
                raise FloatingPointError(f"Nonfinite parameters in {config.run_id}")
        metrics.events[-1]["update_applied"] = updated
        metrics.events[-1]["version_after"] = version

        if event.finish_time < config.time_budget:
            if rule.synchronous:
                # Fast workers stay idle until the slowest finishes the round.
                if updated:
                    for worker in workers:
                        schedule(worker, event.finish_time)
            else:
                # Even after rejection, this worker starts again on current state.
                schedule(workers[event.worker_id], event.finish_time)

        if updated and version % config.log_every_updates == 0:
            metrics.record(problem, parameters, version, event.finish_time, started)

    # All methods receive exactly the same simulated horizon. In-flight tasks
    # and incomplete accepted buffers are counted, never flushed at zero cost.
    metrics.record(problem, parameters, version, config.time_budget, started)
    history = pd.DataFrame(metrics.history)
    events = pd.DataFrame(metrics.events, columns=[
        "event_number", "worker_id", "task_id", "start_time", "finish_time",
        "computation_delay", "snapshot_version", "arrival_version", "staleness",
        "accepted", "update_applied", "version_after",
    ])
    summary = dict(config.metadata(), **history.iloc[-1].to_dict())
    summary["outstanding_model_age_sum"] = sum(version - event.model_version for event in queue)
    summary["mean_computation_delay"] = float(events.computation_delay.mean()) if len(events) else 0.0
    summary["last_update_time"] = float(
        next((e["finish_time"] for e in reversed(metrics.events)
              if e.get("update_applied", False)), 0.0)
    )
    return SimulationResult(config, history, events, parameters.copy(), summary)
