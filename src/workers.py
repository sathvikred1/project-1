"""Independent per-worker streams pair task samples and delays across methods."""
from dataclasses import dataclass, field

import numpy as np


DELAY_DESCRIPTION = {
    "low": "All workers: mean 1.0, lognormal CV 0.10",
    "moderate": "75%: mean 1.0; 25%: mean 2.5; lognormal CV 0.20",
    "high": "75%: mean 1.0; 25%: mean 6.0; lognormal CV 0.35",
}


@dataclass(order=True)
class Completion:
    finish_time: float
    worker_id: int
    task_id: int
    model_version: int = field(compare=False)
    start_time: float = field(compare=False)
    gradient: np.ndarray = field(compare=False, repr=False)
    # Retaining the snapshot makes the stale-gradient semantics inspectable.
    snapshot: np.ndarray = field(compare=False, repr=False)


class Worker:
    def __init__(self, worker_id, config):
        self.worker_id = worker_id
        self.task_id = 0
        self.batch_rng = np.random.default_rng(np.random.SeedSequence([config.seed, worker_id, 0]))
        self.delay_rng = np.random.default_rng(np.random.SeedSequence([config.seed, worker_id, 1]))
        fraction = config.slow_fraction if config.slow_fraction is not None else 0.25
        slow_count = int(round(config.workers * fraction))
        self.slow = worker_id >= config.workers - slow_count
        self.mean_delay = (1.0 if config.scenario == "low" or not self.slow else
                           {"moderate": 2.5, "high": 6.0}[config.scenario])
        self.delay_cv = {"low": 0.10, "moderate": 0.20, "high": 0.35}[config.scenario]

    def start(self, parameters, version, time, problem, batch_size):
        # Take the model snapshot NOW, before any other worker can update it.
        snapshot = parameters.copy()
        indices = self.batch_rng.integers(0, len(problem.y_train), size=batch_size)
        gradient = problem.gradient(snapshot, indices)
        # The lognormal is parameterized by its actual arithmetic mean and CV.
        sigma = np.sqrt(np.log1p(self.delay_cv ** 2))
        mu = np.log(self.mean_delay) - sigma ** 2 / 2
        delay = float(self.delay_rng.lognormal(mu, sigma))
        self.task_id += 1
        # The simulator schedules this stored gradient; it never recalculates it.
        return Completion(time + delay, self.worker_id, self.task_id,
                          version, time, gradient, snapshot)
