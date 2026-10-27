from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class RunConfig:
    method: str
    workers: int = 8
    scenario: str = "high"
    seed: int = 11
    time_budget: float = 150.0
    learning_rate: float = 0.15
    sample_batch_size: int = 32
    threshold: int | None = None
    minibatch_gradients: int = 4
    log_every_updates: int = 1
    # Only used by the separate controlled straggler-count diagnostic.
    slow_fraction: float | None = None

    def __post_init__(self):
        if self.method not in {"sync", "async", "filtered", "minibatch"}:
            raise ValueError(f"Unknown method: {self.method}")
        if self.scenario not in {"low", "moderate", "high"}:
            raise ValueError(f"Unknown scenario: {self.scenario}")
        if self.workers < 1 or self.sample_batch_size < 1 or self.minibatch_gradients < 1:
            raise ValueError("Worker and batch counts must be positive")
        if not math.isfinite(self.time_budget) or self.time_budget <= 0:
            raise ValueError("time_budget must be finite and positive")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if self.seed < 0 or self.log_every_updates < 1:
            raise ValueError("seed must be nonnegative and log interval positive")
        if self.method in {"filtered", "minibatch"}:
            if self.threshold is None or self.threshold < 0:
                raise ValueError("Filtered methods require a nonnegative threshold")
        elif self.threshold is not None:
            raise ValueError("Unfiltered methods must not specify a threshold")
        if self.slow_fraction is not None and not 0 <= self.slow_fraction <= 1:
            raise ValueError("slow_fraction must be between zero and one")

    @property
    def method_label(self):
        if self.method == "filtered":
            return f"Filtered (tau={self.threshold})"
        if self.method == "minibatch":
            return f"Async-MB (B={self.minibatch_gradients}, tau={self.threshold})"
        return {"sync": "Synchronous", "async": "Vanilla async"}[self.method]

    @property
    def run_id(self):
        suffix = "" if self.slow_fraction is None else f"_slow{self.slow_fraction:g}"
        return f"{self.method}_tau{self.threshold}_b{self.minibatch_gradients}_w{self.workers}_{self.scenario}_s{self.seed}{suffix}"

    def metadata(self):
        return dict(asdict(self), run_id=self.run_id, method_label=self.method_label)
