import numpy as np


class Metrics:
    def __init__(self):
        self.accepted = 0
        self.discarded = 0
        self.used = 0
        self.staleness = []
        self.accepted_staleness = []
        self.events = []
        self.history = []

    def arrival(self, event, version, accepted):
        age = version - event.model_version
        self.staleness.append(age)
        if accepted:
            self.accepted += 1
            self.accepted_staleness.append(age)
        else:
            self.discarded += 1
        self.events.append({
            "event_number": len(self.events) + 1, "worker_id": event.worker_id,
            "task_id": event.task_id, "start_time": event.start_time,
            "finish_time": event.finish_time, "computation_delay": event.finish_time - event.start_time,
            "snapshot_version": event.model_version, "arrival_version": version,
            "staleness": age, "accepted": accepted,
        })

    def counters(self):
        computed = self.accepted + self.discarded
        return {
            "gradients_computed": computed, "gradients_accepted": self.accepted,
            "gradients_discarded": self.discarded, "gradients_used": self.used,
            "gradients_pending": self.accepted - self.used,
            "discard_percentage": 100 * self.discarded / computed if computed else 0.0,
            "mean_staleness": float(np.mean(self.staleness)) if computed else 0.0,
            "max_staleness": int(max(self.staleness, default=0)),
            "p95_staleness": float(np.quantile(self.staleness, 0.95)) if computed else 0.0,
            "mean_accepted_staleness": (float(np.mean(self.accepted_staleness))
                                        if self.accepted_staleness else 0.0),
        }

    def record(self, problem, parameters, version, time, started):
        counters = self.counters()
        self.history.append(dict(
            model_updates=version, simulated_time=time,
            gradients_started=started,
            gradients_in_flight=started - counters["gradients_computed"],
            **problem.evaluate(parameters), **counters,
        ))
