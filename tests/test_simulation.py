"""Hand-computable cases catch stale-gradient and barrier errors."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from src.config import RunConfig
from src.objective import LogisticProblem
from src.simulation import simulate
from src.validation import audit_result
from src.workers import Worker


@pytest.fixture
def simple_problem():
    # One constant feature and one intercept; y=1 gives g(w=0)=(-1/2,-1/2).
    x = np.ones((8, 2))
    y = np.ones(8)
    return LogisticProblem(x, y, x.copy(), y.copy(), l2=0)


@pytest.fixture
def fixed_delays(monkeypatch):
    original = Worker.start

    def start(self, parameters, version, time, problem, batch_size):
        event = original(self, parameters, version, time, problem, batch_size)
        event.finish_time = time + (1.0 if self.worker_id == 0 else 2.5)
        return event

    monkeypatch.setattr(Worker, "start", start)


def cfg(method, **kwargs):
    return RunConfig(method, workers=2, time_budget=2.5, learning_rate=0.1,
                     log_every_updates=1, **kwargs)


def test_logistic_gradient_finite_difference():
    rng = np.random.default_rng(3)
    x = np.column_stack([rng.normal(size=(13, 4)), np.ones(13)])
    y = rng.integers(0, 2, 13)
    problem = LogisticProblem(x, y, x, y, l2=0.07)
    w = rng.normal(size=5)
    numerical = np.empty(5)
    for j in range(5):
        step = np.eye(5)[j] * 1e-6
        numerical[j] = (problem.loss(w + step) - problem.loss(w - step)) / 2e-6
    np.testing.assert_allclose(problem.gradient(w, np.arange(13)), numerical, rtol=1e-6, atol=1e-8)


def test_async_applies_original_snapshot(simple_problem, fixed_delays):
    result = simulate(simple_problem, cfg("async"))
    first = np.full(2, 0.05)
    second = first - 0.1 * (expit(first.sum()) - 1)
    expected = second + 0.05  # Slow worker returns its ORIGINAL gradient -0.5.
    np.testing.assert_allclose(result.final_parameters, expected)
    assert list(result.events.staleness) == [0, 0, 2]
    assert not audit_result(result)


def test_filtered_rejects_old_gradient(simple_problem, fixed_delays):
    result = simulate(simple_problem, cfg("filtered", threshold=1))
    assert list(result.events.accepted) == [True, True, False]
    assert result.summary["model_updates"] == 2
    assert result.summary["gradients_discarded"] == 1
    assert not audit_result(result)


def test_threshold_is_inclusive(simple_problem, fixed_delays):
    result = simulate(simple_problem, cfg("filtered", threshold=2))
    assert result.events.accepted.all()
    assert result.events.iloc[-1].staleness == 2


def test_sync_waits_for_slowest(simple_problem, fixed_delays):
    result = simulate(simple_problem, cfg("sync"))
    np.testing.assert_allclose(result.final_parameters, [0.05, 0.05])
    assert result.summary["last_update_time"] == 2.5
    assert result.summary["model_updates"] == 1
    assert result.events.staleness.max() == 0
    assert not audit_result(result)


def test_minibatch_averages_and_keeps_unfinished_buffer(simple_problem, fixed_delays):
    result = simulate(simple_problem, cfg("minibatch", threshold=1, minibatch_gradients=2))
    np.testing.assert_allclose(result.final_parameters, [0.05, 0.05])
    assert result.summary["gradients_used"] == 2
    assert result.summary["gradients_pending"] == 1
    assert result.summary["gradients_computed"] == 3
    assert result.summary["model_updates"] == 1
    assert not audit_result(result)


def test_no_arrivals_before_horizon(simple_problem, fixed_delays):
    result = simulate(simple_problem, replace(cfg("async"), time_budget=0.5))
    assert result.events.empty
    assert result.summary["gradients_in_flight"] == 2
    assert result.summary["model_updates"] == 0
    assert not audit_result(result)


@pytest.mark.parametrize("method,threshold", [("sync", None), ("async", None), ("filtered", 2), ("minibatch", 2)])
def test_exact_reproducibility(simple_problem, method, threshold):
    config = RunConfig(method, threshold=threshold, time_budget=12)
    first, second = simulate(simple_problem, config), simulate(simple_problem, config)
    pd.testing.assert_frame_equal(first.history, second.history, check_exact=True)
    pd.testing.assert_frame_equal(first.events, second.events, check_exact=True)
    np.testing.assert_array_equal(first.final_parameters, second.final_parameters)
    assert not audit_result(first)


def test_more_workers_changes_model_age(simple_problem):
    means = [simulate(simple_problem, RunConfig("async", workers=n, scenario="low", time_budget=20)).summary["mean_staleness"]
             for n in (4, 8, 16)]
    assert means[0] < means[1] < means[2]
