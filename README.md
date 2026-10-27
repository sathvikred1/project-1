# Asynchrony / Delay-Robust Optimization

This project studies how worker delays and stale gradients affect distributed stochastic optimization. We use an event-driven simulation to compare synchronous SGD with several asynchronous optimization approaches under different levels of worker delay.

The implementation is mainly inspired by two recent papers:

1. **Faster Stochastic Optimization with Arbitrary Delays via Adaptive Asynchronous Mini-Batching** — Attia, Gaash & Koren, ICML 2025
2. **Ringmaster ASGD: The First Asynchronous SGD with Optimal Time Complexity** — Maranjyan, Tyurin & Richtárik, ICML 2025

This project is a simplified educational implementation for studying the main ideas. It is not intended to be an exact reproduction of either paper.

---

## Methods Implemented

We compare four main optimization approaches.

### 1. Synchronous SGD

All workers compute gradients, but the system waits for every worker before performing the next model update.

This means that a slow worker can delay the entire optimization process.

### 2. Vanilla Asynchronous SGD

Workers operate independently. When a worker finishes computing a gradient, that gradient is immediately applied to the current model.

The gradient is computed using the model snapshot available when the worker started its task. Therefore, slow workers may return gradients based on older versions of the model.

### 3. Delay-Filtered Asynchronous SGD

This method is similar to vanilla Async-SGD, but very stale gradients can be rejected.

We experiment with three staleness thresholds:

- tau = 2
- tau = 5
- tau = 10

This allows us to study the trade-off between using more gradients and rejecting outdated information.

### 4. Simplified Asynchronous Mini-Batch SGD

This method collects accepted asynchronous gradients and averages them in groups of four before performing a model update.

This is inspired by adaptive asynchronous mini-batching, but our implementation uses a fixed batch size and fixed staleness threshold.

---

# Experimental Setup

The optimization problem is binary logistic regression on a synthetic classification dataset.

The full experiment uses:

- 6,000 total samples
- 30 features
- 4,800 training samples
- 1,200 validation samples
- L2 regularization coefficient = 0.001
- Learning rate = 0.15
- Worker gradient batch size = 32
- Worker counts = 4, 8, and 16
- Three delay conditions: low, moderate, and high heterogeneity
- Three random seeds: 11, 22, and 33
- Full simulated time horizon = 150 units

The full experiment contains **162 main configurations**, along with additional controlled straggler experiments.

The dataset and train/validation split are fixed across the methods so that the methods can be compared under the same data conditions.

---

# Running the Project

## Install Dependencies

From the project directory:

```bash
python -m pip install -r requirements.txt
```

For running the tests:

```bash
python -m pip install pytest
```

---

## Run Tests

Before running the experiments:

```bash
python -m pytest -q
```

The committed experiment artifacts report that all **12 unit tests passed**.

---

## Quick Experiment

For a smaller test run:

```bash
python main.py --quick
```

Quick mode runs:

- 54 configurations
- 1 random seed
- 40 simulated time units

The results are saved under:

```text
quick/results/
quick/plots/
```

This mode is useful for checking that the implementation works before running the complete experiment.

---

## Full Experiment

Run:

```bash
python main.py --full
```

Full mode uses:

- 162 main runs
- 3 random seeds
- 150 simulated time units

The generated results are saved in:

```text
results/
```

and figures are saved in:

```text
plots/
```

---

# Experimental Results

The following results focus on the important case of **8 workers under high delay heterogeneity**.

The values are averages across three random seeds.

| Method | Train Loss | Validation Accuracy | Updates | Mean / Max Staleness | Discard % |
| --- | ---: | ---: | ---: | ---: | ---: |
| Synchronous SGD | 0.4002 | 0.860 | 20.0 | 0.00 / 0.0 | 0.0% |
| Vanilla Async-SGD | 0.3225 | 0.872 | 937.0 | 6.93 / 67.3 | 0.0% |
| Filtered Async, tau=2 | 0.3227 | 0.875 | 464.3 | 3.43 / 35.0 | 50.4% |
| Filtered Async, tau=5 | 0.3222 | 0.871 | 720.0 | 5.32 / 53.3 | 23.2% |
| Filtered Async, tau=10 | 0.3225 | 0.872 | 874.0 | 6.46 / 64.3 | 6.7% |
| Async Mini-Batch, B=4, tau=5 | **0.3210** | **0.876** | 223.0 | **1.65 / 16.3** | 4.6% |

---

# Main Observations

## 1. Synchronous vs. Asynchronous SGD

One of the clearest differences appears when worker speeds are highly heterogeneous.

With 8 workers and the same simulated time budget:

- Synchronous SGD completed about **20 updates**
- Vanilla Async-SGD completed about **937 updates**

Their final mean training losses were:

- Synchronous SGD: **0.4002**
- Vanilla Async-SGD: **0.3225**

The main reason is that synchronous SGD must wait for the slowest worker before continuing.

Async-SGD can continue processing gradients from faster workers instead of waiting.

However, the synchronous and asynchronous methods use different effective update batch sizes, so this result should not be interpreted as a perfectly batch-matched performance comparison.

---

## 2. Effect of the Staleness Threshold

The filtering experiments show an interesting trade-off.

For 8 workers under high heterogeneity:

| Threshold | Train Loss | Gradients Discarded |
| ---: | ---: | ---: |
| tau=2 | 0.3227 | 50.4% |
| tau=5 | **0.3222** | 23.2% |
| tau=10 | 0.3225 | 6.7% |

A very strict threshold was not automatically better.

With tau=2, approximately half of the gradients were discarded. However, this did not produce the lowest final loss.

Among the filtering thresholds tested, **tau=5 produced the lowest mean final training loss**.

This suggests that there is a trade-off between removing stale information and throwing away potentially useful gradient work.

These comparisons are descriptive results from three seeds and are not statistical-significance claims.

---

## 3. Effect of Increasing the Number of Workers

For vanilla Async-SGD under high heterogeneity:

| Workers | Train Loss | Mean Staleness | Max Staleness | Completed Gradients |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 0.3222 | 2.97 | 30.7 | 471.7 |
| 8 | 0.3225 | 6.93 | 67.3 | 937.0 |
| 16 | 0.3225 | 14.87 | 164.3 | 1890.0 |

Increasing the number of workers greatly increased the number of completed gradients.

However, it also increased gradient staleness.

Mean staleness increased approximately from:

```text
2.97 -> 6.93 -> 14.87
```

while maximum staleness increased from:

```text
30.7 -> 67.3 -> 164.3
```

This demonstrates an important trade-off in asynchronous optimization.

More workers can increase the amount of work completed, but gradients can become increasingly outdated before they reach the global model.

In this experiment, increasing the worker count also did not continuously improve the final training loss.

---

# Unexpected Result: Mean vs. Tail Staleness

One interesting result was that increasing worker-delay heterogeneity did **not** noticeably increase the average completed-gradient staleness.

For vanilla Async-SGD with 8 workers:

```text
Low heterogeneity mean staleness  = 6.98
High heterogeneity mean staleness = 6.93
```

At first, we expected the mean staleness to increase.

However, the tail of the staleness distribution changed significantly.

Mean maximum staleness increased from:

```text
11.7 -> 67.3
```

and mean p95 staleness increased from:

```text
9.00 -> 16.53
```

This indicates that worker heterogeneity creates a smaller number of extremely stale gradients even when the overall mean stays relatively stable.

In a closed system with continuously active workers, average update-based staleness can remain close to the number of other active workers. Because of this, statistics such as maximum and 95th-percentile staleness can provide more useful information about severe worker delays than the mean alone.

The experiment keeps this unexpected result instead of changing the simulation to force the originally expected trend.

---

# Simplified Async Mini-Batching

The simplified asynchronous mini-batch method also produced interesting results.

With 8 workers under high heterogeneity:

| Method | Train Loss | Updates | Mean Arrival Staleness |
| --- | ---: | ---: | ---: |
| Vanilla Async-SGD | 0.3225 | 937.0 | 6.93 |
| Async Mini-Batch | **0.3210** | 223.0 | **1.65** |

The mini-batch method performs fewer model updates because it combines four accepted gradients before updating the model.

Therefore, its update count and update-based staleness should not be compared directly with vanilla Async-SGD without considering this difference.

Our implementation is intentionally simplified and does not reproduce the complete adaptive mini-batching algorithm from the original paper.

---

# Verification and Sanity Checks

The generated full-run artifacts contain several checks to make sure that the results come from the actual simulation.

The recorded verification includes:

- **162 main experimental runs**
- **54 aggregate configurations**
- **174 runs checked** including diagnostic experiments
- **12 unit tests passed**
- Algorithm invariants passed for all checked runs
- Exact reproducibility checks passed
- Training loss decreased from its initial value in **162/162 main runs**
- Filtering was active in **76/81 applicable runs**
- Source hashes matched the experiment manifest
- Summary statistics matched independent recalculation
- Quick-run events matched prefixes of the corresponding full runs
- 12 PNG figures generated
- 12 vector PDF figures generated

Detailed verification information can be found in:

```text
results/sanity_checks.json
results/artifact_verification.json
```

---

# Important Result Files

The most useful generated files are:

### `results/week9_summary.md`

Automatically generated readable summary of the experimental setup, results, observations, sanity checks, limitations, and future experiments.

### `results/summary_results.csv`

Individual experimental results for each random seed.

### `results/aggregate_results.csv`

Mean and sample standard deviation across seeds for each experimental configuration.

### `results/raw_results.csv`

Complete optimization history recorded during the experiments.

### `results/gradient_events.csv.gz`

Individual worker gradient-arrival events, including gradient staleness and whether the gradient was accepted.

### `results/paired_comparisons.csv`

Paired comparisons between selected methods.

### `results/empirical_checks.csv`

Checks whether expected experimental trends were actually observed.

### `results/sanity_checks.json`

Algorithm invariants and reproducibility results.

### `results/run_manifest.json`

Experiment configuration, package versions, dataset fingerprint, seeds, and source-code hashes.

### `results/artifact_verification.json`

Verification of the generated experiment artifacts.

---

# Important Figures

The experiment generates several plots.

Some of the most useful figures for the project report are:

```text
plots/02_loss_vs_time.png
```

Shows optimization loss against simulated wall-clock time.

```text
plots/06_worker_scaling.png
```

Shows the effect of using 4, 8, and 16 workers.

```text
plots/07_threshold_comparison.png
```

Shows the effect of different stale-gradient filtering thresholds.

```text
plots/10_tail_staleness.png
```

Shows how heterogeneous worker delays affect the tail of the staleness distribution.

```text
plots/12_loss_vs_gradient_work.png
```

Compares optimization progress relative to gradient computation work.

---

# Limitations

There are several limitations to the current experiment.

First, this is an event-driven simulation rather than a real distributed computing cluster. Computation is performed locally while worker completion times are simulated.

The experiment therefore does not include:

- Network communication cost
- Server bottlenecks
- Hardware contention
- Worker failures
- Scheduling overhead
- Cancellation overhead

The experiments also use one synthetic convex logistic-regression problem and a fixed learning rate.

In addition, the methods do not have identical effective update batch sizes. Synchronous SGD averages gradients from all workers, vanilla Async-SGD applies one gradient at a time, and the simplified mini-batch method combines four gradients.

Therefore, these experiments should be viewed as a controlled study of **worker delays, throughput, gradient staleness, and stale-gradient filtering**, rather than proof that one method is universally better than another.

---

# Future Work

Possible next experiments include:

1. Testing multiple learning rates using a prespecified common learning-rate sweep.
2. Comparing methods under matched effective batch sizes.
3. Increasing the number of random seeds.
4. Testing additional datasets.
5. Studying non-convex optimization problems.
6. Simulating time-varying worker speeds.
7. Adding communication and server overhead.
8. Testing larger numbers of workers.
9. Implementing the complete adaptive mini-batching procedure from the research paper.
10. Implementing more of the theoretical Ringmaster ASGD parameter-selection procedure.

---

# References

**Amit Attia, Ofir Gaash, and Tomer Koren.**  
*Faster Stochastic Optimization with Arbitrary Delays via Adaptive Asynchronous Mini-Batching.*  
International Conference on Machine Learning (ICML), 2025.

**Artavazd Maranjyan, Alexander Tyurin, and Peter Richtárik.**  
*Ringmaster ASGD: The First Asynchronous SGD with Optimal Time Complexity.*  
International Conference on Machine Learning (ICML), 2025.
