# Week 9: Asynchrony / Delay-Robust Optimization

This project is a simplified educational simulation inspired by recent research on asynchronous and delay-robust stochastic optimization. It is not an exact reproduction of the original papers.

Run mode: **full**. All numerical observations below come from executed runs in this output directory. Paper 3 is outside the implementation.

## A. Experimental setup

- Objective: binary logistic regression with L2 coefficient 0.001; intercept unregularized. Losses include the same L2 term on both splits.
- Dataset: `sklearn.datasets.make_classification`, 6,000 samples, 30 features (20 informative, 5 redundant), class separation 1.1, label-flip probability 0.04. Fixed dataset seed 2026.
- Split: stratified 80/20 (4,800 training; 1,200 validation), split seed 2027. Standardization fits training data only. No external download or test-set tuning.
- Every method: zero initialization, constant learning rate 0.15, 32 training examples sampled with replacement per worker gradient. Independent per-worker random streams pair samples and delay draws across methods by seed, worker, and task number.
- Workers: 4, 8, 16. Delay-filter thresholds: inclusive ages 2, 5, 10. Main comparison uses threshold 5, fixed before execution. Async mini-batching averages B=4 accepted task gradients, with inclusive threshold 5.
- Training/delay seeds: [11, 22, 33] (3 seeds); 162 main configurations plus 12 controlled straggler diagnostics.
- Stopping: common fixed horizon of 150 simulated time units, independent of loss and method. Started-but-unfinished jobs and accepted-but-uncommitted buffers are recorded without a free final update.
- Evaluation: initial state, every applied model update, and the horizon. Training/validation losses, accuracy, counts, and staleness are saved. Every gradient arrival is saved separately.
- Dispersion: sample standard deviation across training/delay seeds, with the dataset held fixed. This is neither a confidence interval nor dataset-to-dataset uncertainty. Single-seed quick results have undefined SD.
- Synchronous steps average one task gradient per worker (effective sample batch 32W); vanilla/filtered steps use 32 samples; async mini-batch steps average 128 samples. Equal learning rates do not equalize effective batch size or the number of updates. Time and gradient-work plots show different aspects of this trade-off.

Delay laws are independent positive lognormal draws, parameterized by arithmetic mean and coefficient of variation (CV):

| Scenario | Distribution |
| --- | --- |
| low | All workers: mean 1.0, lognormal CV 0.10 |
| moderate | 75%: mean 1.0; 25%: mean 2.5; lognormal CV 0.20 |
| high | 75%: mean 1.0; 25%: mean 6.0; lognormal CV 0.35 |

The last W/4 worker IDs are slow in moderate/high scenarios; each worker retains its role. The separate diagnostic varies slow-worker counts 0, 1, 2, 4 of 8 at the same high-scenario CV.

## B. Preliminary results

High heterogeneity, 8 workers, equal horizon 150. Values are means ± sample SD where available. Age means and maxima below summarize all completed arrivals, including discarded arrivals; the max-age column averages each seed's maximum.

| Method | Train loss | Val loss | Val accuracy | Updates | Mean / max age | Discard % |
| --- | --- | --- | --- | --- | --- | --- |
| Synchronous | 0.4002 ± 0.0079 | 0.3874 ± 0.0084 | 0.860 ± 0.007 | 20.0 ± 2.0 | 0.00 / 0.0 | 0.0 ± 0.0 |
| Vanilla async | 0.3225 ± 0.0019 | 0.3046 ± 0.0034 | 0.872 ± 0.004 | 937.0 ± 12.2 | 6.93 / 67.3 | 0.0 ± 0.0 |
| Filtered (tau=2) | 0.3227 ± 0.0017 | 0.3054 ± 0.0037 | 0.875 ± 0.002 | 464.3 ± 6.5 | 3.43 / 35.0 | 50.4 ± 0.4 |
| Filtered (tau=5) | 0.3222 ± 0.0015 | 0.3044 ± 0.0044 | 0.871 ± 0.004 | 720.0 ± 9.5 | 5.32 / 53.3 | 23.2 ± 0.4 |
| Filtered (tau=10) | 0.3225 ± 0.0016 | 0.3046 ± 0.0028 | 0.872 ± 0.005 | 874.0 ± 14.1 | 6.46 / 64.3 | 6.7 ± 0.5 |
| Async-MB (B=4, tau=5) | 0.3210 ± 0.0004 | 0.3044 ± 0.0014 | 0.876 ± 0.001 | 223.0 ± 2.6 | 1.65 / 16.3 | 4.6 ± 0.2 |

See `summary_results.csv` for individual seeds, `aggregate_results.csv` for all 54 configuration means and SDs, and `summary_results.md` for the complete readable table.

## C. Main observations

- **Sync vs async, low delay, 8 workers:** vanilla async ended at loss 0.3228 versus synchronous 0.3266 (lower at the same time budget), with 1193.3 versus 129.3 mean updates. These methods use different effective update batch sizes.
- **Sync vs async, moderate delay, 8 workers:** vanilla async ended at loss 0.3222 versus synchronous 0.3506 (lower at the same time budget), with 1011.0 versus 53.0 mean updates. These methods use different effective update batch sizes.
- **Sync vs async, high delay, 8 workers:** vanilla async ended at loss 0.3225 versus synchronous 0.4002 (lower at the same time budget), with 937.0 versus 20.0 mean updates. These methods use different effective update batch sizes.

Prespecified training-loss target: 0.50, high delay, 8 workers. Loss is evaluated at every update, so times are actual first-crossing events; no interpolation or invented times for runs without a crossing. Successful-only times must be interpreted together with the crossing counts.

| Method | Logged target crossings | Mean first logged time |
| --- | --- | --- |
| Synchronous | 3/3 | 44.48 (successful runs only) |
| Vanilla async | 3/3 | 1.41 (successful runs only) |
| Filtered (tau=5) | 3/3 | 1.41 (successful runs only) |
| Async-MB (B=4, tau=5) | 3/3 | 4.32 (successful runs only) |

- **Heterogeneity and vanilla-async staleness, 8 workers:** low: mean 6.98, p95 9.00, max 11.7; moderate: mean 6.97, p95 16.33, max 25.0; high: mean 6.93, p95 16.53, max 67.3. All three statistics are computed over completed arrivals; p95 and max are then averaged across seeds.
- **Filtered (tau=2) vs vanilla async, high delay, 8 workers:** paired final-training-loss difference (method minus vanilla) +0.0002, sample SD 0.0006; higher or equal mean loss, with lower loss in 1/3 paired seeds. This is a descriptive comparison, not a significance claim.
- **Filtered (tau=5) vs vanilla async, high delay, 8 workers:** paired final-training-loss difference (method minus vanilla) -0.0004, sample SD 0.0007; lower mean loss, with lower loss in 2/3 paired seeds. This is a descriptive comparison, not a significance claim.
- **Filtered (tau=10) vs vanilla async, high delay, 8 workers:** paired final-training-loss difference (method minus vanilla) -0.0001, sample SD 0.0005; lower mean loss, with lower loss in 2/3 paired seeds. This is a descriptive comparison, not a significance claim.
- **Async-MB (B=4, tau=5) vs vanilla async, high delay, 8 workers:** paired final-training-loss difference (method minus vanilla) -0.0015, sample SD 0.0016; lower mean loss, with lower loss in 2/3 paired seeds. This is a descriptive comparison, not a significance claim.

Worker scaling for vanilla async under high heterogeneity:

| Workers | Train loss | Mean staleness | Max staleness | Completed gradients |
| --- | --- | --- | --- | --- |
| 4 | 0.3222 ± 0.0014 | 2.97 ± 0.02 | 30.7 ± 6.1 | 471.7 ± 3.2 |
| 8 | 0.3225 ± 0.0019 | 6.93 ± 0.03 | 67.3 ± 2.1 | 937.0 ± 12.2 |
| 16 | 0.3225 ± 0.0011 | 14.87 ± 0.03 | 164.3 ± 10.5 | 1890.0 ± 8.7 |

- Across 4 → 8 → 16 workers, mean final loss did not decrease at every worker-count increase. This finite-grid result cannot establish that adding workers always helps.
- **Threshold trade-off, 8 workers, high delay:** tau=2: loss 0.3227, discarded 50.4%; tau=5: loss 0.3222, discarded 23.2%; tau=10: loss 0.3225, discarded 6.7%. Thresholds were not adjusted after inspecting outcomes.
- **Simplified async mini-batch, moderate delay, 8 workers:** final loss 0.3205, 250.3 updates, mean arrival age 1.73; vanilla: loss 0.3222, 1011.0 updates, age 6.97. Age is measured in model updates, so differing update frequencies change its scale.
- **Simplified async mini-batch, high delay, 8 workers:** final loss 0.3210, 223.0 updates, mean arrival age 1.65; vanilla: loss 0.3225, 937.0 updates, age 6.93. Age is measured in model updates, so differing update frequencies change its scale.

## D. Unexpected results and sanity checks

Algorithm invariants passed for 174 runs, and both actual-data replay configurations matched exactly. Analytic unit tests separately verify that the slow worker's stored gradient is used, not recomputed on arrival.

The following expected trends were **not supported**, and the observations are retained:

| check | expected | observed | status |
| --- | --- | --- | --- |
| High vs low delay: mean_staleness, 4 workers | higher under high heterogeneity | low=2.9900; high=2.9689 | NOT_SUPPORTED |
| High vs low delay: mean_staleness, 8 workers | higher under high heterogeneity | low=6.9765; high=6.9259 | NOT_SUPPORTED |
| High vs low delay: mean_staleness, 16 workers | higher under high heterogeneity | low=14.9498; high=14.8726 | NOT_SUPPORTED |

In a closed pool of W continuously busy workers using unfiltered async SGD, mean completed-gradient age can remain near W−1 even when heterogeneity grows. Each applied update ages the other W−1 jobs. The exact identity `sum(completed ages) + sum(outstanding ages) = (W−1) * updates` was checked for every vanilla-async run. Unusually old unfinished jobs at the horizon do not yet enter completed-arrival statistics. Tail age and actual completion delays therefore carry information the mean can conceal. The simulator was audited rather than altered to force a higher mean.

Controlled straggler-count results (mean across seeds):

| Slow workers / 8 | Mean age | Max age | Completed gradients | Mean completed service time |
| --- | --- | --- | --- | --- |
| 0 | 6.97 | 22.0 | 1186.0 | 1.008 |
| 1 | 6.95 | 75.3 | 1061.3 | 1.122 |
| 2 | 6.93 | 67.3 | 937.0 | 1.267 |
| 4 | 6.89 | 56.0 | 693.0 | 1.705 |

More stragglers need not monotonically increase maximum or mean age: making additional workers slow also reduces the update rate that ages their gradients. Mean service times are deliberately increased by the role assignment; update-based age is an outcome, not a controlled delay parameter.

## E. Current limitations

- Event-driven simulation, not physical distributed workers: compute happens serially on the host; only completion availability is simulated. No communication cost, contention, server bottleneck, scheduling overhead, worker cancellation, or failures.
- A single synthetic convex logistic-regression dataset and fixed learning rate. These results do not test the papers' nonconvex optimal-time guarantees or establish performance on neural networks.
- Main scenarios change both speed dispersion and mean total throughput. The controlled straggler experiment also changes throughput; a future capacity-matched experiment is needed to isolate variance alone.
- Synchronous updates average W task gradients; async updates use one; simplified mini-batch uses four. The common step size controls one variable but leaves an effective-batch/update-frequency confound. The gradient-work plot helps expose this distinction.
- Fixed thresholds and fixed B replace paper-specific theory, automatic batch adaptation, base-optimizer wrappers, and quantile selection. Our inclusive tau rule corresponds to delta < R with R=tau+1 in the no-cancellation Ringmaster rule, but we do not select its theoretical threshold or claim its theorem.
- Seed variation covers task sampling and simulated delays only. Small seed counts cannot establish statistical significance. Event-weighted ages favor workers that finish often; pending jobs are right-censored.
- Target times are actual first-crossing update events. Plot uncertainty bands align stepwise loss values on a common display grid with no interpolation; averaging can hide individual-run fluctuations.

## F. Next experiments

1. Add a prespecified common learning-rate sweep applied equally to all methods, with separate tuning and final evaluation seeds; also compare matched effective batch sizes and gradient-work budgets.
2. Match aggregate worker capacity across delay laws, then vary delay variance or straggler prevalence independently. Increase seeds and add other datasets before making broader claims.
3. Explore stress conditions with larger step sizes, nonconvex objectives, time-varying speeds, and server/communication overhead; retain failures and measure tail staleness.
4. If exact reproduction becomes the goal, implement and verify the paper-specific adaptive mini-batching and Ringmaster parameter choices against their pseudocode. Paper 3 remains deferred.

## Figures for the progress report

- `plots/02_loss_vs_time.png`: primary time-budget comparison; pair with `12_loss_vs_gradient_work.png` to explain differing effective update batches.
- `plots/08_async_heterogeneity.png` and `10_tail_staleness.png`: show what heterogeneity changes and why mean staleness alone is incomplete.
- `plots/07_threshold_comparison.png` together with `05_discard_percentage.png`: show the loss/discard trade-off, including any threshold that underperforms vanilla.
- `plots/06_worker_scaling.png`: worker-count effects for all four main methods; `09_async_minibatch.png` isolates simplified mini-batching.
- `plots/01_loss_vs_updates.png` and `03_staleness_vs_time.png` supply the update-count and cumulative-age views. All figures also have vector PDF copies.

## Research references and implementation scope

- Attia, Gaash & Koren (ICML 2025), [Faster Stochastic Optimization with Arbitrary Delays via Adaptive Asynchronous Mini-Batching](https://proceedings.mlr.press/v267/attia25a.html). This project borrows asynchronous collection and filtering; B and thresholds are fixed, without the paper's automatic quantile adaptation.
- Maranjyan, Tyurin & Richtárik (ICML 2025), [Ringmaster ASGD: The First Asynchronous SGD with Optimal Time Complexity](https://proceedings.mlr.press/v267/maranjyan25b.html), especially [Algorithm 4](https://arxiv.org/html/2501.16168v2). This project studies a configurable completed-gradient rejection rule, without calculation stops or theoretical parameter selection.

Reproducibility: `run_manifest.json` records configurations, package versions, data fingerprint, and source SHA-256 hashes; `sanity_checks.json` records invariant checks and exact replays. All tables and observations above were generated from the accompanying executed CSVs.
