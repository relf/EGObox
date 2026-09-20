# Basin comparison

The tables in the first two sections record the initial integration, before the
COBYLA stopping-policy correction described below. Their timings do not measure
the corrected adapter. A subsequent 100-seed investigation did not reproduce the
original five-seed Ackley median gap.

## Initial integration

Results from prebuilt release binaries on an AMD Ryzen 9 7900, NixOS, Rust
1.98.1, with `RAYON_NUM_THREADS=2`. Five seeds (0, 1, 2, 3, and 42), ten
measured runs per seed. EGO measurements use five rotating blocks of two
measured runs, with one warm-up per seed in each block. Build jobs were stopped
before measurement. Timings are medians of the five per-seed medians.

The baseline is upstream `48ad383`. The executor stage is `f59978f`, which uses
Basin execution with the original numerical solvers. The complete Basin build is
`edfc4f1`. EGO uses EI, a 30-iteration cap, and targets -15.1 (`xsinx`), -5.50
(G24), and 0.1 (3D Ackley). Target success also requires constraint violation <=
1e-4.

  | Problem | Infill | Build    | Median ms | Median iterations | Median observations | Median objective | Target successes | Maximum violation |
  | ---     | ---    | ---      | ---:      | ---:              | ---:                | ---:             | ---:             | ---:              |
  | xsinx   | Cobyla | baseline |     8.892 |                 6 |                   9 |       -15.124954 |              5/5 |                 0 |
  | xsinx   | Cobyla | executor |     9.030 |                 6 |                   9 |       -15.124954 |              5/5 |                 0 |
  | xsinx   | Cobyla | basin    |     5.489 |                 6 |                   9 |       -15.124954 |              5/5 |                 0 |
  | xsinx   | Slsqp  | baseline |     7.470 |                 6 |                   9 |       -15.124957 |              5/5 |                 0 |
  | xsinx   | Slsqp  | executor |     7.375 |                 6 |                   9 |       -15.124957 |              5/5 |                 0 |
  | xsinx   | Slsqp  | basin    |     6.488 |                 6 |                   9 |       -15.124957 |              5/5 |                 0 |
  | g24     | Cobyla | baseline |    93.904 |                13 |                  18 |       -5.5080365 |              5/5 |          6.34e-05 |
  | g24     | Cobyla | executor |    94.124 |                13 |                  18 |       -5.5080365 |              5/5 |          6.34e-05 |
  | g24     | Cobyla | basin    |    19.884 |                 6 |                  11 |       -5.5078816 |              5/5 |          8.53e-06 |
  | g24     | Slsqp  | baseline |    27.838 |                 6 |                  11 |       -5.5080119 |              5/5 |          2.84e-05 |
  | g24     | Slsqp  | executor |    28.176 |                 6 |                  11 |       -5.5080119 |              5/5 |          2.84e-05 |
  | g24     | Slsqp  | basin    |    20.108 |                 6 |                  11 |       -5.5078816 |              5/5 |          8.64e-06 |
  | ackley  | Cobyla | baseline |   202.401 |                30 |                  35 |        2.8549987 |              0/5 |                 0 |
  | ackley  | Cobyla | executor |   199.788 |                30 |                  35 |        2.8549987 |              0/5 |                 0 |
  | ackley  | Cobyla | basin    |   121.257 |                30 |                  35 |        4.5442806 |              0/5 |                 0 |
  | ackley  | Slsqp  | baseline |   137.351 |                30 |                  35 |        4.5974739 |              0/5 |                 0 |
  | ackley  | Slsqp  | executor |   136.572 |                30 |                  35 |        4.5974739 |              0/5 |                 0 |
  | ackley  | Slsqp  | basin    |   131.272 |                30 |                  35 |        3.7229529 |              0/5 |                 0 |

Executor-only numerical differences across 30 problem/algorithm/seed cases: 0.
Median paired executor/baseline time ratio: 0.998. These local timings do not
establish a general speedup; solver changes alter the amount and quality of
work.

## GP fitting

Both builds use the same deterministic hyperparameter starts (`n_start=10`, plus
the initial guess). The configured `max_eval=200` gives an effective cap of 25
callbacks per start for `xsinx` and 30 for Griewank under the existing
dimension-based GP budget rule. The GP baseline uses the unchanged default
numerical backend. Fits use 15 training points for `xsinx` and 40 for 3D
Griewank, with a fixed 200-point validation set. Prediction time is excluded. GP
fitting does not use the outer EGO executor, so its executor-only version is the
baseline.

  | Problem  | Build    | Median fit ms | Median likelihood | Median validation RMSE | RMSE range across seeds |
  | ---      | ---      | ---:          | ---:              | ---:                   | ---:                    |
  | xsinx    | baseline |         0.658 |           62.2735 |            5.16394e-05 | 4.30566e-05–0.000117865 |
  | xsinx    | basin    |         0.555 |           62.2736 |            5.10665e-05 | 4.28672e-05–0.000107645 |
  | griewank | baseline |         3.240 |          0.498096 |               0.370753 |       0.328493–0.422587 |
  | griewank | basin    |         2.050 |          0.445633 |               0.348888 |       0.329272–0.384502 |

## Interpretation

All five seeds reached the `xsinx` and G24 targets with either backend. The
Basin G24 runs met the target with lower constraint violations in this sample;
the COBYLA runs also used fewer outer iterations. The seeded `xsinx` case from
the executor review stops after six iterations and nine observations with both
executors.

Ackley remained difficult for both backends within 30 iterations. Basin COBYLA
had a worse median objective (4.54 versus 2.85), while Basin SLSQP had a better
median (3.72 versus 4.60). Neither backend reached the target of 0.1 on any of
these five seeds. For COBYLA, Basin was worse on three seeds and better on two;
the means were essentially unchanged (3.31 with Basin versus 3.32 originally).
The ten timing repetitions per seed do not add independent optimization trials.
Faster execution therefore does not imply better optimization.

The GP comparisons had similar or lower median validation RMSE with Basin, but
these small workloads are not a general accuracy or performance guarantee. The
executor comparison retains the original numerical solvers and makes no LogEI
changes. The end-to-end benchmarks use EI, so the separate LogEI fixes do not
confound this comparison.

## Ackley investigation and stopping policy

A follow-up on September 18, 2026, used seeds `0..99` with the same 3D Ackley
configuration, Basin 1.13.0, and `cobyla` 1.0.4. It independently switched GP
fitting and acquisition optimization while retaining the Basin outer executor.
The original numerical backends reproduced the five recorded seeds exactly.
These results still use the initial adapter's stopping policy:

  | GP fitting | Acquisition | Median objective | Mean objective | Target successes |
  | ---        | ---         | ---:             | ---:           | ---:             |
  | Original   | Original    |         2.370387 |       2.711366 |            2/100 |
  | Basin      | Original    |         2.379144 |       2.666858 |            1/100 |
  | Original   | Basin       |         2.319712 |       2.732078 |            0/100 |
  | Basin      | Basin       |         2.453991 |       2.569654 |            1/100 |

Full Basin won on 46 seeds and lost on 54. The paired mean difference was
`-0.141712`, with a paired-bootstrap 95% interval of `[-0.658569, 0.386066]`.
The difference of medians was `+0.083604` (about 3.5%), with an interval of
`[-0.786700, 0.756966]`. Both intervals include zero. This comparison does not
establish a general quality regression or statistical equivalence.

The investigation isolated a separate adapter defect. It checked COBYLA's
objective tolerance after each changed feasible incumbent, although the original
backend checks progress at radius reductions. On `f(x) = -exp(-(x - 4.1)^2)`,
starting at zero within `[-5, 5]`, the adapter stopped after three callbacks at
`x = 1`, with objective `-0.000067055`. The original backend reached
approximately `-1` in 23 callbacks, and Basin reached `-1` in 30 with the
adapter's objective tolerances disabled. All used an initial radius of 0.5, a
600-callback budget, and objective tolerances of `1e-4` where enabled.

On 180 fixed acquisition problems along six original-backend trajectories,
disabling that stop improved the best acquisition value across starts on 145
problems and worsened none by more than `1e-4`. These correlated diagnostics are
not independent outer optimization trials. Disabling the acquisition stop did
not improve final Ackley outcomes across 100 seeds: the median was 2.608314 and
the mean was 3.052833. Better local acquisition optimization does not guarantee
better results from a short Bayesian optimization run.

The adapter now compares strictly improving feasible costs between completed
COBYLA radius stages. SLSQP retains its step-based checks. Regression tests
cover the flat-start example, relative tolerance with an objective offset, and
the ability of objective tolerances to limit later refinement. The corrected
adapter reaches `-1` at `x = 4.1` in 30 callbacks on the original reproduction.

### Validation of the correction

Replaying the same 180 fixed acquisition problems with the corrected adapter
improved 145 and worsened none relative to the initial adapter, using the same
absolute `1e-4` comparison threshold. Against the original numerical backend,
the corrected adapter won 32 comparisons, lost seven, and tied within the
threshold on 141. Mean callbacks per start increased from 22.95 to 119.03; the
median increased from 14 to 44. Disabling objective tolerances entirely used
237.28 callbacks on average in the earlier investigation. The correction retains
useful stopping checks while allowing substantially more exploration.

The corrected full Basin integration was also run on seeds `0..99`:

  | Backend | Median objective | Mean objective | Target successes |
  | --- | ---: | ---: | ---: |
  | Original | 2.370387 | 2.711366 | 2/100 |
  | Basin before correction | 2.453991 | 2.569654 | 1/100 |
  | Basin after correction | 2.517719 | 2.943030 | 3/100 |

The corrected integration won on 44 seeds and lost on 56 against the original
backend. Its paired mean difference was `+0.231664`, with a paired-bootstrap 95%
interval of `[-0.313451, 0.750041]`. The median difference was `+0.147332`, with
an interval of `[-0.456009, 0.897472]`. These use 10,000 paired resamples and
RNG seed 521. The correction fixes the demonstrated local stopping error; these
outer results do not establish an Ackley quality improvement.

Initial designs matched the original runs, all observations remained finite and
within bounds, and returned objectives matched observed minima. The fixed
acquisition probes preserved the original outer histories exactly. All five
seeds still met the `xsinx` and G24 targets with either acquisition algorithm;
the largest G24 constraint violation was `9.07e-6`, below `1e-4`. No new timing
claims are made for these validation runs.

## Reproduction

See [the integration guide](basin.md#reproducing-comparisons) for build
commands. The EGO example emits objective values, constraint violations,
iteration and observation counts, exit reasons, and time for every run. The GP
example emits likelihood, hyperparameters, validation RMSE, and fitting time.
Compile before measuring and keep the thread count fixed.

For the rotating EGO measurements above, invoke each prebuilt binary with `2`
measured repetitions in five blocks, rotating the order of the three binaries
between blocks. Each invocation includes its own warm-up. Invoke the GP binaries
with `10` measured repetitions. Preserve the JSON Lines output when reproducing
these results so changes in work and quality can be inspected alongside timing.
