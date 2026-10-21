//! Comparison of the models of the disciplinary GP uncertainty, for several sizes of the
//! initial disciplinary designs of experiments, i.e. several levels of uncertainty:
//! * the solved and linearized random MDAs (decision D37),
//! * the perfect and GP posterior correlations across design points (decision D38).
//!
//! ```text
//! cargo run -p egobox-egmdo --release --example egmdo_uncertainty -- uq [n_seeds] [problem...]
//! cargo run -p egobox-egmdo --release --example egmdo_uncertainty -- opt [n_seeds] [problem...]
//! ```
//!
//! * `uq` compares the uncertainty quantification only: with the same disciplinary GPs and the
//!   same design points, the objective PCEs and `P_min` (or `P_sol`) from both random MDAs,
//!   and `P_min` with both correlations.
//! * `opt` runs the whole algorithm with the four combinations.
//!
//! Problems: `toy`, `sellar_modified`, `compound_cylinder`, `toy_constrained`,
//! `sellar_constrained` (all by default). The results are printed as Markdown tables.

use egobox_doe::{Lhs, SamplingMethod};
use egobox_egmdo::criteria::{Feasibility, SampleMinima, qoi_joint_samples};
use egobox_egmdo::pce::normal_samples;
use egobox_egmdo::problems::{self, TestProblem};
use egobox_egmdo::uq::{
    Correlation, JointSamples, PointUq, RandomMda, UqSettings, coupling_correlations, point_uq,
};
use egobox_egmdo::{DisciplinaryModels, EgmdoConfig, GpSettings, initial_doe, minimize};
use ndarray::{Array1, Axis, array};
use ndarray_rand::rand::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;
use rayon::prelude::*;
use std::time::Instant;

/// Constraint tolerance of the feasibility in `P_sol` (default of the solver)
const TOLERANCE: f64 = 1e-4;

struct Case {
    name: &'static str,
    problem: fn() -> TestProblem,
    /// Sizes of the initial disciplinary DoEs, the same for every discipline
    doe_sizes: Vec<usize>,
    /// The solver configuration of the problem, from the benchmarks
    config: EgmdoConfig,
    /// Relative tolerance of the convergence criterion of the papers, on `E[Ẑ*]` (2020
    /// problems) or on `E[Ŷ_min]` (2024 problems)
    design_criterion: Option<f64>,
    objective_criterion: Option<f64>,
}

fn cases() -> Vec<Case> {
    vec![
        Case {
            name: "toy",
            problem: problems::toy,
            doe_sizes: vec![3, 4, 6, 10],
            config: EgmdoConfig {
                doe_uq: Some(array![[-4.3], [-2.5], [0.5], [3.7]]),
                max_iters: 5,
                ..EgmdoConfig::default()
            },
            design_criterion: Some(0.05),
            objective_criterion: None,
        },
        Case {
            name: "sellar_modified",
            problem: problems::sellar_modified,
            doe_sizes: vec![4, 5, 8, 12],
            config: EgmdoConfig {
                n_doe_uq: 20,
                max_iters: 10,
                ..EgmdoConfig::default()
            },
            design_criterion: Some(0.05),
            objective_criterion: None,
        },
        Case {
            name: "compound_cylinder",
            problem: problems::compound_cylinder,
            doe_sizes: vec![5, 8, 12, 16],
            config: EgmdoConfig {
                n_doe_uq: 20,
                max_iters: 20,
                mda: problems::compound_cylinder().mda,
                ..EgmdoConfig::default()
            },
            design_criterion: Some(0.05),
            objective_criterion: None,
        },
        Case {
            name: "toy_constrained",
            problem: problems::toy_constrained,
            doe_sizes: vec![3, 4, 6, 10],
            config: EgmdoConfig {
                doe_uq: Some(array![[-3.6], [-2.], [1.2], [3.8]]),
                max_iters: 5,
                ..EgmdoConfig::default()
            },
            design_criterion: None,
            objective_criterion: Some(0.01),
        },
        Case {
            name: "sellar_constrained",
            problem: problems::sellar_constrained,
            doe_sizes: vec![4, 5, 8, 12],
            config: EgmdoConfig {
                n_doe_uq: 20,
                max_iters: 10,
                uq: UqSettings {
                    n_mc: Some(200),
                    ..UqSettings::default()
                },
                ..EgmdoConfig::default()
            },
            design_criterion: None,
            objective_criterion: Some(0.01),
        },
    ]
}

fn median(values: &[f64]) -> f64 {
    let mut v: Vec<f64> = values.iter().copied().filter(|v| !v.is_nan()).collect();
    v.sort_by(|a, b| a.total_cmp(b));
    v.get(v.len() / 2).copied().unwrap_or(f64::NAN)
}

fn quantile(values: &[f64], q: f64) -> f64 {
    let mut v: Vec<f64> = values.iter().copied().filter(|v| !v.is_nan()).collect();
    v.sort_by(|a, b| a.total_cmp(b));
    if v.is_empty() {
        return f64::NAN;
    }
    v[((v.len() - 1) as f64 * q).round() as usize]
}

fn mean_std(values: &[f64]) -> (f64, f64) {
    let a = Array1::from_vec(values.to_vec());
    (a.mean().unwrap_or(f64::NAN), a.std(0.))
}

fn feasibility(tp: &TestProblem) -> Feasibility {
    let n_cstr = tp.problem.n_ineq() + tp.problem.n_eq();
    Feasibility::new(
        tp.problem.n_ineq(),
        Array1::from_elem(n_cstr, TOLERANCE),
        0.01,
    )
}

/// Statistics of one UQ comparison: per design point and per run
#[derive(Default)]
struct UqStats {
    /// Standard deviation of the objective, solved random MDA
    std_solved: Vec<f64>,
    /// Ratio of the standard deviations, linearized / solved
    std_ratio: Vec<f64>,
    /// Difference of the means, in standard deviations of the solved random MDA
    mean_shift: Vec<f64>,
    /// Total variation distance between the `P_sol` of both random MDAs, per run
    p_distance: Vec<f64>,
    /// Whether the most probable minimizer is the same, per run
    same_argmax: Vec<bool>,
    /// GP posterior correlations between the design points, all couplings and pairs
    correlations: Vec<f64>,
    /// Total variation distance between the `P_sol` with the GP posterior and the perfect
    /// correlations (solved random MDA), per run
    p_distance_correlation: Vec<f64>,
    /// Whether the most probable minimizer is the same with both correlations, per run
    same_argmax_correlation: Vec<bool>,
    /// Non-converged random MDAs, over all points
    n_failed: usize,
    /// Points where the linearization fell back to the solved random MDA
    n_fallbacks: usize,
    /// Points dropped because a random MDA failed at more than `n_MC - P` samples
    n_dropped: usize,
    n_points: usize,
    time_solved: f64,
    time_linearized: f64,
}

fn argmax(p: &Array1<f64>) -> usize {
    p.iter()
        .enumerate()
        .fold(
            (0, f64::NEG_INFINITY),
            |(k, m), (i, &v)| {
                if v > m { (i, v) } else { (k, m) }
            },
        )
        .0
}

/// Compares the UQ of both random MDAs at the `DoE_UQ` of the problem configuration, with
/// disciplinary GPs trained on `n` initial points per discipline
fn compare_uq(case: &Case, n: usize, seed: u64, stats: &mut UqStats) {
    let tp = (case.problem)();
    let data = (0..tp.disciplines.len())
        .map(|i| {
            let x = initial_doe(&tp.problem, i, n, 100 * seed + i as u64);
            let y = tp.eval_discipline(i, &x);
            (x, y)
        })
        .collect();
    let models = match DisciplinaryModels::fit(&tp.problem, data, GpSettings::default()) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("{} n = {n} seed {seed}: {e}", case.name);
            return;
        }
    };
    let z = case.config.doe_uq.clone().unwrap_or_else(|| {
        Lhs::new(tp.problem.xlimits())
            .with_rng(Xoshiro256Plus::seed_from_u64(seed))
            .sample(case.config.n_doe_uq)
    });
    let settings = &case.config.uq;
    let basis = settings.basis(tp.problem.n_couplings());
    let xi = settings.regression_samples(&basis, seed);
    let mda = &case.config.mda;

    let run = |model: RandomMda| -> (Vec<Option<PointUq>>, f64) {
        let start = Instant::now();
        let uq = (0..z.nrows())
            .into_par_iter()
            .map(|i| {
                point_uq(
                    &tp.problem,
                    &models,
                    &z.row(i),
                    &xi.view(),
                    &basis,
                    mda,
                    model,
                )
                .ok()
            })
            .collect();
        (uq, start.elapsed().as_secs_f64())
    };
    let (solved, t_solved) = run(RandomMda::Solved);
    let (linearized, t_linearized) = run(RandomMda::Linearized);
    stats.time_solved += t_solved;
    stats.time_linearized += t_linearized;

    let (mut uq_s, mut uq_l, mut kept) = (vec![], vec![], vec![]);
    for (i, (s, l)) in solved.into_iter().zip(linearized).enumerate() {
        stats.n_points += 1;
        match (s, l) {
            (Some(s), Some(l)) => {
                stats.n_failed += s.n_failed;
                stats.n_fallbacks += !l.linearized as usize;
                let (m_s, m_l) = (s.pce.mean()[0], l.pce.mean()[0]);
                let (s_s, s_l) = (s.pce.variance()[0].sqrt(), l.pce.variance()[0].sqrt());
                stats.std_solved.push(s_s);
                if s_s > 1e-12 {
                    stats.std_ratio.push(s_l / s_s);
                    stats.mean_shift.push((m_l - m_s).abs() / s_s);
                }
                uq_s.push(s);
                uq_l.push(l);
                kept.push(i);
            }
            _ => stats.n_dropped += 1,
        }
    }
    if uq_s.len() >= 2 {
        let xi_p = normal_samples(10_000, basis.dim(), seed + 1);
        let feas = feasibility(&tp);
        let p_s = SampleMinima::new(&uq_s, &xi_p.view(), &feas).probability_of_solution();
        let p_l = SampleMinima::new(&uq_l, &xi_p.view(), &feas).probability_of_solution();
        stats
            .p_distance
            .push(0.5 * (&p_s - &p_l).mapv(f64::abs).sum());
        stats.same_argmax.push(argmax(&p_s) == argmax(&p_l));

        // GP posterior correlation across the design points (D38), solved random MDA
        let z_kept = z.select(Axis(0), &kept);
        let correlations =
            coupling_correlations(&tp.problem, &models, &z_kept, &uq_s).expect("correlations");
        for r in correlations.iter() {
            for i in 0..r.nrows() {
                for l in i + 1..r.ncols() {
                    stats.correlations.push(r[[i, l]]);
                }
            }
        }
        let samples = JointSamples::draw(
            Correlation::GpPosterior,
            &tp.problem,
            &models,
            &z_kept,
            &uq_s,
            10_000,
            seed + 2,
        )
        .expect("joint samples");
        let p_g = SampleMinima::from_values(&qoi_joint_samples(&uq_s, &samples), &feas)
            .probability_of_solution();
        stats
            .p_distance_correlation
            .push(0.5 * (&p_s - &p_g).mapv(f64::abs).sum());
        stats
            .same_argmax_correlation
            .push(argmax(&p_s) == argmax(&p_g));
    }
}

fn uq_study(n_seeds: u64, selected: &[&String]) {
    println!(
        "UQ only: same disciplinary GPs and design points, {n_seeds} seeds per DoE size.\n\
         Std ratio and mean shift: median [10%, 90%] over the design points.\n"
    );
    println!(
        "| Problem | Initial DoE per discipline | Objective std (solved, median) | Std ratio lin./solved | Mean shift (solved std) | P_min distance lin./solved | Same most probable minimizer lin./solved | Failed random MDAs | Fallbacks | Dropped points | Time solved / lin. (s) | GP correlation between points | P_min distance GP/perfect correlation | Same most probable minimizer GP/perfect |"
    );
    println!("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|");
    for case in cases()
        .iter()
        .filter(|c| selected.is_empty() || selected.iter().any(|s| *s == c.name))
    {
        for &n in case.doe_sizes.iter() {
            let mut stats = UqStats::default();
            for seed in 0..n_seeds {
                compare_uq(case, n, seed, &mut stats);
            }
            let n_mc = case.config.uq.n_mc(
                case.config
                    .uq
                    .basis((case.problem)().problem.n_couplings())
                    .size(),
            );
            println!(
                "| {} | {n} | {:.3} | {:.2} [{:.2}, {:.2}] | {:.2} [{:.2}, {:.2}] | {:.3} | {}/{} | {:.2}% | {}/{} | {} | {:.3} / {:.3} | {:.2} [{:.2}, {:.2}] | {:.3} | {}/{} |",
                case.name,
                median(&stats.std_solved),
                median(&stats.std_ratio),
                quantile(&stats.std_ratio, 0.1),
                quantile(&stats.std_ratio, 0.9),
                median(&stats.mean_shift),
                quantile(&stats.mean_shift, 0.1),
                quantile(&stats.mean_shift, 0.9),
                median(&stats.p_distance),
                stats.same_argmax.iter().filter(|&&b| b).count(),
                stats.same_argmax.len(),
                100. * stats.n_failed as f64 / (stats.n_points * n_mc).max(1) as f64,
                stats.n_fallbacks,
                stats.n_points,
                stats.n_dropped,
                stats.time_solved / n_seeds as f64,
                stats.time_linearized / n_seeds as f64,
                median(&stats.correlations),
                quantile(&stats.correlations, 0.1),
                quantile(&stats.correlations, 0.9),
                median(&stats.p_distance_correlation),
                stats.same_argmax_correlation.iter().filter(|&&b| b).count(),
                stats.same_argmax_correlation.len(),
            );
        }
    }
}

fn opt_study(n_seeds: u64, selected: &[&String]) {
    println!("Whole algorithm, defaults (C-EGMDO options, Matérn 5/2), {n_seeds} seeds per row.\n");
    println!(
        "| Problem | Initial DoE per discipline | Random MDA | Correlation | Converged (papers' criterion) | Best point optimal | Disciplinary evaluations | Failed random MDAs | Solved points (lin. fallbacks) | Median time (s) |"
    );
    println!("|---|---|---|---|---|---|---|---|---|---|");
    for case in cases()
        .iter()
        .filter(|c| selected.is_empty() || selected.iter().any(|s| *s == c.name))
    {
        let tp = (case.problem)();
        for &n in case.doe_sizes.iter() {
            for (model, correlation) in [
                (RandomMda::Solved, Correlation::Perfect),
                (RandomMda::Linearized, Correlation::Perfect),
                (RandomMda::Solved, Correlation::GpPosterior),
                (RandomMda::Linearized, Correlation::GpPosterior),
            ] {
                let (mut converged, mut optimal, mut failed_runs) = (0, 0, 0);
                let (mut evals, mut times, mut failed_mdas, mut solved_points) =
                    (vec![], vec![], 0, 0);
                for seed in 0..n_seeds {
                    let config = EgmdoConfig {
                        n_doe_disciplines: Some(vec![n; tp.disciplines.len()]),
                        uq: UqSettings {
                            random_mda: model,
                            correlation,
                            ..case.config.uq.clone()
                        },
                        seed: Some(seed),
                        ..case.config.clone()
                    };
                    let start = Instant::now();
                    let result = match minimize(&tp.problem, &tp.disciplines, config) {
                        Ok(r) => r,
                        Err(e) => {
                            eprintln!(
                                "{} n = {n} {model:?} {correlation:?} seed {seed}: {e}",
                                case.name
                            );
                            failed_runs += 1;
                            continue;
                        }
                    };
                    times.push(start.elapsed().as_secs_f64());
                    let ok = if let Some(t) = case.design_criterion {
                        let err = &result.z_star_mean() - &tp.z_ref;
                        let norm = |v: &Array1<f64>| v.mapv(|e| e * e).sum().sqrt();
                        norm(&err) / norm(&tp.z_ref) < t
                    } else {
                        let t = case.objective_criterion.expect("criterion");
                        (result.y_min_mean() - tp.f_ref).abs() < t * tp.f_ref.abs()
                    };
                    let (_, qoi, _) = tp.evaluate(&result.z_best.view()).expect("true MDA");
                    let is_optimal = (qoi[0] - tp.f_ref).abs() < 0.01 * tp.f_ref.abs()
                        && qoi.iter().skip(1).all(|&g| g <= 1e-3);
                    converged += ok as usize;
                    optimal += is_optimal as usize;
                    evals.push(result.n_evaluations.iter().sum::<usize>() as f64);
                    solved_points += result.uq.iter().filter(|u| !u.linearized).count();
                    failed_mdas += result
                        .history
                        .iter()
                        .map(|r| r.n_failed_mdas)
                        .max()
                        .unwrap_or(0);
                }
                let (m, s) = mean_std(&evals);
                println!(
                    "| {} | {n} | {model:?} | {correlation:?} | {converged}/{n_seeds} | {optimal}/{n_seeds} | {m:.1} ± {s:.1} | {failed_mdas} | {solved_points} | {:.2} |{}",
                    case.name,
                    median(&times),
                    if failed_runs > 0 {
                        format!(" {failed_runs} failed runs |")
                    } else {
                        String::new()
                    }
                );
            }
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let study = args.first().map(String::as_str).unwrap_or("uq").to_owned();
    let rest = &args[1.min(args.len())..];
    let n_seeds = rest.first().and_then(|a| a.parse().ok()).unwrap_or(10);
    let selected: Vec<&String> = rest.iter().filter(|a| a.parse::<u64>().is_err()).collect();
    match study.as_str() {
        "uq" => uq_study(n_seeds, &selected),
        "opt" => opt_study(n_seeds, &selected),
        other => eprintln!("unknown study '{other}': expected 'uq' or 'opt'"),
    }
}
