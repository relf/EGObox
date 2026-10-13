//! Benchmarks of EGMDO and C-EGMDO on the test problems of the papers, with an `Egor`
//! baseline on the MDF formulation (one true MDA per evaluation of the objective).
//!
//! ```text
//! cargo run -p egobox-egmdo --release --example egmdo_benchmarks -- [n_seeds] [problem...]
//! ```
//!
//! Problems: `toy`, `sellar_modified`, `compound_cylinder`, `toy_constrained`,
//! `sellar_constrained` (all by default). The results are printed as Markdown tables.

use egobox_egmdo::mda::{MdaStatus, TrueCouplings, solve_one};
use egobox_egmdo::problems::{self, TestProblem};
use egobox_egmdo::uq::UqSettings;
use egobox_egmdo::{Acquisition, DisciplineFn, EgmdoConfig, minimize};
use egobox_ego::EgorBuilder;
use ndarray::{Array1, Array2, ArrayView2, Axis, array};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// The convergence criterion of the papers for a problem
#[derive(Clone, Copy)]
enum Criterion {
    /// Relative error of `E[Ẑ*]` (Euclidean norm, D17)
    Design(f64),
    /// Relative error of `E[Ŷ_min]`
    Objective(f64),
}

impl std::fmt::Display for Criterion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Criterion::Design(t) => write!(f, "E[Z*] within {}%", 100. * t),
            Criterion::Objective(t) => write!(f, "E[Y_min] within {}%", 100. * t),
        }
    }
}

struct Case {
    name: &'static str,
    problem: fn() -> TestProblem,
    criterion: Criterion,
    configs: Vec<(&'static str, EgmdoConfig)>,
    egor_iters: usize,
}

/// Tolerance on the true constraints for a point to be feasible
const FEASIBILITY: f64 = 1e-3;
/// Relative tolerance on the true objective for a point to be optimal
const OPTIMALITY: f64 = 0.01;

fn is_optimal(tp: &TestProblem, qoi: &Array1<f64>) -> bool {
    (qoi[0] - tp.f_ref).abs() < OPTIMALITY * tp.f_ref.abs()
        && qoi.iter().skip(1).all(|&g| g <= FEASIBILITY)
}

fn relative_error(x: &Array1<f64>, x_ref: &Array1<f64>) -> f64 {
    let norm = |v: &Array1<f64>| v.mapv(|e| e * e).sum().sqrt();
    norm(&(x - x_ref)) / norm(x_ref)
}

fn cases() -> Vec<Case> {
    let toy_doe = |config: EgmdoConfig| EgmdoConfig {
        n_doe_disciplines: Some(vec![5, 4]),
        doe_uq: Some(array![[-4.3], [-2.5], [0.5], [3.7]]),
        max_iters: 5,
        ..config
    };
    let sellar = |config: EgmdoConfig| EgmdoConfig {
        n_doe_disciplines: Some(vec![5, 5]),
        n_doe_uq: 20,
        max_iters: 10,
        ..config
    };
    let cylinder = |config: EgmdoConfig| EgmdoConfig {
        n_doe_disciplines: Some(vec![6, 8]),
        n_doe_uq: 20,
        max_iters: 20,
        mda: problems::compound_cylinder().mda,
        ..config
    };
    let toy_constrained = |config: EgmdoConfig| EgmdoConfig {
        n_doe_disciplines: Some(vec![5, 4]),
        doe_uq: Some(array![[-3.6], [-2.], [1.2], [3.8]]),
        max_iters: 5,
        ..config
    };
    let sellar_constrained = |config: EgmdoConfig| EgmdoConfig {
        n_doe_disciplines: Some(vec![5, 5]),
        n_doe_uq: 20,
        max_iters: 10,
        uq: UqSettings {
            n_mc: Some(200),
            ..UqSettings::default()
        },
        ..config
    };
    let wb2s = || EgmdoConfig {
        acquisition: Acquisition::Wb2s,
        ..EgmdoConfig::default()
    };
    let cei = || EgmdoConfig {
        acquisition: Acquisition::Ei,
        ..EgmdoConfig::default()
    };
    vec![
        Case {
            name: "toy",
            problem: problems::toy,
            criterion: Criterion::Design(0.05),
            configs: vec![
                ("EGMDO paper", toy_doe(EgmdoConfig::egmdo())),
                ("defaults", toy_doe(EgmdoConfig::default())),
                ("defaults, WB2S", toy_doe(wb2s())),
            ],
            egor_iters: 20,
        },
        Case {
            name: "sellar_modified",
            problem: problems::sellar_modified,
            criterion: Criterion::Design(0.05),
            configs: vec![
                ("EGMDO paper", sellar(EgmdoConfig::egmdo())),
                ("defaults", sellar(EgmdoConfig::default())),
                ("defaults, WB2S", sellar(wb2s())),
            ],
            egor_iters: 40,
        },
        Case {
            name: "compound_cylinder",
            problem: problems::compound_cylinder,
            criterion: Criterion::Design(0.05),
            configs: vec![
                ("EGMDO paper", cylinder(EgmdoConfig::egmdo())),
                ("defaults", cylinder(EgmdoConfig::default())),
            ],
            egor_iters: 40,
        },
        Case {
            name: "toy_constrained",
            problem: problems::toy_constrained,
            criterion: Criterion::Objective(0.01),
            configs: vec![
                ("defaults (WB2)", toy_constrained(EgmdoConfig::default())),
                ("defaults, WB2S", toy_constrained(wb2s())),
                ("defaults, CEI", toy_constrained(cei())),
            ],
            egor_iters: 20,
        },
        Case {
            name: "sellar_constrained",
            problem: problems::sellar_constrained,
            criterion: Criterion::Objective(0.01),
            configs: vec![
                ("defaults (WB2)", sellar_constrained(EgmdoConfig::default())),
                ("defaults, WB2S", sellar_constrained(wb2s())),
                ("defaults, CEI", sellar_constrained(cei())),
            ],
            egor_iters: 40,
        },
    ]
}

fn mean_std(values: &[f64]) -> (f64, f64) {
    let a = Array1::from_vec(values.to_vec());
    (a.mean().unwrap_or(f64::NAN), a.std(0.))
}

fn median(values: &[f64]) -> f64 {
    let mut v = values.to_vec();
    v.sort_by(|a, b| a.total_cmp(b));
    v.get(v.len() / 2).copied().unwrap_or(f64::NAN)
}

/// Runs EGMDO with `config` over `n_seeds` seeds and prints one table row
fn run_egmdo(case: &Case, label: &str, config: &EgmdoConfig, n_seeds: u64) {
    let tp = (case.problem)();
    let (mut converged, mut optimal, mut failed) = (0, 0, 0);
    let (mut evals, mut times) = (vec![], vec![]);
    for seed in 0..n_seeds {
        let config = EgmdoConfig {
            seed: Some(seed),
            ..config.clone()
        };
        let start = Instant::now();
        let result = match minimize(&tp.problem, &tp.disciplines, config) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("{} {label} seed {seed}: {e}", case.name);
                failed += 1;
                continue;
            }
        };
        times.push(start.elapsed().as_secs_f64());
        let ok = match case.criterion {
            Criterion::Design(t) => relative_error(&result.z_star_mean(), &tp.z_ref) < t,
            Criterion::Objective(t) => (result.y_min_mean() - tp.f_ref).abs() < t * tp.f_ref.abs(),
        };
        let (_, qoi, _) = tp.evaluate(&result.z_best.view()).expect("true MDA");
        converged += ok as usize;
        optimal += is_optimal(&tp, &qoi) as usize;
        evals.push(result.n_evaluations.iter().sum::<usize>() as f64);
    }
    let (m, s) = mean_std(&evals);
    println!(
        "| {} | {label} | {converged}/{n_seeds} | {optimal}/{n_seeds} | {m:.1} ± {s:.1} | {:.1} |{}",
        case.name,
        median(&times),
        if failed > 0 {
            format!(" {failed} failed runs |")
        } else {
            String::new()
        }
    );
}

/// Runs `Egor` on the MDF formulation over `n_seeds` seeds and prints one table row: each
/// evaluation of the objective solves the true MDA, and every disciplinary call is counted.
fn run_egor(case: &Case, n_seeds: u64) {
    let tp = Arc::new((case.problem)());
    let n_cstr = tp.problem.n_ineq();
    let (mut reached, mut optimal) = (0, 0);
    let (mut evals_to_target, mut evals_total, mut times) = (vec![], vec![], vec![]);
    for seed in 0..n_seeds {
        let counter = Arc::new(AtomicUsize::new(0));
        let first_optimal = Arc::new(Mutex::new(None::<usize>));
        // True disciplines counting their evaluations
        let counting: Arc<Vec<Box<dyn DisciplineFn>>> = Arc::new(
            (0..tp.disciplines.len())
                .map(|i| {
                    let (tp, counter) = (tp.clone(), counter.clone());
                    Box::new(move |x: &ArrayView2<f64>| {
                        counter.fetch_add(x.nrows(), Ordering::Relaxed);
                        (tp.disciplines[i])(x)
                    }) as Box<dyn DisciplineFn>
                })
                .collect(),
        );
        let objective = {
            let (tp, counting, counter, first_optimal) = (
                tp.clone(),
                counting.clone(),
                counter.clone(),
                first_optimal.clone(),
            );
            move |z: &ArrayView2<f64>| -> Array2<f64> {
                let couplings = TrueCouplings::new(&tp.problem, &counting).expect("disciplines");
                let mut out = Array2::zeros((z.nrows(), 1 + n_cstr));
                for (k, zk) in z.rows().into_iter().enumerate() {
                    let (y, status) = solve_one(&couplings, &zk, &tp.mda).expect("MDA");
                    if status != MdaStatus::NonFinite {
                        let qoi = tp
                            .problem
                            .eval_qoi(&zk.insert_axis(Axis(0)), &y.view().insert_axis(Axis(0)))
                            .expect("qoi");
                        out.row_mut(k).assign(&qoi.row(0));
                        let mut first = first_optimal.lock().expect("lock");
                        if first.is_none() && is_optimal(&tp, &qoi.row(0).to_owned()) {
                            *first = Some(counter.load(Ordering::Relaxed));
                        }
                    } else {
                        out.row_mut(k).fill(f64::MAX.sqrt());
                    }
                }
                out
            }
        };
        let xlimits = tp.problem.xlimits().clone();
        let start = Instant::now();
        let result = EgorBuilder::optimize(objective)
            .configure(|config| config.n_cstr(n_cstr).max_iters(case.egor_iters).seed(seed))
            .min_within(&xlimits)
            .expect("Egor configured")
            .run();
        times.push(start.elapsed().as_secs_f64());
        evals_total.push(counter.load(Ordering::Relaxed) as f64);
        if let Ok(result) = result {
            let (_, qoi, _) = tp.evaluate(&result.x_opt.view()).expect("true MDA");
            optimal += is_optimal(&tp, &qoi) as usize;
        }
        if let Some(n) = *first_optimal.lock().expect("lock") {
            reached += 1;
            evals_to_target.push(n as f64);
        }
    }
    let (m, s) = mean_std(&evals_total);
    println!(
        "| {} | Egor MDF ({} iterations) | target reached {reached}/{n_seeds}, after {:.0} evaluations (median) | {optimal}/{n_seeds} | {m:.1} ± {s:.1} | {:.1} |",
        case.name,
        case.egor_iters,
        median(&evals_to_target),
        median(&times)
    );
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let n_seeds = args.first().and_then(|a| a.parse().ok()).unwrap_or(20);
    let selected: Vec<&String> = args.iter().filter(|a| a.parse::<u64>().is_err()).collect();

    println!(
        "{n_seeds} seeds per configuration (0 to {}).\n",
        n_seeds - 1
    );
    println!(
        "| Problem | Method | Converged (papers' criterion) | Best point optimal | Disciplinary evaluations | Median time (s) |"
    );
    println!("|---|---|---|---|---|---|");
    for case in cases()
        .iter()
        .filter(|c| selected.is_empty() || selected.iter().any(|s| *s == c.name))
    {
        eprintln!("{}: {}", case.name, case.criterion);
        for (label, config) in case.configs.iter() {
            run_egmdo(case, label, config, n_seeds);
        }
        run_egor(case, n_seeds);
    }
}
