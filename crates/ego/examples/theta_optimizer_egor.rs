//! Paired Egor comparison of the GP hyperparameters optimizers (COBYLA vs L-BFGS-B).
//! Emit one JSON record per run.
//!
//! cargo run --release --example theta_optimizer_egor -- [n_seeds] [problem filter]
use egobox_ego::{EgorBuilder, ThetaOptimizer};
use ndarray::{Array, Array2, ArrayView2, array};
use std::time::Instant;

/// (name, dim, max_iters, kpls_dim)
const PROBLEMS: [(&str, usize, usize, Option<usize>); 10] = [
    ("rosenbrock", 2, 30, None),
    ("rosenbrock", 6, 60, None),
    ("g24", 2, 30, None),
    ("styblinski_tang", 4, 40, None),
    ("styblinski_tang", 10, 60, None),
    ("styblinski_tang", 20, 60, None),
    ("styblinski_tang", 20, 60, Some(3)),
    ("ackley", 5, 50, None),
    ("ackley", 10, 60, None),
    ("ackley", 20, 60, None),
];

fn objective(x: &ArrayView2<f64>, name: &str) -> Array2<f64> {
    let mut values = Array2::zeros((x.nrows(), if name == "g24" { 3 } else { 1 }));
    for (x, mut y) in x.rows().into_iter().zip(values.rows_mut()) {
        let x = x.to_vec();
        match name {
            "rosenbrock" => y[0] = argmin_testfunctions::rosenbrock(&x),
            "styblinski_tang" => y[0] = argmin_testfunctions::styblinski_tang(&x),
            "ackley" => y[0] = argmin_testfunctions::ackley(&x),
            "g24" => {
                y[0] = -x[0] - x[1];
                y[1] = -2. * x[0].powi(4) + 8. * x[0].powi(3) - 8. * x[0].powi(2) + x[1] - 2.;
                y[2] = -4. * x[0].powi(4) + 32. * x[0].powi(3) - 88. * x[0].powi(2)
                    + 96. * x[0]
                    + x[1]
                    - 36.;
            }
            _ => unreachable!(),
        }
    }
    values
}

fn bounds(name: &str, dim: usize) -> Array2<f64> {
    let (lo, hi) = match name {
        "g24" => return array![[0., 3.], [0., 4.]],
        "rosenbrock" => (-2., 2.),
        "styblinski_tang" => (-5., 5.),
        _ => (-32.768, 32.768),
    };
    Array::from_shape_fn((dim, 2), |(_, j)| if j == 0 { lo } else { hi })
}

fn main() {
    let n_seeds: u64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);
    let filter = std::env::args().nth(2).unwrap_or_default();
    for (name, dim, max_iters, kpls_dim) in PROBLEMS {
        let problem = format!(
            "{name}_{dim}d{}",
            kpls_dim.map_or(String::new(), |k| format!("_kpls{k}"))
        );
        if !problem.contains(&filter) {
            continue;
        }
        for seed in 0..n_seeds {
            for theta_optimizer in [ThetaOptimizer::Cobyla, ThetaOptimizer::Lbfgsb] {
                let optimizer = EgorBuilder::optimize(|x: &ArrayView2<f64>| objective(x, name))
                    .configure(|cfg| {
                        let cfg = cfg
                            .configure_gp(|gp| {
                                gp.kpls_dim(kpls_dim).theta_optimizer(theta_optimizer)
                            })
                            .max_iters(max_iters)
                            .seed(seed);
                        if name == "g24" { cfg.n_cstr(2) } else { cfg }
                    })
                    .min_within(&bounds(name, dim))
                    .unwrap();
                let start = Instant::now();
                let result = optimizer.run().unwrap();
                let seconds = start.elapsed().as_secs_f64();
                println!(
                    "{}",
                    serde_json::json!({
                        "problem": problem, "optimizer": format!("{theta_optimizer:?}"),
                        "seed": seed, "seconds": seconds, "objective": result.y_opt[0],
                        "max_constraint": result.y_opt.iter().skip(1).copied().fold(0.0, f64::max),
                        "iterations": result.state.iter,
                    })
                );
            }
        }
    }
}
