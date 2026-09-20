//! Reproducible end-to-end comparison. Emit one JSON record per timed run.
use egobox_ego::{EgorBuilder, InfillOptimizer, InfillStrategy};
use ndarray::{Array2, ArrayView2, array};
use std::time::Instant;

fn objective(x: &ArrayView2<f64>, name: &str) -> Array2<f64> {
    let mut values = Array2::zeros((x.nrows(), if name == "g24" { 3 } else { 1 }));
    for (x, mut y) in x.rows().into_iter().zip(values.rows_mut()) {
        match name {
            "xsinx" => y[0] = (x[0] - 3.5) * ((x[0] - 3.5) / std::f64::consts::PI).sin(),
            "g24" => {
                y[0] = -x[0] - x[1];
                y[1] = -2. * x[0].powi(4) + 8. * x[0].powi(3) - 8. * x[0].powi(2) + x[1] - 2.;
                y[2] = -4. * x[0].powi(4) + 32. * x[0].powi(3) - 88. * x[0].powi(2)
                    + 96. * x[0]
                    + x[1]
                    - 36.;
            }
            _ => y[0] = argmin_testfunctions::ackley(&x.to_vec()),
        }
    }
    values
}

fn main() {
    let repetitions = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);
    for name in ["xsinx", "g24", "ackley"] {
        for algorithm in [InfillOptimizer::Cobyla, InfillOptimizer::Slsqp] {
            for seed in [0, 1, 2, 3, 42] {
                for repetition in 0..=repetitions {
                    let bounds = match name {
                        "xsinx" => array![[0., 25.]],
                        "g24" => array![[0., 3.], [0., 4.]],
                        _ => array![[-5., 5.], [-5., 5.], [-5., 5.]],
                    };
                    let optimizer = EgorBuilder::optimize(|x: &ArrayView2<f64>| objective(x, name))
                        .configure(|cfg| {
                            let cfg = cfg
                                .seed(seed)
                                .max_iters(30)
                                .infill_strategy(InfillStrategy::EI)
                                .infill_optimizer(algorithm.clone());
                            match name {
                                "xsinx" => cfg.doe(&array![[0.], [7.], [25.]]).target(-15.1),
                                "g24" => cfg.n_cstr(2).target(-5.50),
                                _ => cfg.target(0.1),
                            }
                        })
                        .min_within(&bounds)
                        .unwrap();
                    let start = Instant::now();
                    let result = optimizer.run().unwrap();
                    let seconds = start.elapsed().as_secs_f64();
                    if repetition == 0 {
                        continue;
                    }
                    println!(
                        "{}",
                        serde_json::json!({
                            "problem": name, "algorithm": format!("{algorithm:?}"), "seed": seed,
                            "repetition": repetition, "seconds": seconds, "objective": result.y_opt[0],
                            "max_constraint": result.y_opt.iter().skip(1).copied().fold(0.0, f64::max),
                            "iterations": result.state.iter, "observations": result.x_doe.nrows(),
                            "exit": format!("{:?}", result.state.termination_status),
                        })
                    );
                }
            }
        }
    }
}
