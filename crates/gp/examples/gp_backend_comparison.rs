//! Compare GP fitting time and prediction quality with fixed data and settings.
use egobox_doe::{Lhs, SamplingMethod};
use egobox_gp::Kriging;
use linfa::prelude::*;
use ndarray::{Array1, Array2, ArrayView2, array};
use ndarray_rand::rand::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;
use std::time::Instant;

fn objective(x: &ArrayView2<f64>, name: &str) -> Array1<f64> {
    x.rows()
        .into_iter()
        .map(|x| match name {
            "xsinx" => (x[0] - 3.5) * ((x[0] - 3.5) / std::f64::consts::PI).sin(),
            _ => {
                let product: f64 = x
                    .iter()
                    .enumerate()
                    .map(|(i, x)| (x / ((i + 1) as f64).sqrt()).cos())
                    .product();
                x.mapv(|v| v * v).sum() / 4000. - product + 1.
            }
        })
        .collect()
}

fn main() {
    let repetitions = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);
    for name in ["xsinx", "griewank"] {
        let (bounds, n_train): (Array2<f64>, _) = match name {
            "xsinx" => (array![[0., 25.]], 15),
            _ => (array![[-5., 5.], [-5., 5.], [-5., 5.]], 40),
        };
        let validation = Lhs::new(&bounds)
            .with_rng(Xoshiro256Plus::seed_from_u64(1234))
            .sample(200);
        let expected = objective(&validation.view(), name);
        for seed in [0, 1, 2, 3, 42] {
            let x = Lhs::new(&bounds)
                .with_rng(Xoshiro256Plus::seed_from_u64(seed))
                .sample(n_train);
            let y = objective(&x.view(), name);
            let dataset = Dataset::new(x, y);
            for repetition in 0..=repetitions {
                let start = Instant::now();
                let model = Kriging::params()
                    .n_start(10)
                    .max_eval(200)
                    .fit(&dataset)
                    .expect("GP fit");
                let seconds = start.elapsed().as_secs_f64();
                let residuals = model.predict(&validation).expect("prediction") - &expected;
                let rmse = residuals.mapv(|r| r * r).mean().unwrap().sqrt();
                if repetition > 0 {
                    println!(
                        "{}",
                        serde_json::json!({
                            "problem": name, "seed": seed, "repetition": repetition,
                            "seconds": seconds, "likelihood": model.likelihood(),
                            "theta": model.theta().to_vec(), "rmse": rmse,
                            "training_points": n_train,
                        })
                    );
                }
            }
        }
    }
}
