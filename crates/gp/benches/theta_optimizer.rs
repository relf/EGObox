use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use egobox_doe::{Lhs, SamplingMethod};
use egobox_gp::correlation_models::Matern52Corr;
use egobox_gp::mean_models::ConstantMean;
use egobox_gp::{GaussianProcess, ThetaOptimizer};
use linfa::prelude::{Dataset, Fit};
use ndarray::{Array1, ArrayView1, Axis, array};
use ndarray_rand::rand::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;

/// GP training time with gradient-free COBYLA vs gradient-based L-BFGS-B
/// theta optimization, using Egor likelihood evaluations budget
fn bench_theta_optimizer(c: &mut Criterion) {
    let mut group = c.benchmark_group("theta_optimizer");
    group.sample_size(10);
    for (dim, nt) in [(5, 60), (10, 120), (20, 200)] {
        let xlimits = array![[-3., 3.]].broadcast((dim, 2)).unwrap().to_owned();
        let xt = Lhs::new(&xlimits)
            .with_rng(Xoshiro256Plus::seed_from_u64(42))
            .sample(nt);
        let yt: Array1<f64> = xt.map_axis(Axis(1), |x: ArrayView1<f64>| {
            x.iter()
                .enumerate()
                .map(|(i, &v): (usize, &f64)| (i as f64 + 1.).sqrt() * v.sin() + 0.1 * v * v)
                .sum()
        });
        for theta_optimizer in [ThetaOptimizer::Cobyla, ThetaOptimizer::Lbfgsb] {
            group.bench_function(
                BenchmarkId::new(format!("{theta_optimizer:?}"), format!("{dim}d_{nt}")),
                |b| {
                    b.iter(|| {
                        std::hint::black_box(
                            GaussianProcess::<f64, ConstantMean, Matern52Corr>::params(
                                ConstantMean::default(),
                                Matern52Corr::default(),
                            )
                            .max_eval(50)
                            .theta_optimizer(theta_optimizer)
                            .fit(&Dataset::new(xt.clone(), yt.clone()))
                            .expect("GP fit error"),
                        )
                    })
                },
            );
        }
    }
    group.finish();
}

criterion_group!(benches, bench_theta_optimizer);
criterion_main!(benches);
