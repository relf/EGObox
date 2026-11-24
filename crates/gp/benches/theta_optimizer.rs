use criterion::{BenchmarkGroup, BenchmarkId, Criterion, criterion_group, criterion_main};
use egobox_doe::{Lhs, SamplingMethod};
use egobox_gp::correlation_models::{CorrelationModel, Matern52Corr, SquaredExponentialCorr};
use egobox_gp::mean_models::ConstantMean;
use egobox_gp::{GaussianProcess, ThetaOptimizer};
use linfa::prelude::{Dataset, Fit};
use ndarray::{Array1, ArrayView1, Axis, array};
use ndarray_rand::rand::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;

/// GP training time with gradient-free COBYLA vs gradient-based L-BFGS-B
/// theta optimization, using Egor likelihood evaluations budget.
/// The likelihood reached by each optimizer is printed once per problem.
fn bench_theta_optimizer(c: &mut Criterion) {
    let mut group = c.benchmark_group("theta_optimizer");
    group.sample_size(10);
    bench_kernel(&mut group, Matern52Corr::default(), "Matern52");
    bench_kernel(
        &mut group,
        SquaredExponentialCorr::default(),
        "SquaredExponential",
    );
    group.finish();
}

fn bench_kernel<Corr: CorrelationModel<f64>>(
    group: &mut BenchmarkGroup<'_, criterion::measurement::WallTime>,
    corr: Corr,
    name: &str,
) {
    for (dim, nt, kpls_dim) in [
        (5, 60, None),
        (10, 120, None),
        (20, 200, None),
        (20, 200, Some(3)),
    ] {
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
        let case = match kpls_dim {
            Some(k) => format!("{name}/{dim}d_{nt}_kpls{k}"),
            None => format!("{name}/{dim}d_{nt}"),
        };
        for theta_optimizer in [ThetaOptimizer::Cobyla, ThetaOptimizer::Lbfgsb] {
            let fit = || {
                GaussianProcess::<f64, ConstantMean, Corr>::params(ConstantMean::default(), corr)
                    .kpls_dim(kpls_dim)
                    .max_eval(50)
                    .theta_optimizer(theta_optimizer)
                    .fit(&Dataset::new(xt.clone(), yt.clone()))
                    .expect("GP fit error")
            };
            println!(
                "theta_optimizer/{theta_optimizer:?}/{case}: likelihood={:.4}",
                fit().likelihood()
            );
            group.bench_function(
                BenchmarkId::new(format!("{theta_optimizer:?}"), &case),
                |b| b.iter(|| std::hint::black_box(fit())),
            );
        }
    }
}

criterion_group!(benches, bench_theta_optimizer);
criterion_main!(benches);
