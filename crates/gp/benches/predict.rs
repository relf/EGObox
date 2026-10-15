use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use egobox_doe::{Lhs, SamplingMethod};
use egobox_gp::correlation_models::Matern52Corr;
use egobox_gp::mean_models::ConstantMean;
use egobox_gp::{GaussianProcess, ThetaTuning};
use linfa::prelude::{Dataset, Fit};
use ndarray::{Array1, Array2, Axis, array};
use ndarray_rand::rand::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;

/// Predictions and their gradients at single points as done in infill criteria optimization
fn bench_predict_with_gradients(c: &mut Criterion) {
    let dim = 5;
    let mut group = c.benchmark_group("predict_with_gradients");
    group.sample_size(20);
    for nt in [100, 400, 800] {
        let xlimits = array![[-5., 5.]].broadcast((dim, 2)).unwrap().to_owned();
        let xt = Lhs::new(&xlimits)
            .with_rng(Xoshiro256Plus::seed_from_u64(42))
            .sample(nt);
        let yt: Array1<f64> = xt.map_axis(Axis(1), |x| x.mapv(|v| v * v).sum());
        let gp = GaussianProcess::<f64, ConstantMean, Matern52Corr>::params(
            ConstantMean::default(),
            Matern52Corr::default(),
        )
        .theta_tuning(ThetaTuning::Fixed(Array1::from_elem(dim, 0.5)))
        .fit(&Dataset::new(xt, yt))
        .expect("GP fit error");
        let x: Array2<f64> = Lhs::new(&xlimits)
            .with_rng(Xoshiro256Plus::seed_from_u64(0))
            .sample(20);

        group.bench_with_input(BenchmarkId::new("separate", nt), &x, |b, x| {
            b.iter(|| {
                for xi in x.rows() {
                    let xi = xi.insert_axis(Axis(0));
                    std::hint::black_box(gp.predict_valvar(&xi).unwrap());
                    std::hint::black_box(gp.predict_valvar_gradients(&xi));
                }
            })
        });
        group.bench_with_input(BenchmarkId::new("fused", nt), &x, |b, x| {
            b.iter(|| {
                for xi in x.rows() {
                    let xi = xi.insert_axis(Axis(0));
                    std::hint::black_box(gp.predict_valvar_with_gradients(&xi).unwrap());
                }
            })
        });
    }
    group.finish();
}

criterion_group!(benches, bench_predict_with_gradients);
criterion_main!(benches);
