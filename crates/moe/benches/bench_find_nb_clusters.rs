use criterion::{Criterion, criterion_group, criterion_main};

use egobox_doe::{Lhs, SamplingMethod};
use egobox_moe::*;
use ndarray::{Array1, Array2, Axis, Zip, array};
use ndarray_rand::rand::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;

fn function_test_1d(x: &Array2<f64>) -> Array1<f64> {
    let mut y = Array2::zeros(x.dim());
    Zip::from(&mut y).and(x).for_each(|yi, &xi| {
        if xi < 0.4 {
            *yi = xi * xi;
        } else if (0.4..0.8).contains(&xi) {
            *yi = 3. * xi + 1.;
        } else {
            *yi = f64::sin(10. * xi);
        }
    });
    y.remove_axis(Axis(1))
}

fn criterion_benchmark(c: &mut Criterion) {
    let rng = Xoshiro256Plus::seed_from_u64(42);
    let doe = Lhs::new(&array![[0., 1.]]).with_rng(rng);
    let xtrain = doe.sample(50);
    let ytrain = function_test_1d(&xtrain);

    let rng = Xoshiro256Plus::seed_from_u64(42);
    let mut group = c.benchmark_group("find_nb_clusters");

    group.sample_size(10);
    group.bench_function("find_nb_clusters", |b| {
        b.iter(|| {
            find_best_number_of_clusters(
                &xtrain,
                &ytrain,
                5,
                None,
                RegressionSpec::ALL,
                CorrelationSpec::ALL,
                rng.clone(),
                &egobox_gp::Cancellation::default(),
            )
        })
    });
    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
