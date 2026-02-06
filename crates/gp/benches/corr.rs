use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use egobox_gp::DiffMatrix;
use egobox_gp::correlation_models::{CorrelationModel, Matern32Corr, Matern52Corr};
use ndarray::{Array1, Array2};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

fn bench_matern32_value(c: &mut Criterion) {
    let mut group = c.benchmark_group("matern32_value");

    let dim = 100;
    let xt = Array2::from_shape_fn((1, dim), |(i, j)| (i as f64 * 0.1 + j as f64 * 0.05) % 1.0);
    let dm = DiffMatrix::new(&xt);
    let theta = Array1::from_elem(dim, f64::sqrt(0.2));
    let weights = Array2::ones((dim, dim));

    let corr = Matern32Corr::default();
    group.bench_function(BenchmarkId::from_parameter(dim), |b| {
        b.iter(|| corr.rval_from_distances(&dm.d, &theta, &weights));
    });
    group.finish();
}

fn bench_matern52_value(c: &mut Criterion) {
    let mut group = c.benchmark_group("matern52_value");

    let dim = 100;
    let xt = Array2::from_shape_fn((1, dim), |(i, j)| (i as f64 * 0.1 + j as f64 * 0.05) % 1.0);
    let dm = DiffMatrix::new(&xt);
    let theta = Array1::from_elem(dim, f64::sqrt(0.2));
    let weights = Array2::ones((dim, dim));

    let corr = Matern52Corr::default();
    group.bench_function(BenchmarkId::from_parameter(dim), |b| {
        b.iter(|| corr.rval_from_distances(&dm.d, &theta, &weights));
    });
    group.finish();
}

fn bench_matern32_jacobian(c: &mut Criterion) {
    let mut group = c.benchmark_group("matern32_jacobian");

    let dim = 100;
    let uniform = Uniform::new(0., 1.);
    let mut rng = ndarray_rand::rand::thread_rng();
    let x = Array1::random_using((dim,), uniform, &mut rng);
    let xtrain = Array2::from_shape_fn((1, dim), |(i, j)| (i as f64 * 0.1 + j as f64 * 0.05) % 1.0);
    let theta = Array1::from_elem(dim, f64::sqrt(0.2));
    let weights = Array2::ones((dim, dim));

    let corr = Matern32Corr::default();

    group.bench_function(BenchmarkId::from_parameter(dim), |b| {
        b.iter(|| corr.jac(&x, &xtrain, &theta, &weights));
    });
    group.finish();
}

fn bench_matern52_jacobian(c: &mut Criterion) {
    let mut group = c.benchmark_group("matern52_jacobian");

    let dim = 100;
    let uniform = Uniform::new(0., 1.);
    let mut rng = ndarray_rand::rand::thread_rng();
    let x = Array1::random_using((dim,), uniform, &mut rng);
    let xtrain = Array2::from_shape_fn((1, dim), |(i, j)| (i as f64 * 0.1 + j as f64 * 0.05) % 1.0);
    let theta = Array1::from_elem(dim, f64::sqrt(0.2));
    let weights = Array2::ones((dim, dim));

    let corr = Matern52Corr::default();

    group.bench_function(BenchmarkId::from_parameter(dim), |b| {
        b.iter(|| corr.jac(&x, &xtrain, &theta, &weights));
    });
    group.finish();
}

fn bench_diff_matrix(c: &mut Criterion) {
    let mut group = c.benchmark_group("diff_matrix");

    let dim = 100;
    let nt = 100;
    let uniform = Uniform::new(0., 1.);
    let mut rng = ndarray_rand::rand::thread_rng();
    let x = Array2::random_using((nt, dim), uniform, &mut rng);

    group.bench_function(format!("{}x{}", nt, dim), |b| {
        b.iter(|| DiffMatrix::new(&x));
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_matern32_value,
    bench_matern52_value,
    bench_matern32_jacobian,
    bench_matern52_jacobian,
    bench_diff_matrix,
);
criterion_main!(benches);
