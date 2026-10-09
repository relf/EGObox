use criterion::{Criterion, criterion_group, criterion_main};
use egobox_ego::{EgorBuilder, InfillStrategy, MooStrategy};
use egobox_moe::{CorrelationSpec, RegressionSpec};
use ndarray::{Array2, ArrayView2, Zip, array};

/// Ackley test function: min f(x)=0 at x=(0, 0, 0)
fn ackley(x: &ArrayView2<f64>) -> Array2<f64> {
    let mut y: Array2<f64> = Array2::zeros((x.nrows(), 1));
    Zip::from(y.rows_mut())
        .and(x.rows())
        .par_for_each(|mut yi, xi| yi.assign(&array![argmin_testfunctions::ackley(&xi.to_vec(),)]));
    y
}

fn criterion_ego(c: &mut Criterion) {
    let xlimits = array![[-32.768, 32.768], [-32.768, 32.768], [-32.768, 32.768]];
    let mut group = c.benchmark_group("ego");
    group.sample_size(20);
    group.bench_function("ego ackley matern52", |b| {
        b.iter(|| {
            std::hint::black_box(
                EgorBuilder::optimize(ackley)
                    .configure(|config| {
                        config
                            .configure_gp(|conf| {
                                conf.regression_spec(RegressionSpec::CONSTANT)
                                    .correlation_spec(CorrelationSpec::MATERN52)
                            })
                            .infill_strategy(InfillStrategy::WB2S)
                            .max_iters(5)
                            .seed(42)
                    })
                    .min_within(&xlimits)
                    .expect("Egor configured")
                    .run()
                    .expect("Minimization"),
            )
        });
    });
    group.bench_function("ego ackley matern32", |b| {
        b.iter(|| {
            std::hint::black_box(
                EgorBuilder::optimize(ackley)
                    .configure(|config| {
                        config
                            .configure_gp(|conf| {
                                conf.regression_spec(RegressionSpec::CONSTANT)
                                    .correlation_spec(CorrelationSpec::MATERN32)
                            })
                            .infill_strategy(InfillStrategy::WB2S)
                            .max_iters(5)
                            .seed(42)
                    })
                    .min_within(&xlimits)
                    .expect("Egor configured")
                    .run()
                    .expect("Minimization"),
            )
        });
    });
    group.bench_function("ego ackley square exp", |b| {
        b.iter(|| {
            std::hint::black_box(
                EgorBuilder::optimize(ackley)
                    .configure(|config| {
                        config
                            .configure_gp(|conf| {
                                conf.regression_spec(RegressionSpec::CONSTANT)
                                    .correlation_spec(CorrelationSpec::SQUAREDEXPONENTIAL)
                            })
                            .infill_strategy(InfillStrategy::WB2S)
                            .max_iters(5)
                            .target(5e-1)
                            .seed(42)
                    })
                    .min_within(&xlimits)
                    .expect("Egor configured")
                    .run()
                    .expect("Minimization"),
            )
        });
    });
    group.bench_function("ego ackley abs exp", |b| {
        b.iter(|| {
            std::hint::black_box(
                EgorBuilder::optimize(ackley)
                    .configure(|config| {
                        config
                            .configure_gp(|conf| {
                                conf.regression_spec(RegressionSpec::CONSTANT)
                                    .correlation_spec(CorrelationSpec::ABSOLUTEEXPONENTIAL)
                            })
                            .infill_strategy(InfillStrategy::WB2S)
                            .max_iters(5)
                            .seed(42)
                    })
                    .min_within(&xlimits)
                    .expect("Egor configured")
                    .run()
                    .expect("Minimization"),
            )
        });
    });

    group.finish();
}

/// ZDT1 bi-objective test function with x in [0, 1]^nx
fn zdt1(x: &ArrayView2<f64>) -> Array2<f64> {
    let mut y = Array2::zeros((x.nrows(), 2));
    Zip::from(y.rows_mut())
        .and(x.rows())
        .for_each(|mut yi, xi| {
            let f1 = xi[0];
            let g = 1. + 9. * xi.slice(ndarray::s![1..]).mean().unwrap();
            yi.assign(&array![f1, g * (1. - (f1 / g).sqrt())]);
        });
    y
}

/// DTLZ2 three-objective test function with x in [0, 1]^nx
fn dtlz2(x: &ArrayView2<f64>) -> Array2<f64> {
    let mut y = Array2::zeros((x.nrows(), 3));
    Zip::from(y.rows_mut())
        .and(x.rows())
        .for_each(|mut yi, xi| {
            let g: f64 = xi
                .slice(ndarray::s![2..])
                .iter()
                .map(|v| (v - 0.5).powi(2))
                .sum();
            let (a, b) = (
                xi[0] * std::f64::consts::FRAC_PI_2,
                xi[1] * std::f64::consts::FRAC_PI_2,
            );
            yi.assign(&array![
                (1. + g) * a.cos() * b.cos(),
                (1. + g) * a.cos() * b.sin(),
                (1. + g) * a.sin()
            ]);
        });
    y
}

/// Multi-objective test function
type Problem = fn(&ArrayView2<f64>) -> Array2<f64>;

fn criterion_moo(c: &mut Criterion) {
    let mut group = c.benchmark_group("moo");
    group.sample_size(10);
    // (name, strategy, batch size)
    let strategies = [
        ("parego", MooStrategy::ParEgo, 1),
        ("eim", MooStrategy::Eim, 1),
        ("ehvi", MooStrategy::Ehvi, 1),
        ("qehvi", MooStrategy::QEhvi, 3),
    ];
    // (name, function, number of objectives, number of variables)
    let problems: [(&str, Problem, usize, usize); 2] =
        [("zdt1", zdt1, 2, 3), ("dtlz2", dtlz2, 3, 4)];
    for (pb_name, f, n_obj, nx) in problems {
        let xlimits = Array2::from_shape_vec((nx, 2), [0., 1.].repeat(nx)).unwrap();
        for (name, strategy, batch) in strategies.iter() {
            group.bench_function(format!("moo {pb_name} {name}"), |b| {
                b.iter(|| {
                    std::hint::black_box(
                        EgorBuilder::optimize(f)
                            .configure(|config| {
                                config
                                    .n_obj(n_obj)
                                    .configure_moo(|moo| moo.strategy(strategy.clone()))
                                    .configure_qei(|qei| qei.batch(*batch))
                                    .n_doe(10)
                                    .max_iters(5)
                                    .seed(42)
                            })
                            .min_within(&xlimits)
                            .expect("Egor configured")
                            .run_pareto()
                            .expect("Multi-objective optimization"),
                    )
                });
            });
        }
    }
    group.finish();
}

criterion_group!(benches, criterion_ego, criterion_moo);
criterion_main!(benches);
