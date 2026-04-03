use criterion::{Criterion, criterion_group, criterion_main};
use egobox_ego::{EgorBuilder, InfillStrategy};
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
                            .max_iters(10)
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
                            .max_iters(10)
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

    group.finish();
}

criterion_group!(benches, criterion_ego);
criterion_main!(benches);
