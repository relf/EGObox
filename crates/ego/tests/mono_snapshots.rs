//! Mono-objective non-regression snapshots (see `common` for the fixture mechanism).

mod common;

use common::{SNAPSHOT_REL_TOL, check_snapshot_data};
use egobox_doe::{Lhs, SamplingMethod};
use egobox_ego::{
    CoegoStatus, CorrelationSpec, Cstr, CstrSpec, EgorBuilder, EgorServiceBuilder,
    FailsafeStrategy, FeasibleInfillStrategy, HotStartMode, InfillOptimizer, InfillStrategy,
    OptimResult, ParamTuning, QEiStrategy, RegressionSpec, XType,
};
use egobox_moe::NbClusters;
use ndarray::{Array2, ArrayView1, ArrayView2, Axis, Zip, array, concatenate};
use ndarray_rand::rand::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;
use serial_test::serial;

fn check_snapshot(name: &str, res: &OptimResult<f64>) {
    check_snapshot_with_tol(name, res, SNAPSHOT_REL_TOL)
}

/// Same as [check_snapshot] with a given relative tolerance, for scenarios whose
/// evaluated points are more sensitive to CPU dependent math library code paths
fn check_snapshot_with_tol(name: &str, res: &OptimResult<f64>, rel_tol: f64) {
    check_snapshot_data(name, &res.x_doe, &res.y_doe, rel_tol)
}

fn xsinx(x: &ArrayView2<f64>) -> Array2<f64> {
    (x - 3.5) * ((x - 3.5) / std::f64::consts::PI).mapv(|v| v.sin())
}

fn g24(x: &ArrayView1<f64>) -> f64 {
    -x[0] - x[1]
}

fn g24_c1(x: &ArrayView1<f64>) -> f64 {
    -2.0 * x[0].powf(4.0) + 8.0 * x[0].powf(3.0) - 8.0 * x[0].powf(2.0) + x[1] - 2.0
}

fn g24_c2(x: &ArrayView1<f64>) -> f64 {
    -4.0 * x[0].powf(4.0) + 32.0 * x[0].powf(3.0) - 88.0 * x[0].powf(2.0) + 96.0 * x[0] + x[1]
        - 36.0
}

fn f_g24(x: &ArrayView2<f64>) -> Array2<f64> {
    let mut y = Array2::zeros((x.nrows(), 3));
    Zip::from(y.rows_mut())
        .and(x.rows())
        .for_each(|mut yi, xi| yi.assign(&array![g24(&xi), g24_c1(&xi), g24_c2(&xi)]));
    y
}

fn f_g24_bare(x: &ArrayView2<f64>) -> Array2<f64> {
    let mut y = Array2::zeros((x.nrows(), 1));
    Zip::from(y.rows_mut())
        .and(x.rows())
        .for_each(|mut yi, xi| yi.assign(&array![g24(&xi)]));
    y
}

fn sphere(x: &ArrayView2<f64>) -> Array2<f64> {
    (x * x).sum_axis(Axis(1)).insert_axis(Axis(1))
}

fn mixsinx(x: &ArrayView2<f64>) -> Array2<f64> {
    (x - 3.5) * ((x - 3.5) / std::f64::consts::PI).mapv(|v| v.sin())
}

/// Branin-like function failing (NaN) in the lower left corner
fn branin_with_nans(x: &ArrayView2<f64>) -> Array2<f64> {
    x.map_axis(Axis(1), |xi| {
        if xi[0] * xi[1] >= 0.2 {
            let (x0, x1) = (15. * xi[0] - 5., 15. * xi[1]);
            let a = x1 - 5.1 / (4. * std::f64::consts::PI.powi(2)) * x0 * x0
                + 5. / std::f64::consts::PI * x0
                - 6.;
            a * a + 10. * (1. - 1. / (8. * std::f64::consts::PI)) * x0.cos() + 10.
        } else {
            f64::NAN
        }
    })
    .insert_axis(Axis(1))
}

fn g24_doe(n: usize) -> Array2<f64> {
    Lhs::new(&array![[0., 3.], [0., 4.]])
        .with_rng(Xoshiro256Plus::seed_from_u64(42))
        .sample(n)
}

fn xsinx_with(name: &str, infill: InfillStrategy) {
    let res = EgorBuilder::optimize(xsinx)
        .configure(|cfg| {
            cfg.doe(&array![[0.], [7.], [25.]])
                .infill_strategy(infill)
                .max_iters(8)
                .seed(42)
        })
        .min_within(&array![[0., 25.]])
        .unwrap()
        .run()
        .unwrap();
    check_snapshot(name, &res);
}

#[test]
#[serial]
fn snapshot_xsinx_ei() {
    xsinx_with("xsinx_ei", InfillStrategy::EI);
}

#[test]
#[serial]
fn snapshot_xsinx_logei() {
    xsinx_with("xsinx_logei", InfillStrategy::LogEI);
}

#[test]
#[serial]
fn snapshot_xsinx_wb2() {
    xsinx_with("xsinx_wb2", InfillStrategy::WB2);
}

#[test]
#[serial]
fn snapshot_g24_n_cstr() {
    let res = EgorBuilder::optimize(f_g24)
        .configure(|cfg| cfg.n_cstr(2).doe(&g24_doe(6)).max_iters(8).seed(42))
        .min_within(&array![[0., 3.], [0., 4.]])
        .unwrap()
        .run()
        .unwrap();
    check_snapshot("g24_n_cstr", &res);
}

#[test]
#[serial]
fn snapshot_g24_cstr_infill_cobyla() {
    let res = EgorBuilder::optimize(f_g24)
        .configure(|cfg| {
            cfg.n_cstr(2)
                .cstr_infill(true)
                .infill_strategy(InfillStrategy::EI)
                .infill_optimizer(InfillOptimizer::Cobyla)
                .doe(&g24_doe(6))
                .max_iters(8)
                .seed(42)
        })
        .min_within(&array![[0., 3.], [0., 4.]])
        .unwrap()
        .run()
        .unwrap();
    check_snapshot("g24_cstr_infill_cobyla", &res);
}

#[test]
#[serial]
fn snapshot_g24_cstr_specs() {
    let res = EgorBuilder::optimize(f_g24)
        .configure(|cfg| {
            cfg.cstr_specs(vec![CstrSpec::Leq(0.0), CstrSpec::Btw(-100., 0.0)])
                .doe(&g24_doe(6))
                .max_iters(8)
                .seed(42)
        })
        .min_within(&array![[0., 3.], [0., 4.]])
        .unwrap()
        .run()
        .unwrap();
    check_snapshot_with_tol("g24_cstr_specs", &res, 5e-5);
}

#[test]
#[serial]
fn snapshot_g24_function_constraints() {
    let c1: Cstr = |x, _g, _u| g24_c1(&ArrayView1::from(x));
    let c2: Cstr = |x, _g, _u| g24_c2(&ArrayView1::from(x));
    let res = EgorBuilder::optimize(f_g24_bare)
        .subject_to(vec![c1, c2])
        .configure(|cfg| {
            cfg.doe(&g24_doe(6))
                .infill_optimizer(InfillOptimizer::Cobyla)
                .max_iters(8)
                .seed(42)
        })
        .min_within(&array![[0., 3.], [0., 4.]])
        .unwrap()
        .run()
        .unwrap();
    check_snapshot("g24_function_constraints", &res);
}

fn g24_qei(name: &str, strategy: QEiStrategy, rel_tol: f64) {
    let res = EgorBuilder::optimize(f_g24)
        .configure(|cfg| {
            cfg.n_cstr(2)
                .configure_qei(|qei| qei.batch(3).strategy(strategy))
                .doe(&g24_doe(6))
                .max_iters(3)
                .seed(42)
        })
        .min_within(&array![[0., 3.], [0., 4.]])
        .unwrap()
        .run()
        .unwrap();
    check_snapshot_with_tol(name, &res, rel_tol);
}

#[test]
#[serial]
fn snapshot_g24_qei_kb() {
    g24_qei("g24_qei_kb", QEiStrategy::KrigingBeliever, 5e-4);
}

#[test]
#[serial]
fn snapshot_g24_qei_clmin() {
    g24_qei(
        "g24_qei_clmin",
        QEiStrategy::ConstantLiarMinimum,
        SNAPSHOT_REL_TOL,
    );
}

#[test]
#[serial]
fn snapshot_mixsinx_mixint() {
    let res = EgorBuilder::optimize(mixsinx)
        .configure(|cfg| {
            cfg.doe(&array![[0.], [7.], [25.]])
                .infill_strategy(InfillStrategy::EI)
                .max_iters(5)
                .seed(42)
        })
        .min_within_mixint_space(&[XType::Int(0, 25)])
        .unwrap()
        .run()
        .unwrap();
    check_snapshot("mixsinx_mixint", &res);
}

fn branin_failsafe(name: &str, strategy: FailsafeStrategy) {
    let xlimits = array![[0., 1.], [0., 1.]];
    let doe = Lhs::new(&xlimits)
        .with_rng(Xoshiro256Plus::seed_from_u64(42))
        .sample(10);
    let res = EgorBuilder::optimize(branin_with_nans)
        .configure(|cfg| {
            cfg.doe(&doe)
                .failsafe_strategy(strategy)
                .max_iters(6)
                .seed(42)
        })
        .min_within(&xlimits)
        .unwrap()
        .run()
        .unwrap();
    check_snapshot(name, &res);
}

#[test]
#[serial]
fn snapshot_branin_imputation() {
    branin_failsafe("branin_imputation", FailsafeStrategy::Imputation);
}

#[test]
#[serial]
fn snapshot_branin_viability() {
    branin_failsafe("branin_viability", FailsafeStrategy::Viability);
}

#[test]
#[serial]
fn snapshot_xsinx_trego() {
    let res = EgorBuilder::optimize(xsinx)
        .configure(|cfg| cfg.trego(true).max_iters(8).seed(1))
        .min_within(&array![[0., 25.]])
        .unwrap()
        .run()
        .unwrap();
    check_snapshot("xsinx_trego", &res);
}

#[test]
#[serial]
fn snapshot_sphere_coego() {
    let dim = 4;
    let xlimits = Array2::from_shape_vec((dim, 2), [-10.0, 10.0].repeat(dim)).unwrap();
    let doe = Lhs::new(&xlimits)
        .with_rng(Xoshiro256Plus::seed_from_u64(0))
        .sample(dim + 1);
    let res = EgorBuilder::optimize(sphere)
        .configure(|cfg| {
            cfg.doe(&doe)
                .coego(CoegoStatus::Enabled(2))
                .max_iters(6)
                .seed(42)
        })
        .min_within(&xlimits)
        .unwrap()
        .run()
        .unwrap();
    check_snapshot("sphere_coego", &res);
}

#[test]
#[serial]
fn snapshot_g24_function_constraints_warm_start() {
    let outdir = "target/snapshot_warm_start";
    let _ = std::fs::remove_dir_all(outdir);
    let c1: Cstr = |x, _g, _u| g24_c1(&ArrayView1::from(x));
    // objective and metamodelized c2, c1 being a function constraint
    let f_g24_c2 = |x: &ArrayView2<f64>| {
        let mut y = Array2::zeros((x.nrows(), 2));
        Zip::from(y.rows_mut())
            .and(x.rows())
            .for_each(|mut yi, xi| yi.assign(&array![g24(&xi), g24_c2(&xi)]));
        y
    };
    let run = |warm_start: bool| {
        EgorBuilder::optimize(f_g24_c2)
            .subject_to(vec![c1])
            .configure(|cfg| {
                cfg.n_cstr(1)
                    .doe(&g24_doe(6))
                    .infill_optimizer(InfillOptimizer::Cobyla)
                    .max_iters(3)
                    .outdir(outdir)
                    .warm_start(warm_start)
                    .seed(42)
            })
            .min_within(&array![[0., 3.], [0., 4.]])
            .unwrap()
            .run()
            .unwrap()
    };
    let _ = run(false);
    let res = run(true);
    let _ = std::fs::remove_dir_all(outdir);
    check_snapshot("g24_function_constraints_warm_start", &res);
}

#[test]
#[serial]
fn snapshot_xsinx_hot_start() {
    let outdir = "target/snapshot_hot_start";
    let _ = std::fs::remove_dir_all(outdir);
    let run = |hot_start: HotStartMode, max_iters: usize| {
        EgorBuilder::optimize(xsinx)
            .configure(|cfg| {
                cfg.doe(&array![[0.], [7.], [25.]])
                    .max_iters(max_iters)
                    .hot_start(hot_start)
                    .outdir(outdir)
                    .seed(42)
            })
            .min_within(&array![[0., 25.]])
            .unwrap()
            .run()
            .unwrap()
    };
    let _ = run(HotStartMode::Enabled, 3);
    let res = run(HotStartMode::ExtendedIters(3), 3);
    let _ = std::fs::remove_dir_all(outdir);
    assert_eq!(res.state.iter, 6);
    check_snapshot("xsinx_hot_start", &res);
}

/// xsinx with a deterministic pseudo-noise of amplitude 0.5
fn noisy_xsinx(x: &ArrayView2<f64>) -> Array2<f64> {
    xsinx(x) + x.mapv(|v| 0.5 * (1000. * v).sin())
}

fn noise_ratio_tuning() -> ParamTuning<f64> {
    ParamTuning::Optimized {
        init: 1e-2,
        bounds: (1e-10, 10.),
    }
}

/// Explicit 1D DOE, independent of the platform (unlike an optimized LHS of a few points)
fn xsinx_doe5() -> Array2<f64> {
    array![[0.], [6.], [12.], [18.], [25.]]
}

fn xsinx_doe(n: usize) -> Array2<f64> {
    Lhs::new(&array![[0., 25.]])
        .with_rng(Xoshiro256Plus::seed_from_u64(42))
        .sample(n)
}

#[test]
#[serial]
fn snapshot_noisy_xsinx_noise_ratio() {
    let res = EgorBuilder::optimize(noisy_xsinx)
        .configure(|cfg| {
            cfg.configure_gp(|gp| gp.noise_ratio(noise_ratio_tuning()))
                .doe(&xsinx_doe(8))
                .max_iters(6)
                .seed(42)
        })
        .min_within(&array![[0., 25.]])
        .unwrap()
        .run()
        .unwrap();
    check_snapshot("noisy_xsinx_noise_ratio", &res);
}

#[test]
#[serial]
fn snapshot_xsinx_trego_local_phase() {
    let res = EgorBuilder::optimize(xsinx)
        .configure(|cfg| {
            cfg.configure_trego(|trego| {
                trego.local_gp(true).local_gp_min_points(4).local_mean(true)
            })
            .max_iters(5)
            .seed(1)
        })
        .min_within(&array![[0., 25.]])
        .unwrap()
        .run()
        .unwrap();
    check_snapshot("xsinx_trego_local_phase", &res);
}

#[test]
#[serial]
fn snapshot_noisy_xsinx_trego_local_phase() {
    let res = EgorBuilder::optimize(noisy_xsinx)
        .configure(|cfg| {
            cfg.configure_gp(|gp| gp.noise_ratio(noise_ratio_tuning()))
                .configure_trego(|trego| trego.local_gp(true).local_mean(true))
                .doe(&xsinx_doe(8))
                .max_iters(4)
                .seed(42)
        })
        .min_within(&array![[0., 25.]])
        .unwrap()
        .run()
        .unwrap();
    check_snapshot("noisy_xsinx_trego_local_phase", &res);
}

#[test]
#[serial]
fn snapshot_sphere_kpls() {
    let dim = 5;
    let xlimits = Array2::from_shape_vec((dim, 2), [-2.0, 2.0].repeat(dim)).unwrap();
    let doe = Lhs::new(&xlimits)
        .with_rng(Xoshiro256Plus::seed_from_u64(0))
        .sample(10);
    let res = EgorBuilder::optimize(sphere)
        .configure(|cfg| {
            cfg.configure_gp(|gp| gp.kpls(2))
                .doe(&doe)
                .max_iters(5)
                .seed(42)
        })
        .min_within(&xlimits)
        .unwrap()
        .run()
        .unwrap();
    check_snapshot("sphere_kpls", &res);
}

/// Automatic clustering bounded to one cluster: surrogates are reclustered every
/// 10 added points while the single-cluster mixture remains deterministic.
/// With 9 initial points, reclustering happens at the second iteration.
#[test]
#[serial]
fn snapshot_xsinx_auto_clustering() {
    let doe = array![[0.], [3.], [6.], [9.], [12.], [15.], [18.], [21.], [25.]];
    let res = EgorBuilder::optimize(xsinx)
        .configure(|cfg| {
            cfg.configure_gp(|gp| gp.n_clusters(NbClusters::automax(1)))
                .doe(&doe)
                .max_iters(3)
                .seed(42)
        })
        .min_within(&array![[0., 25.]])
        .unwrap()
        .run()
        .unwrap();
    check_snapshot("xsinx_auto_clustering", &res);
}

/// All regression and correlation models: experts are selected by cross-validation
#[test]
#[serial]
fn snapshot_xsinx_all_specs() {
    let res = EgorBuilder::optimize(xsinx)
        .configure(|cfg| {
            cfg.configure_gp(|gp| {
                gp.regression_spec(RegressionSpec::ALL)
                    .correlation_spec(CorrelationSpec::ALL)
            })
            .doe(&xsinx_doe5())
            .max_iters(4)
            .seed(42)
        })
        .min_within(&array![[0., 25.]])
        .unwrap()
        .run()
        .unwrap();
    check_snapshot("xsinx_all_specs", &res);
}

#[test]
#[serial]
fn snapshot_branin_efi_fe() {
    let xlimits = array![[0., 1.], [0., 1.]];
    let doe = Lhs::new(&xlimits)
        .with_rng(Xoshiro256Plus::seed_from_u64(42))
        .sample(10);
    let res = EgorBuilder::optimize(branin_with_nans)
        .configure(|cfg| {
            cfg.doe(&doe)
                .infill_strategy(InfillStrategy::EI)
                .feasible_infill_strategy(FeasibleInfillStrategy::EfiFe(0.3))
                .max_iters(6)
                .seed(42)
        })
        .min_within(&xlimits)
        .unwrap()
        .run()
        .unwrap();
    check_snapshot("branin_efi_fe", &res);
}

/// Ask-and-tell: `suggest` on the growing data, the suggested points being evaluated afterwards
#[test]
#[serial]
fn snapshot_g24_suggest() {
    let egor = EgorServiceBuilder::optimize()
        .configure(|cfg| cfg.n_cstr(2).seed(42))
        .min_within(&array![[0., 3.], [0., 4.]])
        .unwrap();
    let mut x_doe = g24_doe(6);
    let mut y_doe = f_g24(&x_doe.view());
    for _ in 0..4 {
        let x_new = egor.suggest(&x_doe, &y_doe);
        let y_new = f_g24(&x_new.view());
        x_doe = concatenate![Axis(0), x_doe, x_new];
        y_doe = concatenate![Axis(0), y_doe, y_new];
    }
    check_snapshot_data("g24_suggest", &x_doe, &y_doe, SNAPSHOT_REL_TOL);
}

/// Ask-and-tell with noisy values: the best point (appended last) is selected from
/// the surrogate estimates
#[test]
#[serial]
fn snapshot_noisy_xsinx_suggest() {
    let egor = EgorServiceBuilder::optimize()
        .configure(|cfg| {
            cfg.configure_gp(|gp| gp.noise_ratio(noise_ratio_tuning()))
                .seed(42)
        })
        .min_within(&array![[0., 25.]])
        .unwrap();
    let mut x_doe = xsinx_doe(8);
    let mut y_doe = noisy_xsinx(&x_doe.view());
    for _ in 0..4 {
        let x_new = egor.suggest(&x_doe, &y_doe);
        let y_new = noisy_xsinx(&x_new.view());
        x_doe = concatenate![Axis(0), x_doe, x_new];
        y_doe = concatenate![Axis(0), y_doe, y_new];
    }
    let best = egor.best_index(&x_doe, &y_doe);
    let x_data = concatenate![Axis(0), x_doe, x_doe.select(Axis(0), &[best])];
    let y_data = concatenate![Axis(0), y_doe, y_doe.select(Axis(0), &[best])];
    check_snapshot_data("noisy_xsinx_suggest", &x_data, &y_data, SNAPSHOT_REL_TOL);
}

/// GP variance portfolio: infill criterion optimized with several weights of the
/// variance, one point per cluster of the found points being added
#[test]
#[serial]
fn snapshot_xsinx_gp_var_portfolio() {
    let res = EgorBuilder::optimize(xsinx)
        .configure(|cfg| {
            cfg.configure_runtime_flags(|flags| flags.use_gp_var_portfolio(true))
                .doe(&xsinx_doe5())
                .max_iters(4)
                .seed(42)
        })
        .min_within(&array![[0., 25.]])
        .unwrap()
        .run()
        .unwrap();
    check_snapshot("xsinx_gp_var_portfolio", &res);
}

/// Mixture of two GP experts, whose clustering is seeded by the optimizer seed
#[test]
#[serial]
fn snapshot_xsinx_two_clusters() {
    let res = EgorBuilder::optimize(xsinx)
        .configure(|cfg| {
            cfg.configure_gp(|gp| gp.n_clusters(NbClusters::fixed(2)))
                .doe(&xsinx_doe(10))
                .max_iters(4)
                .seed(42)
        })
        .min_within(&array![[0., 25.]])
        .unwrap()
        .run()
        .unwrap();
    check_snapshot("xsinx_two_clusters", &res);
}
