//! Mono-objective non-regression snapshots.
//!
//! Each scenario runs a seeded optimization and compares the resulting `[x_doe, y_doe]`
//! bit for bit against a `.npy` fixture stored in `tests/snapshots/<os>-<backend>/`.
//! Floating point results differ across OS and optimizer/linear algebra backends,
//! hence fixtures are recorded per platform: a scenario without fixture for the current
//! platform is skipped, unless `EGOBOX_REQUIRE_SNAPSHOTS=1` is set (reference CI job).
//!
//! Record (or update) fixtures with `EGOBOX_UPDATE_SNAPSHOTS=1 cargo test --release --test mono_snapshots`.
//! A fixture update has to be committed separately and explicitly reviewed.

use egobox_doe::{Lhs, SamplingMethod};
use egobox_ego::{
    CoegoStatus, Cstr, CstrSpec, EgorBuilder, FailsafeStrategy, HotStartMode, InfillOptimizer,
    InfillStrategy, OptimResult, QEiStrategy, XType,
};
use ndarray::{Array2, ArrayView1, ArrayView2, Axis, Zip, array, concatenate};
use ndarray_npy::{read_npy, write_npy};
use ndarray_rand::rand::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;
use serial_test::serial;
use std::path::PathBuf;

const UPDATE_ENV: &str = "EGOBOX_UPDATE_SNAPSHOTS";
const REQUIRE_ENV: &str = "EGOBOX_REQUIRE_SNAPSHOTS";

fn snapshot_dir() -> PathBuf {
    let mut backend = vec![];
    if cfg!(feature = "c-cobyla") {
        backend.push("c-cobyla");
    }
    if cfg!(feature = "c-slsqp") {
        backend.push("c-slsqp");
    }
    if cfg!(feature = "blas") {
        backend.push("blas");
    }
    let backend = if backend.is_empty() {
        "default".to_string()
    } else {
        backend.join("+")
    };
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("snapshots")
        .join(format!("{}-{}", std::env::consts::OS, backend))
}

fn check_snapshot(name: &str, res: &OptimResult<f64>) {
    let data = concatenate![Axis(1), res.x_doe, res.y_doe];
    let filepath = snapshot_dir().join(format!("{name}.npy"));
    if std::env::var(UPDATE_ENV).is_ok() {
        std::fs::create_dir_all(snapshot_dir()).expect("snapshot dir creation");
        write_npy(&filepath, &data).expect("snapshot write");
        return;
    }
    if !filepath.exists() {
        assert!(
            std::env::var(REQUIRE_ENV).as_deref() != Ok("1"),
            "Missing snapshot {filepath:?} while {REQUIRE_ENV}=1"
        );
        eprintln!("No snapshot {filepath:?} for this platform: skipped");
        return;
    }
    let expected: Array2<f64> = read_npy(&filepath).expect("snapshot read");
    assert_eq!(
        expected.dim(),
        data.dim(),
        "{name}: [x_doe, y_doe] shape differs from snapshot"
    );
    for ((i, j), v) in data.indexed_iter() {
        assert_eq!(
            expected[[i, j]].to_bits(),
            v.to_bits(),
            "{name}: [x_doe, y_doe][[{i}, {j}]] = {v} differs from snapshot {}",
            expected[[i, j]]
        );
    }
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
    check_snapshot("g24_cstr_specs", &res);
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

fn g24_qei(name: &str, strategy: QEiStrategy) {
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
    check_snapshot(name, &res);
}

#[test]
#[serial]
fn snapshot_g24_qei_kb() {
    g24_qei("g24_qei_kb", QEiStrategy::KrigingBeliever);
}

#[test]
#[serial]
fn snapshot_g24_qei_clmin() {
    g24_qei("g24_qei_clmin", QEiStrategy::ConstantLiarMinimum);
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
