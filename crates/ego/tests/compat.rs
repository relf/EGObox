//! Persisted optimizer data compatibility: hot start checkpoints and GP model files saved
//! by a previous version (fixtures in `tests/compat/`) must still load. A checkpoint holds
//! the whole solver (configuration with polymorphic strategies and criteria, surrogate
//! builder) and its state (data, surrogate models), so resuming from it guards their
//! serde layouts and typetag tags.
//!
//! Record (or update) fixtures with
//! `EGOBOX_UPDATE_COMPAT_FIXTURES=1 cargo test --release -p egobox-ego --test compat`.
//! A fixture update means saved checkpoints or models of users can no longer be loaded:
//! it has to be committed separately and explicitly reviewed.
//!
//! Checkpoints are only loaded by a solver whose type name (`std::any::type_name`, including
//! module paths) is the one recorded in the checkpoint. As the type name format of a given type
//! may differ from one compiler version to another, the checkpoint resumption is skipped when
//! the recorded type name only differs in its formatting.

use egobox_doe::{Lhs, SamplingMethod};
use egobox_ego::{
    CHECKPOINT_FILE, Cstr, CstrSpec, EGOR_INITIAL_GP_FILENAME, EgorBuilder, EgorSolver,
    HotStartMode, OptimResult, TerminationReason, TerminationStatus, XType, load_gp_models,
};
use egobox_moe::{GpMixtureParams, MixintGpMixtureParams};
use ndarray::{Array2, ArrayView1, ArrayView2, Axis, Zip, array, stack};
use ndarray_npy::{read_npy, write_npy};
use ndarray_rand::rand::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;
use serial_test::serial;
use std::any::type_name;
use std::path::PathBuf;

const UPDATE_ENV: &str = "EGOBOX_UPDATE_COMPAT_FIXTURES";
const ABS_TOL: f64 = 1e-10;
const REL_TOL: f64 = 1e-8;
/// Iterations done before the checkpoint and run again after resuming from it
const N_ITERS: usize = 3;

fn updating() -> bool {
    std::env::var(UPDATE_ENV).is_ok()
}

fn fixture(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("compat");
    if updating() {
        std::fs::create_dir_all(&dir).expect("compat fixture dir creation");
    }
    dir.join(name)
}

fn g24(x: &ArrayView1<f64>) -> f64 {
    -x[0] - x[1]
}

fn g24_c1(x: &ArrayView1<f64>) -> f64 {
    -2.0 * x[0].powf(4.0) + 8.0 * x[0].powf(3.0) - 8.0 * x[0].powf(2.0) + x[1]
}

fn g24_c2(x: &ArrayView1<f64>) -> f64 {
    -4.0 * x[0].powf(4.0) + 32.0 * x[0].powf(3.0) - 88.0 * x[0].powf(2.0) + 96.0 * x[0] + x[1]
}

fn f_g24(x: &ArrayView2<f64>) -> Array2<f64> {
    let mut y = Array2::zeros((x.nrows(), 3));
    Zip::from(y.rows_mut())
        .and(x.rows())
        .for_each(|mut yi, xi| yi.assign(&array![g24(&xi), g24_c1(&xi), g24_c2(&xi)]));
    y
}

fn mixed_xsinx(x: &ArrayView2<f64>) -> Array2<f64> {
    x.map_axis(Axis(1), |xi| {
        (xi[0] - 3.5) * ((xi[0] - 3.5) / std::f64::consts::PI).sin() + 0.1 * xi[1]
    })
    .insert_axis(Axis(1))
}

/// Constrained problem with constraint specifications: the `Btw` specification
/// gives a derived constraint, modeled with an affine transform of a mixture
fn run_g24(outdir: &str, hot_start: HotStartMode) -> OptimResult<f64> {
    let doe = Lhs::new(&array![[0., 3.], [0., 4.]])
        .with_rng(Xoshiro256Plus::seed_from_u64(42))
        .sample(6);
    EgorBuilder::optimize(f_g24)
        .configure(|cfg| {
            cfg.cstr_specs(vec![CstrSpec::Leq(2.0), CstrSpec::Btw(-100., 36.0)])
                .doe(&doe)
                .max_iters(N_ITERS)
                .hot_start(hot_start)
                .outdir(outdir)
                .seed(42)
        })
        .min_within(&array![[0., 3.], [0., 4.]])
        .expect("g24 optimizer")
        .run()
        .expect("g24 optimization")
}

/// Mixed-integer problem with TREGO: mixed-integer surrogate builder and mixtures,
/// TREGO strategy and state
fn run_mixint(outdir: &str, hot_start: HotStartMode) -> OptimResult<f64> {
    EgorBuilder::optimize(mixed_xsinx)
        .configure(|cfg| {
            cfg.n_doe(5)
                .trego(true)
                .max_iters(N_ITERS)
                .hot_start(hot_start)
                .outdir(outdir)
                .seed(42)
        })
        .min_within_mixint_space(&[XType::Float(0., 25.), XType::Int(0, 5)])
        .expect("mixed-integer optimizer")
        .run()
        .expect("mixed-integer optimization")
}

/// Predicted values and variances of each model as columns
fn predictions(path: &PathBuf, n_models: usize, xtest: &Array2<f64>) -> Array2<f64> {
    let models = load_gp_models(path).expect("GP models file");
    assert_eq!(models.len(), n_models, "{path:?}: number of GP models");
    let columns = models
        .iter()
        .flat_map(|model| {
            let (mean, var) = model.predict_valvar(&xtest.view()).expect("prediction");
            [mean, var]
        })
        .collect::<Vec<_>>();
    let views = columns.iter().map(|c| c.view()).collect::<Vec<_>>();
    stack(Axis(1), &views).expect("predictions")
}

/// Type name without elided lifetimes, formatted differently by compiler versions
/// (e.g. `fn(&'_ [f64])` or `fn(&[f64])`)
fn type_name_format_free(name: &str) -> String {
    name.replace("'_ ", "")
}

/// Record (when updating) then check the GP models file (initial models, always saved:
/// TREGO local steps do not save models) and the checkpoint of the `name` run
/// done by a solver of type `solver_type`
fn check_persisted_run(
    name: &str,
    run: impl Fn(&str, HotStartMode) -> OptimResult<f64>,
    solver_type: &str,
    n_models: usize,
    xtest: &Array2<f64>,
) {
    let outdir = format!("target/compat_{name}");
    let checkpoint = fixture(&format!("{name}_checkpoint.bin"));
    let checkpoint_solver_type = fixture(&format!("{name}_checkpoint_solver_type.txt"));
    let gp_models = fixture(&format!("{name}_gp_models.bin"));
    let reference = fixture(&format!("{name}_gp_predictions.npy"));
    let _ = std::fs::remove_dir_all(&outdir);
    if updating() {
        run(&outdir, HotStartMode::Enabled);
        let saved = PathBuf::from(&outdir);
        std::fs::copy(saved.join(CHECKPOINT_FILE), &checkpoint).expect("checkpoint copy");
        std::fs::write(&checkpoint_solver_type, solver_type).expect("solver type save");
        std::fs::copy(saved.join(EGOR_INITIAL_GP_FILENAME), &gp_models).expect("GP models copy");
        write_npy(&reference, &predictions(&gp_models, n_models, xtest)).expect("predictions save");
        std::fs::remove_dir_all(&outdir).expect("output dir cleanup");
    }

    // GP models saved along the optimization
    let expected: Array2<f64> = read_npy(&reference).expect("reference predictions");
    let actual = predictions(&gp_models, n_models, xtest);
    assert_eq!(actual.dim(), expected.dim(), "{name}: predictions shape");
    for ((i, j), v) in actual.indexed_iter() {
        let e = expected[[i, j]];
        assert!(
            (v - e).abs() <= ABS_TOL + REL_TOL * e.abs(),
            "{name}: prediction[[{i}, {j}]] = {v} differs from {e}"
        );
    }

    // Optimization resumed from the saved checkpoint
    let recorded_type = std::fs::read_to_string(&checkpoint_solver_type).expect("solver type");
    if recorded_type != solver_type {
        assert_eq!(
            type_name_format_free(&recorded_type),
            type_name_format_free(solver_type),
            "{name}: solver type changed, previous checkpoints can no longer be loaded"
        );
        eprintln!(
            "{name}: checkpoint resumption skipped, solver type name formatted differently \
             by this compiler ({solver_type}) than by the recording one ({recorded_type})"
        );
        return;
    }
    std::fs::create_dir_all(&outdir).expect("output dir creation");
    std::fs::copy(&checkpoint, PathBuf::from(&outdir).join(CHECKPOINT_FILE))
        .expect("checkpoint restore");
    let res = run(&outdir, HotStartMode::ExtendedIters(2));
    let _ = std::fs::remove_dir_all(&outdir);
    assert_eq!(
        res.state.iter,
        (N_ITERS + 2) as u64,
        "{name}: resumed iterations"
    );
    assert_eq!(
        res.state.termination_status,
        TerminationStatus::Terminated(TerminationReason::MaxItersReached)
    );
}

#[test]
#[serial]
fn compat_g24_cstr_specs_checkpoint() {
    // objective, Leq constraint, Btw constraint and its derived (affine) counterpart
    let xtest = array![[0.5, 1.], [1.5, 2.5], [2.5, 3.5]];
    let solver_type = type_name::<EgorSolver<GpMixtureParams<f64>, Cstr>>();
    check_persisted_run("g24_cstr_specs", run_g24, solver_type, 4, &xtest);
}

#[test]
#[serial]
fn compat_mixint_checkpoint() {
    let xtest = array![[2., 1.], [12., 3.], [19., 4.]];
    let solver_type = type_name::<EgorSolver<MixintGpMixtureParams, Cstr>>();
    check_persisted_run("mixint", run_mixint, solver_type, 1, &xtest);
}
