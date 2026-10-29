//! Non-regression snapshot helpers shared by the snapshot test suites.
//!
//! Each scenario runs a seeded optimization and compares the resulting `[x_doe, y_doe]`
//! against a `.npy` fixture stored in `tests/snapshots/<os>-<backend>/`.
//! Floating point results differ across OS and optimizer/linear algebra backends,
//! hence fixtures are recorded per platform: a scenario without fixture for the current
//! platform is skipped, unless `EGOBOX_REQUIRE_SNAPSHOTS=1` is set (reference CI job).
//! Even on a given platform, math library code paths depend on the CPU, so values are
//! compared with a tight tolerance scaled by the magnitude of each column
//! (`SNAPSHOT_ABS_TOL + SNAPSHOT_REL_TOL * max |column|`), as values close to zero
//! (e.g. active constraints) amplify tiny differences of the evaluated points:
//! an actual behavior change moves the evaluated points far beyond it.
//!
//! Record (or update) fixtures with
//! `EGOBOX_UPDATE_SNAPSHOTS=1 cargo test --release --test mono_snapshots --test moo_snapshots`.
//! A fixture update has to be committed separately and explicitly reviewed.
#![allow(dead_code)]

use ndarray::{Array2, Axis, concatenate};
use ndarray_npy::{read_npy, write_npy};
use std::path::PathBuf;

pub const UPDATE_ENV: &str = "EGOBOX_UPDATE_SNAPSHOTS";
pub const REQUIRE_ENV: &str = "EGOBOX_REQUIRE_SNAPSHOTS";
pub const SNAPSHOT_ABS_TOL: f64 = 1e-8;
pub const SNAPSHOT_REL_TOL: f64 = 1e-6;

pub fn snapshot_dir() -> PathBuf {
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

/// Compare `[x_doe, y_doe]` against the `name` fixture with the given relative tolerance
pub fn check_snapshot_data(name: &str, x_doe: &Array2<f64>, y_doe: &Array2<f64>, rel_tol: f64) {
    let data = concatenate![Axis(1), *x_doe, *y_doe];
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
    let scales: Vec<f64> = expected
        .columns()
        .into_iter()
        .map(|col| {
            col.iter()
                .filter(|v| v.is_finite())
                .fold(0., |m: f64, v| m.max(v.abs()))
        })
        .collect();
    for ((i, j), v) in data.indexed_iter() {
        let e = expected[[i, j]];
        let same = (v.is_nan() && e.is_nan())
            || v == &e
            || (v - e).abs() <= SNAPSHOT_ABS_TOL + rel_tol * scales[j];
        assert!(
            same,
            "{name}: [x_doe, y_doe][[{i}, {j}]] = {v} differs from snapshot {e}"
        );
    }
}
