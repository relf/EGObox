//! Multi-objective non-regression snapshots (see `common` for the fixture mechanism).

mod common;

use common::{SNAPSHOT_REL_TOL, check_snapshot_data};
use egobox_ego::{EgorBuilder, MooStrategy, ParetoResult};
use ndarray::{Array2, ArrayView2, Zip, array};
use serial_test::serial;

/// ZDT1 bi-objective test function: the Pareto front is f2 = 1 - sqrt(f1) with x2 = 0
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

fn zdt1_with(strategy: MooStrategy, batch: usize, max_iters: usize) -> ParetoResult<f64> {
    EgorBuilder::optimize(zdt1)
        .configure(|cfg| {
            cfg.n_obj(2)
                .configure_moo(|moo| moo.strategy(strategy))
                .configure_qei(|qei| qei.batch(batch))
                .n_doe(6)
                .max_iters(max_iters)
                .seed(42)
        })
        .min_within(&array![[0., 1.], [0., 1.]])
        .unwrap()
        .run_pareto()
        .unwrap()
}

fn check_pareto_snapshot(name: &str, res: &ParetoResult<f64>) {
    check_snapshot_data(name, &res.x_doe, &res.y_doe, SNAPSHOT_REL_TOL)
}

#[test]
#[serial]
fn snapshot_zdt1_ehvi() {
    check_pareto_snapshot("zdt1_ehvi", &zdt1_with(MooStrategy::Ehvi, 1, 4));
}

#[test]
#[serial]
fn snapshot_zdt1_eim() {
    check_pareto_snapshot("zdt1_eim", &zdt1_with(MooStrategy::Eim, 1, 4));
}

#[test]
#[serial]
fn snapshot_zdt1_parego() {
    check_pareto_snapshot("zdt1_parego", &zdt1_with(MooStrategy::ParEgo, 1, 4));
}

#[test]
#[serial]
fn snapshot_zdt1_qehvi() {
    check_pareto_snapshot("zdt1_qehvi", &zdt1_with(MooStrategy::QEhvi, 2, 2));
}
