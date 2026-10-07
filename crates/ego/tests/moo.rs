//! Multi-objective optimization (ParEGO) tests

use egobox_ego::{
    CstrSpec, EgorBuilder, EgorServiceBuilder, FailsafeStrategy, HotStartMode, MooStrategy,
    ParetoResult, TerminationReason, TerminationStatus, XType,
};
use ndarray::{Array1, Array2, ArrayView2, Axis, Zip, array};
use serial_test::serial;
use std::f64::consts::PI;

/// Exact 2-D hypervolume of the points dominating `ref_point`
fn hypervolume_2d(front: &Array2<f64>, ref_point: [f64; 2]) -> f64 {
    let mut pts: Vec<(f64, f64)> = front
        .rows()
        .into_iter()
        .map(|r| (r[0], r[1]))
        .filter(|(a, b)| *a < ref_point[0] && *b < ref_point[1])
        .collect();
    pts.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut hv = 0.;
    let mut current = ref_point[1];
    for (a, b) in pts {
        if b < current {
            hv += (ref_point[0] - a) * (current - b);
            current = b;
        }
    }
    hv
}

/// The run is not stopped prematurely (no weight yields a new point)
fn assert_max_iters_reached(res: &ParetoResult<f64>) {
    assert_eq!(
        res.state.termination_status,
        TerminationStatus::Terminated(TerminationReason::MaxItersReached),
        "{} points evaluated",
        res.x_doe.nrows()
    );
}

fn assert_non_dominated(y: &Array2<f64>, n_obj: usize) {
    for (i, a) in y.rows().into_iter().enumerate() {
        for (j, b) in y.rows().into_iter().enumerate() {
            if i != j {
                let dominated = (0..n_obj).all(|k| b[k] <= a[k]) && (0..n_obj).any(|k| b[k] < a[k]);
                assert!(!dominated, "Pareto point {a} dominated by {b}");
            }
        }
    }
}

/// ZDT1 with 2 variables in [0, 1]: front f2 = 1 - sqrt(f1) for x2 = 0
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

/// HV of ZDT1 true front wrt (1.1, 1.1)
const ZDT1_HV_REF: f64 = 0.1 + 2. / 3. + 0.11;

/// Binh and Korn problem: [f1, f2, c1, c2] with c <= 0
fn bnh(x: &ArrayView2<f64>) -> Array2<f64> {
    let mut y = Array2::zeros((x.nrows(), 4));
    Zip::from(y.rows_mut())
        .and(x.rows())
        .for_each(|mut yi, xi| {
            let (x1, x2) = (xi[0], xi[1]);
            yi.assign(&array![
                4. * x1 * x1 + 4. * x2 * x2,
                (x1 - 5.).powi(2) + (x2 - 5.).powi(2),
                (x1 - 5.).powi(2) + x2 * x2 - 25.,
                7.7 - (x1 - 8.).powi(2) - (x2 + 3.).powi(2),
            ]);
        });
    y
}

/// Binh and Korn problem with raw constraints c1 <= 25 and c2 >= 7.7
fn bnh_raw(x: &ArrayView2<f64>) -> Array2<f64> {
    let mut y = bnh(x);
    y.column_mut(2).mapv_inplace(|v| v + 25.);
    y.column_mut(3).mapv_inplace(|v| 7.7 - v);
    y
}

/// DTLZ2 with 3 objectives and 4 variables in [0, 1]: front is the unit sphere octant
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
            let (a, b) = (xi[0] * PI / 2., xi[1] * PI / 2.);
            yi.assign(&array![
                (1. + g) * a.cos() * b.cos(),
                (1. + g) * a.cos() * b.sin(),
                (1. + g) * a.sin()
            ]);
        });
    y
}

fn run_zdt1(max_iters: usize) -> ParetoResult<f64> {
    EgorBuilder::optimize(zdt1)
        .configure(|cfg| cfg.n_obj(2).n_doe(10).max_iters(max_iters).seed(42))
        .min_within(&array![[0., 1.], [0., 1.]])
        .expect("Egor configured")
        .run_pareto()
        .expect("ZDT1 optimization")
}

#[test]
#[serial]
fn test_zdt1_parego() {
    let res = run_zdt1(30);
    assert_max_iters_reached(&res);
    assert_eq!(res.y_doe.ncols(), 2);
    assert_non_dominated(&res.y_pareto, 2);
    let hv = hypervolume_2d(&res.y_pareto, [1.1, 1.1]);
    println!(
        "ZDT1 front ({} points) HV = {hv} ({:.1}% of true front HV)",
        res.y_pareto.nrows(),
        100. * hv / ZDT1_HV_REF
    );
    assert!(hv > 0.85 * ZDT1_HV_REF);
}

#[test]
#[serial]
fn test_zdt1_parego_is_deterministic() {
    let res1 = run_zdt1(5);
    let res2 = run_zdt1(5);
    assert_eq!(res1.x_doe, res2.x_doe);
    assert_eq!(res1.y_pareto, res2.y_pareto);
}

#[test]
#[serial]
fn test_zdt1_run_returns_compromise_point() {
    let res = EgorBuilder::optimize(zdt1)
        .configure(|cfg| cfg.n_obj(2).n_doe(10).max_iters(5).seed(42))
        .min_within(&array![[0., 1.], [0., 1.]])
        .expect("Egor configured")
        .run()
        .expect("ZDT1 optimization");
    assert_eq!(res.y_opt.len(), 2);
    // compromise point is not dominated by any evaluated point
    let dominated = res
        .y_doe
        .rows()
        .into_iter()
        .any(|y| (0..2).all(|k| y[k] <= res.y_opt[k]) && (0..2).any(|k| y[k] < res.y_opt[k]));
    assert!(!dominated);
}

#[test]
#[serial]
fn test_zdt1_parego_hot_start_continues_like_uninterrupted_run() {
    let outdir = "target/test_moo_hot_start";
    let _ = std::fs::remove_dir_all(outdir);
    let run = |hot_start: HotStartMode, max_iters: usize| {
        EgorBuilder::optimize(zdt1)
            .configure(|cfg| {
                cfg.n_obj(2)
                    .n_doe(10)
                    .max_iters(max_iters)
                    .hot_start(hot_start)
                    .outdir(outdir)
                    .seed(42)
            })
            .min_within(&array![[0., 1.], [0., 1.]])
            .expect("Egor configured")
            .run_pareto()
            .expect("ZDT1 optimization")
    };
    let _ = run(HotStartMode::Enabled, 3);
    let resumed = run(HotStartMode::ExtendedIters(3), 3);
    let _ = std::fs::remove_dir_all(outdir);
    let straight = run(HotStartMode::Disabled, 6);
    assert_eq!(resumed.x_doe, straight.x_doe);
    assert_eq!(resumed.y_pareto, straight.y_pareto);
}

fn assert_bnh_front(res: &ParetoResult<f64>, feasible: impl Fn(&ndarray::ArrayView1<f64>) -> bool) {
    assert_non_dominated(&res.y_pareto, 2);
    assert!(
        res.y_pareto.nrows() >= 5,
        "front too small: {}",
        res.y_pareto
    );
    for y in res.y_pareto.rows() {
        assert!(feasible(&y), "infeasible Pareto point {y}");
    }
    let min_f: Array1<f64> =
        res.y_pareto
            .slice(ndarray::s![.., ..2])
            .fold_axis(Axis(0), f64::INFINITY, |m, &v| m.min(v));
    println!(
        "BNH front ({} points) min f = {min_f}",
        res.y_pareto.nrows()
    );
    // extreme points of the true front are (0, 50) and (136, 4)
    assert!(min_f[0] < 10. && min_f[1] < 10.);
}

#[test]
#[serial]
fn test_bnh_parego() {
    let res = EgorBuilder::optimize(bnh)
        .configure(|cfg| cfg.n_obj(2).n_cstr(2).n_doe(10).max_iters(30).seed(42))
        .min_within(&array![[0., 5.], [0., 3.]])
        .expect("Egor configured")
        .run_pareto()
        .expect("BNH optimization");
    assert_eq!(res.y_doe.ncols(), 4);
    assert_bnh_front(&res, |y| y[2] <= 1e-4 && y[3] <= 1e-4);
}

#[test]
#[serial]
fn test_bnh_parego_with_cstr_specs() {
    let res = EgorBuilder::optimize(bnh_raw)
        .configure(|cfg| {
            cfg.n_obj(2)
                .cstr_specs(vec![CstrSpec::Leq(25.), CstrSpec::Geq(7.7)])
                .n_doe(10)
                .max_iters(30)
                .seed(42)
        })
        .min_within(&array![[0., 5.], [0., 3.]])
        .expect("Egor configured")
        .run_pareto()
        .expect("BNH optimization");
    // results in raw constraint layout
    assert_eq!(res.y_doe.ncols(), 4);
    assert_bnh_front(&res, |y| y[2] <= 25. + 1e-4 && y[3] >= 7.7 - 1e-4);
}

#[test]
#[serial]
fn test_dtlz2_parego() {
    let xlimits = Array2::from_shape_vec((4, 2), [0., 1.].repeat(4)).unwrap();
    let res = EgorBuilder::optimize(dtlz2)
        .configure(|cfg| cfg.n_obj(3).n_doe(15).max_iters(30).seed(42))
        .min_within(&xlimits)
        .expect("Egor configured")
        .run_pareto()
        .expect("DTLZ2 optimization");
    assert_max_iters_reached(&res);
    assert_non_dominated(&res.y_pareto, 3);
    let distances = res
        .y_pareto
        .rows()
        .into_iter()
        .map(|y| y.dot(&y).sqrt() - 1.)
        .collect::<Vec<_>>();
    let mean_dist = distances.iter().sum::<f64>() / distances.len() as f64;
    println!(
        "DTLZ2 front ({} points) mean distance to true front = {mean_dist}",
        res.y_pareto.nrows()
    );
    assert!(res.y_pareto.nrows() >= 5);
    assert!(mean_dist < 0.35);
}

#[test]
#[serial]
fn test_zdt1_mixint_parego() {
    // second variable is an integer level in 0..=9 mapped to [0, 1]
    let f = |x: &ArrayView2<f64>| {
        let mut xr = x.to_owned();
        xr.column_mut(1).mapv_inplace(|v| v / 9.);
        zdt1(&xr.view())
    };
    let res = EgorBuilder::optimize(f)
        .configure(|cfg| cfg.n_obj(2).n_doe(10).max_iters(10).seed(42))
        .min_within_mixint_space(&[XType::Float(0., 1.), XType::Int(0, 9)])
        .expect("Egor configured")
        .run_pareto()
        .expect("mixed-integer ZDT1 optimization");
    assert_non_dominated(&res.y_pareto, 2);
    for x in res.x_pareto.rows() {
        assert_eq!(x[1], x[1].round());
    }
}

#[test]
fn test_moo_unsupported_configurations() {
    let xlimits = array![[0., 1.], [0., 1.]];
    assert!(
        EgorBuilder::optimize(zdt1)
            .configure(|cfg| cfg.n_obj(0))
            .min_within(&xlimits)
            .is_err()
    );
    assert!(
        EgorBuilder::optimize(zdt1)
            .configure(|cfg| cfg.n_obj(2).trego(true))
            .min_within(&xlimits)
            .is_err()
    );
    assert!(
        EgorBuilder::optimize(zdt1)
            .configure(|cfg| cfg.n_obj(2).target(0.))
            .min_within(&xlimits)
            .is_err()
    );
    assert!(
        EgorBuilder::optimize(zdt1)
            .configure(|cfg| cfg.n_obj(2).failsafe_strategy(FailsafeStrategy::Imputation))
            .min_within(&xlimits)
            .is_err()
    );
    assert!(
        EgorServiceBuilder::optimize()
            .configure(|cfg| cfg.n_obj(2))
            .min_within(&xlimits)
            .is_err()
    );
    assert!(
        EgorBuilder::optimize(zdt1)
            .configure(|cfg| cfg
                .n_obj(2)
                .configure_moo(|moo| moo.strategy(MooStrategy::ParEgo)))
            .min_within(&xlimits)
            .is_ok()
    );
    for rho in [-1., f64::NAN, f64::INFINITY] {
        assert!(
            EgorBuilder::optimize(zdt1)
                .configure(|cfg| cfg.n_obj(2).configure_moo(|moo| moo.rho(rho)))
                .min_within(&xlimits)
                .is_err()
        );
    }
    assert!(
        EgorBuilder::optimize(zdt1)
            .configure(|cfg| cfg.n_obj(2).configure_moo(|moo| moo.n_divisions(0)))
            .min_within(&xlimits)
            .is_err()
    );
}

#[test]
#[serial]
fn test_zdt1_parego_qei_state_param_and_cost_match() {
    let res = EgorBuilder::optimize(zdt1)
        .configure(|cfg| {
            cfg.n_obj(2)
                .n_doe(10)
                .configure_qei(|qei| qei.batch(3))
                .max_iters(3)
                .seed(42)
        })
        .min_within(&array![[0., 1.], [0., 1.]])
        .expect("Egor configured")
        .run_pareto()
        .expect("ZDT1 optimization");
    let param = res.state.param.as_ref().expect("current param");
    let cost = res.state.cost.as_ref().expect("current cost");
    let index = res
        .x_doe
        .rows()
        .into_iter()
        .position(|x| x == param)
        .expect("current param is an evaluated point");
    assert_eq!(res.y_doe.row(index), cost);
}

#[test]
#[serial]
fn test_mono_objective_run_pareto_returns_optimum() {
    let xsinx = |x: &ArrayView2<f64>| (x - 3.5) * ((x - 3.5) / PI).mapv(|v| v.sin());
    let egor = EgorBuilder::optimize(xsinx)
        .configure(|cfg| cfg.doe(&array![[0.], [7.], [25.]]).max_iters(5).seed(42))
        .min_within(&array![[0., 25.]])
        .expect("Egor configured");
    let res = egor.run().expect("xsinx optimization");
    let front = egor.run_pareto().expect("xsinx optimization");
    assert_eq!(front.x_pareto.nrows(), 1);
    assert_eq!(front.x_pareto.row(0), res.x_opt);
    assert_eq!(front.y_pareto.row(0), res.y_opt);
}

#[test]
#[serial]
fn test_parego_rejected_tries_keep_raw_state() {
    // Constant objectives: the infill criterion is flat, proposed points are the current
    // best one, hence rejected at every try and the solver converges
    let f = |x: &ArrayView2<f64>| Array2::zeros((x.nrows(), 2));
    let res = EgorBuilder::optimize(f)
        .configure(|cfg| {
            cfg.n_obj(2)
                .doe(&array![[0.], [0.5], [1.]])
                .max_iters(3)
                .seed(42)
        })
        .min_within(&array![[0., 1.]])
        .expect("Egor configured")
        .run_pareto()
        .expect("optimization");
    println!(
        "termination: {}, {} points",
        res.state.termination_status,
        res.x_doe.nrows()
    );
    // With the default optimizers, all tries are rejected (the C-ported ones may still find
    // new points on the flat criterion)
    if !cfg!(any(feature = "c-cobyla", feature = "c-slsqp")) {
        assert_eq!(
            res.state.termination_status,
            TerminationStatus::Terminated(TerminationReason::SolverConverged)
        );
    }
    // no scalarized virtual value leaks into the state
    if let Some(cost) = res.state.cost.as_ref() {
        assert_eq!(cost.len(), 2);
    }
}
