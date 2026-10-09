//! Multi-objective optimization (ParEGO) tests

use egobox_ego::{
    CstrSpec, EgorBuilder, EgorConfig, EgorServiceBuilder, EimAggregation, FailsafeStrategy,
    HotStartMode, MooStrategy, ParetoResult, TerminationReason, TerminationStatus, XType,
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
        .configure(|cfg| {
            cfg.n_obj(2)
                .configure_moo(|moo| moo.strategy(MooStrategy::ParEgo))
                .n_doe(10)
                .max_iters(max_iters)
                .seed(42)
        })
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
    // same compromise point with run_pareto, which is a point of the front
    let pareto = EgorBuilder::optimize(zdt1)
        .configure(|cfg| cfg.n_obj(2).n_doe(10).max_iters(5).seed(42))
        .min_within(&array![[0., 1.], [0., 1.]])
        .expect("Egor configured")
        .run_pareto()
        .expect("ZDT1 optimization");
    assert_eq!(pareto.x_opt, res.x_opt);
    assert_eq!(pareto.y_opt, res.y_opt);
    assert!(pareto.x_pareto.rows().into_iter().any(|x| x == res.x_opt));
}

#[test]
fn test_front_helpers() {
    use egobox_ego::{find_compromise_index, find_pareto_front_indices};
    // [f1, f2, c] with c <= 0
    let y = array![
        [1., 4., -1.],
        [2., 2., -1.],
        [4., 1., -1.],
        [3., 3., -1.], // dominated
        [0., 0., 1.],  // infeasible
        [f64::NAN, 0., -1.]
    ];
    let c = Array2::zeros((y.nrows(), 0));
    let tol = array![1e-4];
    assert_eq!(find_pareto_front_indices(&y, &c, 2, &tol), vec![0, 1, 2]);
    assert_eq!(find_compromise_index(&y, &c, 2, &tol), Some(1));
    // function constraint making the point 1 infeasible
    let c = array![[-1.], [1.], [-1.], [-1.], [-1.], [-1.]];
    let tol = array![1e-4, 1e-4];
    assert_eq!(find_pareto_front_indices(&y, &c, 2, &tol), vec![0, 2, 3]);
    // no feasible point: the smallest violation
    let y = array![[1., 4., 2.], [2., 2., 0.5], [4., 1., 3.]];
    let c = Array2::zeros((3, 0));
    let tol = array![1e-4];
    assert_eq!(find_pareto_front_indices(&y, &c, 2, &tol), vec![1]);
    assert_eq!(find_compromise_index(&y, &c, 2, &tol), Some(1));
    assert_eq!(
        find_compromise_index(
            &Array2::<f64>::zeros((0, 3)),
            &Array2::zeros((0, 0)),
            2,
            &tol
        ),
        None
    );
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
                    .configure_moo(|moo| moo.strategy(MooStrategy::ParEgo))
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
        .configure(|cfg| {
            cfg.n_obj(2)
                .configure_moo(|moo| moo.strategy(MooStrategy::ParEgo))
                .n_cstr(2)
                .n_doe(10)
                .max_iters(30)
                .seed(42)
        })
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
                .configure_moo(|moo| moo.strategy(MooStrategy::ParEgo))
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
        .configure(|cfg| {
            cfg.n_obj(3)
                .configure_moo(|moo| moo.strategy(MooStrategy::ParEgo))
                .n_doe(15)
                .max_iters(30)
                .seed(42)
        })
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
        .configure(|cfg| {
            cfg.n_obj(2)
                .configure_moo(|moo| moo.strategy(MooStrategy::ParEgo))
                .n_doe(10)
                .max_iters(10)
                .seed(42)
        })
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
            .configure(|cfg| {
                cfg.n_obj(2)
                    .configure_moo(|moo| moo.strategy(MooStrategy::ParEgo))
                    .failsafe_strategy(FailsafeStrategy::Imputation)
            })
            .min_within(&xlimits)
            .is_err()
    );
    assert!(
        EgorServiceBuilder::optimize()
            .configure(|cfg| cfg.n_obj(2))
            .min_within(&xlimits)
            .is_ok()
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
                .configure_moo(|moo| moo.strategy(MooStrategy::ParEgo))
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
                .configure_moo(|moo| moo.strategy(MooStrategy::ParEgo))
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

#[test]
#[serial]
fn test_solver_suggest_with_several_objectives() {
    use egobox_ego::{Cstr, EgorConfig, EgorSolver, to_xtypes};
    use egobox_moe::GpMixtureParams;

    let xlimits = array![[0., 5.], [0., 3.]];
    for strategy in [MooStrategy::ParEgo, MooStrategy::Eim, MooStrategy::Ehvi] {
        let config = EgorConfig::default()
            .xtypes(&to_xtypes(&xlimits))
            .n_obj(2)
            .n_cstr(2)
            .configure_moo(|moo| moo.strategy(strategy))
            .seed(42)
            .check()
            .expect("valid config");
        let solver = EgorSolver::<GpMixtureParams<f64>, Cstr>::new(config);
        let mut x = array![
            [0., 0.],
            [1., 1.],
            [2.5, 1.5],
            [4., 2.],
            [5., 3.],
            [3., 0.5]
        ];
        for _ in 0..3 {
            let y = bnh(&x.view());
            let x_new = solver.suggest(&x, &y);
            assert_eq!(x_new.dim(), (1, 2));
            assert!(
                Zip::from(x_new.row(0))
                    .and(xlimits.rows())
                    .all(|v, lim| lim[0] <= *v && *v <= lim[1])
            );
            x = ndarray::concatenate![Axis(0), x, x_new];
        }
    }
}

fn run_zdt1_eim(aggregation: EimAggregation, max_iters: usize) -> ParetoResult<f64> {
    EgorBuilder::optimize(zdt1)
        .configure(|cfg| {
            cfg.n_obj(2)
                .configure_moo(|moo| moo.strategy(MooStrategy::Eim).eim_aggregation(aggregation))
                .n_doe(10)
                .max_iters(max_iters)
                .seed(42)
        })
        .min_within(&array![[0., 1.], [0., 1.]])
        .expect("Egor configured")
        .run_pareto()
        .expect("ZDT1 optimization")
}

#[test]
#[serial]
fn test_zdt1_eim() {
    for aggregation in [
        EimAggregation::Euclidean,
        EimAggregation::Maximin,
        EimAggregation::Hypervolume,
    ] {
        let res = run_zdt1_eim(aggregation, 30);
        assert_non_dominated(&res.y_pareto, 2);
        let hv = hypervolume_2d(&res.y_pareto, [1.1, 1.1]);
        println!(
            "ZDT1 EIM {aggregation:?} front ({} points, {} evaluations, {}) HV = {hv} ({:.1}% of true front HV)",
            res.y_pareto.nrows(),
            res.x_doe.nrows(),
            res.state.termination_status,
            100. * hv / ZDT1_HV_REF
        );
        assert_max_iters_reached(&res);
        assert!(hv > 0.8 * ZDT1_HV_REF);
    }
}

#[test]
#[serial]
fn test_zdt1_eim_is_deterministic() {
    let res1 = run_zdt1_eim(EimAggregation::Euclidean, 5);
    let res2 = run_zdt1_eim(EimAggregation::Euclidean, 5);
    assert_eq!(res1.x_doe, res2.x_doe);
}

#[test]
#[serial]
fn test_zdt1_eim_hot_start_continues_like_uninterrupted_run() {
    let outdir = "target/test_moo_eim_hot_start";
    let _ = std::fs::remove_dir_all(outdir);
    let run = |hot_start: HotStartMode, max_iters: usize| {
        EgorBuilder::optimize(zdt1)
            .configure(|cfg| {
                cfg.n_obj(2)
                    .configure_moo(|moo| {
                        moo.strategy(MooStrategy::Eim)
                            .eim_aggregation(EimAggregation::Euclidean)
                    })
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
}

#[test]
#[serial]
fn test_bnh_eim() {
    for cstr_infill in [false, true] {
        let res = EgorBuilder::optimize(bnh)
            .configure(|cfg| {
                cfg.n_obj(2)
                    .n_cstr(2)
                    .cstr_infill(cstr_infill)
                    .configure_moo(|moo| {
                        moo.strategy(MooStrategy::Eim)
                            .eim_aggregation(EimAggregation::Hypervolume)
                    })
                    .n_doe(10)
                    .max_iters(30)
                    .seed(42)
            })
            .min_within(&array![[0., 5.], [0., 3.]])
            .expect("Egor configured")
            .run_pareto()
            .expect("BNH optimization");
        assert_bnh_front(&res, |y| y[2] <= 1e-4 && y[3] <= 1e-4);
    }
}

#[test]
#[serial]
fn test_dtlz2_eim() {
    let xlimits = Array2::from_shape_vec((4, 2), [0., 1.].repeat(4)).unwrap();
    let res = EgorBuilder::optimize(dtlz2)
        .configure(|cfg| {
            cfg.n_obj(3)
                .configure_moo(|moo| {
                    moo.strategy(MooStrategy::Eim)
                        .eim_aggregation(EimAggregation::Hypervolume)
                })
                .n_doe(15)
                .max_iters(30)
                .seed(42)
        })
        .min_within(&xlimits)
        .expect("Egor configured")
        .run_pareto()
        .expect("DTLZ2 optimization");
    assert_max_iters_reached(&res);
    assert_non_dominated(&res.y_pareto, 3);
    let mean_dist = res
        .y_pareto
        .rows()
        .into_iter()
        .map(|y| y.dot(&y).sqrt() - 1.)
        .sum::<f64>()
        / res.y_pareto.nrows() as f64;
    println!(
        "DTLZ2 EIM front ({} points) mean distance to true front = {mean_dist}",
        res.y_pareto.nrows()
    );
    assert!(res.y_pareto.nrows() >= 5);
    assert!(mean_dist < 0.35);
}

#[test]
#[serial]
fn test_zdt1_eim_kriging_believer_batch() {
    let res = EgorBuilder::optimize(zdt1)
        .configure(|cfg| {
            cfg.n_obj(2)
                .configure_moo(|moo| {
                    moo.strategy(MooStrategy::Eim)
                        .eim_aggregation(EimAggregation::Euclidean)
                })
                .configure_qei(|qei| qei.batch(3))
                .n_doe(10)
                .max_iters(8)
                .seed(42)
        })
        .min_within(&array![[0., 1.], [0., 1.]])
        .expect("Egor configured")
        .run_pareto()
        .expect("ZDT1 optimization");
    assert_max_iters_reached(&res);
    // batch points are distinct new points
    assert!(res.x_doe.nrows() > 10 + 8);
    let hv = hypervolume_2d(&res.y_pareto, [1.1, 1.1]);
    println!(
        "ZDT1 EIM Kriging believer batch front HV = {hv} ({:.1}% of true front HV)",
        100. * hv / ZDT1_HV_REF
    );
    assert!(hv > 0.7 * ZDT1_HV_REF);
}

#[test]
fn test_eim_unsupported_configurations() {
    assert!(
        EgorBuilder::optimize(zdt1)
            .configure(|cfg| {
                cfg.n_obj(2)
                    .configure_moo(|moo| {
                        moo.strategy(MooStrategy::Eim)
                            .eim_aggregation(EimAggregation::Euclidean)
                    })
                    .infill_strategy(egobox_ego::InfillStrategy::EI)
                    .feasible_infill_strategy(egobox_ego::FeasibleInfillStrategy::EfiP)
            })
            .min_within(&array![[0., 1.], [0., 1.]])
            .is_err()
    );
}

fn ehvi(cfg: EgorConfig) -> EgorConfig {
    cfg.configure_moo(|moo| moo.strategy(MooStrategy::Ehvi))
}

#[test]
#[serial]
fn test_zdt1_ehvi() {
    let res = EgorBuilder::optimize(zdt1)
        .configure(|cfg| ehvi(cfg.n_obj(2).n_doe(10).max_iters(30).seed(42)))
        .min_within(&array![[0., 1.], [0., 1.]])
        .expect("Egor configured")
        .run_pareto()
        .expect("ZDT1 optimization");
    assert_max_iters_reached(&res);
    assert_non_dominated(&res.y_pareto, 2);
    let hv = hypervolume_2d(&res.y_pareto, [1.1, 1.1]);
    println!(
        "ZDT1 EHVI front ({} points) HV = {hv} ({:.1}% of true front HV)",
        res.y_pareto.nrows(),
        100. * hv / ZDT1_HV_REF
    );
    assert!(hv > 0.85 * ZDT1_HV_REF);
}

#[test]
#[serial]
fn test_dtlz2_ehvi() {
    let xlimits = Array2::from_shape_vec((4, 2), [0., 1.].repeat(4)).unwrap();
    let res = EgorBuilder::optimize(dtlz2)
        .configure(|cfg| ehvi(cfg.n_obj(3).n_doe(15).max_iters(30).seed(42)))
        .min_within(&xlimits)
        .expect("Egor configured")
        .run_pareto()
        .expect("DTLZ2 optimization");
    assert_max_iters_reached(&res);
    assert_non_dominated(&res.y_pareto, 3);
    let mean_dist = res
        .y_pareto
        .rows()
        .into_iter()
        .map(|y| y.dot(&y).sqrt() - 1.)
        .sum::<f64>()
        / res.y_pareto.nrows() as f64;
    println!(
        "DTLZ2 EHVI front ({} points) mean distance to true front = {mean_dist}",
        res.y_pareto.nrows()
    );
    assert!(res.y_pareto.nrows() >= 5);
    assert!(mean_dist < 0.35);
}

#[test]
#[serial]
fn test_bnh_ehvi() {
    let res = EgorBuilder::optimize(bnh)
        .configure(|cfg| ehvi(cfg.n_obj(2).n_cstr(2).n_doe(10).max_iters(30).seed(42)))
        .min_within(&array![[0., 5.], [0., 3.]])
        .expect("Egor configured")
        .run_pareto()
        .expect("BNH optimization");
    assert_bnh_front(&res, |y| y[2] <= 1e-4 && y[3] <= 1e-4);
}

#[test]
#[serial]
fn test_zdt1_ehvi_kriging_believer_batch() {
    let res = EgorBuilder::optimize(zdt1)
        .configure(|cfg| {
            ehvi(cfg.n_obj(2).n_doe(10).max_iters(8).seed(42)).configure_qei(|qei| qei.batch(3))
        })
        .min_within(&array![[0., 1.], [0., 1.]])
        .expect("Egor configured")
        .run_pareto()
        .expect("ZDT1 optimization");
    assert_max_iters_reached(&res);
    assert!(res.x_doe.nrows() > 10 + 8);
    let hv = hypervolume_2d(&res.y_pareto, [1.1, 1.1]);
    println!(
        "ZDT1 EHVI Kriging believer batch front HV = {hv} ({:.1}% of true front HV)",
        100. * hv / ZDT1_HV_REF
    );
    assert!(hv > 0.7 * ZDT1_HV_REF);
}

#[test]
#[serial]
fn test_zdt1_hv_stop() {
    let max_iters = 100;
    let res = EgorBuilder::optimize(zdt1)
        .configure(|cfg| {
            cfg.n_obj(2)
                .n_doe(10)
                .max_iters(max_iters)
                .seed(42)
                .configure_moo(|moo| moo.strategy(MooStrategy::Ehvi).hv_stop(1e-3, 5))
        })
        .min_within(&array![[0., 1.], [0., 1.]])
        .expect("Egor configured")
        .run_pareto()
        .expect("ZDT1 optimization");
    println!(
        "ZDT1 EHVI with hv_stop: {} after {} iterations",
        res.state.termination_status, res.state.iter
    );
    assert_eq!(
        res.state.termination_status,
        TerminationStatus::Terminated(TerminationReason::SolverConverged)
    );
    assert!((res.state.iter as usize) < max_iters);
    let hv = hypervolume_2d(&res.y_pareto, [1.1, 1.1]);
    assert!(hv > 0.85 * ZDT1_HV_REF);
}

#[test]
fn test_ehvi_too_many_objectives() {
    let xlimits = array![[0., 1.], [0., 1.]];
    let config = |n_obj: usize| {
        EgorBuilder::optimize(zdt1)
            .configure(|cfg| ehvi(cfg.n_obj(n_obj)))
            .min_within(&xlimits)
    };
    assert!(config(8).is_ok());
    assert!(config(9).is_err());
}

#[test]
fn test_hv_stop_invalid_configurations() {
    for (tol, n_iters) in [(-1., 3), (f64::NAN, 3), (1e-3, 0)] {
        assert!(
            EgorBuilder::optimize(zdt1)
                .configure(|cfg| cfg.n_obj(2).configure_moo(|moo| moo.hv_stop(tol, n_iters)))
                .min_within(&array![[0., 1.], [0., 1.]])
                .is_err()
        );
    }
}

#[test]
#[serial]
fn test_zdt1_ask_and_tell() {
    use egobox_doe::SamplingMethod;
    for strategy in [MooStrategy::ParEgo, MooStrategy::Eim, MooStrategy::Ehvi] {
        let egor = EgorServiceBuilder::optimize()
            .configure(|cfg| {
                cfg.n_obj(2)
                    .configure_moo(|moo| moo.strategy(strategy.clone()))
                    .seed(42)
            })
            .min_within(&array![[0., 1.], [0., 1.]])
            .expect("Egor service configured");
        let mut x = egobox_doe::Lhs::new(&array![[0., 1.], [0., 1.]])
            .with_rng(
                <rand_xoshiro::Xoshiro256Plus as ndarray_rand::rand::SeedableRng>::seed_from_u64(
                    42,
                ),
            )
            .sample(10);
        for _ in 0..15 {
            let y = zdt1(&x.view());
            let x_new = egor.suggest(&x, &y);
            x = ndarray::concatenate![Axis(0), x, x_new];
        }
        let y = zdt1(&x.view());
        let front: Vec<usize> = (0..y.nrows())
            .filter(|&i| {
                !(0..y.nrows()).any(|j| {
                    (0..2).all(|k| y[[j, k]] <= y[[i, k]]) && (0..2).any(|k| y[[j, k]] < y[[i, k]])
                })
            })
            .collect();
        let hv = hypervolume_2d(&y.select(Axis(0), &front), [1.1, 1.1]);
        println!(
            "ZDT1 ask-and-tell {strategy:?}: front of {} points, HV = {hv} ({:.1}% of true front HV)",
            front.len(),
            100. * hv / ZDT1_HV_REF
        );
        assert!(hv > 0.6 * ZDT1_HV_REF);
    }
}

/// Branin-like objective failing (NaN) in the lower left corner, and distance to (1, 1)
fn branin_distance_with_nans(x: &ArrayView2<f64>) -> Array2<f64> {
    let mut y = Array2::zeros((x.nrows(), 2));
    Zip::from(y.rows_mut())
        .and(x.rows())
        .for_each(|mut yi, xi| {
            if xi[0] * xi[1] >= 0.2 {
                let (x0, x1) = (15. * xi[0] - 5., 15. * xi[1]);
                let a = x1 - 5.1 / (4. * PI * PI) * x0 * x0 + 5. / PI * x0 - 6.;
                let f1 = a * a + 10. * (1. - 1. / (8. * PI)) * x0.cos() + 10.;
                let f2 = (xi[0] - 1.).powi(2) + (xi[1] - 1.).powi(2);
                yi.assign(&array![f1, f2]);
            } else {
                yi.fill(f64::NAN);
            }
        });
    y
}

#[test]
#[serial]
fn test_imputation_with_per_objective_strategies() {
    let xlimits = array![[0., 1.], [0., 1.]];
    for strategy in [MooStrategy::Eim, MooStrategy::Ehvi] {
        let res = EgorBuilder::optimize(branin_distance_with_nans)
            .configure(|cfg| {
                cfg.n_obj(2)
                    .configure_moo(|moo| moo.strategy(strategy.clone()))
                    .failsafe_strategy(FailsafeStrategy::Imputation)
                    .n_doe(10)
                    .max_iters(15)
                    .seed(42)
            })
            .min_within(&xlimits)
            .expect("Egor configured")
            .run_pareto()
            .expect("optimization with failures");
        let x_fail = res.state.surrogate.x_fail.clone().expect("failed points");
        println!(
            "{strategy:?} with imputation: {} points, {} failed, front of {} points, {}",
            res.x_doe.nrows(),
            x_fail.nrows(),
            res.x_pareto.nrows(),
            res.state.termination_status
        );
        assert!(x_fail.nrows() > 0);
        assert!(res.x_pareto.nrows() > 0);
        assert_non_dominated(&res.y_pareto, 2);
        // failed points (stored with imputed values) are never Pareto points
        for x in res.x_pareto.rows() {
            assert!(x[0] * x[1] >= 0.2, "failed point {x} in the Pareto set");
        }
    }
}

#[test]
#[serial]
fn test_zdt1_eim_constant_liar_batch() {
    let res = EgorBuilder::optimize(zdt1)
        .configure(|cfg| {
            cfg.n_obj(2)
                .configure_moo(|moo| moo.strategy(MooStrategy::Eim))
                .configure_qei(|qei| {
                    qei.batch(3)
                        .strategy(egobox_ego::QEiStrategy::ConstantLiarMinimum)
                })
                .n_doe(10)
                .max_iters(8)
                .seed(42)
        })
        .min_within(&array![[0., 1.], [0., 1.]])
        .expect("Egor configured")
        .run_pareto()
        .expect("ZDT1 optimization");
    assert_max_iters_reached(&res);
    assert!(res.x_doe.nrows() > 10 + 8);
    assert_non_dominated(&res.y_pareto, 2);
    let hv = hypervolume_2d(&res.y_pareto, [1.1, 1.1]);
    println!(
        "ZDT1 EIM constant liar batch: {} points, HV = {hv} ({:.1}% of true front HV)",
        res.x_doe.nrows(),
        100. * hv / ZDT1_HV_REF
    );
    assert!(hv > 0.7 * ZDT1_HV_REF);
}

fn qehvi(cfg: EgorConfig) -> EgorConfig {
    cfg.configure_moo(|moo| moo.strategy(MooStrategy::QEhvi))
}

fn run_zdt1_qehvi(batch: usize, max_iters: usize) -> ParetoResult<f64> {
    EgorBuilder::optimize(zdt1)
        .configure(|cfg| {
            qehvi(cfg.n_obj(2).n_doe(10).max_iters(max_iters).seed(42))
                .configure_qei(|qei| qei.batch(batch))
        })
        .min_within(&array![[0., 1.], [0., 1.]])
        .expect("Egor configured")
        .run_pareto()
        .expect("ZDT1 optimization")
}

#[test]
#[serial]
fn test_zdt1_qehvi_batch() {
    let res = run_zdt1_qehvi(3, 8);
    assert_max_iters_reached(&res);
    assert!(res.x_doe.nrows() > 10 + 8);
    // points of a batch are distinct
    for i in 0..res.x_doe.nrows() {
        for j in 0..i {
            let d = (&res.x_doe.row(i) - &res.x_doe.row(j)).mapv(f64::abs).sum();
            assert!(
                d > 1e-6,
                "duplicated points {} and {}",
                res.x_doe.row(i),
                res.x_doe.row(j)
            );
        }
    }
    assert_non_dominated(&res.y_pareto, 2);
    let hv = hypervolume_2d(&res.y_pareto, [1.1, 1.1]);
    println!(
        "ZDT1 qEHVI batch: {} points, front of {} points, HV = {hv} ({:.1}% of true front HV)",
        res.x_doe.nrows(),
        res.y_pareto.nrows(),
        100. * hv / ZDT1_HV_REF
    );
    assert!(hv > 0.7 * ZDT1_HV_REF);
}

#[test]
#[serial]
fn test_zdt1_qehvi_is_deterministic() {
    let res1 = run_zdt1_qehvi(3, 3);
    let res2 = run_zdt1_qehvi(3, 3);
    assert_eq!(res1.x_doe, res2.x_doe);
}

#[test]
#[serial]
fn test_zdt1_qehvi_without_batch_is_ehvi() {
    let res = run_zdt1_qehvi(1, 5);
    let ehvi_res = EgorBuilder::optimize(zdt1)
        .configure(|cfg| ehvi(cfg.n_obj(2).n_doe(10).max_iters(5).seed(42)))
        .min_within(&array![[0., 1.], [0., 1.]])
        .expect("Egor configured")
        .run_pareto()
        .expect("ZDT1 optimization");
    assert_eq!(res.x_doe, ehvi_res.x_doe);
}

#[test]
#[serial]
fn test_zdt1_qehvi_hot_start_continues_like_uninterrupted_run() {
    let outdir = "target/test_moo_qehvi_hot_start";
    let _ = std::fs::remove_dir_all(outdir);
    let run = |hot_start: HotStartMode, max_iters: usize| {
        EgorBuilder::optimize(zdt1)
            .configure(|cfg| {
                qehvi(cfg.n_obj(2).n_doe(10).max_iters(max_iters))
                    .configure_qei(|qei| qei.batch(2))
                    .hot_start(hot_start)
                    .outdir(outdir)
                    .seed(42)
            })
            .min_within(&array![[0., 1.], [0., 1.]])
            .expect("Egor configured")
            .run_pareto()
            .expect("ZDT1 optimization")
    };
    let _ = run(HotStartMode::Enabled, 2);
    let resumed = run(HotStartMode::ExtendedIters(2), 2);
    let _ = std::fs::remove_dir_all(outdir);
    let straight = run(HotStartMode::Disabled, 4);
    assert_eq!(resumed.x_doe, straight.x_doe);
}

#[test]
#[serial]
fn test_bnh_qehvi_batch() {
    let res = EgorBuilder::optimize(bnh)
        .configure(|cfg| {
            qehvi(cfg.n_obj(2).n_cstr(2).n_doe(10).max_iters(12).seed(42))
                .configure_qei(|qei| qei.batch(2))
        })
        .min_within(&array![[0., 5.], [0., 3.]])
        .expect("Egor configured")
        .run_pareto()
        .expect("BNH optimization");
    assert_bnh_front(&res, |y| y[2] <= 1e-4 && y[3] <= 1e-4);
}

#[test]
#[serial]
fn test_qehvi_batch_with_imputation() {
    let res = EgorBuilder::optimize(branin_distance_with_nans)
        .configure(|cfg| {
            qehvi(cfg.n_obj(2).n_doe(10).max_iters(5).seed(42))
                .configure_qei(|qei| qei.batch(2))
                .failsafe_strategy(FailsafeStrategy::Imputation)
        })
        .min_within(&array![[0., 1.], [0., 1.]])
        .expect("Egor configured")
        .run_pareto()
        .expect("optimization with failures");
    assert!(res.x_pareto.nrows() > 0);
    assert_non_dominated(&res.y_pareto, 2);
    for x in res.x_pareto.rows() {
        assert!(x[0] * x[1] >= 0.2, "failed point {x} in the Pareto set");
    }
}

#[test]
fn test_qehvi_unsupported_configurations() {
    use egobox_moe::NbClusters;
    let xlimits = array![[0., 1.], [0., 1.]];
    let config = |n_obj: usize, batch: usize, n_clusters: NbClusters| {
        EgorBuilder::optimize(zdt1)
            .configure(|cfg| {
                qehvi(cfg.n_obj(n_obj))
                    .configure_qei(|qei| qei.batch(batch))
                    .configure_gp(|gp| gp.n_clusters(n_clusters))
            })
            .min_within(&xlimits)
    };
    assert!(config(2, 4, NbClusters::fixed(1)).is_ok());
    assert!(config(8, 2, NbClusters::fixed(1)).is_ok());
    assert!(config(2, 5, NbClusters::fixed(1)).is_err());
    assert!(config(9, 2, NbClusters::fixed(1)).is_err());
    assert!(config(2, 2, NbClusters::fixed(2)).is_err());
    assert!(config(2, 2, NbClusters::auto()).is_err());
}

#[test]
#[serial]
fn test_zdt1_mixint_qehvi_batch() {
    // second variable is an integer level in 0..=9 mapped to [0, 1]
    let f = |x: &ArrayView2<f64>| {
        let mut xr = x.to_owned();
        xr.column_mut(1).mapv_inplace(|v| v / 9.);
        zdt1(&xr.view())
    };
    let res = EgorBuilder::optimize(f)
        .configure(|cfg| {
            qehvi(cfg.n_obj(2).n_doe(10).max_iters(6).seed(42)).configure_qei(|qei| qei.batch(2))
        })
        .min_within_mixint_space(&[XType::Float(0., 1.), XType::Int(0, 9)])
        .expect("Egor configured")
        .run_pareto()
        .expect("mixed-integer ZDT1 optimization");
    assert_non_dominated(&res.y_pareto, 2);
    for x in res.x_pareto.rows() {
        assert_eq!(x[1], x[1].round());
    }
    // the front is found at the lowest level of the integer variable
    let best_level = res.x_pareto.column(1).fold(f64::INFINITY, |m, v| m.min(*v));
    println!(
        "Mixed-integer ZDT1 qEHVI front of {} points, levels {}",
        res.x_pareto.nrows(),
        res.x_pareto.column(1)
    );
    assert_eq!(best_level, 0.);
}
