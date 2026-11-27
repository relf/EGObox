//! Optimization within hierarchical design spaces: the objective function only
//! sees valid points (excluded variables at their default value, value
//! constraints satisfied), and the optimizer makes progress.

use egobox_doe::{Lhs, SamplingMethod};
use egobox_ego::{CoegoStatus, DesignSpace, EgorBuilder, EgorServiceFactory, HotStartMode, XType};
use egobox_moe::{HierarchicalSampling, as_continuous_limits};
use ndarray::{Array2, ArrayView2, Zip, array};
use ndarray_rand::rand::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Number of rows of `x` (folded space) changed by the correction
fn n_invalid(ds: &DesignSpace, x: &ArrayView2<f64>) -> usize {
    let (xcorr, _) = ds.correct_get_included(x).unwrap();
    Zip::from(xcorr.rows())
        .and(x.rows())
        .fold(0, |n, a, b| n + usize::from(a != b))
}

/// Hierarchical Goldstein problem (Pelamatti et al.), as defined in SMT's
/// `HierarchicalGoldstein`: a meta variable with 4 levels decrees 2
/// continuous and 2 integer variables
fn goldstein_space() -> DesignSpace {
    let float = XType::Float(0., 100.);
    let int = XType::Int(0, 2);
    let mut ds = DesignSpace::new(&[
        XType::Enum(4),
        XType::Ord(vec![0., 1.]),
        float.clone(),
        float.clone(),
        float.clone(),
        float.clone(),
        float,
        int.clone(),
        int.clone(),
        int.clone(),
        int,
    ]);
    ds.decreed_var(4, 0, &[1., 3.])
        .unwrap()
        .decreed_var(5, 0, &[2., 3.])
        .unwrap()
        .decreed_var(7, 0, &[0., 2.])
        .unwrap()
        .decreed_var(8, 0, &[0., 1.])
        .unwrap();
    ds
}

#[allow(clippy::too_many_arguments)]
fn goldstein_h(x1: f64, x2: f64, x3: f64, x4: f64, z3: f64, z4: f64, x5: f64, cos: bool) -> f64 {
    let mut h = 53.3108 + 0.184901 * x1 - 5.02914 * x1.powi(3) * 1e-6
        + 7.72522 * x1.powf(z3) * 1e-8
        - 0.0870775 * x2
        - 0.106959 * x3
        + 7.98772 * x3.powf(z4) * 1e-6
        + 0.00242482 * x4
        + 1.32851 * x4.powi(3) * 1e-6
        - 0.00146393 * x1 * x2
        - 0.00301588 * x1 * x3
        - 0.00272291 * x1 * x4
        + 0.0017004 * x2 * x3
        + 0.0038428 * x2 * x4
        - 0.000198969 * x3 * x4
        + 1.86025 * x1 * x2 * x3 * 1e-5
        - 1.88719 * x1 * x2 * x4 * 1e-6
        + 2.50923 * x1 * x3 * x4 * 1e-5
        - 5.62199 * x2 * x3 * x4 * 1e-5;
    if cos {
        h += 5. * (2. * std::f64::consts::PI * x5 / 100.).cos() - 2.;
    }
    h
}

fn goldstein(x: &[f64]) -> f64 {
    let level = |z: f64| [20., 50., 80.][z as usize];
    let (x3, x4) = match x[0] as usize {
        0 => (level(x[7]), level(x[8])),
        1 => (x[4], level(x[8])),
        2 => (level(x[7]), x[5]),
        _ => (x[4], x[5]),
    };
    goldstein_h(x[2], x[3], x3, x4, x[9], x[10], x[6], x[1] == 1.)
}

/// Objective counting the invalid points it is given
fn counting<F: Fn(&[f64]) -> f64 + Clone + Sync>(
    ds: &DesignSpace,
    f: F,
    invalid: &Arc<AtomicUsize>,
) -> impl Fn(&ArrayView2<f64>) -> Array2<f64> + Clone + Sync {
    let (ds, invalid) = (ds.clone(), invalid.clone());
    move |x: &ArrayView2<f64>| {
        invalid.fetch_add(n_invalid(&ds, x), Ordering::Relaxed);
        Array2::from_shape_fn((x.nrows(), 1), |(i, _)| f(&x.row(i).to_vec()))
    }
}

#[test]
fn test_goldstein() {
    let ds = goldstein_space();
    let invalid = Arc::new(AtomicUsize::new(0));
    let res = EgorBuilder::optimize(counting(&ds, goldstein, &invalid))
        .configure(|config| config.n_doe(22).max_iters(25).seed(42))
        .min_within_design_space(&ds)
        .expect("Egor configured")
        .run()
        .expect("Egor minimization");

    assert_eq!(invalid.load(Ordering::Relaxed), 0);
    assert_eq!(n_invalid(&ds, &res.x_doe.view()), 0);
    // The global minimum is 8.94 (meta = 3, cosine term, floats at their
    // upper bound but x5 = 50, z3 = z4 = 0). With seeds 0 to 5, the best DoE
    // value is 28 to 31, and 25 iterations reach 9.2 to 19.7.
    let doe_best = res
        .y_doe
        .slice(ndarray::s![..22, 0])
        .fold(f64::INFINITY, |a, &b| a.min(b));
    assert!(
        res.y_opt[0] < 25. && res.y_opt[0] < doe_best - 5.,
        "y_opt = {}, best of DoE = {doe_best}",
        res.y_opt
    );
}

const ASGD: f64 = 0.;
const ADAM: f64 = 1.;

/// The MLP design space of the reference (Table 1, Fig. 14): no 3 layers
/// with Adam, no 40 or 45 units with ASGD
fn mlp_space() -> DesignSpace {
    let units = XType::Ord(vec![25., 30., 35., 40., 45.]);
    let mut ds = DesignSpace::named(&[
        ("learning_rate", XType::Float(0., 1.)),
        ("activation", XType::Enum(3)),
        ("optimizer", XType::Enum(2)),
        ("decay", XType::Float(0., 1.)),
        ("power_update", XType::Float(0., 1.)),
        ("average_start", XType::Float(0., 1.)),
        ("running_average_1", XType::Float(0., 1.)),
        ("running_average_2", XType::Float(0., 1.)),
        ("numerical_stability", XType::Float(0., 1.)),
        ("layers", XType::Int(1, 3)),
        ("units_1", units.clone()),
        ("units_2", units.clone()),
        ("units_3", units),
    ])
    .unwrap();
    for var in ["decay", "power_update", "average_start"] {
        ds.decreed_var(var, "optimizer", &[ASGD]).unwrap();
    }
    for var in [
        "running_average_1",
        "running_average_2",
        "numerical_stability",
    ] {
        ds.decreed_var(var, "optimizer", &[ADAM]).unwrap();
    }
    ds.decreed_var("units_1", "layers", &[1., 2., 3.])
        .unwrap()
        .decreed_var("units_2", "layers", &[2., 3.])
        .unwrap()
        .decreed_var("units_3", "layers", &[3.])
        .unwrap()
        .forbid("layers", &[3.], "optimizer", &[ADAM])
        .unwrap();
    for var in ["units_1", "units_2", "units_3"] {
        ds.forbid(var, &[40., 45.], "optimizer", &[ASGD]).unwrap();
    }
    ds
}

/// A loss on the MLP design space, 0 at Adam with 2 layers of 45 units
fn mlp_loss(x: &[f64]) -> f64 {
    let layers = x[9] as usize;
    let mut loss = (x[0] - 0.2).powi(2) + 0.1 * f64::from(x[1] != 1.);
    loss += if x[2] == ASGD {
        0.3 + (x[3] - 0.5).powi(2) + (x[4] - 0.5).powi(2)
    } else {
        (x[6] - 0.9).powi(2) + (x[7] - 0.9).powi(2) + (x[8] - 0.1).powi(2)
    };
    loss += 0.1 * (layers as f64 - 2.).abs();
    loss + x[10..10 + layers]
        .iter()
        .map(|u| (45. - u) / 100.)
        .sum::<f64>()
}

#[test]
fn test_mlp_with_value_constraints() {
    let ds = mlp_space();
    let invalid = Arc::new(AtomicUsize::new(0));
    let res = EgorBuilder::optimize(counting(&ds, mlp_loss, &invalid))
        .configure(|config| config.n_doe(30).max_iters(30).seed(0))
        .min_within_design_space(&ds)
        .expect("Egor configured")
        .run()
        .expect("Egor minimization");

    // No 3 layers with Adam nor 40 or 45 units with ASGD was evaluated
    assert_eq!(invalid.load(Ordering::Relaxed), 0);
    // The minimum 0 is at Adam with 2 layers of 45 units. Any loss below 0.3
    // is an Adam configuration (ASGD costs at least 0.3), and one layer more
    // or less, or another activation, costs 0.1. Observed: 0.007 without MKL,
    // 0.1006 with MKL (results depend on the linear algebra backend).
    assert!(res.y_opt[0] < 0.2, "y_opt = {}", res.y_opt);
    assert_eq!(res.x_opt[2], ADAM);
}

/// A wing with 0, 1 or 2 motors and the radius of each motor's propeller,
/// with a minimum at one motor of radius 1.2
fn motors_space() -> DesignSpace {
    let mut ds = DesignSpace::named(&[
        ("m", XType::Int(0, 2)),
        ("r_1", XType::Float(0.5, 2.)),
        ("r_2", XType::Float(0.5, 2.)),
    ])
    .unwrap();
    ds.decreed_var("r_1", "m", &[1., 2.])
        .unwrap()
        .decreed_var("r_2", "m", &[2.])
        .unwrap();
    ds
}

fn motors(x: &[f64]) -> f64 {
    let m = x[0] as usize;
    (m as f64 - 1.).powi(2) + x[1..1 + m].iter().map(|r| (r - 1.2).powi(2)).sum::<f64>()
}

#[test]
fn test_given_doe_is_corrected() {
    let ds = motors_space();
    // Excluded radii given values other than their default
    let doe = array![
        [0., 1.5, 1.5],
        [1., 0.9, 1.8],
        [2., 0.7, 1.9],
        [1., 1.6, 0.6]
    ];
    let invalid = Arc::new(AtomicUsize::new(0));
    let res = EgorBuilder::optimize(counting(&ds, motors, &invalid))
        .configure(|config| config.doe(&doe).max_iters(5).seed(1))
        .min_within_design_space(&ds)
        .expect("Egor configured")
        .run()
        .expect("Egor minimization");

    assert_eq!(invalid.load(Ordering::Relaxed), 0);
    let (expected, _) = ds.correct_get_included(&doe).unwrap();
    assert_eq!(res.x_doe.slice(ndarray::s![..4, ..]), expected);
}

#[test]
fn test_coego_is_rejected() {
    let ds = motors_space();
    let res = EgorBuilder::optimize(|x: &ArrayView2<f64>| Array2::zeros((x.nrows(), 1)))
        .configure(|config| config.coego(CoegoStatus::Enabled(2)))
        .min_within_design_space(&ds);
    assert!(res.is_err());
}

#[test]
fn test_service_suggests_valid_points() {
    let ds = mlp_space();
    let lhs = Lhs::new(&as_continuous_limits::<f64>(ds.xtypes()))
        .with_rng(Xoshiro256Plus::seed_from_u64(42));
    let mut sampler =
        HierarchicalSampling::new(lhs, ds.clone()).with_rng(Xoshiro256Plus::seed_from_u64(42));
    sampler.work_in_folded_space(true);
    let x_data = sampler.sample(20);
    let y_data = Array2::from_shape_fn((20, 1), |(i, _)| mlp_loss(&x_data.row(i).to_vec()));

    let egor = EgorServiceFactory::<egobox_ego::Cstr>::optimize()
        .configure(|config| config.seed(42))
        .min_within_design_space(&ds)
        .expect("Egor service configured");
    let x = egor.suggest(&x_data, &y_data).expect("suggestion");
    assert_eq!(x.ncols(), 13);
    assert_eq!(n_invalid(&ds, &x.view()), 0);
}

#[test]
fn test_hot_start_keeps_the_design_space() {
    let ds = goldstein_space();
    let outdir = std::env::temp_dir().join(format!("egobox-hierarchical-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&outdir);
    let outdir = outdir.to_str().unwrap().to_string();
    let run = |hot_start: HotStartMode, invalid: &Arc<AtomicUsize>| {
        EgorBuilder::optimize(counting(&ds, goldstein, invalid))
            .configure(|config| {
                config
                    .n_doe(15)
                    .max_iters(2)
                    .seed(3)
                    .outdir(outdir.clone())
                    .hot_start(hot_start)
            })
            .min_within_design_space(&ds)
            .expect("Egor configured")
            .run()
            .expect("Egor minimization")
    };
    let invalid = Arc::new(AtomicUsize::new(0));
    run(HotStartMode::Enabled, &invalid);
    // Resumed from the checkpoint, which does not hold the design space
    let res = run(HotStartMode::ExtendedIters(3), &invalid);
    assert_eq!(invalid.load(Ordering::Relaxed), 0);
    assert_eq!(res.x_doe.nrows(), 15 + 2 + 3);
    let _ = std::fs::remove_dir_all(&outdir);
}
