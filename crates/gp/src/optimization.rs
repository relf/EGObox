use crate::errors::{GpError, Result};
use egobox_doe::{Lhs, SamplingMethod};
use linfa_pls::PlsRegression;
use ndarray::{ArrayBase, Data, Ix2, arr1, s};
use ndarray_rand::rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256Plus;

use ndarray::{Array, Array1, Array2, Zip};

use linfa::prelude::{Dataset, Fit, Float};

use log::debug;
use rayon::prelude::*;
use std::time::Instant;

/// Check that the dimension reduction `kpls_dim`, if any, is not greater
/// than the training input dimension `nx`
pub(crate) fn check_kpls_dim(kpls_dim: Option<&usize>, nx: usize) -> Result<()> {
    if let Some(d) = kpls_dim
        && *d > nx
    {
        return Err(GpError::InvalidValueError(format!(
            "Dimension reduction {d} should be smaller than actual                     training input dimensions {nx}"
        )));
    }
    Ok(())
}

/// Weights of the training inputs: PLS rotations reducing their dimension to `kpls_dim`
/// when specified (zeros if PLS fails on constant residuals), identity otherwise
pub(crate) fn pls_rotation<F: Float>(
    x: &ArrayBase<impl Data<Elem = F>, Ix2>,
    y: &Array2<F>,
    kpls_dim: Option<&usize>,
) -> Result<Array2<F>> {
    let Some(n_components) = kpls_dim else {
        return Ok(Array2::eye(x.ncols()));
    };
    let ds = Dataset::new(x.to_owned(), y.to_owned());
    let w_star = PlsRegression::params(*n_components).fit(&ds).map_or_else(
        |e| match e {
            linfa_pls::PlsError::PowerMethodConstantResidualError() => {
                Ok(Array2::zeros((x.ncols(), *n_components)))
            }
            err => Err(err),
        },
        |v| Ok(v.rotations().0.to_owned()),
    )?;
    Ok(w_star)
}

/// Initial guess of theta of dimension `dim` given either one value for all components
/// or one value per component
pub(crate) fn broadcast_theta0<F: Float>(init: &Array1<F>, dim: usize) -> Array1<F> {
    let theta0_dim = init.len();
    if theta0_dim == 1 {
        Array1::from_elem(dim, init[0])
    } else if theta0_dim == dim {
        init.to_owned()
    } else {
        panic!(
            "Initial guess for theta should be either 1-dim or dim of xtrain (w_star.ncols()), got {theta0_dim}"
        )
    }
}

/// Bounds of theta of dimension `dim` given either one bounds for all components
/// or one bounds per component
pub(crate) fn broadcast_bounds<F: Float>(bounds: &Array1<(F, F)>, dim: usize) -> Vec<(F, F)> {
    let bounds_dim = bounds.len();
    if bounds_dim == 1 {
        vec![bounds[0]; dim]
    } else if bounds_dim == dim {
        bounds.to_vec()
    } else {
        panic!("Bounds for theta should be either 1-dim or dim of xtrain ({dim}), got {bounds_dim}")
    }
}

#[derive(Clone, Copy)]
pub(crate) struct CobylaParams {
    pub rhobeg: f64,
    pub ftol_rel: f64,
    pub maxeval: usize,
    #[cfg(not(feature = "c-cobyla"))]
    pub bounded_evaluations: bool,
}

impl Default for CobylaParams {
    fn default() -> Self {
        CobylaParams {
            rhobeg: 0.5,
            ftol_rel: 1e-4,
            maxeval: 200,
            #[cfg(not(feature = "c-cobyla"))]
            bounded_evaluations: false,
        }
    }
}

pub(crate) fn prepare_multistart<F: Float>(
    n_start: usize,
    theta0: &Array1<F>,
    bounds: &[(F, F)],
) -> (Array2<F>, Vec<(F, F)>) {
    // Use log10 theta as optimization parameter
    let bounds: Vec<(F, F)> = bounds
        .iter()
        .map(|(lo, up)| (lo.log10(), up.log10()))
        .collect();

    // Multistart: user/default defined theta0 + values on log10 scale
    let mut theta0s = Array2::zeros((n_start + 1, theta0.len()));
    theta0s.row_mut(0).assign(&theta0.mapv(|v| F::log10(v)));

    match n_start.cmp(&1) {
        std::cmp::Ordering::Equal => {
            //let mut rng = Xoshiro256Plus::seed_from_u64(42);
            let mut rng = Xoshiro256Plus::from_entropy();
            let vals = bounds.iter().map(|(a, b)| rng.gen_range(*a..*b)).collect();
            theta0s.row_mut(1).assign(&Array::from_vec(vals))
        }
        std::cmp::Ordering::Greater => {
            let mut xlimits: Array2<F> = Array2::zeros((bounds.len(), 2));
            // for mut row in xlimits.rows_mut() {
            //     row.assign(&arr1(&[limits.0, limits.1]));
            // }
            Zip::from(xlimits.rows_mut())
                .and(&bounds)
                .for_each(|mut row, limits| row.assign(&arr1(&[limits.0, limits.1])));
            // Use a seed here for reproducibility. Do we need to make it truly random
            // Probably no, as it is just to get init values spread over
            // [lower bound, upper bound] for multistart thanks to LHS method.

            let seeds = Lhs::new(&xlimits)
                .kind(egobox_doe::LhsKind::Maximin)
                .with_rng(Xoshiro256Plus::seed_from_u64(42))
                .sample(n_start);
            Zip::from(theta0s.slice_mut(s![1.., ..]).rows_mut())
                .and(seeds.rows())
                .par_for_each(|mut theta, row| theta.assign(&row));
        }
        std::cmp::Ordering::Less => (),
    };
    (theta0s, bounds)
}

/// Minimize `objfn` with COBYLA started in parallel from each row of `inits` (log10 scale)
/// within `bounds`, returns the best value and parameters, (inf, ones) when all fail
pub(crate) fn multistart_minimize<ObjF, F>(
    objfn: ObjF,
    inits: &Array2<F>,
    bounds: &[(F, F)],
    cobyla: CobylaParams,
) -> (f64, Array1<f64>)
where
    ObjF: Fn(&[f64], Option<&mut [f64]>, &mut ()) -> f64 + Sync,
    F: Float,
{
    debug!("Optimize with multistart params = {inits:?} and bounds = {bounds:?}");
    let now = Instant::now();
    let opt = (0..inits.nrows())
        .into_par_iter()
        .map(|i| optimize_params(&objfn, &inits.row(i).to_owned(), bounds, cobyla))
        .reduce(
            || (f64::INFINITY, Array::ones((inits.ncols(),))),
            |a, b| if b.0 < a.0 { b } else { a },
        );
    debug!("elapsed optim = {:?}", now.elapsed().as_millis());
    opt
}

/// Optimize gp hyper parameters given an initial guess and bounds with cobyla
#[cfg(feature = "c-cobyla")]
pub(crate) fn optimize_params<ObjF, F>(
    objfn: ObjF,
    param0: &Array1<F>,
    bounds: &[(F, F)],
    cobyla: CobylaParams,
) -> (f64, Array1<f64>)
where
    ObjF: Fn(&[f64], Option<&mut [f64]>, &mut ()) -> f64,
    F: Float,
{
    use cobyla::{Func, StopTols, minimize};

    let cons: Vec<&dyn Func<()>> = vec![];
    let param0 = param0.map(|v| into_f64(v)).into_raw_vec_and_offset().0;

    let bounds: Vec<_> = bounds
        .iter()
        .map(|(lo, up)| (into_f64(lo), into_f64(up)))
        .collect();

    match minimize(
        |x, u| objfn(x, None, u),
        &param0,
        &bounds,
        &cons,
        (),
        cobyla.maxeval,
        cobyla::RhoBeg::All(cobyla.rhobeg),
        Some(StopTols {
            ftol_rel: cobyla.ftol_rel,
            ..StopTols::default()
        }),
    ) {
        Ok((_, x_opt, fval)) => {
            let params_opt = arr1(&x_opt);
            let fval = if f64::is_nan(fval) {
                f64::INFINITY
            } else {
                fval
            };
            (fval, params_opt)
        }
        Err((status, x_opt, _)) => {
            log::warn!("ERROR Cobyla optimizer in GP status={status:?}");
            (f64::INFINITY, arr1(&x_opt))
        }
    }
}

/// Conversion of a float value to f64, used by the f64 optimizers
#[inline(always)]
pub(crate) fn into_f64<F: Float>(v: &F) -> f64 {
    num_traits::ToPrimitive::to_f64(v).expect("float convertible to f64")
}

/// Optimize log10 GP hyperparameters using Basin's bounded COBYLA adapter.
#[cfg(not(feature = "c-cobyla"))]
pub(crate) fn optimize_params<ObjF, F>(
    objfn: ObjF,
    param0: &Array1<F>,
    bounds: &[(F, F)],
    cobyla: CobylaParams,
) -> (f64, Array1<f64>)
where
    ObjF: Fn(&[f64], Option<&mut [f64]>, &mut ()) -> f64,
    F: Float,
{
    use crate::basin_optimizer::{Algorithm, Settings, minimize};
    let xinit: Vec<_> = param0.iter().map(into_f64).collect();
    let bounds: Vec<_> = bounds
        .iter()
        .map(|(lo, hi)| (into_f64(lo), into_f64(hi)))
        .collect();
    let (value, params) = minimize(
        Algorithm::Cobyla,
        &|x, g| objfn(x, g, &mut ()),
        &|_, _, _| unreachable!("GP fitting has no nonlinear constraints"),
        Settings {
            xinit: &xinit,
            bounds: &bounds,
            n_constraints: 0,
            max_evals: cobyla.maxeval,
            ftol_abs: 0.,
            ftol_rel: cobyla.ftol_rel,
            initial_radius: cobyla.rhobeg,
            bounded_evaluations: cobyla.bounded_evaluations,
        },
    );
    (value, arr1(&params))
}
