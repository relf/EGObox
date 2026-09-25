use egobox_doe::{Lhs, SamplingMethod};
use ndarray::{arr1, s};
use ndarray_rand::rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256Plus;

use ndarray::{Array, Array1, Array2, Zip};

use linfa::prelude::Float;

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

#[inline(always)]
fn into_f64<F: Float>(v: &F) -> f64 {
    unsafe { *(v as *const F as *const f64) }
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
