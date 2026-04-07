use egobox_gp::{ThetaTuning, correlation_models::*};
use egobox_moe::CorrelationSpec;
use ndarray::Array1;

pub fn theta_bounds(
    tuning: &ThetaTuning<f64>,
    dim: usize,
    spec: CorrelationSpec,
) -> Array1<(f64, f64)> {
    if let Some(bounds) = tuning.bounds() {
        if bounds.len() == 1 {
            // If bounds is a single value, use it for all dimensions

            let b = if bounds[0] == ThetaTuning::<f64>::DEFAULT_BOUNDS {
                // Use bounds which depends on dim and kernel instead of constant default
                special_bounds(dim, spec)
            } else {
                // Use given default
                bounds[0]
            };
            Array1::from_elem(1, b)
        } else {
            // If bounds is a vector, use it as is
            bounds.clone()
        }
    } else {
        // Use bounds which depends on dim and kernel instead of constant default
        Array1::from_elem(1, special_bounds(dim, spec))
    }
}

/// Theta bounds
#[allow(dead_code)]
fn special_bounds(dim: usize, spec: CorrelationSpec) -> (f64, f64) {
    let (theta_inf, theta_sup) = (
        ThetaTuning::<f64>::DEFAULT_BOUNDS.0,
        ThetaTuning::<f64>::DEFAULT_BOUNDS.1,
    );
    let (theta_inf, theta_sup) = if spec.contains(CorrelationSpec::SQUAREDEXPONENTIAL) {
        let influence: (f64, f64) = SquaredExponentialCorr::default().theta_influence_factors();
        (theta_inf.min(influence.0), theta_sup.max(influence.1))
    } else {
        (theta_inf, theta_sup)
    };
    let (theta_inf, theta_sup) = if spec.contains(CorrelationSpec::ABSOLUTEEXPONENTIAL) {
        let influence: (f64, f64) = AbsoluteExponentialCorr::default().theta_influence_factors();
        (theta_inf.min(influence.0), theta_sup.max(influence.1))
    } else {
        (theta_inf, theta_sup)
    };
    let (theta_inf, theta_sup) = if spec.contains(CorrelationSpec::MATERN32) {
        let influence: (f64, f64) = Matern32Corr::default().theta_influence_factors();
        (theta_inf.min(influence.0), theta_sup.max(influence.1))
    } else {
        (theta_inf, theta_sup)
    };
    let (theta_inf, theta_sup) = if spec.contains(CorrelationSpec::MATERN32) {
        let influence: (f64, f64) = Matern32Corr::default().theta_influence_factors();
        (theta_inf.min(influence.0), theta_sup.max(influence.1))
    } else {
        (theta_inf, theta_sup)
    };

    if dim < 10 {
        (
            ThetaTuning::<f64>::DEFAULT_BOUNDS.0,
            ThetaTuning::<f64>::DEFAULT_BOUNDS.1,
        )
    } else {
        let d = dim as f64;
        let s = 1. / f64::sqrt(12.); // uniform design on [0, 1]^d standard deviation
        let k = 9. / 5.; // uniform distribution kurtosis
        let interval = 1.96 * f64::sqrt(2. * (k + 1.) * d);
        let rmin = f64::sqrt(2. * d - interval);
        let rmax = f64::sqrt(2. * d + interval);
        //dbg!("rmin = {}, rmax={}", rmin, rmax);
        let lmin = s * rmin * theta_inf;
        let lmax = s * rmax * theta_sup;
        //dbg!("lmin = {}, lmax={}", lmin, lmax);
        let tlo = 1. / lmax;
        let tup = 1. / lmin;
        //dbg!("tlo = {}, tup={}", tlo, tup);
        (tlo, tup)
    }
}

#[cfg(test)]
mod tests {
    use paste::paste;

    use super::*;

    #[allow(unused_macros)]
    macro_rules! test_theta_bounds {
        ($dim:expr_2021, $corr:ident ) => {
            paste! {

                #[test]
                fn [<test_theta_bounds_  $corr:snake _ $dim:snake>]() {
                    let bounds = special_bounds($dim, CorrelationSpec::[< $corr:upper >]);
                    println!("bounds({}) = {:?}", $dim, bounds);
                    assert!(bounds.0 <= bounds.1);
                }
            }
        };
    }

    test_theta_bounds!(1, SquaredExponential);
    test_theta_bounds!(10, SquaredExponential);
    test_theta_bounds!(20, SquaredExponential);
    test_theta_bounds!(50, SquaredExponential);
    test_theta_bounds!(100, SquaredExponential);
    test_theta_bounds!(200, SquaredExponential);
    test_theta_bounds!(1, AbsoluteExponential);
    test_theta_bounds!(10, AbsoluteExponential);
    test_theta_bounds!(20, AbsoluteExponential);
    test_theta_bounds!(50, AbsoluteExponential);
    test_theta_bounds!(100, AbsoluteExponential);
    test_theta_bounds!(200, AbsoluteExponential);
    test_theta_bounds!(1, Matern32);
    test_theta_bounds!(10, Matern32);
    test_theta_bounds!(20, Matern32);
    test_theta_bounds!(50, Matern32);
    test_theta_bounds!(100, Matern32);
    test_theta_bounds!(200, Matern32);
    test_theta_bounds!(1, Matern52);
    test_theta_bounds!(10, Matern52);
    test_theta_bounds!(20, Matern52);
    test_theta_bounds!(50, Matern52);
    test_theta_bounds!(100, Matern52);
    test_theta_bounds!(200, Matern52);

    // test for MOPTA08 124D
    test_theta_bounds!(124, SquaredExponential);
}
