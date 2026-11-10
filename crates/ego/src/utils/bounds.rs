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

/// Theta bounds depending on the dimension and the kernels used.
///
/// Below 10 dimensions, the default bounds are used. Otherwise, the bounds of the
/// lengthscale are derived from the distribution of distances between points of a
/// uniform design in [0, 1]^dim and from the influence factors of the kernels,
/// see <https://hal.science/hal-03812073v2/document>, p. 9.
/// When several kernels are used, the widest factors are taken.
fn special_bounds(dim: usize, spec: CorrelationSpec) -> (f64, f64) {
    if dim < 10 {
        return ThetaTuning::<f64>::DEFAULT_BOUNDS;
    }
    let influence_factors: [(CorrelationSpec, (f64, f64)); 4] = [
        (
            CorrelationSpec::SQUAREDEXPONENTIAL,
            SquaredExponentialCorr::default().theta_influence_factors(),
        ),
        (
            CorrelationSpec::ABSOLUTEEXPONENTIAL,
            AbsoluteExponentialCorr::default().theta_influence_factors(),
        ),
        (
            CorrelationSpec::MATERN32,
            Matern32Corr::default().theta_influence_factors(),
        ),
        (
            CorrelationSpec::MATERN52,
            Matern52Corr::default().theta_influence_factors(),
        ),
    ];
    let (factor_inf, factor_sup) = influence_factors
        .iter()
        .filter(|(kernel, _)| spec.contains(*kernel))
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(inf, sup), (_, f)| {
            (inf.min(f.0), sup.max(f.1))
        });
    if factor_inf > factor_sup {
        // no kernel selected
        return ThetaTuning::<f64>::DEFAULT_BOUNDS;
    }

    let d = dim as f64;
    let s = 1. / f64::sqrt(12.); // uniform design on [0, 1]^d standard deviation
    let k = 9. / 5.; // uniform distribution kurtosis
    let interval = 1.96 * f64::sqrt(2. * (k + 1.) * d);
    let rmin = f64::sqrt(2. * d - interval);
    let rmax = f64::sqrt(2. * d + interval);
    let lmin = s * rmin * factor_inf;
    let lmax = s * rmax * factor_sup;
    (1. / lmax, 1. / lmin)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_theta_bounds() {
        let specs = [
            CorrelationSpec::SQUAREDEXPONENTIAL,
            CorrelationSpec::ABSOLUTEEXPONENTIAL,
            CorrelationSpec::MATERN32,
            CorrelationSpec::MATERN52,
        ];
        // 124: MOPTA08 dimension
        for dim in [1, 10, 20, 50, 100, 124, 200] {
            for spec in specs {
                let bounds = special_bounds(dim, spec);
                assert!(bounds.0 <= bounds.1, "{spec:?} in {dim}D: {bounds:?}");
            }
        }
    }
}
