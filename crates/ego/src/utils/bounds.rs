use egobox_gp::ThetaTuning;
use ndarray::Array1;

pub fn theta_bounds(tuning: &ThetaTuning<f64>, dim: usize) -> Array1<(f64, f64)> {
    if let Some(bounds) = tuning.bounds() {
        if bounds.len() == 1 {
            // If bounds is a single value, use it for all dimensions

            let b = if bounds[0] == ThetaTuning::<f64>::DEFAULT_BOUNDS {
                // Use bounds which depends on dim instead of constant default
                special_bounds(dim)
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
        // Use bounds which depends on dim instead of constant default
        Array1::from_elem(1, special_bounds(dim))
    }
}

/// Theta bounds depending on the dimension.
///
/// Below 10 dimensions, the default bounds are used. Otherwise, the lengthscale
/// bounds are derived from the typical distances between points of a uniform design
/// in [0, 1]^dim, following eq. (16) of Appriou, Rullière, Gaudrie, "Combination of
/// Optimization-free Kriging Models for High-Dimensional Problems" (Computational
/// Statistics, 2024), <https://hal.science/hal-03812073v2/document>.
///
/// The default bounds play the role of the kernel influence factors of the paper.
/// The per-kernel factors (`CorrelationModel::theta_influence_factors`) are not used:
/// benchmarks gave mixed results (better on Rosenbrock, worse on Michalewicz),
/// see <https://github.com/relf/EGObox/pull/560>.
fn special_bounds(dim: usize) -> (f64, f64) {
    let (factor_inf, factor_sup) = ThetaTuning::<f64>::DEFAULT_BOUNDS;
    if dim < 10 {
        return (factor_inf, factor_sup);
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
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_theta_bounds() {
        assert_eq!(special_bounds(9), ThetaTuning::<f64>::DEFAULT_BOUNDS);
        // 124: MOPTA08 dimension
        for dim in [1, 10, 20, 50, 100, 124, 200] {
            let bounds = special_bounds(dim);
            assert!(bounds.0 <= bounds.1, "{dim}D: {bounds:?}");
        }
        let (lo, hi) = special_bounds(10);
        assert_abs_diff_eq!(lo, 0.0588, epsilon = 1e-4);
        assert_abs_diff_eq!(hi, 150.01, epsilon = 1e-2);
        let (lo, hi) = special_bounds(20);
        assert_abs_diff_eq!(lo, 0.0444, epsilon = 1e-4);
        assert_abs_diff_eq!(hi, 78.94, epsilon = 1e-2);
    }
}
