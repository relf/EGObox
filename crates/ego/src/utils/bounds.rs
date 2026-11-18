use egobox_gp::ThetaTuning;
use ndarray::Array1;

/// Theta bounds used by Egor to fit GPs: the bounds of the given `tuning`, either a
/// single pair for all components or one pair per component, or
/// [`ThetaTuning::DEFAULT_BOUNDS`] when the tuning has none.
///
/// The bounds do not depend on the dimension of the problem: dimension-dependent
/// default bounds did not improve optimization results compared to the default ones,
/// see <https://github.com/relf/EGObox/pull/560> and <https://github.com/relf/EGObox/pull/562>.
pub fn theta_bounds(tuning: &ThetaTuning<f64>) -> Array1<(f64, f64)> {
    tuning
        .bounds()
        .cloned()
        .unwrap_or_else(|| Array1::from_elem(1, ThetaTuning::<f64>::DEFAULT_BOUNDS))
}

/// Clamp the initial theta `init` within `bounds`. Each of them is given either as a
/// single value for all components or as one value per component.
pub fn clamp_theta_init(init: &Array1<f64>, bounds: &Array1<(f64, f64)>) -> Array1<f64> {
    let n = init.len().max(bounds.len());
    assert!(
        (init.len() == 1 || init.len() == n) && (bounds.len() == 1 || bounds.len() == n),
        "theta init (size {}) and bounds (size {}) are not compatible",
        init.len(),
        bounds.len()
    );
    Array1::from_shape_fn(n, |i| {
        let v = init[i.min(init.len() - 1)];
        let (lo, up) = bounds[i.min(bounds.len() - 1)];
        v.clamp(lo, up)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_theta_bounds() {
        let default = array![ThetaTuning::<f64>::DEFAULT_BOUNDS];
        assert_eq!(theta_bounds(&ThetaTuning::default()), default);
        assert_eq!(
            theta_bounds(&ThetaTuning::Fixed(array![1.])),
            default,
            "no bounds"
        );
        let given = array![(0.1, 5.)];
        let tuning = ThetaTuning::Full {
            init: array![1.],
            bounds: given.clone(),
        };
        assert_eq!(theta_bounds(&tuning), given);
        let given = array![(0.1, 5.), (1., 20.)];
        let tuning = ThetaTuning::Partial {
            init: array![1., 1.],
            bounds: given.clone(),
            active: vec![0],
        };
        assert_eq!(theta_bounds(&tuning), given);
    }

    #[test]
    fn test_clamp_theta_init() {
        let init = array![0.01, 0.5, 20.];
        assert_eq!(
            clamp_theta_init(&init, &array![(0.1, 10.)]),
            array![0.1, 0.5, 10.]
        );
        assert_eq!(
            clamp_theta_init(&init, &array![(0.1, 1.), (0.6, 1.), (1., 30.)]),
            array![0.1, 0.6, 20.]
        );
        assert_eq!(
            clamp_theta_init(&array![0.1], &array![(0.2, 1.), (0.01, 1.)]),
            array![0.2, 0.1]
        );
        assert_eq!(
            clamp_theta_init(&array![0.1], &array![(0.2, 1.)]),
            array![0.2]
        );
    }
}
