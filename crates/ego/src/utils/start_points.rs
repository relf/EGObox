use ndarray::{Array1, Array2, ArrayBase, Axis, Data, Ix1, Ix2, stack};
use std::cmp::Ordering;

/// Determine starting points as midpoints between training points that are
/// far enough from other training points.
///
pub fn start_points(
    x: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    xl: &ArrayBase<impl Data<Elem = f64>, Ix1>,
    xu: &ArrayBase<impl Data<Elem = f64>, Ix1>,
    n_max: Option<usize>,
) -> Array2<f64> {
    let n = x.nrows();
    let d = x.ncols();
    let xrange = xu - xl;

    // Collect all (i, j, distance) triplets
    let mut pairs = Vec::new();
    for i in 1..n {
        for j in 0..i {
            let diff = &x.row(i) - &x.row(j);
            let scaled = &diff / &xrange;
            let dist = scaled.dot(&scaled).sqrt();
            pairs.push((i, j, dist));
        }
    }

    // Sort pairs by distance
    pairs.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(Ordering::Equal));

    // Determine good midpoints
    let mut xstart: Vec<Array1<f64>> = Vec::new();

    for (i, j, _) in pairs {
        // Consider xij middle of [xi, xj] to be a starting point
        let xi = x.row(i);
        let xj = x.row(j);
        let xij = (xi.to_owned() + xj) / 2.0;

        // Distance from xi (and xj)
        let d_ij = ((xi.to_owned() - &xij) / &xrange)
            .mapv(|v| v.powi(2))
            .sum()
            .sqrt();

        // xij good to be added
        let mut good = true;

        // Discard xij if there is a training point xk closer than xi and xj from it
        for k in 0..n {
            if k != i && k != j {
                let xk = x.row(k);
                let d_k = ((&xk - &xij) / &xrange).mapv(|v| v.powi(2)).sum().sqrt();
                if d_k < d_ij {
                    good = false;
                    break;
                }
            }
        }

        // Discard xij if there is already a starting point xk closer than xi and xj from it
        for xk in &xstart {
            let d_k = ((xk.to_owned() - &xij) / &xrange)
                .mapv(|v| v.powi(2))
                .sum()
                .sqrt();
            if d_k < d_ij {
                good = false;
                break;
            }
        }

        if good {
            // ok picked
            xstart.push(xij);
        }
        if let Some(max) = n_max
            && xstart.len() >= max
        {
            // Reached maximum number of starting points
            break;
        }
    }

    if xstart.is_empty() {
        Array2::<f64>::zeros((0, d))
    } else {
        let vstarts = xstart.iter().map(|p| p.view()).collect::<Vec<_>>();
        stack(Axis(0), &vstarts).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_midpoint_between_two_distant_points() {
        let x = array![[0.1, 0.2], [0.9, 0.8]];
        let xl = array![0.0, 0.0];
        let xu = array![1.0, 1.0];

        let result = start_points(&x, &xl, &xu, None);
        assert_eq!(result.nrows(), 1);
        assert_eq!(result.shape(), &[1, 2]);

        let expected = array![[0.5, 0.5]];
        assert!(result.abs_diff_eq(&expected, 1e-12));
    }

    #[test]
    fn test_multiple_points() {
        let x = array![[0.1, 0.2], [0.9, 0.2], [0.5, 0.8]];
        let xl = array![0.0, 0.0];
        let xu = array![1.0, 1.0];

        let result = start_points(&x, &xl, &xu, None);
        println!("result={result}");
        assert!(result.nrows() >= 1);
    }

    #[test]
    fn test_n_start_with_max() {
        let x = array![[0.1, 0.2], [0.9, 0.2], [0.5, 0.8]];
        let xl = array![0.0, 0.0];
        let xu = array![1.0, 1.0];

        let result = start_points(&x, &xl, &xu, Some(1));
        assert_eq!(result.nrows(), 1);

        let result = start_points(&x, &xl, &xu, Some(2));
        assert_eq!(result.nrows(), 2);
    }
}
