//! Linear algebra helpers

use ndarray::Array2;

/// Lower Cholesky factor `L` of a symmetric positive semi-definite matrix `a` (`L L^T = a`):
/// degenerated directions (null pivots, e.g. perfectly correlated points) get null columns
pub(crate) fn cholesky_psd(a: &Array2<f64>) -> Array2<f64> {
    let n = a.nrows();
    let max_diag = a.diag().iter().fold(0., |m: f64, v| m.max(*v));
    let tol = 1e-10 * max_diag;
    let mut l = Array2::zeros((n, n));
    for j in 0..n {
        let pivot = a[[j, j]] - (0..j).map(|k| l[[j, k]] * l[[j, k]]).sum::<f64>();
        if pivot <= tol {
            continue;
        }
        let ljj = pivot.sqrt();
        l[[j, j]] = ljj;
        for i in j + 1..n {
            l[[i, j]] = (a[[i, j]] - (0..j).map(|k| l[[i, k]] * l[[j, k]]).sum::<f64>()) / ljj;
        }
    }
    l
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_cholesky_psd() {
        let a = array![[4., 2., 2.], [2., 2., 1.], [2., 1., 1.]];
        let l = cholesky_psd(&a);
        assert_abs_diff_eq!(l.dot(&l.t()), a, epsilon = 1e-12);
        // rank deficient: duplicated point
        let a = array![[2., 1., 2.], [1., 3., 1.], [2., 1., 2.]];
        let l = cholesky_psd(&a);
        assert_abs_diff_eq!(l.dot(&l.t()), a, epsilon = 1e-12);
    }
}
