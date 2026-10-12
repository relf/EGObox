//! Polynomial chaos expansions (PCE) in independent standard normal variables.
//!
//! A quantity `Y(Ξ)` is approximated by `Σ_j a_j H_j(Ξ)`, where the `H_j` are the
//! orthonormal multivariate Hermite polynomials of total degree at most `d`
//! (specification, section 4). Orthonormality gives the moments and the Sobol
//! sensitivity indices directly from the coefficients (sections 4 and 20).

use crate::errors::{EgmdoError, Result};
use linfa_linalg::qr::LeastSquaresQrInto;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use ndarray_rand::RandomExt;
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand_distr::StandardNormal;
use rand_xoshiro::Xoshiro256Plus;

/// Draws `n` samples of `dim` independent standard normal variables, as a `(n, dim)` matrix.
pub fn normal_samples(n: usize, dim: usize, seed: u64) -> Array2<f64> {
    let mut rng = Xoshiro256Plus::seed_from_u64(seed);
    Array2::random_using((n, dim), StandardNormal, &mut rng)
}

/// The orthonormal multivariate Hermite polynomials of total degree at most `degree`
/// in `dim` variables.
///
/// The multi-indices are ordered by total degree, then in reverse lexicographic order,
/// so that the first polynomial is the constant `H_1 = 1`.
#[derive(Clone, Debug)]
pub struct HermiteBasis {
    dim: usize,
    degree: usize,
    multi_indices: Vec<Vec<usize>>,
}

impl HermiteBasis {
    /// The basis of total degree `degree` in `dim` variables
    pub fn new(dim: usize, degree: usize) -> Self {
        let mut multi_indices = vec![];
        for total in 0..=degree {
            let mut current = vec![0; dim];
            graded(&mut multi_indices, &mut current, 0, total);
        }
        HermiteBasis {
            dim,
            degree,
            multi_indices,
        }
    }

    /// The number of variables
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// The maximum total degree
    pub fn degree(&self) -> usize {
        self.degree
    }

    /// The number of polynomials `P = (dim + degree)! / (dim! degree!)`
    pub fn size(&self) -> usize {
        self.multi_indices.len()
    }

    /// The multi-indices of the polynomials
    pub fn multi_indices(&self) -> &[Vec<usize>] {
        &self.multi_indices
    }

    /// Evaluates the polynomials at samples `xi` of shape `(n, dim)`, as a `(n, P)` matrix.
    pub fn eval(&self, xi: &ArrayView2<f64>) -> Array2<f64> {
        let mut psi = Array2::ones((xi.nrows(), self.size()));
        for (k, x) in xi.rows().into_iter().enumerate() {
            // He_m(x_q) / sqrt(m!) for every variable q and degree m
            let univariate: Vec<Vec<f64>> = x.iter().map(|&v| hermite(v, self.degree)).collect();
            for (j, alpha) in self.multi_indices.iter().enumerate() {
                psi[[k, j]] = alpha
                    .iter()
                    .enumerate()
                    .map(|(q, &m)| univariate[q][m])
                    .product();
            }
        }
        psi
    }
}

/// Recursively enumerates the multi-indices of total degree `remaining` over
/// variables `q..`, in reverse lexicographic order.
fn graded(out: &mut Vec<Vec<usize>>, current: &mut Vec<usize>, q: usize, remaining: usize) {
    if q == current.len() - 1 {
        current[q] = remaining;
        out.push(current.clone());
        current[q] = 0;
        return;
    }
    for m in (0..=remaining).rev() {
        current[q] = m;
        graded(out, current, q + 1, remaining - m);
    }
    current[q] = 0;
}

/// The normalized probabilists' Hermite polynomials `He_m(x) / sqrt(m!)` for `m = 0..=degree`
fn hermite(x: f64, degree: usize) -> Vec<f64> {
    let mut he = vec![1.; degree + 1];
    if degree >= 1 {
        he[1] = x;
    }
    for m in 1..degree {
        he[m + 1] = x * he[m] - m as f64 * he[m - 1];
    }
    let mut factorial = 1.;
    for (m, v) in he.iter_mut().enumerate().skip(1) {
        factorial *= m as f64;
        *v /= factorial.sqrt();
    }
    he
}

/// The PCEs of several quantities in the same basis
#[derive(Clone, Debug)]
pub struct Pce {
    basis: HermiteBasis,
    /// Coefficients, one column per quantity: `(P, n_quantities)`
    coefs: Array2<f64>,
}

impl Pce {
    /// PCEs with the given coefficients `(P, n_quantities)`
    pub fn new(basis: HermiteBasis, coefs: Array2<f64>) -> Result<Self> {
        if coefs.nrows() != basis.size() {
            return Err(EgmdoError::InvalidData(format!(
                "PCE coefficients must have {} rows, got {}",
                basis.size(),
                coefs.nrows()
            )));
        }
        Ok(Pce { basis, coefs })
    }

    /// Fits by least squares the PCEs of the quantities `values` `(n, n_quantities)`
    /// observed at samples `xi` `(n, dim)`. Requires `n > P`.
    pub fn fit(
        basis: &HermiteBasis,
        xi: &ArrayView2<f64>,
        values: &ArrayView2<f64>,
    ) -> Result<Self> {
        if xi.nrows() != values.nrows() || xi.ncols() != basis.dim() {
            return Err(EgmdoError::InvalidData(format!(
                "PCE regression expects (n, {}) samples and n values, got {:?} and {:?}",
                basis.dim(),
                xi.shape(),
                values.shape()
            )));
        }
        if xi.nrows() <= basis.size() {
            return Err(EgmdoError::InvalidData(format!(
                "PCE regression needs more than {} samples, got {}",
                basis.size(),
                xi.nrows()
            )));
        }
        let psi = basis.eval(xi);
        let coefs = psi
            .least_squares_into(values.to_owned())
            .map_err(|e| EgmdoError::InvalidData(format!("PCE regression failed: {e}")))?;
        Pce::new(basis.clone(), coefs)
    }

    /// The basis
    pub fn basis(&self) -> &HermiteBasis {
        &self.basis
    }

    /// The coefficients `(P, n_quantities)`
    pub fn coefs(&self) -> &Array2<f64> {
        &self.coefs
    }

    /// The number of quantities
    pub fn n_quantities(&self) -> usize {
        self.coefs.ncols()
    }

    /// Evaluates the PCEs at samples `xi` `(n, dim)`, as a `(n, n_quantities)` matrix
    pub fn eval(&self, xi: &ArrayView2<f64>) -> Array2<f64> {
        self.basis.eval(xi).dot(&self.coefs)
    }

    /// The means of the quantities
    pub fn mean(&self) -> Array1<f64> {
        self.coefs.row(0).to_owned()
    }

    /// The variances of the quantities
    pub fn variance(&self) -> Array1<f64> {
        self.coefs
            .slice(ndarray::s![1.., ..])
            .mapv(|a| a * a)
            .sum_axis(Axis(0))
    }

    /// The closed Sobol index of quantity `k` for the group of variables `vars`:
    /// `Var[E[Y | ξ_q, q ∈ vars]] / Var[Y]`, i.e. the share of the variance explained by
    /// the polynomials depending on these variables only (decision D28).
    /// For a single variable, it is the first-order Sobol index.
    /// Returns 0 when the variance is 0 up to round-off, i.e. negligible with respect to
    /// the square of the quantity (the regression of a constant leaves coefficients of
    /// the order of the machine precision).
    pub fn closed_sobol_index(&self, k: usize, vars: &[usize]) -> f64 {
        let coefs = self.coefs.column(k);
        let total = self.variance()[k];
        let magnitude = coefs.mapv(|a| a * a).sum();
        if total <= f64::EPSILON * magnitude || total <= f64::MIN_POSITIVE {
            return 0.;
        }
        let explained: f64 = self
            .basis
            .multi_indices
            .iter()
            .zip(coefs.iter())
            .skip(1)
            .filter(|(alpha, _)| {
                alpha
                    .iter()
                    .enumerate()
                    .all(|(q, &m)| m == 0 || vars.contains(&q))
            })
            .map(|(_, a)| a * a)
            .sum();
        explained / total
    }

    /// The coefficients of quantity `k`
    pub fn coefs_of(&self, k: usize) -> ArrayView1<'_, f64> {
        self.coefs.column(k)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_basis_size_and_order() {
        let basis = HermiteBasis::new(2, 3);
        assert_eq!(basis.size(), 10);
        assert_eq!(
            basis.multi_indices(),
            &[
                vec![0, 0],
                vec![1, 0],
                vec![0, 1],
                vec![2, 0],
                vec![1, 1],
                vec![0, 2],
                vec![3, 0],
                vec![2, 1],
                vec![1, 2],
                vec![0, 3]
            ]
        );
        assert_eq!(HermiteBasis::new(10, 3).size(), 286);
        assert_eq!(HermiteBasis::new(1, 4).size(), 5);
    }

    #[test]
    fn test_hermite_values() {
        let basis = HermiteBasis::new(1, 3);
        let psi = basis.eval(&array![[2.]].view());
        // He_0 = 1, He_1 = x, He_2 = x² - 1, He_3 = x³ - 3x, normalized by sqrt(m!)
        assert_abs_diff_eq!(
            psi,
            array![[1., 2., 3. / 2f64.sqrt(), 2. / 6f64.sqrt()]],
            epsilon = 1e-12
        );
    }

    #[test]
    fn test_orthonormality() {
        let basis = HermiteBasis::new(2, 3);
        let xi = normal_samples(200_000, 2, 0);
        let psi = basis.eval(&xi.view());
        let gram = psi.t().dot(&psi) / xi.nrows() as f64;
        assert_abs_diff_eq!(gram, Array2::eye(basis.size()), epsilon = 0.05);
    }

    #[test]
    fn test_exact_recovery_and_moments() {
        let basis = HermiteBasis::new(2, 3);
        let xi = normal_samples(50, 2, 1);
        // Y1 = 1 + ξ1 + 2 ξ2, Y2 = ξ1 ξ2 + (ξ2² - 1)
        let y = ndarray::stack![
            Axis(1),
            xi.column(0).mapv(|v| 1. + v) + &xi.column(1).mapv(|v| 2. * v),
            &xi.column(0) * &xi.column(1) + &xi.column(1).mapv(|v| v * v - 1.)
        ];
        let pce = Pce::fit(&basis, &xi.view(), &y.view()).unwrap();
        let expected = array![
            [1., 0.],
            [1., 0.],
            [2., 0.],
            [0., 0.],
            [0., 1.],
            [0., 2f64.sqrt()],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.]
        ];
        assert_abs_diff_eq!(*pce.coefs(), expected, epsilon = 1e-10);
        assert_abs_diff_eq!(pce.mean(), array![1., 0.], epsilon = 1e-10);
        assert_abs_diff_eq!(pce.variance(), array![5., 3.], epsilon = 1e-10);
        let other = normal_samples(5, 2, 2);
        let y1 = other.column(0).mapv(|v| 1. + v) + &other.column(1).mapv(|v| 2. * v);
        assert_abs_diff_eq!(pce.eval(&other.view()).column(0), y1, epsilon = 1e-10);
    }

    #[test]
    fn test_sobol_indices() {
        let basis = HermiteBasis::new(2, 3);
        let xi = normal_samples(50, 2, 3);
        // Y = ξ1 + 2 ξ2: first-order indices 0.2 and 0.8
        let y = (&xi.column(0) + &xi.column(1).mapv(|v| 2. * v)).insert_axis(Axis(1));
        let pce = Pce::fit(&basis, &xi.view(), &y.view()).unwrap();
        assert_abs_diff_eq!(pce.closed_sobol_index(0, &[0]), 0.2, epsilon = 1e-10);
        assert_abs_diff_eq!(pce.closed_sobol_index(0, &[1]), 0.8, epsilon = 1e-10);
        assert_abs_diff_eq!(pce.closed_sobol_index(0, &[0, 1]), 1., epsilon = 1e-10);

        // Y = ξ1 ξ2: no first-order effect, only the interaction
        let y = (&xi.column(0) * &xi.column(1)).insert_axis(Axis(1));
        let pce = Pce::fit(&basis, &xi.view(), &y.view()).unwrap();
        assert_abs_diff_eq!(pce.closed_sobol_index(0, &[0]), 0., epsilon = 1e-10);
        assert_abs_diff_eq!(pce.closed_sobol_index(0, &[0, 1]), 1., epsilon = 1e-10);

        // Constant: zero variance
        let y = Array2::from_elem((50, 1), 3.);
        let pce = Pce::fit(&basis, &xi.view(), &y.view()).unwrap();
        assert_eq!(pce.closed_sobol_index(0, &[0]), 0.);
    }

    #[test]
    fn test_invalid_regression() {
        let basis = HermiteBasis::new(2, 3);
        let xi = normal_samples(10, 2, 0);
        let y = Array2::zeros((10, 1));
        assert!(Pce::fit(&basis, &xi.view(), &y.view()).is_err(), "n = P");
        let xi = normal_samples(20, 3, 0);
        let y = Array2::zeros((20, 1));
        assert!(
            Pce::fit(&basis, &xi.view(), &y.view()).is_err(),
            "wrong dim"
        );
    }
}
