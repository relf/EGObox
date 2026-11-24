//! Dense linear algebra used by the GP model computed either with the pure Rust
//! `linfa-linalg` backend (default) or with LAPACK through `ndarray-linalg`
//! (`blas` feature). This module is the only place in the workspace where both backends
//! differ: `egobox-moe` and `egobox-ego` only forward the `blas` feature to `egobox-gp`.

use crate::errors::Result;
use linfa::Float;
use ndarray::{Array1, Array2, ArrayBase, Data, Ix2};

#[cfg(not(feature = "blas"))]
use linfa_linalg::{cholesky::*, eigh::*, qr::*, svd::*, triangular::*};

#[cfg(feature = "blas")]
use linfa::dataset::{WithLapack, WithoutLapack};
#[cfg(feature = "blas")]
use ndarray_linalg::{cholesky::*, eigh::*, qr::*, svd::*, triangular::*};

/// Which triangular part of a matrix is used
#[cfg(not(feature = "blas"))]
pub(crate) use linfa_linalg::triangular::UPLO;
/// Which triangular part of a matrix is used
#[cfg(feature = "blas")]
pub(crate) use ndarray_linalg::UPLO;

/// Lower triangular Cholesky factor `l` of a symmetric positive definite matrix `a = l . l^T`
#[cfg(not(feature = "blas"))]
pub(crate) fn cholesky<F: Float>(a: Array2<F>) -> Result<Array2<F>> {
    Ok(a.cholesky_into()?)
}

/// See non blas version
#[cfg(feature = "blas")]
pub(crate) fn cholesky<F: Float>(a: Array2<F>) -> Result<Array2<F>> {
    Ok(a.with_lapack().cholesky_into(UPLO::Lower)?.without_lapack())
}

/// Solve `a . x = b` where `a` is a triangular matrix whose non-zero part is given by `uplo`
#[cfg(not(feature = "blas"))]
pub(crate) fn solve_triangular<F: Float>(
    a: &ArrayBase<impl Data<Elem = F>, Ix2>,
    b: Array2<F>,
    uplo: UPLO,
) -> Result<Array2<F>> {
    Ok(a.solve_triangular_into(b, uplo)?)
}

/// See non blas version
#[cfg(feature = "blas")]
pub(crate) fn solve_triangular<F: Float>(
    a: &ArrayBase<impl Data<Elem = F>, Ix2>,
    b: Array2<F>,
    uplo: UPLO,
) -> Result<Array2<F>> {
    Ok(a.view()
        .with_lapack()
        .solve_triangular_into(uplo, Diag::NonUnit, b.with_lapack())?
        .without_lapack())
}

/// Solve `a . x = b` where `a` is a triangular factor computed at training time
/// whose non-zero part is given by `uplo`
pub(crate) fn tri_solve<F: Float>(
    a: &ArrayBase<impl Data<Elem = F>, Ix2>,
    b: Array2<F>,
    uplo: UPLO,
) -> Array2<F> {
    solve_triangular(a, b, uplo).unwrap()
}

/// Inverse of a symmetric positive definite matrix `a` given its lower triangular
/// Cholesky factor `l` (`a = l . l^T`): `a^-1 = l^-T . l^-1`
#[cfg(not(feature = "blas"))]
pub(crate) fn cholesky_inverse<F: Float>(
    l: &ArrayBase<impl Data<Elem = F>, Ix2>,
) -> Result<Array2<F>> {
    let l_inv = lower_triangular_inverse(l);
    Ok(l_inv.t().dot(&l_inv))
}

/// See non blas version
#[cfg(feature = "blas")]
pub(crate) fn cholesky_inverse<F: Float>(
    l: &ArrayBase<impl Data<Elem = F>, Ix2>,
) -> Result<Array2<F>> {
    let factor = CholeskyFactorized {
        factor: l.to_owned().with_lapack(),
        uplo: UPLO::Lower,
    };
    Ok(factor.invc_into()?.without_lapack())
}

/// Inverse of a lower triangular matrix `l` with a non-zero diagonal, by blocks
/// `l = [[a, 0], [b, c]]`, `l^-1 = [[a^-1, 0], [-c^-1 . b . a^-1, c^-1]]`:
/// most of the work is done by matrix products.
#[cfg(not(feature = "blas"))]
fn lower_triangular_inverse<F: Float>(l: &ArrayBase<impl Data<Elem = F>, Ix2>) -> Array2<F> {
    use ndarray::linalg::general_mat_mul;
    use ndarray::s;
    const BLOCK: usize = 32;
    let n = l.nrows();
    if n <= BLOCK {
        return lower_triangular_inverse_by_rows(l);
    }
    let k = n / 2;
    let a_inv = lower_triangular_inverse(&l.slice(s![..k, ..k]));
    let c_inv = lower_triangular_inverse(&l.slice(s![k.., k..]));
    let mut x = Array2::<F>::zeros((n, n));
    let b_a_inv = l.slice(s![k.., ..k]).dot(&a_inv);
    general_mat_mul(
        -F::one(),
        &c_inv,
        &b_a_inv,
        F::zero(),
        &mut x.slice_mut(s![k.., ..k]),
    );
    x.slice_mut(s![..k, ..k]).assign(&a_inv);
    x.slice_mut(s![k.., k..]).assign(&c_inv);
    x
}

/// Inverse `x` of a lower triangular matrix `l` with a non-zero diagonal, computed row by row
/// as `x_i = (e_i - sum_{k<i} l_ik x_k) / l_ii` where the rows `x_k` of the lower triangular
/// inverse are non-zero up to column k: contiguous updates skipping the upper triangle zeros.
#[cfg(not(feature = "blas"))]
fn lower_triangular_inverse_by_rows<F: Float>(
    l: &ArrayBase<impl Data<Elem = F>, Ix2>,
) -> Array2<F> {
    use ndarray::{Axis, s};
    let n = l.nrows();
    let mut x = Array2::<F>::zeros((n, n));
    for i in 0..n {
        let (done, mut rest) = x.view_mut().split_at(Axis(0), i);
        let mut x_i = rest.row_mut(0);
        for k in 0..i {
            let l_ik = l[[i, k]];
            if l_ik != F::zero() {
                x_i.slice_mut(s![..=k])
                    .scaled_add(-l_ik, &done.row(k).slice(s![..=k]));
            }
        }
        x_i[i] = F::one();
        let inv_l_ii = F::one() / l[[i, i]];
        x_i.slice_mut(s![..=i]).mapv_inplace(|v| v * inv_l_ii);
    }
    x
}

/// Reduced QR decomposition `(q, r)` of a matrix `a = q . r`
#[cfg(not(feature = "blas"))]
pub(crate) fn qr<F: Float>(
    a: &ArrayBase<impl Data<Elem = F>, Ix2>,
) -> Result<(Array2<F>, Array2<F>)> {
    Ok(a.qr()?.into_decomp())
}

/// See non blas version
#[cfg(feature = "blas")]
pub(crate) fn qr<F: Float>(
    a: &ArrayBase<impl Data<Elem = F>, Ix2>,
) -> Result<(Array2<F>, Array2<F>)> {
    // LAPACK accepts wide matrices but the triangular solves on such R factor panic:
    // fail as the default backend does
    if a.nrows() < a.ncols() {
        return Err(linfa_linalg::LinalgError::NotThin {
            rows: a.nrows(),
            cols: a.ncols(),
        }
        .into());
    }
    let (q, r) = a.view().with_lapack().qr()?;
    Ok((q.without_lapack(), r.without_lapack()))
}

/// Singular values of a matrix
#[cfg(not(feature = "blas"))]
pub(crate) fn singular_values<F: Float>(
    a: &ArrayBase<impl Data<Elem = F>, Ix2>,
) -> Result<Array1<F>> {
    let (_, sv, _) = a.svd(false, false)?;
    Ok(sv)
}

/// See non blas version
#[cfg(feature = "blas")]
pub(crate) fn singular_values<F: Float>(
    a: &ArrayBase<impl Data<Elem = F>, Ix2>,
) -> Result<Array1<F>> {
    let (_, sv, _) = a.view().with_lapack().svd(false, false)?;
    Ok(sv.mapv(F::cast))
}

/// Eigenvalues and eigenvectors (as columns) of a symmetric matrix
#[cfg(not(feature = "blas"))]
pub(crate) fn eigh<F: Float>(a: Array2<F>) -> Result<(Array1<F>, Array2<F>)> {
    Ok(a.eigh_into()?)
}

/// See non blas version
#[cfg(feature = "blas")]
pub(crate) fn eigh<F: Float>(a: Array2<F>) -> Result<(Array1<F>, Array2<F>)> {
    let (vals, vecs) = a.with_lapack().eigh_into(UPLO::Lower)?;
    Ok((vals.mapv(F::cast), vecs.without_lapack()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray_rand::RandomExt;
    use ndarray_rand::rand::SeedableRng;
    use ndarray_rand::rand_distr::Uniform;
    use rand_xoshiro::Xoshiro256Plus;

    #[test]
    fn test_cholesky_inverse() {
        let mut rng = Xoshiro256Plus::seed_from_u64(42);
        // sizes below, at and above the block size of the triangular inverse recursion
        for n in [1, 2, 7, 32, 33, 77, 150] {
            let m = Array2::random_using((n, n), Uniform::new(-1., 1.), &mut rng);
            let a = m.t().dot(&m) + Array2::<f64>::eye(n);
            let a_inv = cholesky_inverse(&cholesky(a.clone()).unwrap()).unwrap();
            assert_abs_diff_eq!(a_inv.dot(&a), Array2::eye(n), epsilon = 1e-8);
            assert_abs_diff_eq!(a_inv, a_inv.t(), epsilon = 1e-10);
        }
    }
}
