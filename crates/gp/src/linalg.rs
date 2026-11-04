//! Dense linear algebra used by the GP model computed either with the pure Rust
//! `linfa-linalg` backend (default) or with LAPACK through `ndarray-linalg`
//! (`blas` feature). This module is the only place where both backends differ.

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
