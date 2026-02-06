//! A module for correlation models with PLS weighting to model the error term of the GP model.
//!
//! The following correlation models are implemented:
//! * squared exponential,
//! * absolute exponential,
//! * matern 3/2,
//! * matern 5/2.

use crate::utils::differences;
use linfa::Float;
use ndarray::{Array1, Array2, ArrayBase, Axis, Data, Ix1, Ix2, Zip};
use ndarray_einsum::einsum;
#[cfg(feature = "serializable")]
use serde::{Deserialize, Serialize};
use std::convert::TryFrom;
use std::fmt;

/// A trait for using a correlation model in GP regression
pub trait CorrelationModel<F: Float>: Clone + Copy + Default + fmt::Display + Sync {
    /// Compute correlation function matrix r(x, x') given x and a set of `x'` training samples, aka `xtrain`
    /// `theta` parameters, and PLS `weights` with:
    ///
    /// * `x`      : point at which to compute correlation (shape nx)
    /// * `xtrain` : training samples (shape nt x nx)
    ///   where nx is the dimension of x and nt is the number of training samples (aka xtrain.nrows()).
    /// * `theta`   : hyperparameters (shape 1 x nx)
    /// * `weights` : PLS weights (shape nx x h) where h is the reduced dimension when PLS is used (kpls_dim).
    ///
    /// The returned correlation function matrix has shape (nt x 1) and corresponds to r(x, xtrain)
    /// where r is the correlation function defined by the model.
    fn rval(
        &self,
        x: &ArrayBase<impl Data<Elem = F>, Ix1>,
        xtrain: &ArrayBase<impl Data<Elem = F>, Ix2>,
        theta: &ArrayBase<impl Data<Elem = F>, Ix1>,
        weights: &ArrayBase<impl Data<Elem = F>, Ix2>,
    ) -> Array2<F> {
        let d = differences(x, xtrain);
        self.rval_from_distances(&d, theta, weights)
    }

    /// Compute correlation function matrix r(x, x') given distances `distances` between x and x',
    /// `theta` parameters, and PLS `weights` with:
    ///
    /// * `distances`     : distances (nxd)
    /// * `theta`   : hyperparameters (1xd)
    /// * `weights` : PLS weights (dxh)
    ///   where d is the initial dimension and h (<d) is the reduced dimension when PLS is used (kpls_dim)
    ///
    /// The returned correlation function matrix has shape (nt x 1) and corresponds to r(x, xtrain)
    /// where r is the correlation function defined by the model.
    fn rval_from_distances(
        &self,
        distances: &ArrayBase<impl Data<Elem = F>, Ix2>,
        theta: &ArrayBase<impl Data<Elem = F>, Ix1>,
        weights: &ArrayBase<impl Data<Elem = F>, Ix2>,
    ) -> Array2<F>;

    /// Compute gradients of `r(x, x')` at given `x` given a set of `x`in`` training samples, aka `xtra
    /// `theta` parameters, and PLS `weights`.
    /// The returned jacobian matrix is dr/dx where r is the correlation function vector between x and xtrain (shape nt).
    /// Gradients are computed with respect to `x` and returned as a matrix of shape (nt, nx)
    /// where nt is the number of training samples (aka xtrain.nrows()) and nx is the dimension of x.
    fn jac(
        &self,
        x: &ArrayBase<impl Data<Elem = F>, Ix1>,
        xtrain: &ArrayBase<impl Data<Elem = F>, Ix2>,
        theta: &ArrayBase<impl Data<Elem = F>, Ix1>,
        weights: &ArrayBase<impl Data<Elem = F>, Ix2>,
    ) -> Array2<F>;

    /// Compute both the correlation function matrix `r(x, x')` and its jacobian at given `x`
    /// given a set of `xtrain` training samples, `theta` parameters, and PLS `weights`.
    /// Used to avoid redundant computations when both correlation and jacobian are needed.
    fn rval_with_jac(
        &self,
        x: &ArrayBase<impl Data<Elem = F>, Ix1>,
        xtrain: &ArrayBase<impl Data<Elem = F>, Ix2>,
        theta: &ArrayBase<impl Data<Elem = F>, Ix1>,
        weights: &ArrayBase<impl Data<Elem = F>, Ix2>,
    ) -> (Array2<F>, Array2<F>);

    /// Returns the theta influence factors for the correlation model.
    /// See <https://hal.science/hal-03812073v2/document>
    fn theta_influence_factors(&self) -> (F, F) {
        (F::one(), F::one())
    }
}

/// Squared exponential correlation models
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[cfg_attr(
    feature = "serializable",
    derive(Serialize, Deserialize),
    serde(into = "String"),
    serde(try_from = "String")
)]
pub struct SquaredExponentialCorr();

impl From<SquaredExponentialCorr> for String {
    fn from(_item: SquaredExponentialCorr) -> String {
        "SquaredExponential".to_string()
    }
}

impl TryFrom<String> for SquaredExponentialCorr {
    type Error = &'static str;
    fn try_from(s: String) -> Result<Self, Self::Error> {
        if s == "SquaredExponential" {
            Ok(Self::default())
        } else {
            Err("Bad string value for SquaredExponentialCorr, should be \'SquaredExponential\'")
        }
    }
}

impl<F: Float> CorrelationModel<F> for SquaredExponentialCorr {
    ///   d    h
    /// prod prod exp( - |theta_l * weight_j_l * d_j|^2 / 2 )
    ///  j=1  l=1
    fn rval_from_distances(
        &self,
        d: &ArrayBase<impl Data<Elem = F>, Ix2>,
        theta: &ArrayBase<impl Data<Elem = F>, Ix1>,
        weights: &ArrayBase<impl Data<Elem = F>, Ix2>,
    ) -> Array2<F> {
        let theta_w = (theta * weights)
            .mapv(|v| v.powf(F::cast(2.)))
            .sum_axis(Axis(1));
        let r = d.mapv(|v| v.powf(F::cast(2.))).dot(&theta_w);
        r.mapv(|v| F::exp(F::cast(-0.5) * v))
            .into_shape_with_order((d.nrows(), 1))
            .unwrap()
    }

    fn jac(
        &self,
        x: &ArrayBase<impl Data<Elem = F>, Ix1>,
        xtrain: &ArrayBase<impl Data<Elem = F>, Ix2>,
        theta: &ArrayBase<impl Data<Elem = F>, Ix1>,
        weights: &ArrayBase<impl Data<Elem = F>, Ix2>,
    ) -> Array2<F> {
        let d = differences(x, xtrain);
        let r = self.rval_from_distances(&d, theta, weights);

        let dtheta_w = (theta * weights)
            .mapv(|v| v.powf(F::cast(2)))
            .sum_axis(Axis(1))
            .mapv(|v| F::cast(-v));

        d * &dtheta_w * &r
    }

    fn rval_with_jac(
        &self,
        x: &ArrayBase<impl Data<Elem = F>, Ix1>,
        xtrain: &ArrayBase<impl Data<Elem = F>, Ix2>,
        theta: &ArrayBase<impl Data<Elem = F>, Ix1>,
        weights: &ArrayBase<impl Data<Elem = F>, Ix2>,
    ) -> (Array2<F>, Array2<F>) {
        let d = differences(x, xtrain);
        let r = self.rval_from_distances(&d, theta, weights);

        let dtheta_w = (theta * weights)
            .mapv(|v| v.powf(F::cast(2)))
            .sum_axis(Axis(1))
            .mapv(|v| F::cast(-v));

        let jr = d * &dtheta_w * &r;
        (r, jr)
    }

    fn theta_influence_factors(&self) -> (F, F) {
        (F::cast(0.29), F::cast(1.96))
    }
}

impl fmt::Display for SquaredExponentialCorr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "SquaredExponential")
    }
}

/// Absolute exponential correlation models
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(
    feature = "serializable",
    derive(Serialize, Deserialize),
    serde(into = "String"),
    serde(try_from = "String")
)]
pub struct AbsoluteExponentialCorr();

impl From<AbsoluteExponentialCorr> for String {
    fn from(_item: AbsoluteExponentialCorr) -> String {
        "AbsoluteExponential".to_string()
    }
}

impl TryFrom<String> for AbsoluteExponentialCorr {
    type Error = &'static str;
    fn try_from(s: String) -> Result<Self, Self::Error> {
        if s == "AbsoluteExponential" {
            Ok(Self::default())
        } else {
            Err("Bad string value for AbsoluteExponentialCorr, should be \'AbsoluteExponential\'")
        }
    }
}

impl<F: Float> CorrelationModel<F> for AbsoluteExponentialCorr {
    ///   d    h
    /// prod prod exp( - theta_l * |weight_j_l * d_j| )
    ///  j=1  l=1
    fn rval_from_distances(
        &self,
        d: &ArrayBase<impl Data<Elem = F>, Ix2>,
        theta: &ArrayBase<impl Data<Elem = F>, Ix1>,
        weights: &ArrayBase<impl Data<Elem = F>, Ix2>,
    ) -> Array2<F> {
        let theta_w = weights.mapv(|v| v.abs()).dot(theta);
        let r = d.mapv(|v| v.abs()).dot(&theta_w);
        r.mapv(|v| F::exp(-v))
            .into_shape_with_order((d.nrows(), 1))
            .unwrap()
    }

    fn jac(
        &self,
        x: &ArrayBase<impl Data<Elem = F>, Ix1>,
        xtrain: &ArrayBase<impl Data<Elem = F>, Ix2>,
        theta: &ArrayBase<impl Data<Elem = F>, Ix1>,
        weights: &ArrayBase<impl Data<Elem = F>, Ix2>,
    ) -> Array2<F> {
        let d = differences(x, xtrain);
        let r = self.rval_from_distances(&d, theta, weights);
        let sign_d = d.mapv(|v| v.signum());

        let dtheta_w = sign_d
            * (theta * weights.mapv(|v| v.abs()))
                .sum_axis(Axis(1))
                .mapv(|v| F::cast(-1.) * v);
        &dtheta_w * &r
    }

    fn rval_with_jac(
        &self,
        x: &ArrayBase<impl Data<Elem = F>, Ix1>,
        xtrain: &ArrayBase<impl Data<Elem = F>, Ix2>,
        theta: &ArrayBase<impl Data<Elem = F>, Ix1>,
        weights: &ArrayBase<impl Data<Elem = F>, Ix2>,
    ) -> (Array2<F>, Array2<F>) {
        let d = differences(x, xtrain);
        let r = self.rval_from_distances(&d, theta, weights);
        let sign_d = d.mapv(|v| v.signum());

        let dtheta_w = sign_d
            * (theta * weights.mapv(|v| v.abs()))
                .sum_axis(Axis(1))
                .mapv(|v| F::cast(-1.) * v);
        let jr = &dtheta_w * &r;
        (r, jr)
    }

    fn theta_influence_factors(&self) -> (F, F) {
        (F::cast(0.15), F::cast(3.76))
    }
}

impl fmt::Display for AbsoluteExponentialCorr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "AbsoluteExponential")
    }
}

/// Matern 3/2 correlation model
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(
    feature = "serializable",
    derive(Serialize, Deserialize),
    serde(into = "String"),
    serde(try_from = "String")
)]
pub struct Matern32Corr();

impl From<Matern32Corr> for String {
    fn from(_item: Matern32Corr) -> String {
        "Matern32".to_string()
    }
}

impl TryFrom<String> for Matern32Corr {
    type Error = &'static str;
    fn try_from(s: String) -> Result<Self, Self::Error> {
        if s == "Matern32" {
            Ok(Self::default())
        } else {
            Err("Bad string value for Matern32Corr, should be \'Matern32\'")
        }
    }
}

impl<F: Float> CorrelationModel<F> for Matern32Corr {
    ///   d    h         
    /// prod prod (1 + sqrt(3) * theta_l * |d_j . weight_j|) exp( - sqrt(3) * theta_l * |d_j . weight_j| )
    ///  j=1  l=1
    fn rval_from_distances(
        &self,
        d: &ArrayBase<impl Data<Elem = F>, Ix2>,
        theta: &ArrayBase<impl Data<Elem = F>, Ix1>,
        weights: &ArrayBase<impl Data<Elem = F>, Ix2>,
    ) -> Array2<F> {
        let (a, b) = self._compute_r_factors(d, theta, weights);
        let r = a * b;
        r.into_shape_with_order((d.nrows(), 1)).unwrap()
    }

    fn jac(
        &self,
        x: &ArrayBase<impl Data<Elem = F>, Ix1>,
        xtrain: &ArrayBase<impl Data<Elem = F>, Ix2>,
        theta: &ArrayBase<impl Data<Elem = F>, Ix1>,
        weights: &ArrayBase<impl Data<Elem = F>, Ix2>,
    ) -> Array2<F> {
        let d = differences(x, xtrain);
        let (a, b) = self._compute_r_factors(&d, theta, weights);
        self._jac_helper(&d, &a, &b, xtrain, theta, weights)
    }

    fn rval_with_jac(
        &self,
        x: &ArrayBase<impl Data<Elem = F>, Ix1>,
        xtrain: &ArrayBase<impl Data<Elem = F>, Ix2>,
        theta: &ArrayBase<impl Data<Elem = F>, Ix1>,
        weights: &ArrayBase<impl Data<Elem = F>, Ix2>,
    ) -> (Array2<F>, Array2<F>) {
        let d = differences(x, xtrain);
        let (a, b) = self._compute_r_factors(&d, theta, weights);
        let r = &a * &b;
        let jr = self._jac_helper(&d, &a, &b, xtrain, theta, weights);
        (r.into_shape_with_order((d.nrows(), 1)).unwrap(), jr)
    }

    fn theta_influence_factors(&self) -> (F, F) {
        (F::cast(0.21), F::cast(2.74))
    }
}

impl fmt::Display for Matern32Corr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Matern32")
    }
}

impl Matern32Corr {
    fn _compute_r_factors<F: Float>(
        &self,
        d: &ArrayBase<impl Data<Elem = F>, Ix2>,
        theta: &ArrayBase<impl Data<Elem = F>, Ix1>,
        weights: &ArrayBase<impl Data<Elem = F>, Ix2>,
    ) -> (Array1<F>, Array1<F>) {
        let sqrt3 = F::cast(3.).sqrt();
        let theta_w = theta * weights.mapv(|v| v.abs());

        let abs_d = d.mapv(|v| v.abs());
        let d_theta_w = abs_d.dot(&theta_w);

        let mut a = Array1::ones(d.nrows());
        Zip::from(&mut a)
            .and(abs_d.rows())
            .for_each(|a_i, abs_d_i| {
                Zip::from(abs_d_i)
                    .and(theta_w.rows())
                    .for_each(|abs_d_ij, theta_w_j| {
                        *a_i *= theta_w_j
                            .mapv(|v| F::one() + sqrt3 * v * *abs_d_ij)
                            .product();
                    });
            });

        let b = d_theta_w.sum_axis(Axis(1)).mapv(|v| F::exp(-sqrt3 * v));
        (a, b)
    }

    fn _jac_helper<F: Float>(
        &self,
        d: &ArrayBase<impl Data<Elem = F>, Ix2>,
        a: &ArrayBase<impl Data<Elem = F>, Ix1>,
        b: &ArrayBase<impl Data<Elem = F>, Ix1>,
        xtrain: &ArrayBase<impl Data<Elem = F>, Ix2>,
        theta: &ArrayBase<impl Data<Elem = F>, Ix1>,
        weights: &ArrayBase<impl Data<Elem = F>, Ix2>,
    ) -> Array2<F> {
        let sqrt3 = F::cast(3.).sqrt();
        let theta_w = weights.mapv(|v| v.abs()).dot(theta);
        let sign_d = d.mapv(|v| v.signum());
        let mut db = Array2::<F>::zeros((xtrain.nrows(), xtrain.ncols()));
        let abs_d = d.mapv(|v| v.abs());
        Zip::from(db.rows_mut())
            .and(a)
            .and(b)
            .and(sign_d.rows())
            .for_each(|mut db_i, ai, bi, si| {
                Zip::from(&mut db_i)
                    .and(&si)
                    .and(&theta_w)
                    .for_each(|db_ij, sij, theta_wj| {
                        *db_ij = -sqrt3 * *theta_wj * *sij * *bi * *ai;
                    });
            });
        let theta_w = theta * weights.mapv(|v| v.abs());
        let mut da = Array2::<F>::zeros((xtrain.nrows(), xtrain.ncols()));
        Zip::from(da.rows_mut())
            .and(abs_d.rows())
            .and(sign_d.rows())
            .for_each(|mut da_i, abs_d_i, sign_i| {
                Zip::indexed(&mut da_i)
                    .and(&sign_i)
                    .for_each(|j, da_ij, sign_ij| {
                        Zip::indexed(theta_w.columns()).for_each(|k, theta_w_k| {
                            let mut term = F::one();
                            let deriv = sqrt3 * theta_w_k[j] * *sign_ij;
                            Zip::indexed(theta_w.rows()).and(abs_d_i).for_each(
                                |p, theta_w_p, abs_d_ip| {
                                    Zip::indexed(theta_w_p).for_each(|l, theta_w_pl| {
                                        if l != k || p != j {
                                            let v = *theta_w_pl * *abs_d_ip;
                                            term *= F::one() + sqrt3 * v
                                        }
                                    });
                                },
                            );
                            *da_ij += deriv * term;
                        });
                    });
            });
        let da = einsum("i,ij->ij", &[b, &da])
            .unwrap()
            .into_shape_with_order((xtrain.nrows(), xtrain.ncols()))
            .unwrap();
        db + da
    }
}

/// Matern 5/2 correlation model
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(
    feature = "serializable",
    derive(Serialize, Deserialize),
    serde(into = "String"),
    serde(try_from = "String")
)]
pub struct Matern52Corr();

impl From<Matern52Corr> for String {
    fn from(_item: Matern52Corr) -> String {
        "Matern52".to_string()
    }
}

impl TryFrom<String> for Matern52Corr {
    type Error = &'static str;
    fn try_from(s: String) -> Result<Self, Self::Error> {
        if s == "Matern52" {
            Ok(Self::default())
        } else {
            Err("Bad string value for Matern52Corr, should be \'Matern52\'")
        }
    }
}

impl<F: Float> CorrelationModel<F> for Matern52Corr {
    ///   d    h         
    /// prod prod (1 + sqrt(5) * theta_l * |d_j . weight_j| + (5./3.) * theta_l^2 * |d_j . weight_j|^2) exp( - sqrt(5) * theta_l * |d_j . weight_j| )
    ///  j=1  l=1
    fn rval_from_distances(
        &self,
        d: &ArrayBase<impl Data<Elem = F>, Ix2>,
        theta: &ArrayBase<impl Data<Elem = F>, Ix1>,
        weights: &ArrayBase<impl Data<Elem = F>, Ix2>,
    ) -> Array2<F> {
        let (a, b) = self.compute_r_factors(d, theta, weights);
        let r = a * b;
        r.into_shape_with_order((d.nrows(), 1)).unwrap()
    }

    fn jac(
        &self,
        x: &ArrayBase<impl Data<Elem = F>, Ix1>,
        xtrain: &ArrayBase<impl Data<Elem = F>, Ix2>,
        theta: &ArrayBase<impl Data<Elem = F>, Ix1>,
        weights: &ArrayBase<impl Data<Elem = F>, Ix2>,
    ) -> Array2<F> {
        let d = differences(x, xtrain);
        let (a, b) = self.compute_r_factors(&d, theta, weights);

        self._jac_helper(&d, &a, &b, xtrain, theta, weights)
    }

    fn rval_with_jac(
        &self,
        x: &ArrayBase<impl Data<Elem = F>, Ix1>,
        xtrain: &ArrayBase<impl Data<Elem = F>, Ix2>,
        theta: &ArrayBase<impl Data<Elem = F>, Ix1>,
        weights: &ArrayBase<impl Data<Elem = F>, Ix2>,
    ) -> (Array2<F>, Array2<F>) {
        let d = differences(x, xtrain);
        let (a, b) = self.compute_r_factors(&d, theta, weights);
        let r = &a * &b;
        let jr = self._jac_helper(&d, &a, &b, xtrain, theta, weights);

        (r.into_shape_with_order((d.nrows(), 1)).unwrap(), jr)
    }

    fn theta_influence_factors(&self) -> (F, F) {
        (F::cast(0.23), F::cast(2.44))
    }
}

impl fmt::Display for Matern52Corr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Matern52")
    }
}

impl Matern52Corr {
    fn compute_r_factors<F: Float>(
        &self,
        d: &ArrayBase<impl Data<Elem = F>, Ix2>,
        theta: &ArrayBase<impl Data<Elem = F>, Ix1>,
        weights: &ArrayBase<impl Data<Elem = F>, Ix2>,
    ) -> (Array1<F>, Array1<F>) {
        let sqrt5 = F::cast(5).sqrt();
        let div5_3 = F::cast(5. / 3.);
        let theta_w = theta * weights.mapv(|v| v.abs());

        let mut a = Array1::ones(d.nrows());
        Zip::from(&mut a).and(d.rows()).for_each(|a_i, d_i| {
            Zip::from(&d_i)
                .and(theta_w.rows())
                .for_each(|d_ij, theta_w_j| {
                    *a_i *= theta_w_j
                        .mapv(|v| {
                            F::one() + sqrt5 * v * d_ij.abs() + div5_3 * (v * v * *d_ij * *d_ij)
                        })
                        .product();
                });
        });

        let d_theta_w = d.mapv(|v| v.abs()).dot(&theta_w);
        let b = d_theta_w.sum_axis(Axis(1)).mapv(|v| F::exp(-sqrt5 * v));
        (a, b)
    }

    fn _jac_helper<F: Float>(
        &self,
        d: &ArrayBase<impl Data<Elem = F>, Ix2>,
        a: &ArrayBase<impl Data<Elem = F>, Ix1>,
        b: &ArrayBase<impl Data<Elem = F>, Ix1>,
        xtrain: &ArrayBase<impl Data<Elem = F>, Ix2>,
        theta: &ArrayBase<impl Data<Elem = F>, Ix1>,
        weights: &ArrayBase<impl Data<Elem = F>, Ix2>,
    ) -> Array2<F> {
        let sqrt5 = F::cast(5).sqrt();
        let div5_3 = F::cast(5. / 3.);
        let div10_3 = F::cast(2.) * div5_3;

        let theta_w = weights.mapv(|v| v.abs()).dot(theta);
        let sign_d = d.mapv(|v| v.signum());
        let mut db = Array2::<F>::zeros((xtrain.nrows(), xtrain.ncols()));
        let abs_d = d.mapv(|v| v.abs());
        Zip::from(db.rows_mut())
            .and(a)
            .and(b)
            .and(sign_d.rows())
            .for_each(|mut db_i, ai, bi, si| {
                Zip::from(&mut db_i)
                    .and(&si)
                    .and(&theta_w)
                    .for_each(|db_ij, sij, theta_wj| {
                        *db_ij = -sqrt5 * *theta_wj * *sij * *bi * *ai;
                    });
            });
        let theta_w = theta * weights.mapv(|v| v.abs());
        let mut da = Array2::<F>::zeros((xtrain.nrows(), xtrain.ncols()));
        Zip::from(da.rows_mut())
            .and(abs_d.rows())
            .and(sign_d.rows())
            .for_each(|mut da_i, abs_d_i, sign_i| {
                Zip::indexed(&mut da_i).and(&abs_d_i).and(&sign_i).for_each(
                    |j, da_ij, abs_d_ij, sign_ij| {
                        Zip::indexed(theta_w.columns()).for_each(|k, theta_w_k| {
                            let mut term = F::one();
                            let deriv = sqrt5 * theta_w_k[j] * *sign_ij
                                + div10_3 * theta_w_k[j] * theta_w_k[j] * *sign_ij * *abs_d_ij;
                            Zip::indexed(theta_w.rows()).and(abs_d_i).for_each(
                                |p, theta_w_p, abs_d_ip| {
                                    Zip::indexed(theta_w_p).for_each(|l, theta_w_pl| {
                                        if l != k || p != j {
                                            let v = *theta_w_pl * *abs_d_ip;
                                            term *= F::one() + sqrt5 * v + div5_3 * v * v;
                                        }
                                    });
                                },
                            );
                            *da_ij += deriv * term;
                        });
                    },
                );
            });
        let da = einsum("i,ij->ij", &[b, &da])
            .unwrap()
            .into_shape_with_order((xtrain.nrows(), xtrain.ncols()))
            .unwrap();
        db + da
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::{DiffMatrix, NormalizedData};
    use approx::assert_abs_diff_eq;
    use ndarray::{arr1, array};
    use paste::paste;

    #[test]
    fn test_squared_exponential() {
        let xt = array![[4.5], [1.2], [2.0], [3.0], [4.0]];
        let dm = DiffMatrix::new(&xt);
        let res = SquaredExponentialCorr::default().rval_from_distances(
            &dm.d,
            &arr1(&[f64::sqrt(0.2)]),
            &array![[1.]],
        );
        let expected = array![
            [0.336552878364737],
            [0.5352614285189903],
            [0.7985162187593771],
            [0.9753099120283326],
            [0.9380049995307295],
            [0.7232502423798424],
            [0.4565760496233148],
            [0.9048374180359595],
            [0.6703200460356393],
            [0.9048374180359595]
        ];
        assert_abs_diff_eq!(res, expected, epsilon = 1e-6);
    }

    #[test]
    fn test_squared_exponential_2d() {
        let xt = array![[0., 1.], [2., 3.], [4., 5.]];
        let dm = DiffMatrix::new(&xt);
        dbg!(&dm);
        let res = SquaredExponentialCorr::default().rval_from_distances(
            &dm.d,
            &arr1(&[f64::sqrt(2.), 2.]),
            &array![[1., 0.], [0., 1.]],
        );
        let expected = array![[6.14421235e-06], [1.42516408e-21], [6.14421235e-06]];
        assert_abs_diff_eq!(res, expected, epsilon = 1e-6);
    }

    #[test]
    fn test_matern32_2d() {
        let xt = array![[0., 1.], [2., 3.], [4., 5.]];
        let dm = DiffMatrix::new(&xt);
        dbg!(&dm);
        let res = Matern32Corr::default().rval_from_distances(
            &dm.d,
            &arr1(&[1., 2.]),
            &array![[1., 0.], [0., 1.]],
        );
        let expected = array![[1.08539595e-03], [1.10776401e-07], [1.08539595e-03]];
        assert_abs_diff_eq!(res, expected, epsilon = 1e-6);
    }

    macro_rules! test_correlation {
        ($corr:ident, $kpls:expr_2021) => {
            paste! {
                #[test]
                fn [<test_corr_ $corr:lower _kpls_ $kpls _derivatives>]() {
                    let x = array![3., 5.];
                    let xt = array![
                        [-9.375, -5.625],
                        [-5.625, -4.375],
                        [9.375, 1.875],
                        [8.125, 5.625],
                        [-4.375, -0.625],
                        [6.875, -3.125],
                        [4.375, 9.375],
                        [3.125, 4.375],
                        [5.625, -8.125],
                        [-8.125, 3.125],
                        [1.875, -6.875],
                        [-0.625, 8.125],
                        [-1.875, -1.875],
                        [0.625, 0.625],
                        [-6.875, -9.375],
                        [-3.125, 6.875]
                    ];
                    let xtrain = NormalizedData::new(&xt);
                    let xnorm = (x.to_owned() - &xtrain.mean) / &xtrain.std;
                    let (theta, weights) = if $kpls {
                        (array![0.31059002],
                            array![[-0.02701716],
                            [-0.99963497]])
                    } else {
                        (array![0.34599115925909146, 0.32083374253611624],
                         array![[1., 0.], [0., 1.]])
                    };

                    let corr = [< $corr Corr >]::default();
                    let jac = corr.jac(&xnorm, &xtrain.data, &theta, &weights) / &xtrain.std;
                    println!("Jacobian: \n{:?}", jac);
                    let xa: f64 = x[0];
                    let xb: f64 = x[1];
                    let e = 1e-5;
                    let x = array![
                        [xa, xb],
                        [xa + e, xb],
                        [xa - e, xb],
                        [xa, xb + e],
                        [xa, xb - e]
                    ];

                    let mut rxx = Array2::zeros((xtrain.data.nrows(), x.nrows()));
                    Zip::from(rxx.columns_mut())
                        .and(x.rows())
                        .for_each(|mut rxxi, xi| {
                            let xnorm = (xi.to_owned() - &xtrain.mean) / &xtrain.std;
                            let d = differences(&xnorm, &xtrain.data);
                            rxxi.assign(&(corr.rval_from_distances( &d, &theta, &weights).column(0)));
                        });
                    let fdiffa = (rxx.column(1).to_owned() - rxx.column(2)).mapv(|v| v / (2. * e));
                    assert_abs_diff_eq!(fdiffa, jac.column(0), epsilon=1e-6);
                    let fdiffb = (rxx.column(3).to_owned() - rxx.column(4)).mapv(|v| v / (2. * e));
                    assert_abs_diff_eq!(fdiffb, jac.column(1), epsilon=1e-6);
                }
            }
        };
    }

    test_correlation!(SquaredExponential, false);
    test_correlation!(AbsoluteExponential, false);
    test_correlation!(Matern32, false);
    test_correlation!(Matern52, false);
    test_correlation!(SquaredExponential, true);
    test_correlation!(AbsoluteExponential, true);
    test_correlation!(Matern32, true);
    test_correlation!(Matern52, true);

    #[test]
    fn test_matern52_2d() {
        let xt = array![[0., 1.], [2., 3.], [4., 5.]];
        let dm = DiffMatrix::new(&xt);
        let res = Matern52Corr::default().rval_from_distances(
            &dm.d,
            &arr1(&[1., 2.]),
            &array![[1., 0.], [0., 1.]],
        );
        let expected = array![[6.62391590e-04], [1.02117882e-08], [6.62391590e-04]];
        assert_abs_diff_eq!(res, expected, epsilon = 1e-6);
    }
}
