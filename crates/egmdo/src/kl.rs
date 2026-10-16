//! Karhunen–Loève (KL) decomposition of the objective PCEs and KL-GP random field.
//!
//! The objective PCEs at the points of `DoE_UQ` form a random vector whose KL decomposition
//! is computed from the PCE coefficients. Its mean and eigenvectors are then interpolated
//! over the design space by GPs, the KL-GPs, which gives a continuous model of the random
//! objective (specification, section 5).

use crate::errors::{EgmdoError, Result};
use crate::pce::HermiteBasis;
use crate::surrogates::{DisciplinaryGp, GpSettings};
use linfa_linalg::eigh::EighInto;
use ndarray::{Array1, Array2, ArrayBase, ArrayView2, Axis, Data, Ix2, s};

/// The truncated KL decomposition of the random vector `Ŷ_obj(Ξ)` over `DoE_UQ`:
///
/// ```text
/// Ŷ_obj(Ξ) = μ + Σ_k ζ_k(Ξ) φ_k,   ζ_k(Ξ) = Σ_{j>=2} b_kj H_j(Ξ),   b_kj = a_jᵀ φ_k
/// ```
#[derive(Clone, Debug)]
pub struct KlDecomposition {
    mean: Array1<f64>,
    eigenvalues: Array1<f64>,
    modes: Array2<f64>,
    b: Array2<f64>,
}

impl KlDecomposition {
    /// Decomposes the PCE coefficients `coefs` `(n_UQ, P)` of the objective at the `DoE_UQ`
    /// points, keeping the smallest number of modes explaining more than `1 - threshold`
    /// of the total variance. No mode is kept if the variance is zero (decision D8).
    pub fn new(coefs: &ArrayBase<impl Data<Elem = f64>, Ix2>, threshold: f64) -> Result<Self> {
        let mean = coefs.column(0).to_owned();
        let a = coefs.slice(s![.., 1..]);
        let cov = a.dot(&a.t());
        let (values, vectors) = cov
            .eigh_into()
            .map_err(|e| EgmdoError::InvalidData(format!("KL eigendecomposition failed: {e}")))?;
        let mut order: Vec<usize> = (0..values.len()).collect();
        order.sort_by(|&i, &j| values[j].total_cmp(&values[i]));
        let values = values.select(Axis(0), &order).mapv(|v| v.max(0.));
        let vectors = vectors.select(Axis(1), &order);

        let total = values.sum();
        let n_modes = if total > 0. {
            let mut cumulated = 0.;
            values
                .iter()
                .position(|v| {
                    cumulated += v;
                    cumulated / total > 1. - threshold
                })
                .map_or(values.len(), |k| k + 1)
        } else {
            0
        };
        let modes = vectors.slice(s![.., ..n_modes]).to_owned();
        let b = modes.t().dot(&a);
        Ok(KlDecomposition {
            mean,
            eigenvalues: values.slice(s![..n_modes]).to_owned(),
            modes,
            b,
        })
    }

    /// The number of retained modes `M`
    pub fn n_modes(&self) -> usize {
        self.modes.ncols()
    }

    /// The mean vector `μ` over `DoE_UQ`
    pub fn mean(&self) -> &Array1<f64> {
        &self.mean
    }

    /// The retained eigenvalues, in decreasing order
    pub fn eigenvalues(&self) -> &Array1<f64> {
        &self.eigenvalues
    }

    /// The retained eigenvectors `φ_k`, as columns of a `(n_UQ, M)` matrix
    pub fn modes(&self) -> &Array2<f64> {
        &self.modes
    }

    /// The coefficients `b_kj` of the KL random variables in the PCE basis without its
    /// constant term, as a `(M, P - 1)` matrix
    pub fn b(&self) -> &Array2<f64> {
        &self.b
    }

    /// The KL random variables `ζ_k(Ξ)` at samples `xi` `(n, dim)`, as a `(n, M)` matrix
    pub fn zeta(&self, basis: &HermiteBasis, xi: &ArrayView2<f64>) -> Array2<f64> {
        basis.eval(xi).slice(s![.., 1..]).dot(&self.b.t())
    }

    /// The truncated random vector `Ŷ_obj(Ξ)` over `DoE_UQ` at samples `xi`, as a
    /// `(n_UQ, n)` matrix
    pub fn eval(&self, basis: &HermiteBasis, xi: &ArrayView2<f64>) -> Array2<f64> {
        let zeta = self.zeta(basis, xi);
        &self.modes.dot(&zeta.t()) + &self.mean.view().insert_axis(Axis(1))
    }
}

/// The predictions of the KL-GPs at design points: column 0 for the mean, column `k`
/// for eigenvector `k`
#[derive(Clone, Debug)]
pub struct KlGpPrediction {
    /// GP means `(n, M + 1)`
    pub mean: Array2<f64>,
    /// GP standard deviations `(n, M + 1)`
    pub std: Array2<f64>,
}

/// The continuous random field of the objective (14), built by GP interpolation of the
/// mean and eigenvectors of a [KlDecomposition]
#[derive(Clone)]
pub struct KlGpField {
    basis: HermiteBasis,
    kl: KlDecomposition,
    gps: Vec<DisciplinaryGp>,
}

impl KlGpField {
    /// Interpolates the mean and eigenvectors of `kl` over the `DoE_UQ` points `z_uq`
    /// `(n_UQ, nx)`.
    pub fn fit(
        z_uq: &Array2<f64>,
        kl: KlDecomposition,
        basis: HermiteBasis,
        settings: &GpSettings,
    ) -> Result<Self> {
        if z_uq.nrows() != kl.mean.len() {
            return Err(EgmdoError::InvalidData(format!(
                "KL decomposition over {} points but {} design points given",
                kl.mean.len(),
                z_uq.nrows()
            )));
        }
        let mut gps = vec![settings.train(z_uq, &kl.mean)?];
        for mode in kl.modes.columns() {
            gps.push(settings.train(z_uq, &mode.to_owned())?);
        }
        Ok(KlGpField { basis, kl, gps })
    }

    /// The KL decomposition
    pub fn kl(&self) -> &KlDecomposition {
        &self.kl
    }

    /// The PCE basis
    pub fn basis(&self) -> &HermiteBasis {
        &self.basis
    }

    /// The KL random variables `ζ_k(Ξ)` at samples `xi` `(n_s, dim)`, as a `(n_s, M)` matrix
    pub fn zeta(&self, xi: &ArrayView2<f64>) -> Array2<f64> {
        self.kl.zeta(&self.basis, xi)
    }

    /// Predicts the KL-GPs at design points `z` `(n, nx)`
    pub fn predict(&self, z: &Array2<f64>) -> Result<KlGpPrediction> {
        let mut mean = Array2::zeros((z.nrows(), self.gps.len()));
        let mut std = Array2::zeros((z.nrows(), self.gps.len()));
        for (k, gp) in self.gps.iter().enumerate() {
            mean.column_mut(k).assign(&gp.predict(z)?);
            std.column_mut(k)
                .assign(&gp.predict_var(z)?.mapv(|v| v.max(0.).sqrt()));
        }
        Ok(KlGpPrediction { mean, std })
    }

    /// Samples the random field (14) at design points `z` `(n_z, nx)` for the samples
    /// `xi` `(n_s, dim)` of the disciplinary uncertainty and `eta` `(n_s, M + 1)` of the
    /// KL-GP interpolation uncertainty. Returns a `(n_z, n_s)` matrix.
    pub fn sample(
        &self,
        z: &Array2<f64>,
        xi: &ArrayView2<f64>,
        eta: &ArrayView2<f64>,
    ) -> Result<Array2<f64>> {
        if eta.nrows() != xi.nrows() || eta.ncols() != self.gps.len() {
            return Err(EgmdoError::InvalidData(format!(
                "eta samples must have shape ({}, {}), got {:?}",
                xi.nrows(),
                self.gps.len(),
                eta.shape()
            )));
        }
        let pred = self.predict(z)?;
        Ok(self.sample_from(&pred, &self.zeta(xi).view(), eta))
    }

    /// Same as [KlGpField::sample] from KL-GP predictions and precomputed `ζ(Ξ)`, to reuse
    /// them over many evaluations.
    pub fn sample_from(
        &self,
        pred: &KlGpPrediction,
        zeta: &ArrayView2<f64>,
        eta: &ArrayView2<f64>,
    ) -> Array2<f64> {
        let (m, sd) = (&pred.mean, &pred.std);
        // Mean term: m_0(z) + s_0(z) η_0
        let mut field = &m.column(0).insert_axis(Axis(1))
            + &(&sd.column(0).insert_axis(Axis(1)) * &eta.column(0).insert_axis(Axis(0)));
        // Modes: Σ_k ζ_k(Ξ) (m_k(z) + s_k(z) η_k)
        for k in 0..self.kl.n_modes() {
            let phi = &m.column(k + 1).insert_axis(Axis(1))
                + &(&sd.column(k + 1).insert_axis(Axis(1))
                    * &eta.column(k + 1).insert_axis(Axis(0)));
            field += &(&phi * &zeta.column(k).insert_axis(Axis(0)));
        }
        field
    }

    /// The objective model (19), using the KL-GP means only, at design points `z`
    /// `(n_z, nx)` for samples `xi` `(n_s, dim)`. Returns a `(n_z, n_s)` matrix.
    pub fn mean_model(&self, z: &Array2<f64>, xi: &ArrayView2<f64>) -> Result<Array2<f64>> {
        let pred = self.predict(z)?;
        let zeta = self.zeta(xi);
        Ok(
            &pred.mean.column(0).insert_axis(Axis(1))
                + &pred.mean.slice(s![.., 1..]).dot(&zeta.t()),
        )
    }

    /// The expectation over `Ξ` of the objective model (19) at design points `z`,
    /// i.e. the KL-GP mean `m_0(z)`
    pub fn expectation(&self, z: &Array2<f64>) -> Result<Array1<f64>> {
        self.gps[0].predict(z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pce::normal_samples;
    use crate::uq::{RandomMda, UqSettings, coefficients, doe_uq};
    use approx::assert_abs_diff_eq;
    use ndarray::{Array, array};

    #[test]
    fn test_kl_of_low_rank_coefficients() {
        let basis = HermiteBasis::new(2, 3);
        // Coefficients over 5 points: a mean and two independent directions
        let mut coefs = Array2::zeros((5, basis.size()));
        coefs.column_mut(0).assign(&array![1., 2., 3., 4., 5.]);
        coefs.column_mut(1).assign(&array![1., 0., -1., 0., 1.]);
        coefs.column_mut(4).assign(&array![0., 2., 0., 2., 0.]);
        coefs.column_mut(2).assign(&array![0.5, 0., -0.5, 0., 0.5]);
        let kl = KlDecomposition::new(&coefs, 1e-6).unwrap();
        assert_eq!(kl.n_modes(), 2);
        assert!(kl.eigenvalues()[0] >= kl.eigenvalues()[1]);
        assert_abs_diff_eq!(
            kl.modes().t().dot(kl.modes()),
            Array2::eye(2),
            epsilon = 1e-10
        );

        // The truncated vector reproduces the PCEs exactly
        let xi = normal_samples(10, 2, 0);
        let pce = basis.eval(&xi.view()).dot(&coefs.t()).reversed_axes();
        assert_abs_diff_eq!(kl.eval(&basis, &xi.view()), pce, epsilon = 1e-10);
    }

    #[test]
    fn test_kl_truncation() {
        let basis = HermiteBasis::new(2, 3);
        let mut coefs = Array2::zeros((4, basis.size()));
        coefs.column_mut(1).assign(&array![1., 1., 1., 1.]);
        coefs.column_mut(2).assign(&array![1e-5, 0., 0., 0.]);
        // The second mode explains ~ 2.5e-11 of the variance
        assert_eq!(KlDecomposition::new(&coefs, 1e-6).unwrap().n_modes(), 1);
        assert_eq!(KlDecomposition::new(&coefs, 1e-12).unwrap().n_modes(), 2);

        // No disciplinary uncertainty: no mode, the vector is its mean
        let mut coefs = Array2::zeros((4, basis.size()));
        coefs.column_mut(0).assign(&array![1., 2., 3., 4.]);
        let kl = KlDecomposition::new(&coefs, 1e-6).unwrap();
        assert_eq!(kl.n_modes(), 0);
        let xi = normal_samples(3, 2, 0);
        assert_abs_diff_eq!(
            kl.eval(&basis, &xi.view()),
            array![[1., 1., 1.], [2., 2., 2.], [3., 3., 3.], [4., 4., 4.]]
        );
    }

    #[test]
    fn test_kl_gp_field_on_toy_problem() {
        let (tp, models) = crate::uq::tests::toy_models();
        let settings = UqSettings::default();
        let basis = settings.basis(2);
        let xi = settings.regression_samples(&basis, 0);
        // Paper's DoE_UQ
        let z_uq = array![[-4.3], [-2.5], [0.5], [3.7]];
        let uq = doe_uq(
            &tp.problem,
            &models,
            &z_uq.view(),
            &xi.view(),
            &basis,
            &tp.mda,
            RandomMda::Solved,
        )
        .unwrap();
        let coefs = coefficients(&uq, 0);
        let kl = KlDecomposition::new(&coefs, settings.kl_threshold).unwrap();
        assert!(kl.n_modes() >= 1 && kl.n_modes() <= 4);

        let field = KlGpField::fit(&z_uq, kl, basis.clone(), &GpSettings::default()).unwrap();

        // At DoE_UQ points, the KL-GPs interpolate and the model (19) reproduces the PCEs
        let pred = field.predict(&z_uq).unwrap();
        assert_abs_diff_eq!(pred.mean.column(0), field.kl().mean(), epsilon = 1e-6);
        assert!(
            pred.std.iter().all(|&s| s < 1e-3),
            "KL-GP std at DoE_UQ: {}",
            pred.std
        );
        let xi_test = normal_samples(50, 2, 3);
        let pce = basis.eval(&xi_test.view()).dot(&coefs.t()).reversed_axes();
        let model = field.mean_model(&z_uq, &xi_test.view()).unwrap();
        assert_abs_diff_eq!(
            model,
            pce,
            epsilon = 1e-3 * pce.mapv(f64::abs).mean().unwrap()
        );

        // With η = 0, the random field (14) is the model (19); with η ≠ 0 it is spread
        // between DoE_UQ points
        let z = Array::linspace(-5., 5., 11).insert_axis(Axis(1));
        let eta0 = Array2::zeros((50, field.kl().n_modes() + 1));
        assert_abs_diff_eq!(
            field.sample(&z, &xi_test.view(), &eta0.view()).unwrap(),
            field.mean_model(&z, &xi_test.view()).unwrap(),
            epsilon = 1e-12
        );
        let eta = normal_samples(50, field.kl().n_modes() + 1, 4);
        let sampled = field.sample(&z, &xi_test.view(), &eta.view()).unwrap();
        let model = field.mean_model(&z, &xi_test.view()).unwrap();
        assert!((&sampled - &model).mapv(f64::abs).sum() > 1e-3);

        // The expectation over Ξ of the model (19) is the KL-GP mean
        let xi_many = normal_samples(20_000, 2, 5);
        let mc_mean = field
            .mean_model(&z, &xi_many.view())
            .unwrap()
            .mean_axis(Axis(1))
            .unwrap();
        let expectation = field.expectation(&z).unwrap();
        let scale = field.kl().eigenvalues()[0].sqrt();
        assert_abs_diff_eq!(mc_mean, expectation, epsilon = 0.05 * scale.max(1e-3));

        // η must have one column per KL-GP
        let eta_bad = Array2::zeros((50, field.kl().n_modes() + 2));
        assert!(field.sample(&z, &xi_test.view(), &eta_bad.view()).is_err());
    }
}
