use crate::criteria::{InfillCriterion, ei::EI};

use egobox_moe::MixtureGpSurrogate;
use ndarray::{Array1, ArrayView, ArrayView2, Axis};
use ndarray_stats::QuantileExt;

use serde::{Deserialize, Serialize};

/// A structure for WB2 infill criterion specification
#[derive(Clone, Serialize, Deserialize)]
pub struct WB2Criterion(pub Option<f64>);

#[typetag::serde]
impl InfillCriterion for WB2Criterion {
    fn name(&self) -> &'static str {
        if let Some(1.) = self.0 { "WB2" } else { "WB2S" }
    }

    fn uses_dynamic_scaling(&self) -> bool {
        self.0.is_none()
    }

    /// Compute WB2S infill criterion at given `x` point using the surrogate model `obj_model`
    /// and the current minimum of the objective function.
    ///
    /// Uses Probability of Violation (POV) from viability_model to weight the prediction term:
    /// WB2 = scale * EI - pov * y_pred
    /// where pov = viability_model.predict(x) clamped to [0, 1], defaulting to 1.0 if no viability_model.
    fn value(
        &self,
        x: &[f64],
        obj_model: &dyn MixtureGpSurrogate,
        fmin: f64,
        viability_model: Option<&dyn MixtureGpSurrogate>,
        alpha: Option<f64>,
        sigma_weight: Option<f64>,
        scale: Option<f64>,
    ) -> f64 {
        let scale = scale.unwrap_or(self.0.unwrap_or(1.0));
        let pt = ArrayView::from_shape((1, x.len()), x).unwrap();
        let ei = EI.value(
            x,
            obj_model,
            fmin,
            viability_model,
            alpha,
            sigma_weight,
            None,
        );
        let pov = if let Some(viab_model) = viability_model {
            viab_model.predict(&pt).unwrap()[0].clamp(0.0, 1.0)
        } else {
            1.0
        };
        let y_pred = obj_model.predict(&pt).unwrap()[0];
        scale * ei - pov * y_pred
    }

    /// Computes derivatives of WB2S infill criterion wrt to x components at given `x` point
    /// using the surrogate model `obj_model` and the current minimum of the objective function.
    ///
    /// Gradient accounts for POV weighting:
    /// grad(WB2) = scale * grad(EI) - (grad(pov) * y_pred + pov * grad(y_pred))
    fn grad(
        &self,
        x: &[f64],
        obj_model: &dyn MixtureGpSurrogate,
        fmin: f64,
        viability_model: Option<&dyn MixtureGpSurrogate>,
        _alpha: Option<f64>,
        sigma_weight: Option<f64>,
        scale: Option<f64>,
    ) -> Array1<f64> {
        let scale = scale.unwrap_or(self.0.unwrap_or(1.0));
        let pt = ArrayView::from_shape((1, x.len()), x).unwrap();
        let grad_ei = EI.grad(
            x,
            obj_model,
            fmin,
            viability_model,
            _alpha,
            sigma_weight,
            None,
        ) * scale;

        let pov = if let Some(viab_model) = viability_model {
            viab_model.predict(&pt).unwrap()[0].clamp(0.0, 1.0)
        } else {
            1.0
        };

        let y_pred = obj_model.predict(&pt).unwrap()[0];
        let grad_y_pred = obj_model.predict_gradients(&pt).unwrap().row(0).to_owned();

        // Get grad(pov) from viability model if available
        let grad_pov = if let Some(viab_model) = viability_model {
            viab_model.predict_gradients(&pt).unwrap().row(0).to_owned()
        } else {
            Array1::zeros(x.len())
        };

        // Product rule: grad(pov * y_pred) = grad(pov) * y_pred + pov * grad(y_pred)
        let grad_pov_y_pred = &grad_pov * y_pred + &grad_y_pred * pov;

        grad_ei - grad_pov_y_pred
    }

    fn scaling(
        &self,
        x: &ndarray::ArrayView2<f64>,
        obj_model: &dyn MixtureGpSurrogate,
        fmin: f64,
        _viability_model: Option<&dyn MixtureGpSurrogate>,
        _alpha: Option<f64>,
        sigma_weight: Option<f64>,
    ) -> f64 {
        if let Some(scale) = self.0 {
            scale
        } else {
            compute_wb2s_scale(x, obj_model, fmin, sigma_weight)
        }
    }
}

/// Computes the scaling factor used to scale WB2 infill criteria.
pub(crate) fn compute_wb2s_scale(
    x: &ArrayView2<f64>,
    obj_model: &dyn MixtureGpSurrogate,
    fmin: f64,
    sigma_weight: Option<f64>,
) -> f64 {
    let ratio = 100.; // TODO: make it a parameter
    let ei_x = x.map_axis(Axis(1), |xi| {
        let xi = xi.as_standard_layout();
        EI.value(
            xi.as_slice().unwrap(),
            obj_model,
            fmin,
            None,
            None,
            sigma_weight,
            None,
        )
    });
    let i_max = ei_x.argmax().unwrap();
    let pred_max = obj_model
        .predict(&x.row(i_max).insert_axis(Axis(0)))
        .unwrap()[0];
    let ei_max = ei_x[i_max];
    if ei_max.abs() > 100. * f64::EPSILON {
        ratio * pred_max.abs() / ei_max
    } else {
        1.
    }
}

/// WB2 infill criterion
pub const WB2: WB2Criterion = WB2Criterion(Some(1.0));
/// WB2 scaled infill criterion
pub const WB2S: WB2Criterion = WB2Criterion(None);
