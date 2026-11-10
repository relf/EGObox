use crate::criteria::{InfillComposition, InfillCriterion};
use crate::utils::{d_log_ei_helper, log_ei_helper, norm_cdf, norm_pdf};
use egobox_moe::MixtureGpSurrogate;
use ndarray::{Array1, ArrayView};

use serde::{Deserialize, Serialize};

const ALPHA0: f64 = 0.3;

/// A structure for Expected Improvement implementation
#[derive(Clone, Serialize, Deserialize)]
pub struct ExpectedImprovement;

#[typetag::serde]
impl InfillCriterion for ExpectedImprovement {
    fn name(&self) -> &'static str {
        "EI"
    }

    /// Compute EI infill criterion at given `x` point using the surrogate model `obj_model`
    /// and the current minimum of the objective function.
    fn value(
        &self,
        x: &[f64],
        obj_model: &dyn MixtureGpSurrogate,
        fmin: f64,
        viability_model: Option<&dyn MixtureGpSurrogate>,
        alpha: Option<f64>,
        sigma_weight: Option<f64>,
        _scale: Option<f64>,
    ) -> f64 {
        let pt = ArrayView::from_shape((1, x.len()), x).unwrap();
        match obj_model.predict_valvar(&pt) {
            Ok((p, s)) => {
                if s[0] < f64::EPSILON {
                    0.0
                } else {
                    let (pov, alpha) = if let Some(pov_model) = viability_model {
                        let s_c = s[0].sqrt();
                        let pov = pov_model.predict(&pt).unwrap()[0].clamp(0.0, 1.0);
                        let alpha = alpha.unwrap_or({
                            if s_c > f64::EPSILON && pov < f64::EPSILON {
                                0.0
                            } else if s_c < f64::EPSILON {
                                1.0
                            } else {
                                ALPHA0
                            }
                        });
                        (pov, alpha)
                    } else {
                        (1.0, 1.0)
                    };

                    let pred = p[0];
                    let k = sigma_weight.unwrap_or(1.0);
                    let sigma = k * s[0].sqrt();
                    let arg = (fmin - pred) / sigma;

                    let arg1 = pov * arg * norm_cdf(arg);
                    let arg2 = pov.powf(alpha) * norm_pdf(arg);
                    sigma * (arg1 + arg2)
                }
            }
            _ => 0.0,
        }
    }

    /// Computes derivatives of EI infill criterion wrt to x components at given `x` point
    /// using the surrogate model `obj_model` and the current minimum of the objective function.
    fn grad(
        &self,
        x: &[f64],
        obj_model: &dyn MixtureGpSurrogate,
        fmin: f64,
        viability_model: Option<&dyn MixtureGpSurrogate>,
        alpha: Option<f64>,
        sigma_weight: Option<f64>,
        _scale: Option<f64>,
    ) -> Array1<f64> {
        let pt = ArrayView::from_shape((1, x.len()), x).unwrap();
        match obj_model.predict_valvar_with_gradients(&pt) {
            Ok((p, s, y_prime, var_prime)) => {
                if s[0] < f64::EPSILON {
                    Array1::zeros(pt.len())
                } else {
                    let (pov, alpha) = if let Some(pov_model) = viability_model {
                        let s_c = s[0].sqrt();
                        let pov = pov_model.predict(&pt).unwrap()[0].clamp(0.0, 1.0);
                        let alpha = alpha.unwrap_or({
                            if s_c > f64::EPSILON && pov < f64::EPSILON {
                                0.0
                            } else if s_c < f64::EPSILON {
                                1.0
                            } else {
                                ALPHA0
                            }
                        });
                        (pov, alpha)
                    } else {
                        (1.0, 1.0)
                    };
                    if pov <= f64::EPSILON {
                        // Shortcut: consider gradient is null
                        return Array1::zeros(pt.len());
                    }

                    let pred = p[0];
                    let diff_y = fmin - pred;
                    let k = sigma_weight.unwrap_or(1.0);
                    let sigma = s[0].sqrt();
                    let weighted_sigma = k * sigma;
                    let arg = diff_y / weighted_sigma;

                    let y_prime = y_prime.row(0);
                    let sig_2_prime = var_prime.row(0);
                    let sig_prime = sig_2_prime.mapv(|v| k * v / (2. * sigma));
                    let arg_prime = y_prime.mapv(|v| v / (-weighted_sigma))
                        - diff_y.to_owned()
                            * sig_prime.mapv(|v| v / (weighted_sigma * weighted_sigma));

                    // EI = sigma * (pov * arg * norm_cdf(arg) + pov^alpha * norm_pdf(arg))
                    // Let f_ei = pov * arg * norm_cdf(arg) + pov^alpha * norm_pdf(arg)
                    // d(EI)/dx = d(sigma)/dx * f_ei + sigma * df_ei/dx

                    let norm_cdf_arg = norm_cdf(arg);
                    let norm_pdf_arg = norm_pdf(arg);

                    // f_ei = pov * arg * norm_cdf(arg) + pov^alpha * norm_pdf(arg)
                    let pov_alpha = pov.powf(alpha);
                    let f_ei = pov * arg * norm_cdf_arg + pov_alpha * norm_pdf_arg;

                    // df_ei/dx = d(pov)/dx * arg * norm_cdf(arg) + pov * d(arg * norm_cdf(arg))/dx
                    //          + d(pov^alpha)/dx * norm_pdf(arg) + pov^alpha * d(norm_pdf(arg))/dx

                    // d(arg * norm_cdf(arg))/dx = arg_prime * (norm_cdf(arg) + arg * norm_pdf(arg))
                    let d_arg_cdf = arg_prime.clone() * (norm_cdf_arg + arg * norm_pdf_arg);

                    // d(norm_pdf(arg))/dx = -arg * norm_pdf(arg) * arg_prime
                    let d_pdf = -arg * norm_pdf_arg * arg_prime;

                    // d(pov^alpha)/dx = alpha * pov^(alpha-1) * d(pov)/dx
                    let pov_grad = if let Some(pov_model) = viability_model {
                        pov_model.predict_gradients(&pt).unwrap().row(0).to_owned()
                    } else {
                        Array1::zeros(x.len())
                    };

                    let d_pov_alpha = if alpha > 0.0 && pov > f64::EPSILON {
                        alpha * pov.powf(alpha - 1.0) * pov_grad.clone()
                    } else {
                        Array1::zeros(x.len())
                    };

                    // df_ei/dx = pov_grad * arg * norm_cdf(arg) + pov * d_arg_cdf
                    //          + d_pov_alpha * norm_pdf(arg) + pov_alpha * d_pdf
                    let df_ei_dx = &pov_grad * arg * norm_cdf_arg
                        + pov * d_arg_cdf
                        + d_pov_alpha * norm_pdf_arg
                        + pov_alpha * d_pdf;

                    // d(sigma)/dx = sig_prime (already weighted by k)
                    // d(EI)/dx = sig_prime * f_ei + sigma * df_ei/dx
                    sig_prime.to_owned() * f_ei + k * sigma * df_ei_dx
                }
            }
            _ => Array1::zeros(pt.len()),
        }
    }
}

/// Expected Improvement infill criterion
pub const EI: ExpectedImprovement = ExpectedImprovement {};

/// A structure for Log of Expected Improvement implementation
#[derive(Clone, Serialize, Deserialize)]
pub struct LogExpectedImprovement;

#[typetag::serde]
impl InfillCriterion for LogExpectedImprovement {
    fn name(&self) -> &'static str {
        "LogEI"
    }

    fn composition(&self) -> InfillComposition {
        InfillComposition::Log
    }

    /// Compute LogEI infill criterion at given `x` point using the surrogate model `obj_model`
    /// and the current minimum of the objective function.
    fn value(
        &self,
        x: &[f64],
        obj_model: &dyn MixtureGpSurrogate,
        fmin: f64,
        _viability_model: Option<&dyn MixtureGpSurrogate>,
        _alpha: Option<f64>,
        sigma_weight: Option<f64>,
        _scale: Option<f64>,
    ) -> f64 {
        let pt = ArrayView::from_shape((1, x.len()), x).unwrap();

        match obj_model.predict_valvar(&pt) {
            Ok((p, s)) => {
                if s[0] < f64::EPSILON {
                    f64::MIN
                } else {
                    let pred = p[0];
                    let k = sigma_weight.unwrap_or(1.0);
                    let sigma = k * s[0].sqrt();
                    let u = (fmin - pred) / sigma;
                    log_ei_helper(u) + sigma.ln()
                }
            }
            _ => f64::MIN,
        }
    }

    /// Computes derivatives of LogEI infill criterion wrt to x components at given `x` point
    /// using the surrogate model `obj_model` and the current minimum of the objective function.
    fn grad(
        &self,
        x: &[f64],
        obj_model: &dyn MixtureGpSurrogate,
        fmin: f64,
        _viability_model: Option<&dyn MixtureGpSurrogate>,
        _alpha: Option<f64>,
        sigma_weight: Option<f64>,
        _scale: Option<f64>,
    ) -> Array1<f64> {
        let pt = ArrayView::from_shape((1, x.len()), x).unwrap();

        match obj_model.predict_valvar_with_gradients(&pt) {
            Ok((p, s, y_prime, var_prime)) => {
                if s[0] < f64::EPSILON {
                    Array1::from_elem(pt.len(), f64::MIN)
                } else {
                    let pred = p[0];
                    let diff_y = fmin - pred;
                    let k = sigma_weight.unwrap_or(1.0);
                    let sigma = s[0].sqrt();
                    let weighted_sigma = k * sigma;
                    let arg = diff_y / weighted_sigma;

                    let y_prime = y_prime.row(0);
                    let sig_2_prime = var_prime.row(0);
                    let sig_prime = sig_2_prime.mapv(|v| k * v / (2. * sigma));

                    let arg_prime = y_prime.mapv(|v| v / (-weighted_sigma))
                        - diff_y.to_owned()
                            * sig_prime.mapv(|v| v / (weighted_sigma * weighted_sigma));

                    let dhelper = d_log_ei_helper(arg);
                    let arg1 = arg_prime.mapv(|v| dhelper * v);

                    let arg2 = sig_prime / weighted_sigma;
                    arg1 + arg2
                }
            }
            _ => Array1::from_elem(pt.len(), f64::MIN),
        }
    }
}

/// Log of Expected Improvement infill criterion
pub const LOG_EI: LogExpectedImprovement = LogExpectedImprovement {};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::*;
    use approx::assert_abs_diff_eq;
    use egobox_moe::{MixintContext, MoeBuilder};
    use finitediff::vec;
    use linfa::Dataset;
    use ndarray::{Array2, ArrayView2, array};

    #[test]
    fn test_ei_gradients() {
        let xtypes = vec![XType::Float(0., 25.)];

        let mixi = MixintContext::new(&xtypes);

        let surrogate_builder = MoeBuilder::new();
        let xt = array![[0.], [2.], [5.], [10.], [25.]];
        let yt = array![0., 0.2, -0.3, 0.5, -1.];
        let ds = Dataset::new(xt, yt);
        let mixi_moe = mixi
            .create_surrogate(&surrogate_builder, &ds)
            .expect("Mixint surrogate creation");

        let x = vec![3.];
        let grad = EI.grad(&x, &mixi_moe, 0., None, None, Some(0.75), None);

        let f = |x: &Vec<f64>| -> std::result::Result<f64, anyhow::Error> {
            Ok(EI.value(x, &mixi_moe, 0., None, None, Some(0.75), None))
        };
        let grad_central = (vec::central_diff(&f)(&x)).unwrap();
        assert_abs_diff_eq!(grad[0], grad_central[0], epsilon = 1e-6);

        // let cx = Array1::linspace(0., 25., 100);
        // write_npy("ei_cx.npy", &cx).expect("save x");

        // let mut cy = Array1::zeros(cx.len());
        // Zip::from(&mut cy).and(&cx).for_each(|yi, xi| {
        //     *yi = EI.value(&[*xi], &mixi_moe, 0., None);
        // });
        // write_npy("ei_cy.npy", &cy).expect("save y");

        // let mut cdy = Array1::zeros(cx.len());
        // Zip::from(&mut cdy).and(&cx).for_each(|yi, xi| {
        //     *yi = EI.grad(&[*xi], &mixi_moe, 0., None)[0];
        // });
        // write_npy("ei_cdy.npy", &cdy).expect("save y");

        // let cytrue = mixi_moe
        //     .predict(&cx.insert_axis(Axis(1)).view())
        //     .expect("prediction");
        // write_npy("ei_cytrue.npy", &cytrue).expect("save cstr");

        // println!("thetas = {}", mixi_moe);
    }

    fn xsinx(x: &ArrayView2<f64>) -> Array2<f64> {
        (x - 3.5) * ((x - 3.5) / std::f64::consts::PI).mapv(|v| v.sin())
    }

    #[test]
    fn test_logei_gradients() {
        let xtypes = vec![XType::Float(0., 25.)];
        let mixi = MixintContext::new(&xtypes);

        let surrogate_builder = MoeBuilder::new();
        let xt = array![[0.0], [7.0], [25.0]];
        let yt = xsinx(&xt.view()).into_iter().collect::<Array1<_>>();
        let ds = Dataset::new(xt, yt);
        let mixi_moe = mixi
            .create_surrogate(&surrogate_builder, &ds)
            .expect("Mixint surrogate creation");

        let x = Array1::linspace(0., 25., 100);

        let grad = x.mapv(|v| LOG_EI.grad(&[v], &mixi_moe, 0., None, None, Some(0.75), None)[0]);

        let f = |x: &Vec<f64>| -> std::result::Result<f64, anyhow::Error> {
            Ok(LOG_EI.value(x, &mixi_moe, 0., None, None, Some(0.75), None))
        };
        let grad_central = x.mapv(|v| vec::central_diff(&f)(&vec![v]).unwrap()[0]);

        // check relative error between finite difference and analytical gradient
        for (i, v) in grad.iter().enumerate() {
            if v.abs() < 1e6 {
                let rel_error = (v - grad_central[i]).abs() / (v.abs() + 1e-10);
                println!("v={v} fdiff={}", grad_central[i]);
                assert!(
                    rel_error < 5e-1,
                    "Relative error too high at index {i}: {rel_error} {v} - {}",
                    grad_central[i]
                );
            }
        }
    }

    #[test]
    fn test_logei_sigma_weight_changes_value() {
        let xtypes = vec![XType::Float(0., 25.)];
        let mixi = MixintContext::new(&xtypes);

        let surrogate_builder = MoeBuilder::new();
        let xt = array![[0.0], [7.0], [25.0]];
        let yt = xsinx(&xt.view()).into_iter().collect::<Array1<_>>();
        let ds = Dataset::new(xt, yt);
        let mixi_moe = mixi
            .create_surrogate(&surrogate_builder, &ds)
            .expect("Mixint surrogate creation");

        let x = [5.0];
        let low_weight = LOG_EI.value(&x, &mixi_moe, 0.0, None, None, Some(0.5), None);
        let high_weight = LOG_EI.value(&x, &mixi_moe, 0.0, None, None, Some(2.0), None);

        assert_ne!(low_weight, high_weight);
    }
}
