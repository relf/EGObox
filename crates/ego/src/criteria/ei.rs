use crate::criteria::InfillCriterion;
use crate::utils::{d_log_ei_helper, log_ei_helper, norm_cdf, norm_pdf};
use egobox_moe::MixtureGpSurrogate;
use ndarray::{Array1, ArrayView};

use serde::{Deserialize, Serialize};

const SQRT_2PI: f64 = 2.5066282746310007;

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
        sigma_weight: Option<f64>,
        _scale: Option<f64>,
    ) -> f64 {
        let pt = ArrayView::from_shape((1, x.len()), x).unwrap();
        match obj_model.predict_valvar(&pt) {
            Ok((p, s)) => {
                if s[0] < f64::EPSILON {
                    0.0
                } else {
                    let pred = p[0];
                    let k = sigma_weight.unwrap_or(1.0);
                    let sigma = k * s[0].sqrt();
                    let args0 = (fmin - pred) / sigma;
                    let args1 = args0 * norm_cdf(args0);
                    let args2 = norm_pdf(args0);
                    sigma * (args1 + args2)
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
        sigma_weight: Option<f64>,
        _scale: Option<f64>,
    ) -> Array1<f64> {
        let pt = ArrayView::from_shape((1, x.len()), x).unwrap();
        match obj_model.predict_valvar(&pt) {
            Ok((p, s)) => {
                if s[0] < f64::EPSILON {
                    Array1::zeros(pt.len())
                } else {
                    let pred = p[0];
                    let diff_y = fmin - pred;
                    let k = sigma_weight.unwrap_or(1.0);
                    let sigma = s[0].sqrt();
                    let arg = (fmin - pred) / (k * sigma);

                    let (y_prime, var_prime) = obj_model.predict_valvar_gradients(&pt).unwrap();
                    let y_prime = y_prime.row(0);
                    let sig_2_prime = var_prime.row(0);
                    let sig_prime = sig_2_prime.mapv(|v| k * v / (2. * sigma));
                    let arg_prime = y_prime.mapv(|v| v / (-k * sigma))
                        - diff_y.to_owned() * sig_prime.mapv(|v| v / (k * sigma * k * sigma));
                    let factor = k * sigma * (-arg / SQRT_2PI) * (-(arg * arg) / 2.).exp();

                    let arg1 = y_prime.mapv(|v| v * (-norm_cdf(arg)));
                    let arg2 = diff_y * norm_pdf(arg) * arg_prime.to_owned();
                    let arg3 = sig_prime.to_owned() * norm_pdf(arg);
                    let arg4 = factor * arg_prime;
                    arg1 + arg2 + arg3 + arg4
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

    /// Compute LogEI infill criterion at given `x` point using the surrogate model `obj_model`
    /// and the current minimum of the objective function.
    fn value(
        &self,
        x: &[f64],
        obj_model: &dyn MixtureGpSurrogate,
        fmin: f64,
        _sigma_weight: Option<f64>,
        _scale: Option<f64>,
    ) -> f64 {
        let pt = ArrayView::from_shape((1, x.len()), x).unwrap();

        match obj_model.predict_valvar(&pt) {
            Ok((p, s)) => {
                if s[0] < f64::EPSILON {
                    f64::MIN
                } else {
                    let pred = p[0];
                    let sigma = s[0].sqrt();
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
        _sigma_weight: Option<f64>,
        _scale: Option<f64>,
    ) -> Array1<f64> {
        let pt = ArrayView::from_shape((1, x.len()), x).unwrap();

        match obj_model.predict_valvar(&pt) {
            Ok((p, s)) => {
                if s[0] < f64::EPSILON {
                    Array1::from_elem(pt.len(), f64::MIN)
                } else {
                    let pred = p[0];
                    let diff_y = fmin - pred;
                    let sigma = s[0].sqrt();
                    let arg = diff_y / sigma;

                    let (y_prime, var_prime) = obj_model.predict_valvar_gradients(&pt).unwrap();
                    let y_prime = y_prime.row(0);
                    let sig_2_prime = var_prime.row(0);
                    let sig_prime = sig_2_prime.mapv(|v| v / (2. * sigma));

                    let arg_prime = y_prime.mapv(|v| v / (-sigma))
                        - diff_y.to_owned() * sig_prime.mapv(|v| v / (sigma * sigma));

                    let dhelper = d_log_ei_helper(arg);
                    let arg1 = arg_prime.mapv(|v| dhelper * v);

                    let arg2 = sig_prime / sigma;
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
    use egobox_moe::{MixintContext, MoeBuilder};
    use approx::assert_abs_diff_eq;
    // use egobox_moe::GpSurrogate;
    use finitediff::vec;
    use linfa::Dataset;
    use ndarray::{Array2, ArrayView2, array};
    use ndarray_npy::write_npy;
    // use ndarray_npy::write_npy;

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
        let grad = EI.grad(&x, &mixi_moe, 0., Some(0.75), None);

        let f = |x: &Vec<f64>| -> std::result::Result<f64, anyhow::Error> {
            Ok(EI.value(x, &mixi_moe, 0., Some(0.75), None))
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
        write_npy("logei_x.npy", &x).expect("save x");

        let grad = x.mapv(|v| LOG_EI.grad(&[v], &mixi_moe, 0., None, None)[0]);
        write_npy("logei_grad.npy", &grad).expect("save grad log ei");

        let f = |x: &Vec<f64>| -> std::result::Result<f64, anyhow::Error> {
            Ok(LOG_EI.value(x, &mixi_moe, 0., None, None))
        };
        let grad_central = x.mapv(|v| vec::central_diff(&f)(&vec![v]).unwrap()[0]);
        write_npy("logei_fdiff.npy", &grad_central).expect("save fdiff log ei");

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
    fn test_d_log_ei() {
        let x = Array1::linspace(-10., 10., 100);
        write_npy("logei_x.npy", &x).expect("save x");

        let fx = x.mapv(log_ei_helper);
        write_npy("logei_fx.npy", &fx).expect("save fx");

        let dfx = x.mapv(d_log_ei_helper);
        write_npy("logei_dfx.npy", &dfx).expect("save dfx");

        let gradfx = x.mapv(|x| finite_diff_log_ei(x, 1e-6));
        write_npy("logei_gradfx.npy", &gradfx).expect("save dfx");
    }

    fn finite_diff_log_ei(u: f64, eps: f64) -> f64 {
        (log_ei_helper(u + eps) - log_ei_helper(u - eps)) / (2.0 * eps)
    }
}
