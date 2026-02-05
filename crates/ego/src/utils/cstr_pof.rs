use std::iter::zip;

use crate::utils::{norm_cdf, norm_pdf};
use egobox_moe::MixtureGpSurrogate;
use ndarray::{Array1, ArrayView};

/// Compute probability of feasibility using the given surrogate of
/// a constraint function wrt the tolerance (ie cstr <= cstr_tol)
fn pof(x: &[f64], cstr_model: &dyn MixtureGpSurrogate, cstr_tol: f64) -> f64 {
    let pt = ArrayView::from_shape((1, x.len()), x).unwrap();
    match cstr_model.predict_valvar(&pt) {
        Ok((p, s)) => {
            if s[0] < f64::EPSILON {
                0.0
            } else {
                let pred = p[0];
                let sigma = s[0].sqrt();
                let args0 = (cstr_tol - pred) / sigma;
                norm_cdf(args0)
            }
        }
        _ => 0.0,
    }
}

/// Computes the derivative of the probability of feasibility of the given
/// constraint surrogate model.
fn pof_grad(x: &[f64], cstr_model: &dyn MixtureGpSurrogate, cstr_tol: f64) -> Array1<f64> {
    let pt = ArrayView::from_shape((1, x.len()), x).unwrap();
    match cstr_model.predict_valvar(&pt) {
        Ok((p, s)) => {
            if s[0] < f64::EPSILON {
                Array1::zeros(pt.len())
            } else {
                let pred = p[0];
                let sigma = s[0].sqrt();
                let arg = (cstr_tol - pred) / sigma;
                let (y_prime, var_prime) = cstr_model.predict_valvar_gradients(&pt).unwrap();
                let y_prime = y_prime.row(0);
                let sig_2_prime = var_prime.row(0);
                let sig_prime = sig_2_prime.mapv(|v| v / (2. * sigma));
                let arg_prime =
                    y_prime.mapv(|v| v / (-sigma)) + sig_prime.mapv(|v| v * pred / (sigma * sigma));
                norm_pdf(arg) * arg_prime.to_owned()
            }
        }
        _ => Array1::zeros(pt.len()),
    }
}

pub fn pofs(x: &[f64], cstr_models: &[Box<dyn MixtureGpSurrogate>], cstr_tols: &[f64]) -> f64 {
    if cstr_models.is_empty() {
        1.
    } else {
        zip(cstr_models, cstr_tols).fold(1., |acc, (cstr_model, cstr_tol)| {
            let pof = pof(x, &**cstr_model, *cstr_tol);
            acc * pof
        })
    }
}

pub fn logpofs(x: &[f64], cstr_models: &[Box<dyn MixtureGpSurrogate>], cstr_tols: &[f64]) -> f64 {
    if cstr_models.is_empty() {
        0.
    } else {
        zip(cstr_models, cstr_tols).fold(0., |acc, (cstr_model, cstr_tol)| {
            let logpof = (pof(x, &**cstr_model, *cstr_tol).max(f64::EPSILON)).ln();
            acc + logpof
        })
    }
}

pub fn pofs_grad(
    x: &[f64],
    cstr_models: &[Box<dyn MixtureGpSurrogate>],
    cstr_tols: &[f64],
) -> Array1<f64> {
    if cstr_models.is_empty() {
        Array1::zeros(x.len())
    } else {
        let pof_vals = zip(cstr_models, cstr_tols)
            .map(|(cstr_model, cstr_tol)| pof(x, &**cstr_model, *cstr_tol))
            .collect::<Vec<_>>();

        let pof_grads = zip(cstr_models, cstr_tols)
            .map(|(cstr_model, cstr_tol)| pof_grad(x, &**cstr_model, *cstr_tol))
            .collect::<Vec<_>>();

        zip(pof_vals.clone(), pof_grads).enumerate().fold(
            Array1::zeros(x.len()),
            |acc, (i, (_, pof_grad_i))| {
                let pof_others_product = pof_vals
                    .iter()
                    .enumerate()
                    .filter(|(j, _)| *j != i)
                    .fold(1., |acc_j, (_, pof_j)| acc_j * pof_j);
                acc + pof_grad_i * pof_others_product
            },
        )
    }
}

pub fn logpof_grad(x: &[f64], cstr_model: &dyn MixtureGpSurrogate, cstr_tol: f64) -> Array1<f64> {
    let num = pof_grad(x, cstr_model, cstr_tol);
    let denom = pof(x, cstr_model, cstr_tol).max(f64::EPSILON);
    num / denom
}

pub fn logpofs_grad(
    x: &[f64],
    cstr_models: &[Box<dyn MixtureGpSurrogate>],
    cstr_tols: &[f64],
) -> Array1<f64> {
    zip(cstr_models, cstr_tols).fold(Array1::zeros(x.len()), |acc, (cstr_model, cstr_tol)| {
        let logpof_grad = logpof_grad(x, &**cstr_model, *cstr_tol);
        acc + logpof_grad
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::*;
    use egobox_moe::{MixintContext, MoeBuilder};
    use approx::assert_abs_diff_eq;
    use finitediff::vec;
    use linfa::Dataset;
    use ndarray::array;

    #[test]
    fn test_pof_gradients() {
        let xtypes = vec![XType::Float(0., 25.)];

        let mixi = MixintContext::new(&xtypes);

        let surrogate_builder = MoeBuilder::new();
        let xt = array![[0.], [2.], [5.], [10.], [25.]];
        let yt = array![0., 0.2, -0.3, 0.5, -1.];
        let ds = Dataset::new(xt, yt);
        let mixi_moe = mixi
            .create_surrogate(&surrogate_builder, &ds)
            .expect("Mixint surrogate creation");

        let x = vec![0.3];
        let grad = pof_grad(&x, &mixi_moe, 0.);

        let f =
            |x: &Vec<f64>| -> std::result::Result<f64, anyhow::Error> { Ok(pof(x, &mixi_moe, 0.)) };
        let grad_central = vec::central_diff(&f)(&x).unwrap();

        assert_abs_diff_eq!(grad[0], grad_central[0], epsilon = 1e-6);
    }

    #[test]
    fn test_pofs_grad() {
        let xtypes = vec![XType::Float(0., 25.)];

        let mixi = MixintContext::new(&xtypes);

        let surrogate_builder = MoeBuilder::new();
        let xt = array![[0.], [2.], [5.], [10.], [25.]];
        let yt1 = array![0., 0.2, -0.3, 0.5, -1.];
        let yt2 = array![1., -0.5, 0.3, -0.2, 0.8];
        // let yt2 = array![0., 0.2, -0.3, 0.5, -1.];
        let ds1 = Dataset::new(xt.clone(), yt1);
        let ds2 = Dataset::new(xt, yt2);
        let mixi_moe1 = mixi
            .create_surrogate(&surrogate_builder, &ds1)
            .expect("Mixint surrogate creation");
        let mixi_moe2 = mixi
            .create_surrogate(&surrogate_builder, &ds2)
            .expect("Mixint surrogate creation");

        let cstr_models: Vec<Box<dyn MixtureGpSurrogate>> =
            vec![Box::new(mixi_moe1), Box::new(mixi_moe2)];
        let cstr_tols = vec![0., 0.];

        let x = vec![0.3];
        let grad = pofs_grad(&x, &cstr_models, &cstr_tols);

        let f = |x: &Vec<f64>| -> std::result::Result<f64, anyhow::Error> {
            Ok(pofs(x, &cstr_models, &cstr_tols))
        };
        let grad_central = vec::central_diff(&f)(&x).unwrap();

        let term1 =
            pof_grad(&x, &*cstr_models[0], cstr_tols[0]) * pof(&x, &*cstr_models[1], cstr_tols[1]);
        let term2 =
            pof_grad(&x, &*cstr_models[1], cstr_tols[1]) * pof(&x, &*cstr_models[0], cstr_tols[0]);
        let expected = term1 + term2;
        println!("expected = {expected:?}");
        println!("grad = {grad:?}");
        println!("grad_central = {grad_central:?}");
        assert_abs_diff_eq!(grad[0], grad_central[0], epsilon = 1e-6);
    }
}
