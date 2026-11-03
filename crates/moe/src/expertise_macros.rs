#[doc(hidden)]
// Create a GP with given regression and correlation models.
macro_rules! make_gp_params {
    ($regr:ident, $corr:ident) => {
        paste! {
            GaussianProcess::<f64, [<$regr Mean>], [<$corr Corr>] >::params(
                [<$regr Mean>]::default(),
                [<$corr Corr>]::default(),
            )
        }
    };
}

macro_rules! compute_error {
    ($self:ident, $regr:ident, $corr:ident, $dataset:ident) => {{
        debug!(
            "Surrogate {}_{} on dataset size = {}",
            stringify!($regr),
            stringify!($corr),
            $dataset.nsamples()
        );
        let params = make_gp_params!($regr, $corr)
            .kpls_dim($self.kpls_dim())
            .noise_ratio($self.noise_ratio().clone());
        let mut errors = Vec::new();
        let input_dim = $dataset.records().shape()[1];
        let n_fold = std::cmp::min($dataset.nsamples(), 5);
        trace!("Cross validation N fold = {n_fold}");
        if ($dataset.nsamples() < 4 * input_dim && stringify!($regr) == "Quadratic") {
            f64::INFINITY // not enough points => huge error
        } else if ($dataset.nsamples() < 3 * input_dim && stringify!($regr) == "Linear") {
            f64::INFINITY // not enough points => huge error
        } else {
            for (gp, valid) in $dataset.iter_fold(n_fold, |train| {
                params.clone().kpls_dim($self.kpls_dim()).fit(&train)
            }) {
                // A fit failure (e.g. not enough training points for the regression
                // terms of the fold) disqualifies the surrogate => huge error
                let error = match gp {
                    Ok(gp) => (valid.targets() - gp.predict(valid.records()).unwrap()).norm_l2(),
                    Err(err) => {
                        trace!("GP training failed: {err}");
                        f64::INFINITY
                    }
                };
                trace!("Prediction error = {error}");
                errors.push(error);
            }
            let mean_err = errors.iter().sum::<f64>() / errors.len() as f64;
            trace!("-> mean error = {}", mean_err);
            mean_err
        }
    }};
}

macro_rules! compute_errors_with_corr {
    ($self:ident, $allowed_corr_models:ident, $dataset:ident, $map_error:ident, $regr:ident, $corr:ident) => {{
        if $allowed_corr_models.contains(&stringify!($corr)) {
            $map_error.push((
                format!("{}_{}", stringify!($regr), stringify!($corr)),
                compute_error!($self, $regr, $corr, $dataset),
            ));
        }
    }};
}

macro_rules! compute_errors_with_regr {
    ($self:ident, $allowed_mean_models:ident, $allowed_corr_models:ident, $dataset:ident, $map_error:ident, $regr:ident) => {{
        if $allowed_mean_models.contains(&stringify!($regr)) {
            compute_errors_with_corr!(
                $self,
                $allowed_corr_models,
                $dataset,
                $map_error,
                $regr,
                SquaredExponential
            );
            compute_errors_with_corr!(
                $self,
                $allowed_corr_models,
                $dataset,
                $map_error,
                $regr,
                AbsoluteExponential
            );
            compute_errors_with_corr!(
                $self,
                $allowed_corr_models,
                $dataset,
                $map_error,
                $regr,
                Matern32
            );
            compute_errors_with_corr!(
                $self,
                $allowed_corr_models,
                $dataset,
                $map_error,
                $regr,
                Matern52
            );
        }
    }};
}

macro_rules! compute_errors {
    ($self:ident, $allowed_mean_models:ident, $allowed_corr_models:ident, $dataset:ident, $map_error:ident) => {{
        compute_errors_with_regr!(
            $self,
            $allowed_mean_models,
            $allowed_corr_models,
            $dataset,
            $map_error,
            Constant
        );
        compute_errors_with_regr!(
            $self,
            $allowed_mean_models,
            $allowed_corr_models,
            $dataset,
            $map_error,
            Linear
        );
        compute_errors_with_regr!(
            $self,
            $allowed_mean_models,
            $allowed_corr_models,
            $dataset,
            $map_error,
            Quadratic
        );
    }};
}

pub(crate) use compute_error;
pub(crate) use compute_errors;
pub(crate) use compute_errors_with_corr;
pub(crate) use compute_errors_with_regr;
pub(crate) use make_gp_params;
