//! Method bodies shared by the `Gpx` and `SparseGpx` Python classes.
//!
//! Each function implements the Python method of the same name
//! for any trained mixture of Gaussian processes.
use crate::errors::{gp_file_format, input_x, moe_err, moe_file_err};
use egobox_moe::MixtureGpSurrogate;
use ndarray::{Array1, Array2, Zip};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArrayDyn};
use pyo3::prelude::*;

pub(crate) fn save(model: &dyn MixtureGpSurrogate, filename: &str) -> PyResult<bool> {
    model
        .save(filename, gp_file_format(filename))
        .map_err(|e| moe_file_err(e, filename))?;
    Ok(true)
}

pub(crate) fn predict<'py>(
    model: &dyn MixtureGpSurrogate,
    py: Python<'py>,
    x: PyReadonlyArrayDyn<f64>,
    return_std: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let x = input_x(x.as_array(), model.dims().0)?;
    let mean = model.predict(&x).map_err(moe_err)?.into_pyarray(py);
    if return_std {
        let std = model
            .predict_var(&x)
            .map_err(moe_err)?
            .mapv(|v| v.max(0.).sqrt())
            .into_pyarray(py);
        Ok((mean, std).into_pyobject(py)?.into_any())
    } else {
        Ok(mean.into_any())
    }
}

pub(crate) fn predict_var<'py>(
    model: &dyn MixtureGpSurrogate,
    py: Python<'py>,
    x: PyReadonlyArrayDyn<f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let x = input_x(x.as_array(), model.dims().0)?;
    Ok(model.predict_var(&x).map_err(moe_err)?.into_pyarray(py))
}

pub(crate) fn predict_gradients<'py>(
    model: &dyn MixtureGpSurrogate,
    py: Python<'py>,
    x: PyReadonlyArrayDyn<f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let x = input_x(x.as_array(), model.dims().0)?;
    Ok(model
        .predict_gradients(&x)
        .map_err(moe_err)?
        .into_pyarray(py))
}

pub(crate) fn predict_var_gradients<'py>(
    model: &dyn MixtureGpSurrogate,
    py: Python<'py>,
    x: PyReadonlyArrayDyn<f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let x = input_x(x.as_array(), model.dims().0)?;
    Ok(model
        .predict_var_gradients(&x)
        .map_err(moe_err)?
        .into_pyarray(py))
}

pub(crate) fn sample<'py>(
    model: &dyn MixtureGpSurrogate,
    py: Python<'py>,
    x: PyReadonlyArrayDyn<f64>,
    n_traj: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let x = input_x(x.as_array(), model.dims().0)?;
    Ok(model.sample(&x, n_traj).map_err(moe_err)?.into_pyarray(py))
}

pub(crate) fn training_data<'py>(
    model: &dyn MixtureGpSurrogate,
    py: Python<'py>,
) -> (Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>) {
    let (xdata, ydata) = model.training_data();
    (
        xdata.to_owned().into_pyarray(py),
        ydata.to_owned().into_pyarray(py),
    )
}

pub(crate) fn thetas<'py>(
    model: &dyn MixtureGpSurrogate,
    py: Python<'py>,
) -> Bound<'py, PyArray2<f64>> {
    let experts = model.experts();
    let proto = experts.first().expect("Mixture should contain an expert");
    let mut thetas = Array2::zeros((model.n_clusters(), proto.theta().len()));
    Zip::from(thetas.rows_mut())
        .and(experts)
        .for_each(|mut theta, expert| theta.assign(expert.theta()));
    thetas.into_pyarray(py)
}

pub(crate) fn variances<'py>(
    model: &dyn MixtureGpSurrogate,
    py: Python<'py>,
) -> Bound<'py, PyArray1<f64>> {
    let experts = model.experts();
    let mut variances = Array1::zeros(model.n_clusters());
    Zip::from(&mut variances)
        .and(experts)
        .for_each(|var, expert| *var = expert.variance());
    variances.into_pyarray(py)
}

pub(crate) fn likelihoods<'py>(
    model: &dyn MixtureGpSurrogate,
    py: Python<'py>,
) -> Bound<'py, PyArray1<f64>> {
    let experts = model.experts();
    let mut likelihoods = Array1::zeros(model.n_clusters());
    Zip::from(&mut likelihoods)
        .and(experts)
        .for_each(|lkh, expert| *lkh = expert.likelihood());
    likelihoods.into_pyarray(py)
}
