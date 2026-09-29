use crate::types::{XSpec, XType};
use numpy::{PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

#[derive(FromPyObject)]
pub(crate) enum Domain<'py> {
    Xlists(Vec<Vec<f64>>),
    Xrows(PyReadonlyArray2<'py, f64>),
    Xspecs(Vec<XSpec>),
}

impl Domain<'_> {
    /// Returns true if the domain is empty.
    pub fn is_empty(&self) -> bool {
        match self {
            Domain::Xlists(v) => v.is_empty(),
            Domain::Xrows(arr) => arr.shape()[0] == 0 || arr.shape()[1] == 0,
            Domain::Xspecs(v) => v.is_empty(),
        }
    }
}

/// Translate Python domain specifications into a vector of `XType`
pub(crate) fn parse(py: Python, xspecs: Py<PyAny>) -> PyResult<Vec<egobox_moe::XType>> {
    let domain: Domain = xspecs.extract(py).map_err(|_| {
        PyTypeError::new_err(
            "xspecs should be a list of XSpec, a list of [lower, upper] float bounds \
             or a float array of shape (nx, 2)",
        )
    })?;
    if domain.is_empty() {
        return Err(PyValueError::new_err("xspecs should not be empty"));
    }

    match domain {
        Domain::Xspecs(xspecs) => xtypes_from_xspecs(xspecs),
        Domain::Xrows(xlimits) => xtypes_from_ndarray(xlimits),
        Domain::Xlists(floats) => xtypes_from_floats(floats),
    }
}

fn float_bounds(i: usize, bounds: &[f64]) -> PyResult<egobox_moe::XType> {
    match bounds {
        [lower, upper] => Ok(egobox_moe::XType::Float(*lower, *upper)),
        _ => Err(PyValueError::new_err(format!(
            "xspecs[{i}]: float bounds should be [lower, upper], got {bounds:?}"
        ))),
    }
}

fn xtypes_from_floats(floats: Vec<Vec<f64>>) -> PyResult<Vec<egobox_moe::XType>> {
    floats
        .iter()
        .enumerate()
        .map(|(i, v)| float_bounds(i, v))
        .collect()
}

fn xtypes_from_ndarray(xlimits: PyReadonlyArray2<f64>) -> PyResult<Vec<egobox_moe::XType>> {
    let ary = xlimits.as_array();
    if ary.ncols() != 2 {
        return Err(PyValueError::new_err(format!(
            "xspecs as array should be of shape (nx, 2), got {:?}",
            ary.shape()
        )));
    }
    Ok(ary
        .outer_iter()
        .map(|row| egobox_moe::XType::Float(row[0], row[1]))
        .collect())
}

fn xtypes_from_xspecs(xspecs: Vec<XSpec>) -> PyResult<Vec<egobox_moe::XType>> {
    xspecs
        .iter()
        .enumerate()
        .map(|(i, spec)| match spec.xtype {
            XType::Float => float_bounds(i, &spec.xlimits),
            XType::Int => match spec.xlimits[..] {
                [lower, upper] => Ok(egobox_moe::XType::Int(lower as i32, upper as i32)),
                _ => Err(PyValueError::new_err(format!(
                    "xspecs[{i}]: INT xlimits should be [lower, upper], got {:?}",
                    spec.xlimits
                ))),
            },
            XType::Ord => {
                if spec.xlimits.is_empty() {
                    Err(PyValueError::new_err(format!(
                        "xspecs[{i}]: ORD xlimits should list at least one value"
                    )))
                } else {
                    Ok(egobox_moe::XType::Ord(spec.xlimits.clone()))
                }
            }
            XType::Enum => {
                if !spec.tags.is_empty() {
                    Ok(egobox_moe::XType::Enum(spec.tags.len()))
                } else if let Some(&n) = spec.xlimits.first() {
                    Ok(egobox_moe::XType::Enum(n as usize))
                } else {
                    Err(PyValueError::new_err(format!(
                        "xspecs[{i}]: ENUM requires either tags or xlimits=[size]"
                    )))
                }
            }
        })
        .collect()
}
