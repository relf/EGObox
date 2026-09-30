//! Helpers to deprecate Python API names (kwargs, config fields, methods, dict keys)
//! while keeping them working during a transition period.
use pyo3::exceptions::{PyDeprecationWarning, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Version since which the names handled with this module are deprecated
const DEPRECATED_SINCE: &str = "0.38.0";

/// Emit a `DeprecationWarning` telling to use `new` instead of `old`.
///
/// Stack level 1 points at the user's Python line as native frames do not count.
pub(crate) fn warn_deprecated(py: Python, old: &str, new: &str) -> PyResult<()> {
    let msg = format!(
        "`{old}` is deprecated since {DEPRECATED_SINCE} and will be removed in a future release, use `{new}` instead"
    );
    let category = py.get_type::<PyDeprecationWarning>();
    PyErr::warn(py, &category, &std::ffi::CString::new(msg)?, 1)
}

/// Resolve a value given either under its `new_name` or its deprecated `old_name`.
///
/// Returns `TypeError` if both are given, warns if only the old one is given,
/// otherwise returns the new value or `default`.
pub(crate) fn resolve_renamed<T>(
    py: Python,
    old_name: &str,
    old: Option<T>,
    new_name: &str,
    new: Option<T>,
    default: T,
) -> PyResult<T> {
    match (old, new) {
        (Some(_), Some(_)) => Err(PyTypeError::new_err(format!(
            "`{old_name}` and `{new_name}` cannot be both given, `{old_name}` is deprecated, use `{new_name}` only"
        ))),
        (Some(old), None) => {
            warn_deprecated(py, old_name, new_name)?;
            Ok(old)
        }
        (None, new) => Ok(new.unwrap_or(default)),
    }
}

/// Resolve the dict key to be used for a config field given in dict form,
/// either under its `new` key or its deprecated `old` key.
///
/// Returns `TypeError` if both keys are present, warns if `key` is the old one,
/// otherwise returns `key` unchanged.
pub(crate) fn resolve_renamed_key<'k>(
    dict: &Bound<'_, PyDict>,
    key: &'k str,
    renamed: &[(&str, &'k str)],
) -> PyResult<&'k str> {
    for (old, new) in renamed {
        if key == *old {
            if dict.contains(*new)? {
                return Err(PyTypeError::new_err(format!(
                    "keys '{old}' and '{new}' cannot be both given, '{old}' is deprecated, use '{new}' only"
                )));
            }
            warn_deprecated(dict.py(), &format!("'{old}' key"), &format!("'{new}' key"))?;
            return Ok(new);
        }
    }
    Ok(key)
}
