//! Types for mixed-integer optimization

#[cfg(feature = "serializable")]
use serde::{Deserialize, Serialize};

/// An enumeration to define the type of an input variable component
/// with its domain definition
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serializable", derive(Serialize, Deserialize))]
pub enum XType {
    /// Continuous variable in [lower bound, upper bound]
    Float(f64, f64),
    /// Integer variable in lower bound .. upper bound
    Int(i32, i32),
    /// An Ordered variable in { float_1, float_2, ..., float_n }
    Ord(Vec<f64>),
    /// An Enum variable in { 1, 2, ..., int_n }
    Enum(usize),
}

/// Check if xtypes contains discrete variables (Int, Ord, or Enum)
pub fn discrete(xtypes: &[XType]) -> bool {
    xtypes
        .iter()
        .any(|t| matches!(t, &XType::Int(_, _) | &XType::Ord(_) | &XType::Enum(_)))
}
