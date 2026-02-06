//! Mixed-integer variable types
//!
//! This module defines the `XType` enum for specifying variable domains
//! and helper functions for working with mixed-integer spaces.

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

/// Returns true if xtypes contains at least one discrete type (Int, Ord, or Enum)
pub fn discrete(xtypes: &[XType]) -> bool {
    xtypes
        .iter()
        .any(|t| matches!(t, &XType::Int(_, _) | &XType::Ord(_) | &XType::Enum(_)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_discrete_with_float_only() {
        let xtypes = vec![XType::Float(0.0, 1.0), XType::Float(-1.0, 1.0)];
        assert!(!discrete(&xtypes));
    }

    #[test]
    fn test_discrete_with_int() {
        let xtypes = vec![XType::Float(0.0, 1.0), XType::Int(0, 10)];
        assert!(discrete(&xtypes));
    }

    #[test]
    fn test_discrete_with_ord() {
        let xtypes = vec![XType::Ord(vec![1.0, 2.0, 3.0])];
        assert!(discrete(&xtypes));
    }

    #[test]
    fn test_discrete_with_enum() {
        let xtypes = vec![XType::Float(0.0, 1.0), XType::Enum(3)];
        assert!(discrete(&xtypes));
    }
}
