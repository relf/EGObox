//! Hierarchical design spaces
//!
//! In a hierarchical design space, some variables are *included* only for
//! some values of other variables, and *excluded* otherwise. For instance, the
//! radius of the second propeller of a motor only exists when that motor has
//! two propellers.
//!
//! A [`DesignSpace`] holds the type of each variable ([`XType`]) and, for each
//! *decreed* variable, the [`Condition`] under which it is included. Conditions
//! refer to *meta* variables, which may themselves be decreed: together they
//! form the role graph, which must be acyclic.
//!
//! Variables are given by name or by index ([`Var`]). Names are optional but
//! preferred: a design space built with [`DesignSpace::named`] accepts them
//! wherever a variable is expected, and its error messages show them.
//!
//! Points keep a fixed size, with one entry for every variable. The correction
//! ([`DesignSpace::correct_get_included`]) casts each variable to a value of its
//! type, works out which variables are included, and sets excluded variables to
//! a fixed default value, so that points that only differ in excluded variables
//! become identical.
//!
//! # Reference
//!
//! Saves, P., Hallé-Hannan, E., Bussemaker, J., Diouane, Y., Bartoli, N.
//! (2026). Modeling hierarchical spaces: a review and unified framework for
//! surrogate-based architecture design. Structural and Multidisciplinary
//! Optimization, 69, 65. <https://doi.org/10.1007/s00158-026-04249-2>

use crate::errors::{MoeError, Result};
use crate::mixint::space::{
    cast_variable_mut, check_enum_indices, fold_with_enum_index, unfold_with_enum_mask,
    unfolded_width,
};
use crate::xtypes::XType;
use bitflags::bitflags;
use ndarray::{Array2, ArrayBase, ArrayView1, ArrayViewMut1, Data, Ix2, s};
use ndarray_stats::QuantileExt;
use serde::{Deserialize, Serialize};
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashSet};

bitflags! {
    /// Roles of a variable in a [`DesignSpace`].
    ///
    /// Roles are flags that can be combined: a variable that decrees other
    /// variables while being decreed itself is both [`META`](Role::META) and
    /// [`DECREED`](Role::DECREED). A variable with no role
    /// ([`is_empty()`](Role::is_empty)) is *neutral*: it is always included
    /// and decrees no other variable.
    #[derive(PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Clone, Copy)]
    pub struct Role: u8 {
        /// Decrees the inclusion of other variables
        const META = 1;
        /// Included only for some values of other variables
        const DECREED = 1 << 1;
    }
}

/// A variable of a [`DesignSpace`], given by index or by name.
///
/// Methods that take a variable accept `impl Into<Var>`: an index (`usize`)
/// or a name (`&str` or `&String`). Names only work with a design space built
/// with [`DesignSpace::named`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Var<'a> {
    /// Index of the variable
    Index(usize),
    /// Name of the variable
    Name(&'a str),
}

impl From<usize> for Var<'_> {
    fn from(index: usize) -> Self {
        Var::Index(index)
    }
}

impl<'a> From<&'a str> for Var<'a> {
    fn from(name: &'a str) -> Self {
        Var::Name(name)
    }
}

impl<'a> From<&'a String> for Var<'a> {
    fn from(name: &'a String) -> Self {
        Var::Name(name)
    }
}

/// Condition under which a decreed variable is included.
///
/// Values are given in folded space: integers for `Int` variables, levels
/// for `Ord` variables, and level indices for `Enum` variables. Only discrete
/// variables can appear in a condition.
///
/// [`And`](Condition::And) and [`Or`](Condition::Or) combine conditions:
/// they are the intermediate nodes of the role graph.
///
/// Conditions refer to variables by index. [`DesignSpace::is_in`] builds a
/// condition from a variable name.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Condition {
    /// Holds when the variable `var` is included and takes one of `values`
    In {
        /// Index of the meta variable
        var: usize,
        /// Values of the meta variable for which the condition holds
        values: Vec<f64>,
    },
    /// Holds when all the conditions hold
    And(Vec<Condition>),
    /// Holds when at least one of the conditions holds
    Or(Vec<Condition>),
}

impl Condition {
    /// Condition that holds when the variable of index `var` is included and
    /// takes one of `values`
    pub fn is_in(var: usize, values: &[f64]) -> Self {
        Condition::In {
            var,
            values: values.to_vec(),
        }
    }

    /// Sorted indices of the variables the condition depends on
    fn vars(&self) -> Vec<usize> {
        fn collect(c: &Condition, out: &mut Vec<usize>) {
            match c {
                Condition::In { var, .. } => out.push(*var),
                Condition::And(cs) | Condition::Or(cs) => cs.iter().for_each(|c| collect(c, out)),
            }
        }
        let mut vars = vec![];
        collect(self, &mut vars);
        vars.sort_unstable();
        vars.dedup();
        vars
    }

    /// Whether the condition holds for a point, given the folded values and
    /// the inclusion of the variables it depends on
    fn holds(&self, folded: ArrayView1<f64>, included: ArrayView1<bool>) -> bool {
        match self {
            Condition::In { var, values } => included[*var] && values.contains(&folded[*var]),
            Condition::And(cs) => cs.iter().all(|c| c.holds(folded, included)),
            Condition::Or(cs) => cs.iter().any(|c| c.holds(folded, included)),
        }
    }

    /// Check the condition against the variable types, for the variable `decreed`
    fn validate(&self, xtypes: &[XType], names: Option<&[String]>, decreed: usize) -> Result<()> {
        let label = |i| label(names, i);
        match self {
            Condition::In { var, values } => {
                let xtype = xtypes.get(*var).ok_or_else(|| {
                    invalid(format!(
                        "meta variable {var} out of range: the design space has {} variables",
                        xtypes.len()
                    ))
                })?;
                if *var == decreed {
                    return Err(invalid(format!(
                        "variable {} cannot decree itself",
                        label(*var)
                    )));
                }
                if values.is_empty() {
                    return Err(invalid(format!(
                        "no value given for meta variable {}: variable {} would never be included",
                        label(*var),
                        label(decreed)
                    )));
                }
                for &v in values {
                    let valid = match xtype {
                        XType::Float(_, _) => {
                            return Err(invalid(format!(
                                "variable {} is continuous: only discrete variables (Int, Ord, Enum) can decree other variables",
                                label(*var)
                            )));
                        }
                        XType::Int(lb, ub) => v.fract() == 0. && v >= *lb as f64 && v <= *ub as f64,
                        XType::Ord(levels) => levels.contains(&v),
                        XType::Enum(n) => v.fract() == 0. && v >= 0. && v < *n as f64,
                    };
                    if !valid {
                        return Err(invalid(format!(
                            "{v} is not a value of meta variable {}: {xtype:?}",
                            label(*var)
                        )));
                    }
                }
                Ok(())
            }
            Condition::And(cs) | Condition::Or(cs) => {
                if cs.is_empty() {
                    return Err(invalid(format!(
                        "empty And/Or condition for variable {}",
                        label(decreed)
                    )));
                }
                cs.iter()
                    .try_for_each(|c| c.validate(xtypes, names, decreed))
            }
        }
    }
}

/// A hierarchical design space: variable types, optional variable names, and
/// the conditions under which decreed variables are included.
///
/// # Example
///
/// A wing carries `m` motors, `m` in {0, 1, 2}. Motor `i` has `p_i`
/// propellers, `p_i` in {1, 2}, and propeller `j` of motor `i` has a radius
/// `r_ij`.
///
/// ```
/// use egobox_moe::{DesignSpace, Role, XType};
/// use ndarray::array;
///
/// let r = XType::Float(0.5, 2.);
/// let mut ds = DesignSpace::named(&[
///     ("m", XType::Int(0, 2)),
///     ("p_1", XType::Int(1, 2)),
///     ("p_2", XType::Int(1, 2)),
///     ("r_11", r.clone()),
///     ("r_12", r.clone()),
///     ("r_21", r.clone()),
///     ("r_22", r),
/// ])?;
/// ds.decreed_var("p_1", "m", &[1., 2.])?
///     .decreed_var("p_2", "m", &[2.])?
///     .decreed_var("r_11", "p_1", &[1., 2.])?
///     .decreed_var("r_12", "p_1", &[2.])?
///     .decreed_var("r_21", "p_2", &[1., 2.])?
///     .decreed_var("r_22", "p_2", &[2.])?;
/// assert_eq!(ds.role("p_1"), Role::META | Role::DECREED);
///
/// // One motor with two propellers: the second motor is excluded
/// let x = array![[1., 2., 2., 1.2, 1.3, 1.4, 1.5]];
/// let (x, included) = ds.correct_get_included(&x)?;
/// assert_eq!(x, array![[1., 2., 1., 1.2, 1.3, 0.5, 0.5]]);
/// assert_eq!(included, array![[true, true, false, true, true, false, false]]);
/// # Ok::<(), egobox_moe::MoeError>(())
/// ```
///
/// Variables can also be given by index, which is the only way for a design
/// space built with [`DesignSpace::new`]: `ds.decreed_var(1, 0, &[1., 2.])`.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(try_from = "DesignSpaceSpec", into = "DesignSpaceSpec")]
pub struct DesignSpace {
    /// Type of each variable
    xtypes: Vec<XType>,
    /// Name of each variable, `None` when the variables are not named
    names: Option<Vec<String>>,
    /// Inclusion condition of each variable, `None` when always included
    conditions: Vec<Option<Condition>>,
    /// Variable indices in topological order of the role graph
    order: Vec<usize>,
}

/// Serialized form of a [`DesignSpace`], checked on deserialization
#[derive(Serialize, Deserialize)]
struct DesignSpaceSpec {
    xtypes: Vec<XType>,
    /// Can be left out in JSON when the variables are not named
    #[serde(default)]
    names: Option<Vec<String>>,
    conditions: Vec<Option<Condition>>,
}

impl From<DesignSpace> for DesignSpaceSpec {
    fn from(ds: DesignSpace) -> Self {
        DesignSpaceSpec {
            xtypes: ds.xtypes,
            names: ds.names,
            conditions: ds.conditions,
        }
    }
}

impl TryFrom<DesignSpaceSpec> for DesignSpace {
    type Error = MoeError;

    fn try_from(spec: DesignSpaceSpec) -> Result<Self> {
        if let Some(names) = &spec.names {
            check_names(names, spec.xtypes.len())?;
        }
        if spec.conditions.len() != spec.xtypes.len() {
            return Err(invalid(format!(
                "{} conditions given for {} variables",
                spec.conditions.len(),
                spec.xtypes.len()
            )));
        }
        let names = spec.names.as_deref();
        for (i, condition) in spec.conditions.iter().enumerate() {
            if let Some(condition) = condition {
                condition.validate(&spec.xtypes, names, i)?;
            }
        }
        let order = topological_order(&spec.conditions, names)?;
        Ok(DesignSpace {
            xtypes: spec.xtypes,
            names: spec.names,
            conditions: spec.conditions,
            order,
        })
    }
}

impl DesignSpace {
    /// A design space with the given variable types, where all variables are
    /// included (no hierarchy yet). Variables have no names and are given by
    /// index.
    pub fn new(xtypes: &[XType]) -> Self {
        DesignSpace {
            xtypes: xtypes.to_vec(),
            names: None,
            conditions: vec![None; xtypes.len()],
            order: (0..xtypes.len()).collect(),
        }
    }

    /// A design space with the given variable names and types, where all
    /// variables are included (no hierarchy yet). Variables can be given by
    /// name or by index.
    ///
    /// Fails when a name is empty or used twice.
    pub fn named<S: AsRef<str>>(vars: &[(S, XType)]) -> Result<Self> {
        let names: Vec<String> = vars.iter().map(|(n, _)| n.as_ref().to_string()).collect();
        check_names(&names, vars.len())?;
        let xtypes: Vec<XType> = vars.iter().map(|(_, t)| t.clone()).collect();
        Ok(DesignSpace {
            names: Some(names),
            ..DesignSpace::new(&xtypes)
        })
    }

    /// Declare that variable `decreed` is included only when variable `meta`
    /// takes one of `values`, and is itself included.
    ///
    /// Variables are given by name or by index (see [`Var`]). `meta` must be
    /// discrete (Int, Ord or Enum), and `values` are given in folded space
    /// (see [`Condition`]). Several declarations for the same decreed
    /// variable are combined with AND.
    ///
    /// Fails, leaving the design space unchanged, when a variable is unknown,
    /// when a value is invalid or when the declaration creates a cycle of
    /// decrees.
    pub fn decreed_var<'a>(
        &mut self,
        decreed: impl Into<Var<'a>>,
        meta: impl Into<Var<'a>>,
        values: &[f64],
    ) -> Result<&mut Self> {
        let condition = self.is_in(meta, values)?;
        self.decreed_var_when(decreed, condition)
    }

    /// Declare that variable `decreed` is included only when `condition` holds.
    ///
    /// `decreed` is given by name or by index (see [`Var`]); conditions on
    /// named variables are built with [`is_in`](DesignSpace::is_in). Several
    /// declarations for the same decreed variable are combined with AND.
    ///
    /// Fails, leaving the design space unchanged, when a variable is unknown,
    /// when a value is invalid or when the declaration creates a cycle of
    /// decrees.
    pub fn decreed_var_when<'a>(
        &mut self,
        decreed: impl Into<Var<'a>>,
        condition: Condition,
    ) -> Result<&mut Self> {
        let decreed = self.resolve(decreed.into())?;
        let names = self.names.as_deref();
        condition.validate(&self.xtypes, names, decreed)?;
        let mut conditions = self.conditions.clone();
        conditions[decreed] = Some(match conditions[decreed].take() {
            None => condition,
            Some(Condition::And(mut cs)) => {
                cs.push(condition);
                Condition::And(cs)
            }
            Some(previous) => Condition::And(vec![previous, condition]),
        });
        self.order = topological_order(&conditions, names)?;
        self.conditions = conditions;
        Ok(self)
    }

    /// Condition that holds when the variable `var` is included and takes one
    /// of `values`: [`Condition::is_in`] with the variable given by name or by
    /// index (see [`Var`]).
    ///
    /// Fails when the variable is unknown. The values are checked when the
    /// condition is declared.
    pub fn is_in<'a>(&self, var: impl Into<Var<'a>>, values: &[f64]) -> Result<Condition> {
        Ok(Condition::is_in(self.resolve(var.into())?, values))
    }

    /// Types of the variables
    pub fn xtypes(&self) -> &[XType] {
        &self.xtypes
    }

    /// Names of the variables, `None` when they are not named
    pub fn names(&self) -> Option<&[String]> {
        self.names.as_deref()
    }

    /// Name of the variable of index `var`, `None` when the variables are not
    /// named or `var` is out of range
    pub fn name(&self, var: usize) -> Option<&str> {
        self.names()?.get(var).map(String::as_str)
    }

    /// Index of the variable called `name`, `None` when there is none
    pub fn index_of(&self, name: &str) -> Option<usize> {
        self.names()?.iter().position(|n| n == name)
    }

    /// Inclusion condition of variable `var`, `None` when it is always included
    ///
    /// # Panics
    ///
    /// When `var` is out of range or an unknown name
    pub fn condition<'a>(&self, var: impl Into<Var<'a>>) -> Option<&Condition> {
        self.conditions[self.index(var.into())].as_ref()
    }

    /// Sorted indices of the variables whose values decide whether `var` is included
    ///
    /// # Panics
    ///
    /// When `var` is out of range or an unknown name
    pub fn parents<'a>(&self, var: impl Into<Var<'a>>) -> Vec<usize> {
        self.conditions[self.index(var.into())]
            .as_ref()
            .map_or(vec![], Condition::vars)
    }

    /// Roles of variable `var`
    ///
    /// # Panics
    ///
    /// When `var` is out of range or an unknown name
    pub fn role<'a>(&self, var: impl Into<Var<'a>>) -> Role {
        let var = self.index(var.into());
        let mut role = Role::empty();
        if self.conditions[var].is_some() {
            role |= Role::DECREED;
        }
        if self
            .conditions
            .iter()
            .flatten()
            .any(|c| c.vars().contains(&var))
        {
            role |= Role::META;
        }
        role
    }

    /// Variable indices in topological order of the role graph: each variable
    /// comes after the variables its inclusion depends on. Ties are broken by
    /// variable index.
    pub fn topological_order(&self) -> &[usize] {
        &self.order
    }

    /// Correct points given in folded space (one column per variable, enum
    /// variables given by their level index).
    ///
    /// Returns the corrected points and, for each point and each variable,
    /// whether the variable is included. Each variable is cast to the nearest
    /// value of its type (as [`to_discrete_space`](crate::to_discrete_space)
    /// does), and each excluded variable is set to a default value: the lower
    /// bound for Float and Int variables, the smallest level for Ord
    /// variables, and the first level for Enum variables.
    pub fn correct_get_included(
        &self,
        x: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    ) -> Result<(Array2<f64>, Array2<bool>)> {
        check_enum_indices(&self.xtypes, x)?;
        let xunfold = unfold_with_enum_mask(&self.xtypes, x);
        let (xcorr, included) = self.correct_get_included_unfolded(&xunfold)?;
        Ok((fold_with_enum_index(&self.xtypes, &xcorr), included))
    }

    /// Correct points given in the continuously relaxed space, where each enum
    /// variable takes one column per level
    /// (see [`to_continuous_space`](crate::to_continuous_space)).
    ///
    /// Same as [`correct_get_included`](DesignSpace::correct_get_included),
    /// except that the corrected points stay in the relaxed space, with one-hot
    /// enum columns. The inclusion mask has one column per variable.
    pub fn correct_get_included_unfolded(
        &self,
        x: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    ) -> Result<(Array2<f64>, Array2<bool>)> {
        let widths: Vec<usize> = self.xtypes.iter().map(unfolded_width).collect();
        let starts: Vec<usize> = widths
            .iter()
            .scan(0, |next, w| {
                let start = *next;
                *next += w;
                Some(start)
            })
            .collect();
        let dim: usize = widths.iter().sum();
        if x.ncols() != dim {
            return Err(invalid(format!(
                "input should have {dim} columns in the relaxed space, got {}",
                x.ncols()
            )));
        }

        let (n, nv) = (x.nrows(), self.xtypes.len());
        let mut xcorr = x.to_owned();
        let mut folded = Array2::<f64>::zeros((n, nv));
        let mut included = Array2::from_elem((n, nv), true);
        for &i in &self.order {
            let xtype = &self.xtypes[i];
            let mut block = xcorr.slice_mut(s![.., starts[i]..starts[i] + widths[i]]);
            cast_variable_mut(xtype, block.view_mut());
            if let Some(condition) = &self.conditions[i] {
                for r in 0..n {
                    if !condition.holds(folded.row(r), included.row(r)) {
                        included[[r, i]] = false;
                        set_excluded_default(xtype, block.row_mut(r));
                    }
                }
            }
            for (r, row) in block.rows().into_iter().enumerate() {
                folded[[r, i]] = match xtype {
                    XType::Enum(_) => row.argmax().unwrap() as f64,
                    _ => row[0],
                };
            }
        }
        Ok((xcorr, included))
    }

    /// Index of `var`. Fails when it is out of range or an unknown name.
    fn resolve(&self, var: Var) -> Result<usize> {
        let n = self.xtypes.len();
        match var {
            Var::Index(i) if i < n => Ok(i),
            Var::Index(i) => Err(invalid(format!(
                "variable {i} out of range: the design space has {n} variables"
            ))),
            Var::Name(name) if self.names.is_none() => Err(invalid(format!(
                "unknown variable `{name}`: the variables of the design space are not named"
            ))),
            Var::Name(name) => self
                .index_of(name)
                .ok_or_else(|| invalid(format!("unknown variable `{name}`"))),
        }
    }

    /// Index of `var`, for accessors that panic on an unknown variable
    fn index(&self, var: Var) -> usize {
        self.resolve(var).unwrap_or_else(|e| panic!("{e}"))
    }
}

/// Check that there is one name per variable, and that names are non-empty
/// and unique
fn check_names(names: &[String], nvars: usize) -> Result<()> {
    if names.len() != nvars {
        return Err(invalid(format!(
            "{} names given for {nvars} variables",
            names.len()
        )));
    }
    let mut seen = HashSet::new();
    for (i, name) in names.iter().enumerate() {
        if name.is_empty() {
            return Err(invalid(format!("variable {i} has an empty name")));
        }
        if !seen.insert(name.as_str()) {
            return Err(invalid(format!("variable name `{name}` is used twice")));
        }
    }
    Ok(())
}

/// Variable label for messages: `3 (r_11)` when the variables are named, `3`
/// otherwise
fn label(names: Option<&[String]>, var: usize) -> String {
    match names.and_then(|names| names.get(var)) {
        Some(name) => format!("{var} ({name})"),
        None => var.to_string(),
    }
}

/// Set the relaxed columns `row` of an excluded variable to its default value
fn set_excluded_default(xtype: &XType, mut row: ArrayViewMut1<f64>) {
    match xtype {
        XType::Float(lb, _) => row[0] = *lb,
        XType::Int(lb, _) => row[0] = *lb as f64,
        XType::Ord(levels) => row[0] = levels.iter().copied().fold(f64::INFINITY, f64::min),
        XType::Enum(_) => {
            row.fill(0.);
            row[0] = 1.;
        }
    }
}

/// Topological order of the role graph with Kahn's algorithm, breaking ties by
/// variable index. Fails when the graph has a cycle.
///
/// The variable indices in `conditions` must be in range.
fn topological_order(
    conditions: &[Option<Condition>],
    names: Option<&[String]>,
) -> Result<Vec<usize>> {
    let n = conditions.len();
    let mut children = vec![vec![]; n];
    let mut in_degree = vec![0; n];
    for (i, condition) in conditions.iter().enumerate() {
        let parents = condition.as_ref().map_or(vec![], Condition::vars);
        in_degree[i] = parents.len();
        for p in parents {
            children[p].push(i);
        }
    }
    let mut ready: BinaryHeap<Reverse<usize>> =
        (0..n).filter(|&i| in_degree[i] == 0).map(Reverse).collect();
    let mut order = Vec::with_capacity(n);
    while let Some(Reverse(i)) = ready.pop() {
        order.push(i);
        for &c in &children[i] {
            in_degree[c] -= 1;
            if in_degree[c] == 0 {
                ready.push(Reverse(c));
            }
        }
    }
    if order.len() < n {
        let blocked: Vec<String> = (0..n)
            .filter(|&i| in_degree[i] > 0)
            .map(|i| label(names, i))
            .collect();
        return Err(invalid(format!(
            "the decrees form a cycle: variables {} are in or depend on a cycle",
            blocked.join(", ")
        )));
    }
    Ok(order)
}

fn invalid(msg: String) -> MoeError {
    MoeError::InvalidValueError(msg)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// Motors and propellers (Fig. 1 of the reference): `m` motors on the wing,
    /// `p_i` propellers on motor `i`, radius `r_ij` of propeller `j` of motor `i`
    fn motors_and_propellers() -> DesignSpace {
        let r = XType::Float(0.5, 2.);
        let mut ds = DesignSpace::named(&[
            ("m", XType::Int(0, 2)),
            ("p_1", XType::Int(1, 2)),
            ("p_2", XType::Int(1, 2)),
            ("r_11", r.clone()),
            ("r_12", r.clone()),
            ("r_21", r.clone()),
            ("r_22", r),
        ])
        .unwrap();
        ds.decreed_var("p_1", "m", &[1., 2.])
            .unwrap()
            .decreed_var("p_2", "m", &[2.])
            .unwrap()
            .decreed_var("r_11", "p_1", &[1., 2.])
            .unwrap()
            .decreed_var("r_12", "p_1", &[2.])
            .unwrap()
            .decreed_var("r_21", "p_2", &[1., 2.])
            .unwrap()
            .decreed_var("r_22", "p_2", &[2.])
            .unwrap();
        ds
    }

    const FUEL: f64 = 0.;
    const ELECTRIC: f64 = 1.;
    const HYBRID: f64 = 2.;

    /// Hybrid-electric technology choice (Fig. 7 of the reference), with a
    /// discrete fuel reserve instead of the continuous threshold: the back-up
    /// battery is included when the energy source is electric, or when it is
    /// hybrid and the fuel reserve is small.
    fn hybrid_electric() -> DesignSpace {
        let mut ds = DesignSpace::named(&[
            ("source", XType::Enum(3)),
            ("reserve", XType::Ord(vec![100., 200., 300.])),
            ("battery", XType::Float(10., 50.)),
        ])
        .unwrap();
        let battery = Condition::Or(vec![
            ds.is_in("source", &[ELECTRIC]).unwrap(),
            Condition::And(vec![
                ds.is_in("source", &[HYBRID]).unwrap(),
                ds.is_in("reserve", &[100.]).unwrap(),
            ]),
        ]);
        ds.decreed_var("reserve", "source", &[FUEL, HYBRID])
            .unwrap()
            .decreed_var_when("battery", battery)
            .unwrap();
        ds
    }

    #[test]
    fn test_motors_and_propellers_roles() {
        let ds = motors_and_propellers();
        assert_eq!(ds.role(0), Role::META);
        assert_eq!(ds.role("p_1"), Role::META | Role::DECREED);
        assert_eq!(ds.role(2), Role::META | Role::DECREED);
        (3..7).for_each(|i| assert_eq!(ds.role(i), Role::DECREED));
        assert_eq!(ds.parents("r_12"), vec![1]);
        assert_eq!(ds.topological_order(), &[0, 1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn test_motors_and_propellers_correction() {
        let ds = motors_and_propellers();
        let x = array![
            [0.4, 2., 2., 1., 1., 1., 1.],
            [1., 1.7, 1., 1.2, 1.3, 1.4, 1.5],
            [1.6, 1., 2., 1.2, 1.3, 1.4, 1.5],
        ];
        let (xcorr, included) = ds.correct_get_included(&x).unwrap();
        assert_eq!(
            xcorr,
            array![
                [0., 1., 1., 0.5, 0.5, 0.5, 0.5],
                [1., 2., 1., 1.2, 1.3, 0.5, 0.5],
                [2., 1., 2., 1.2, 0.5, 1.4, 1.5],
            ]
        );
        assert_eq!(
            included,
            array![
                [true, false, false, false, false, false, false],
                [true, true, false, true, true, false, false],
                [true, true, true, true, false, true, true],
            ]
        );
    }

    #[test]
    fn test_hybrid_electric_correction() {
        let ds = hybrid_electric();
        assert_eq!(ds.role("source"), Role::META);
        assert_eq!(ds.role("reserve"), Role::META | Role::DECREED);
        assert_eq!(ds.role("battery"), Role::DECREED);
        assert_eq!(ds.parents("battery"), vec![0, 1]);

        let x = array![
            [ELECTRIC, 250., 30.],
            [FUEL, 120., 30.],
            [HYBRID, 120., 30.],
            [HYBRID, 280., 30.],
        ];
        let (xcorr, included) = ds.correct_get_included(&x).unwrap();
        // With an electric source, the excluded reserve gets its default 100,
        // the "small reserve" level, which must not include the battery by itself
        assert_eq!(
            xcorr,
            array![
                [ELECTRIC, 100., 30.],
                [FUEL, 100., 10.],
                [HYBRID, 100., 30.],
                [HYBRID, 300., 10.],
            ]
        );
        assert_eq!(
            included,
            array![
                [true, false, true],
                [true, true, false],
                [true, true, true],
                [true, true, false],
            ]
        );
    }

    #[test]
    fn test_correction_in_relaxed_space() {
        let ds = hybrid_electric();
        // Relaxed one-hot energy source: hybrid has the largest component
        let x = array![[0.2, 0.1, 0.7, 120., 30.], [0.5, 0.4, 0.1, 280., 30.]];
        let (xcorr, included) = ds.correct_get_included_unfolded(&x).unwrap();
        assert_eq!(
            xcorr,
            array![[0., 0., 1., 100., 30.], [1., 0., 0., 300., 10.]]
        );
        assert_eq!(included, array![[true, true, true], [true, true, false]]);
    }

    #[test]
    fn test_excluded_enum_gets_first_level() {
        let mut ds = DesignSpace::new(&[XType::Int(0, 1), XType::Enum(3)]);
        ds.decreed_var(1, 0, &[1.]).unwrap();
        let x = array![[0.2, 0.1, 0.3, 0.6], [0.8, 0.1, 0.3, 0.6]];
        let (xcorr, included) = ds.correct_get_included_unfolded(&x).unwrap();
        assert_eq!(xcorr, array![[0., 1., 0., 0.], [1., 0., 0., 1.]]);
        assert_eq!(included, array![[true, false], [true, true]]);
    }

    #[test]
    fn test_order_with_meta_after_decreed() {
        // Variable 0 is decreed by variable 2, which comes later
        let mut ds = DesignSpace::new(&[XType::Float(0., 10.), XType::Int(0, 1), XType::Int(0, 1)]);
        ds.decreed_var(0, 2, &[1.]).unwrap();
        assert_eq!(ds.topological_order(), &[1, 2, 0]);
        let (xcorr, included) = ds
            .correct_get_included(&array![[5., 0.4, 0.2], [5., 0.4, 0.8]])
            .unwrap();
        assert_eq!(xcorr, array![[0., 0., 0.], [5., 0., 1.]]);
        assert_eq!(included, array![[false, true, true], [true, true, true]]);
    }

    #[test]
    fn test_declarations_combine_with_and() {
        let mut ds = DesignSpace::new(&[XType::Int(0, 1), XType::Int(0, 1), XType::Float(0., 1.)]);
        ds.decreed_var(2, 0, &[1.])
            .unwrap()
            .decreed_var(2, 1, &[1.])
            .unwrap();
        assert_eq!(
            ds.condition(2),
            Some(&Condition::And(vec![
                Condition::is_in(0, &[1.]),
                Condition::is_in(1, &[1.]),
            ]))
        );
        let x = array![[1., 0., 0.5], [0., 1., 0.5], [1., 1., 0.5]];
        let (_, included) = ds.correct_get_included(&x).unwrap();
        assert_eq!(included.column(2), array![false, false, true]);
    }

    #[test]
    fn test_cycle_is_rejected_and_space_unchanged() {
        let mut ds = DesignSpace::new(&[XType::Int(0, 1), XType::Int(0, 1), XType::Int(0, 1)]);
        ds.decreed_var(1, 0, &[1.])
            .unwrap()
            .decreed_var(2, 1, &[1.])
            .unwrap();
        assert!(ds.decreed_var(0, 2, &[1.]).is_err());
        assert_eq!(ds.condition(0), None);
        assert_eq!(ds.role(0), Role::META);
        assert_eq!(ds.topological_order(), &[0, 1, 2]);
    }

    #[test]
    fn test_invalid_declarations() {
        let mut ds = DesignSpace::new(&[
            XType::Float(0., 1.),
            XType::Int(0, 3),
            XType::Ord(vec![1., 2., 4.]),
            XType::Enum(2),
        ]);
        // Indices out of range or self decree
        assert!(ds.decreed_var(4, 1, &[1.]).is_err());
        assert!(ds.decreed_var(0, 4, &[1.]).is_err());
        assert!(ds.decreed_var(1, 1, &[1.]).is_err());
        // Continuous meta variable
        assert!(ds.decreed_var(1, 0, &[0.5]).is_err());
        // Values that the meta variable cannot take
        assert!(ds.decreed_var(0, 1, &[]).is_err());
        assert!(ds.decreed_var(0, 1, &[4.]).is_err());
        assert!(ds.decreed_var(0, 1, &[1.5]).is_err());
        assert!(ds.decreed_var(0, 2, &[3.]).is_err());
        assert!(ds.decreed_var(0, 3, &[2.]).is_err());
        assert!(ds.decreed_var(0, 3, &[f64::NAN]).is_err());
        // Empty combinations
        assert!(ds.decreed_var_when(0, Condition::Or(vec![])).is_err());
        assert!(
            ds.decreed_var_when(0, Condition::And(vec![Condition::Or(vec![])]))
                .is_err()
        );
        assert!((0..4).all(|i| ds.condition(i).is_none()));
        // Valid ones
        assert!(ds.decreed_var(0, 1, &[0., 3.]).is_ok());
        assert!(ds.decreed_var(0, 2, &[4.]).is_ok());
        assert!(ds.decreed_var(0, 3, &[1.]).is_ok());
    }

    #[test]
    fn test_names() {
        let ds = motors_and_propellers();
        assert_eq!(ds.name(3), Some("r_11"));
        assert_eq!(ds.name(7), None);
        assert_eq!(ds.index_of("r_11"), Some(3));
        assert_eq!(ds.index_of("r_33"), None);
        assert_eq!(ds.condition("r_12"), ds.condition(4));
        assert_eq!(ds.parents(&String::from("p_2")), vec![0]);

        let unnamed = DesignSpace::new(&[XType::Int(0, 1)]);
        assert_eq!(unnamed.names(), None);
        assert_eq!(unnamed.name(0), None);
        assert_eq!(unnamed.index_of("x0"), None);
    }

    #[test]
    fn test_invalid_names() {
        let int = XType::Int(0, 1);
        assert!(DesignSpace::named(&[("a", int.clone()), ("a", int.clone())]).is_err());
        assert!(DesignSpace::named(&[("", int.clone())]).is_err());
        // Owned names
        let names = [String::from("a"), String::from("b")];
        let ds = DesignSpace::named(&[(&names[0], int.clone()), (&names[1], int)]).unwrap();
        assert_eq!(ds.names(), Some(&names[..]));
    }

    #[test]
    fn test_unknown_names_leave_space_unchanged() {
        let mut ds = motors_and_propellers();
        let before = serde_json::to_string(&ds).unwrap();
        assert!(ds.decreed_var("r_33", "m", &[1.]).is_err());
        assert!(ds.decreed_var("r_11", "q", &[1.]).is_err());
        assert!(
            ds.decreed_var_when("r_33", Condition::is_in(0, &[1.]))
                .is_err()
        );
        assert!(ds.is_in("q", &[1.]).is_err());
        assert_eq!(serde_json::to_string(&ds).unwrap(), before);

        let mut unnamed = DesignSpace::new(&[XType::Int(0, 1), XType::Float(0., 1.)]);
        let err = unnamed.decreed_var("y", "x", &[1.]).unwrap_err();
        assert!(err.to_string().contains("not named"), "{err}");
        assert!(unnamed.condition(1).is_none());
    }

    #[test]
    #[should_panic(expected = "unknown variable `q`")]
    fn test_accessor_panics_on_unknown_name() {
        motors_and_propellers().role("q");
    }

    #[test]
    fn test_messages_show_names() {
        let mut ds = motors_and_propellers();
        let err = ds.decreed_var("m", "r_11", &[1.]).unwrap_err().to_string();
        assert!(err.contains("variable 3 (r_11) is continuous"), "{err}");
        // m -> p_2 -> m
        let err = ds.decreed_var("m", "p_2", &[1.]).unwrap_err().to_string();
        assert!(err.contains("0 (m), 1 (p_1), 2 (p_2)"), "{err}");
    }

    #[test]
    fn test_serde_roundtrip() {
        for ds in [hybrid_electric(), {
            let mut ds = DesignSpace::new(&[XType::Int(0, 1), XType::Float(0., 1.)]);
            ds.decreed_var(1, 0, &[1.]).unwrap();
            ds
        }] {
            let json = serde_json::to_string(&ds).unwrap();
            let loaded: DesignSpace = serde_json::from_str(&json).unwrap();
            assert_eq!(serde_json::to_string(&loaded).unwrap(), json);
            assert_eq!(loaded.names(), ds.names());
            assert_eq!(loaded.topological_order(), ds.topological_order());
            assert_eq!(loaded.condition(1), ds.condition(1));

            let bytes = postcard::to_allocvec(&ds).unwrap();
            let loaded: DesignSpace = postcard::from_bytes(&bytes).unwrap();
            assert_eq!(serde_json::to_string(&loaded).unwrap(), json);
        }
    }

    #[test]
    fn test_deserialization_checks_the_graph() {
        // Names can be left out
        let unnamed = r#"{"xtypes":[{"Int":[0,1]},{"Float":[0,1]}],
            "conditions":[null,{"In":{"var":0,"values":[1.0]}}]}"#;
        let ds = serde_json::from_str::<DesignSpace>(unnamed).unwrap();
        assert_eq!(ds.names(), None);
        assert_eq!(ds.role(1), Role::DECREED);

        let cyclic = r#"{"xtypes":[{"Int":[0,1]},{"Int":[0,1]}],
            "conditions":[{"In":{"var":1,"values":[1.0]}},{"In":{"var":0,"values":[1.0]}}]}"#;
        assert!(serde_json::from_str::<DesignSpace>(cyclic).is_err());
        let bad_value = r#"{"xtypes":[{"Int":[0,1]},{"Float":[0,1]}],
            "conditions":[null,{"In":{"var":0,"values":[2.0]}}]}"#;
        assert!(serde_json::from_str::<DesignSpace>(bad_value).is_err());
        let bad_len = r#"{"xtypes":[{"Int":[0,1]}],"conditions":[]}"#;
        assert!(serde_json::from_str::<DesignSpace>(bad_len).is_err());
        let twice = r#"{"xtypes":[{"Int":[0,1]},{"Int":[0,1]}],"names":["a","a"],
            "conditions":[null,null]}"#;
        assert!(serde_json::from_str::<DesignSpace>(twice).is_err());
        let missing = r#"{"xtypes":[{"Int":[0,1]},{"Int":[0,1]}],"names":["a"],
            "conditions":[null,null]}"#;
        assert!(serde_json::from_str::<DesignSpace>(missing).is_err());
    }
}
