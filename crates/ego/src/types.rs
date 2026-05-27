use crate::{EgoError, EgorState, Result};
use argmin::core::CostFunction;
use linfa::Float;
use ndarray::{Array1, Array2, ArrayView2};
use serde::{Deserialize, Serialize};

// Re-export from egobox_moe for backward compatibility
#[deprecated(since = "0.36.1", note = "Use `egobox_moe::SurrogateBuilder` instead")]
pub use egobox_moe::SurrogateBuilder;
#[deprecated(since = "0.36.1", note = "Use `egobox_moe::XType` instead")]
pub use egobox_moe::XType;

/// Optimization result
#[derive(Clone, Debug)]
pub struct OptimResult<F: Float> {
    /// Optimum x value
    pub x_opt: Array1<F>,
    /// Optimum y value (e.g. f(x_opt))
    pub y_opt: Array1<F>,
    /// History of successive x values
    pub x_doe: Array2<F>,
    /// History of successive y values (e.g f(x_doe))
    pub y_doe: Array2<F>,
    /// EgorSolver final state
    pub state: EgorState<F>,
}

/// Infill criterion used to select next promising point
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum InfillStrategy {
    /// Expected Improvement
    EI,
    /// Log of Expected Improvement
    LogEI,
    /// Locating the regional extreme
    WB2,
    /// Scaled WB2
    WB2S,
}

/// Constraint criterion used to select next promising point
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConstraintStrategy {
    /// Use the mean value
    MeanConstraint,
    /// Use the upper bound (ie mean + 3*sigma)
    UpperTrustBound,
}

/// Optimizer used to optimize the infill criteria
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum InfillOptimizer {
    /// SLSQP optimizer (gradient based)
    Slsqp,
    /// Cobyla optimizer (gradient free)
    Cobyla,
}

/// Strategy to choose several points at each iteration
/// to benefit from parallel evaluation of the objective function
/// (The Multi-points Expected Improvement (q-EI) Criterion)
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum QEiStrategy {
    /// Take the mean of the kriging predictor for q points
    KrigingBeliever,
    /// Take the minimum of kriging predictor for q points
    KrigingBelieverLowerBound,
    /// Take the maximum kriging value for q points
    KrigingBelieverUpperBound,
    /// Take the current minimum of the function found so far
    ConstantLiarMinimum,
}

/// Strategy to handle objective computation failure at a given x point
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[non_exhaustive]
pub enum FailsafeStrategy {
    /// Failure point x is ignored
    #[default]
    Rejection,
    /// Use objective surrogate prediction: y <- prediction(x) + variance(x)
    Imputation,
    /// Use a surrogate to model viability (ie probability of evaluation success)
    Viability,
}

/// Constraint specification allowing the user to define how each constraint
/// should be interpreted. Instead of requiring constraints to be formulated
/// as `c <= 0`, users can specify bounds directly.
///
/// Internally, every specification is converted to one or more `c' <= 0`
/// constraints before entering the optimization pipeline.
///
/// # Examples
/// ```
/// use egobox_ego::CstrSpec;
///
/// // c <= 5.0  ->  c - 5 <= 0
/// let spec = CstrSpec::Leq(5.0);
/// assert_eq!(spec.n_internal(), 1);
/// let transformed: Vec<f64> = spec
///     .terms()
///     .into_iter()
///     .map(|(scale, offset)| scale * 3.0 + offset)
///     .collect();
/// assert_eq!(transformed, vec![-2.0]);
///
/// // c >= 2.0  ->  2 - c <= 0
/// let spec = CstrSpec::Geq(2.0);
/// let transformed: Vec<f64> = spec
///     .terms()
///     .into_iter()
///     .map(|(scale, offset)| scale * 3.0 + offset)
///     .collect();
/// assert_eq!(transformed, vec![-1.0]);
///
/// // c = 4.0  ->  c - 4 <= 0  AND  4 - c <= 0
/// let spec = CstrSpec::Eq(4.0);
/// assert_eq!(spec.n_internal(), 2);
///
/// // 1.0 <= c <= 3.0  ->  1 - c <= 0  AND  c - 3 <= 0
/// let spec = CstrSpec::Btw(1.0, 3.0);
/// assert_eq!(spec.n_internal(), 2);
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum CstrSpec {
    /// Constraint `c <= Z`, transformed to `c - Z <= 0`
    Leq(f64),
    /// Constraint `c >= Z`, transformed to `Z - c <= 0`
    Geq(f64),
    /// Constraint `c = Z`, expanded to `c - Z <= 0` AND `Z - c <= 0`
    Eq(f64),
    /// Constraint `LO <= c <= HI`, expanded to `LO - c <= 0` AND `c - HI <= 0`
    Btw(f64, f64),
}

impl CstrSpec {
    /// Number of internal `<= 0` constraints this spec expands to
    pub fn n_internal(&self) -> usize {
        match self {
            CstrSpec::Leq(_) | CstrSpec::Geq(_) => 1,
            CstrSpec::Eq(_) | CstrSpec::Btw(_, _) => 2,
        }
    }

    /// Canonical internal affine terms as `(scale, offset)` pairs.
    ///
    /// Each internal constraint is `scale * raw + offset <= 0`.
    pub fn terms(&self) -> Vec<(f64, f64)> {
        match self {
            CstrSpec::Leq(z) => vec![(1.0, -z)],
            CstrSpec::Geq(z) => vec![(-1.0, *z)],
            CstrSpec::Eq(z) => vec![(1.0, -z), (-1.0, *z)],
            CstrSpec::Btw(lo, hi) => vec![(-1.0, *lo), (1.0, -hi)],
        }
    }
}

/// Compute the total number of internal constraints from a slice of [`CstrSpec`]
pub fn n_internal_cstrs(specs: &[CstrSpec]) -> usize {
    specs.iter().map(|s| s.n_internal()).sum()
}

/// Describes how an internal constraint column is obtained.
///
/// Used during surrogate training to decide whether a column requires
/// a fresh GP training or can be derived from another surrogate.
#[derive(Clone, Debug)]
pub enum InternalCstrKind {
    /// A directly-trained constraint (internal column index used as training target).
    Primary,
    /// An affine transform of another internal constraint column.
    Derived {
        /// Internal column index of the primary source.
        source: usize,
        /// Scale factor applied to source predictions.
        scale: f64,
        /// Constant added after scaling: `pred = scale * source_pred + offset`.
        offset: f64,
    },
}

/// Build a mapping from internal constraint column indices to their kind.
///
/// Index 0 is the objective (always `Primary`).
/// Subsequent indices correspond to expanded internal constraints.
///
/// For `Leq`/`Geq` specs: one `Primary` column.
/// For `Eq(z)` specs: `Primary` then `Derived { source, scale: -1.0, offset: 0.0 }`.
/// For `Btw(lo, hi)` specs: `Primary` then `Derived { source, scale: -1.0, offset: lo - hi }`.
pub fn internal_cstr_mapping(specs: &[CstrSpec]) -> Vec<InternalCstrKind> {
    let mut mapping = vec![InternalCstrKind::Primary]; // index 0 = objective
    for spec in specs {
        match spec {
            CstrSpec::Leq(_) | CstrSpec::Geq(_) => {
                mapping.push(InternalCstrKind::Primary);
            }
            CstrSpec::Eq(_) => {
                let source = mapping.len();
                mapping.push(InternalCstrKind::Primary);
                mapping.push(InternalCstrKind::Derived {
                    source,
                    scale: -1.0,
                    offset: 0.0,
                });
            }
            CstrSpec::Btw(lo, hi) => {
                let source = mapping.len();
                mapping.push(InternalCstrKind::Primary);
                mapping.push(InternalCstrKind::Derived {
                    source,
                    scale: -1.0,
                    offset: lo - hi,
                });
            }
        }
    }
    mapping
}

/// Transform raw constraint columns in `y` according to `specs`.
///
/// Input `y` has shape `(nrows, 1 + n_user_cstrs)` where column 0 is the objective
/// and columns `1..` are the raw user constraint values.
///
/// Returns a new array with shape `(nrows, 1 + n_internal_cstrs)` where the constraint
/// columns have been transformed (and potentially expanded for Equal/Between specs).
pub fn transform_constraints(y: &Array2<f64>, specs: &[CstrSpec]) -> Array2<f64> {
    let nrows = y.nrows();
    let n_intern = n_internal_cstrs(specs);
    let mut result = Array2::zeros((nrows, 1 + n_intern));

    // Copy objective column
    result.column_mut(0).assign(&y.column(0));

    // Transform constraint columns
    for row_idx in 0..nrows {
        let mut col = 1usize;
        for (i, spec) in specs.iter().enumerate() {
            let raw = y[[row_idx, 1 + i]];
            let transformed = spec
                .terms()
                .into_iter()
                .map(|(scale, offset)| scale * raw + offset)
                .collect::<Vec<_>>();
            for v in &transformed {
                result[[row_idx, col]] = *v;
                col += 1;
            }
        }
    }
    result
}

/// Transform raw function-constraint columns in `c_data` according to `specs`.
///
/// Input `c_data` has shape `(nrows, n_user_fcstrs)` with one column per user
/// function constraint. Output has shape `(nrows, n_internal_fcstrs)` where
/// Equal/Between specs expand to two columns.
pub fn transform_function_constraints(c_data: &Array2<f64>, specs: &[CstrSpec]) -> Array2<f64> {
    let nrows = c_data.nrows();
    let n_intern = n_internal_cstrs(specs);
    let mut result = Array2::zeros((nrows, n_intern));

    for row_idx in 0..nrows {
        let mut col = 0usize;
        for (i, spec) in specs.iter().enumerate() {
            let raw = c_data[[row_idx, i]];
            for v in spec
                .terms()
                .into_iter()
                .map(|(scale, offset)| scale * raw + offset)
            {
                result[[row_idx, col]] = v;
                col += 1;
            }
        }
    }
    result
}

/// Build affine expansion mapping for function constraints.
///
/// Returns tuples `(raw_index, scale, offset)` describing each internal function
/// constraint as `scale * raw_fcstr[raw_index] + offset <= 0`.
///
/// If `specs` is `None`, legacy behavior is preserved with one identity mapping
/// per function constraint.
pub fn function_cstr_affine_mapping(
    n_fcstrs: usize,
    specs: Option<&[CstrSpec]>,
) -> Result<Vec<(usize, f64, f64)>> {
    match specs {
        Some(specs) => {
            if specs.len() != n_fcstrs {
                return Err(EgoError::InvalidConfigError(format!(
                    "fcstr_specs length ({}) does not match fcstrs length ({})",
                    specs.len(),
                    n_fcstrs
                )));
            }
            let mut mapping = Vec::new();
            for (idx, spec) in specs.iter().enumerate() {
                for (scale, offset) in spec.terms() {
                    mapping.push((idx, scale, offset));
                }
            }
            Ok(mapping)
        }
        None => Ok((0..n_fcstrs).map(|idx| (idx, 1.0, 0.0)).collect()),
    }
}

/// A trait for types that can be returned by an objective function.
///
/// This allows objective functions to return either `Array2<f64>` directly
/// (infallible) or `Result<Array2<f64>, E>` (fallible) where E implements `Display`.
pub trait ObjFnResponse {
    /// Convert the response into a `Result<Array2<f64>, EgoError>`.
    fn into_obj_result(self) -> Result<Array2<f64>>;
}

impl ObjFnResponse for Array2<f64> {
    fn into_obj_result(self) -> Result<Array2<f64>> {
        Ok(self)
    }
}

impl<E: std::fmt::Display> ObjFnResponse for std::result::Result<Array2<f64>, E> {
    fn into_obj_result(self) -> Result<Array2<f64>> {
        self.map_err(|e| EgoError::UserFnError(e.to_string()))
    }
}

/// An interface for objective function to be optimized
///
/// The function is expected to return a matrix allowing nrows evaluations at once.
/// A row of the output matrix is expected to contain [objective, cstr_1, ... cstr_n] values.
///
/// The function can return either `Array2<f64>` directly (infallible evaluation)
/// or `Result<Array2<f64>, E>` (fallible evaluation) where `E` implements `Display`.
/// On error, the optimizer handles the failure according to the configured [`FailsafeStrategy`].
pub trait ObjFn: Clone {
    /// Evaluate the objective function at the given points.
    fn eval(&self, x: &ArrayView2<f64>) -> crate::errors::Result<Array2<f64>>;
}

impl<T, R: ObjFnResponse> ObjFn for T
where
    T: Clone + Fn(&ArrayView2<f64>) -> R,
{
    fn eval(&self, x: &ArrayView2<f64>) -> crate::errors::Result<Array2<f64>> {
        (self)(x).into_obj_result()
    }
}

/// A trait to retrieve functions constraints
/// provided by the user and used by the internal optimizer
pub trait Constraints<C: CstrFn> {
    /// Returns the list of constraints functions
    fn constraints(&self) -> &[C];

    /// Optional specifications for function constraints.
    fn constraint_specs(&self) -> Option<&[CstrSpec]>;
}

/// As structure to handle the objective and constraints functions for implementing
/// the optimization problem and `argmin::CostFunction` to be used with argmin framework.
#[derive(Clone)]
pub struct ProblemFunc<O: ObjFn, C: CstrFn> {
    fobj: O,
    fcstrs: Vec<C>,
    fcstr_specs: Option<Vec<CstrSpec>>,
}

impl<O: ObjFn, C: CstrFn> ProblemFunc<O, C> {
    /// Constructor given the objective function
    pub fn new(fobj: O) -> Self {
        ProblemFunc {
            fobj,
            fcstrs: vec![],
            fcstr_specs: None,
        }
    }

    /// Add constraints functions
    pub fn subject_to(mut self, fcstrs: Vec<C>) -> Self {
        self.fcstrs = fcstrs;
        self.fcstr_specs = None;
        self
    }

    /// Add constraints functions with corresponding constraint specifications.
    pub fn subject_to_with_specs(mut self, fcstrs: Vec<C>, fcstr_specs: Vec<CstrSpec>) -> Self {
        self.fcstrs = fcstrs;
        self.fcstr_specs = Some(fcstr_specs);
        self
    }
}

impl<O: ObjFn, C: CstrFn> CostFunction for ProblemFunc<O, C> {
    /// Type of the parameter vector
    type Param = Array2<f64>;
    /// Type of the return value computed by the cost function
    type Output = Array2<f64>;

    /// Apply the cost function to a parameter `p`
    fn cost(&self, p: &Self::Param) -> std::result::Result<Self::Output, argmin::core::Error> {
        // Evaluate objective function, forward error on failure
        self.fobj
            .eval(&p.view())
            .map_err(|e| argmin::core::Error::msg(e.to_string()))
    }
}

impl<O: ObjFn, C: CstrFn> Constraints<C> for ProblemFunc<O, C> {
    fn constraints(&self) -> &[C] {
        &self.fcstrs
    }

    fn constraint_specs(&self) -> Option<&[CstrSpec]> {
        self.fcstr_specs.as_deref()
    }
}

/// A trait for functions provided by the user
/// Functions are expected to be defined as `g(x, g, u)` where
/// * `x` is the input information,
/// * `g` an optional gradient information to be updated if present
/// * `u` information provided by the user
#[cfg(not(feature = "nlopt"))]
pub trait UserFn<U>: Fn(&[f64], Option<&mut [f64]>, &mut U) -> f64 {}

#[cfg(not(feature = "nlopt"))]
impl<T, U> UserFn<U> for T where T: Fn(&[f64], Option<&mut [f64]>, &mut U) -> f64 {}

/// A function trait for constraints provided by the user and used by the internal optimizer
/// It is a specialized version of [`UserFn`] with [`InfillObjData`] as user information
#[cfg(not(feature = "nlopt"))]
pub trait CstrFn: Clone + UserFn<InfillObjData<f64>> + Sync {}
#[cfg(not(feature = "nlopt"))]
impl<T> CstrFn for T where T: Clone + UserFn<InfillObjData<f64>> + Sync {}
/// A function trait for constraints used by the internal optimizer
/// It is a specialized version of [`ObjFn`] with [`InfillObjData`] as user informati
#[cfg(feature = "nlopt")]
pub trait CstrFn: Clone + nlopt::ObjFn<InfillObjData<f64>> + Sync {}
#[cfg(feature = "nlopt")]
impl<T> CstrFn for T where T: Clone + nlopt::ObjFn<InfillObjData<f64>> + Sync {}

/// A function type for domain constraints which will be used by the internal optimizer
/// which is the default value for [`crate::EgorFactory`] generic `C` parameter.
pub type Cstr = fn(&[f64], Option<&mut [f64]>, &mut InfillObjData<f64>) -> f64;

/// Data used by internal infill criteria optimization
/// Internally this type is used to carry the information required to
/// compute the various infill criteria implemented by [`crate::Egor`].
///
/// See [`crate::criteria`]
#[derive(Clone, Serialize, Deserialize)]
pub struct InfillObjData<F: Float> {
    /// Current objective minimum found
    #[serde(default = "F::max_value")]
    pub fmin: F,
    /// Current location of objective minimum
    pub xbest: Vec<F>,
    /// Scaling of infill obj (aka value which once scaled is equal to one)
    #[serde(default = "F::one")]
    pub scale_infill_obj: F,
    /// Scaling of constraints (aka value which once scaled is equal to one)
    pub scale_cstr: Option<Array1<F>>,
    /// Scaling of WB2 criterion (aka value which once scaled is equal to one)
    #[serde(default = "F::one")]
    pub scale_wb2: F,
    /// Whether a feasible point is found so far (all constraints within tolerances)
    pub feasibility: bool,
    /// Sigma weighting portfolio
    #[serde(default = "F::one")]
    pub sigma_weight: F,
}

impl<F: Float> Default for InfillObjData<F> {
    fn default() -> Self {
        Self {
            fmin: F::max_value(),
            xbest: vec![],
            scale_infill_obj: F::one(),
            scale_cstr: None,
            scale_wb2: F::one(),
            feasibility: false,
            sigma_weight: F::one(),
        }
    }
}

impl<F: Float> std::fmt::Debug for InfillObjData<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InfillObjData")
            .field("fmin", &self.fmin)
            .field("xbest", &self.xbest)
            .field("scale_infill_obj", &self.scale_infill_obj)
            .field(
                "scale_cstr",
                &self
                    .scale_cstr
                    .as_ref()
                    .map(|sc| sc.to_vec())
                    .unwrap_or_default(),
            )
            .field("scale_wb2", &self.scale_wb2)
            .field("feasibility", &self.feasibility)
            .field("sigma_weight", &self.sigma_weight)
            .finish()
    }
}
