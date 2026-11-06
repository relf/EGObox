//! Infill objective minimized by the infill optimizer: an acquisition criterion
//! (mono-objective infill criterion or multi-objective criterion) to be maximized,
//! negated, scaled and optionally weighted by the probability of feasibility (PoF)
//! of the constraint surrogates.
use crate::criteria::{InfillComposition, InfillCriterion};
use crate::moo::criterion::MooCriterion;
use crate::types::InfillObjData;
use crate::utils::{logpofs, logpofs_grad, pofs, pofs_grad};

use egobox_moe::MixtureGpSurrogate;
use ndarray::Array1;

/// Criterion to be maximized at a point, before scaling and feasibility weighting
pub(crate) trait Acquisition: Sync {
    /// How the criterion combines with the probability of feasibility
    fn composition(&self) -> InfillComposition;

    /// Criterion value at `x`
    fn value(&self, x: &[f64], data: &InfillObjData<f64>) -> f64;

    /// Criterion value and gradient at `x`
    fn value_grad(&self, x: &[f64], data: &InfillObjData<f64>) -> (f64, Array1<f64>);
}

/// Mono-objective infill criterion of the objective surrogate
pub(crate) struct MonoAcquisition<'a> {
    pub(crate) criterion: &'a dyn InfillCriterion,
    pub(crate) obj_model: &'a dyn MixtureGpSurrogate,
    pub(crate) viability_model: Option<&'a dyn MixtureGpSurrogate>,
    pub(crate) alpha: Option<f64>,
}

impl Acquisition for MonoAcquisition<'_> {
    fn composition(&self) -> InfillComposition {
        self.criterion.composition()
    }

    fn value(&self, x: &[f64], data: &InfillObjData<f64>) -> f64 {
        self.criterion.value(
            x,
            self.obj_model,
            data.fmin,
            self.viability_model,
            self.alpha,
            Some(data.sigma_weight),
            Some(data.scale_wb2),
        )
    }

    fn value_grad(&self, x: &[f64], data: &InfillObjData<f64>) -> (f64, Array1<f64>) {
        let grad = self.criterion.grad(
            x,
            self.obj_model,
            data.fmin,
            self.viability_model,
            self.alpha,
            Some(data.sigma_weight),
            Some(data.scale_wb2),
        );
        (self.value(x, data), grad)
    }
}

impl Acquisition for MooCriterion<'_> {
    fn composition(&self) -> InfillComposition {
        InfillComposition::Linear
    }

    fn value(&self, x: &[f64], _data: &InfillObjData<f64>) -> f64 {
        MooCriterion::value(self, x)
    }

    fn value_grad(&self, x: &[f64], _data: &InfillObjData<f64>) -> (f64, Array1<f64>) {
        MooCriterion::value_grad(self, x)
    }
}

/// Constraint surrogates whose probability of feasibility weights the infill objective
pub(crate) struct PofWeighting<'a> {
    pub(crate) models: &'a [Box<dyn MixtureGpSurrogate>],
    /// Tolerances of the constraints, the ones of the models first
    pub(crate) tols: &'a [f64],
}

/// Infill objective value when no feasible point is found yet: a neutral factor
/// of the probability of feasibility, -1 for linear composition, 0 for log composition
pub(crate) fn neutral_infill_val(composition: InfillComposition) -> f64 {
    if composition.is_log() { 0.0 } else { -1.0 }
}

/// Infill objective value and gradient when no feasible point is found yet
pub(crate) fn neutral_infill(composition: InfillComposition, dim: usize) -> (f64, Array1<f64>) {
    (neutral_infill_val(composition), Array1::zeros(dim))
}

/// Infill objective to be minimized at `x`, and its gradient when `with_grad`:
/// the opposite of the acquisition criterion scaled by `data.scale_infill_obj`,
/// weighted by the probability of feasibility of the constraint surrogates when `pof` is given.
/// With `pof`, while no feasible point is found (`data.feasibility` false),
/// the criterion is replaced by a neutral value so that the probability of feasibility
/// alone drives the search.
pub(crate) fn infill_objective(
    acquisition: &dyn Acquisition,
    pof: Option<&PofWeighting>,
    x: &[f64],
    data: &InfillObjData<f64>,
    with_grad: bool,
) -> (f64, Option<Array1<f64>>) {
    let composition = acquisition.composition();
    let (mut obj, mut grad) = if pof.is_some() && !data.feasibility {
        let (value, grad) = neutral_infill(composition, x.len());
        (value, with_grad.then_some(grad))
    } else if with_grad {
        let (value, grad) = acquisition.value_grad(x, data);
        (
            -value / data.scale_infill_obj,
            Some(-grad / data.scale_infill_obj),
        )
    } else {
        (-acquisition.value(x, data) / data.scale_infill_obj, None)
    };
    if let Some(PofWeighting { models, tols }) = pof {
        // gradient weighted only when there are models (null weighting gradient otherwise)
        let weight_grad = !models.is_empty();
        if composition.is_log() {
            if let Some(g) = grad.take() {
                grad = Some(if weight_grad {
                    g - logpofs_grad(x, models, tols)
                } else {
                    g
                });
            }
            obj -= logpofs(x, models, tols);
        } else {
            let pof = pofs(x, models, tols);
            if let Some(g) = grad.take() {
                grad = Some(if weight_grad {
                    g * pof + pofs_grad(x, models, tols) * obj
                } else {
                    g
                });
            }
            obj *= pof;
        }
    }
    (obj, grad)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::criteria::{EI, LOG_EI, WB2S};
    use crate::moo::ehvi::EhviCriterion;
    use crate::moo::ehvi::tests::models;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_neutral_infill_matches_composition() {
        assert_eq!(neutral_infill_val(InfillComposition::Linear), -1.0);
        assert_eq!(neutral_infill_val(InfillComposition::Log), 0.0);

        let (linear_value, linear_grad) = neutral_infill(InfillComposition::Linear, 3);
        assert_eq!(linear_value, -1.0);
        assert_eq!(linear_grad, Array1::<f64>::zeros(3));

        let (log_value, log_grad) = neutral_infill(InfillComposition::Log, 2);
        assert_eq!(log_value, 0.0);
        assert_eq!(log_grad, Array1::<f64>::zeros(2));
    }

    /// Check the gradient against central finite differences of the value at `xs`
    fn check_gradient(
        name: &str,
        acquisition: &dyn Acquisition,
        pof: Option<&PofWeighting>,
        data: &InfillObjData<f64>,
        xs: &[f64],
    ) {
        let value = |x: f64| infill_objective(acquisition, pof, &[x], data, false).0;
        for &x in xs {
            let (v, grad) = infill_objective(acquisition, pof, &[x], data, true);
            assert_eq!(v, value(x), "{name}: value with or without gradient at {x}");
            let h = 1e-6;
            let fdiff = (value(x + h) - value(x - h)) / (2. * h);
            assert_abs_diff_eq!(grad.unwrap()[0], fdiff, epsilon = 1e-2 * fdiff.abs() + 1e-6);
        }
    }

    #[test]
    fn test_infill_objective_gradients() {
        // objective and constraint surrogates of x in [0, 1]
        let (models, objs) = models(3);
        let (obj_model, cstr_models) = (&*models[0], &models[1..2]);
        let tols = [1e-4];
        let xs = [0.1, 0.4, 0.62];
        let x_ref = array![[0.4]];
        let (pred, var) = obj_model.predict_valvar(&x_ref.view()).unwrap();
        // GP standard deviation weighted as with the GP variance portfolio
        let sigma_weight = 0.1;
        let fmin = pred[0] - sigma_weight * var[0].sqrt();

        let with_models = PofWeighting {
            models: cstr_models,
            tols: &tols,
        };
        let without_models = PofWeighting {
            models: &[],
            tols: &[],
        };
        let weightings: [(&str, Option<&PofWeighting>); 3] = [
            ("plain", None),
            ("PoF", Some(&with_models)),
            ("PoF without models", Some(&without_models)),
        ];
        let criteria: [(&str, &dyn InfillCriterion); 3] =
            [("EI", &EI), ("LogEI", &LOG_EI), ("WB2S", &WB2S)];
        for (crit_name, criterion) in criteria {
            let acquisition = MonoAcquisition {
                criterion,
                obj_model,
                viability_model: None,
                alpha: None,
            };
            for feasibility in [true, false] {
                let data = InfillObjData {
                    fmin,
                    scale_infill_obj: 2.,
                    sigma_weight,
                    feasibility,
                    ..Default::default()
                };
                for (pof_name, pof) in weightings {
                    let name = format!("{crit_name} {pof_name} feasibility={feasibility}");
                    check_gradient(&name, &acquisition, pof, &data, &xs);
                }
            }
        }

        // multi-objective criterion
        let objs = objs.slice(ndarray::s![.., ..2]).to_owned();
        let front_rows = crate::moo::pareto::non_dominated_indices(&objs);
        let ehvi = MooCriterion::Ehvi(EhviCriterion::new(&models[..2], &objs, &front_rows));
        for feasibility in [true, false] {
            let data = InfillObjData {
                scale_infill_obj: 2.,
                feasibility,
                ..Default::default()
            };
            for (pof_name, pof) in weightings {
                let name = format!("EHVI {pof_name} feasibility={feasibility}");
                check_gradient(&name, &ehvi, pof, &data, &xs);
            }
        }
    }

    #[test]
    fn test_neutral_infill_objective_without_feasible_point() {
        // with PoF weighting but no constraint surrogate (e.g. function constraints only)
        // while no feasible point is found: constant neutral value, null gradient
        let (models, _) = models(1);
        let pof = PofWeighting {
            models: &[],
            tols: &[],
        };
        let data = InfillObjData {
            fmin: 0.,
            feasibility: false,
            ..Default::default()
        };
        let criteria: [&dyn InfillCriterion; 3] = [&EI, &LOG_EI, &WB2S];
        for criterion in criteria {
            let acquisition = MonoAcquisition {
                criterion,
                obj_model: &*models[0],
                viability_model: None,
                alpha: None,
            };
            let neutral = neutral_infill_val(criterion.composition());
            for x in [0.1, 0.62] {
                let (v, g) = infill_objective(&acquisition, Some(&pof), &[x], &data, true);
                assert_eq!(v, neutral);
                assert_eq!(g.unwrap(), array![0.]);
            }
        }
    }
}
