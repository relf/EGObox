//! Expected Hypervolume Improvement infill criterion
//!
//! Emmerich, M. T. M., Giannakoglou, K. C., & Naujoks, B. (2006). Single- and multiobjective
//! evolutionary optimization assisted by Gaussian random field metamodels.
//! IEEE Transactions on Evolutionary Computation, 10(4), 421–439.
//!
//! The region below the reference point not dominated by the Pareto front is decomposed into the
//! cells of the grid built with the front coordinates. The hypervolume improvement of a point `y`
//! is the sum over these cells `[l, u]` of `prod_j (u_j - max(y_j, l_j))^+`: with independent
//! normal predictions, the expectation of each factor is `EI_j(u_j) - EI_j(l_j)` where
//! `EI_j(b) = E[(b - Y_j)^+]`, hence an exact closed form for any number of objectives.

use super::criterion::{ei_and_derivatives, normalized_front, predict_normalized};
use super::scalarization::Normalization;
use egobox_moe::MixtureGpSurrogate;
use ndarray::{Array1, Array2, ArrayBase, Data, Ix2};

/// Expected Hypervolume Improvement criterion of the objective surrogates wrt a Pareto front.
///
/// Objectives are normalized with their observed bounds, the criterion is to be maximized.
pub(crate) struct EhviCriterion<'a> {
    obj_models: &'a [Box<dyn MixtureGpSurrogate>],
    normalization: Normalization,
    /// Per objective, sorted grid values: -inf, front coordinates, reference point coordinate
    grids: Vec<Vec<f64>>,
    /// Non-dominated cells given by the grid indices of their lower corner
    cells: Vec<Vec<usize>>,
    front_size: usize,
}

impl<'a> EhviCriterion<'a> {
    /// EHVI criterion given the objective surrogates, the objective values `objs` of the data
    /// (used to normalize the objectives) and the rows of `objs` forming the Pareto front
    pub(crate) fn new(
        obj_models: &'a [Box<dyn MixtureGpSurrogate>],
        objs: &ArrayBase<impl Data<Elem = f64>, Ix2>,
        front_rows: &[usize],
    ) -> Self {
        let (normalization, front, ref_point) = normalized_front(objs, front_rows);
        let n_obj = objs.ncols();
        let grids: Vec<Vec<f64>> = (0..n_obj)
            .map(|j| {
                let mut values: Vec<f64> = front
                    .column(j)
                    .iter()
                    .copied()
                    .filter(|v| *v < ref_point[j])
                    .collect();
                values.sort_by(f64::total_cmp);
                values.dedup();
                let mut grid = vec![f64::NEG_INFINITY];
                grid.extend(values);
                grid.push(ref_point[j]);
                grid
            })
            .collect();

        // Enumerate the grid cells, keeping those whose interior is not dominated by the front:
        // a cell is dominated iff a front point is lower or equal to its lower corner
        let mut cells = vec![];
        let mut index = vec![0; n_obj];
        loop {
            let dominated = front
                .rows()
                .into_iter()
                .any(|p| (0..n_obj).all(|j| p[j] <= grids[j][index[j]]));
            if !dominated {
                cells.push(index.clone());
            }
            // next cell index (odometer)
            let mut j = 0;
            loop {
                if j == n_obj {
                    return EhviCriterion {
                        obj_models,
                        normalization,
                        grids,
                        cells,
                        front_size: front.nrows(),
                    };
                }
                index[j] += 1;
                if index[j] < grids[j].len() - 1 {
                    break;
                }
                index[j] = 0;
                j += 1;
            }
        }
    }

    /// Criterion value at `x`
    pub(crate) fn value(&self, x: &[f64]) -> f64 {
        self.eval(x, false).0
    }

    /// Criterion value and gradient at `x`
    pub(crate) fn value_grad(&self, x: &[f64]) -> (f64, Array1<f64>) {
        self.eval(x, true)
    }

    /// Number of points of the Pareto front
    pub(crate) fn front_size(&self) -> usize {
        self.front_size
    }

    fn eval(&self, x: &[f64], with_grad: bool) -> (f64, Array1<f64>) {
        let nx = x.len();
        let Some(pred) = predict_normalized(self.obj_models, &self.normalization, x, with_grad)
        else {
            return (0., Array1::zeros(nx));
        };
        // Expected improvements below each grid value (and their gradients)
        let mut ei: Vec<Vec<f64>> = vec![];
        let mut dei: Vec<Array2<f64>> = vec![];
        for (j, grid) in self.grids.iter().enumerate() {
            let mut ei_j = vec![0.; grid.len()];
            let mut dei_j = Array2::zeros((grid.len(), nx));
            for (k, &g) in grid.iter().enumerate() {
                if g.is_finite() {
                    let (e, de_dmu, de_dsigma) = ei_and_derivatives(g, pred.mu[j], pred.sigma[j]);
                    ei_j[k] = e;
                    if with_grad {
                        dei_j
                            .row_mut(k)
                            .assign(&(&pred.dmu.row(j) * de_dmu + &pred.dsigma.row(j) * de_dsigma));
                    }
                }
            }
            ei.push(ei_j);
            dei.push(dei_j);
        }

        let n_obj = self.grids.len();
        let mut value = 0.;
        let mut grad = Array1::zeros(nx);
        let mut factors = vec![0.; n_obj];
        for cell in &self.cells {
            for j in 0..n_obj {
                factors[j] = ei[j][cell[j] + 1] - ei[j][cell[j]];
            }
            value += factors.iter().product::<f64>();
            if with_grad {
                for j in 0..n_obj {
                    let others: f64 = (0..n_obj).filter(|&k| k != j).map(|k| factors[k]).product();
                    if others != 0. {
                        let dfactor = &dei[j].row(cell[j] + 1) - &dei[j].row(cell[j]);
                        grad = grad + dfactor * others;
                    }
                }
            }
        }
        (value, grad)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::moo::hypervolume::hypervolume;
    use approx::assert_abs_diff_eq;
    use egobox_moe::{GpMixture, NbClusters};
    use linfa::{Dataset, traits::Fit};
    use ndarray::{Axis, array, concatenate};
    use ndarray_rand::RandomExt;
    use ndarray_rand::rand::SeedableRng;
    use ndarray_rand::rand_distr::StandardNormal;
    use rand_xoshiro::Xoshiro256Plus;

    /// GP models of `n_obj` objectives of x in [0, 1]
    fn models(n_obj: usize) -> (Vec<Box<dyn MixtureGpSurrogate>>, Array2<f64>) {
        let xt: Array2<f64> = array![[0.0], [0.25], [0.5], [0.75], [1.0]];
        let fs = [
            xt.column(0).mapv(|v| v + 0.2 * (5. * v).sin()),
            xt.column(0).mapv(|v| 1. - v.sqrt() + 0.3 * (6. * v).sin()),
            xt.column(0)
                .mapv(|v| (v - 0.4).powi(2) + 0.1 * (9. * v).cos()),
        ];
        let mut models: Vec<Box<dyn MixtureGpSurrogate>> = vec![];
        let mut objs = Array2::zeros((xt.nrows(), 0));
        for f in fs.iter().take(n_obj) {
            let gp = GpMixture::params()
                .n_clusters(NbClusters::fixed(1))
                .fit(&Dataset::new(xt.clone(), f.clone()))
                .unwrap();
            models.push(Box::new(gp));
            objs = concatenate![Axis(1), objs, f.clone().insert_axis(Axis(1))];
        }
        (models, objs)
    }

    /// Monte Carlo estimate of EHVI in the normalized space
    fn ehvi_monte_carlo(
        ehvi: &EhviCriterion,
        front: &Array2<f64>,
        ref_point: &Array1<f64>,
        x: f64,
    ) -> f64 {
        let pred = predict_normalized(ehvi.obj_models, &ehvi.normalization, &[x], false).unwrap();
        let hv0 = hypervolume(front, ref_point);
        let n = 200_000;
        let mut rng = Xoshiro256Plus::seed_from_u64(42);
        let z = Array2::<f64>::random_using((n, front.ncols()), StandardNormal, &mut rng);
        let mut sum = 0.;
        for zi in z.rows() {
            let y = &pred.mu + &(&pred.sigma * &zi);
            let ext = concatenate![Axis(0), front.view(), y.insert_axis(Axis(0)).view()];
            sum += hypervolume(&ext, ref_point) - hv0;
        }
        sum / n as f64
    }

    #[test]
    fn test_ehvi_vs_monte_carlo() {
        for n_obj in [2, 3] {
            let (models, objs) = models(n_obj);
            // front of the data and its normalized version
            let front_rows = crate::moo::pareto::non_dominated_indices(&objs);
            let ehvi = EhviCriterion::new(&models, &objs, &front_rows);
            let (_, front, ref_point) = normalized_front(&objs, &front_rows);
            for x in [0.1, 0.4, 0.62] {
                let v = ehvi.value(&[x]);
                let mc = ehvi_monte_carlo(&ehvi, &front, &ref_point, x);
                assert_abs_diff_eq!(v, mc, epsilon = 2e-3 * mc.abs() + 1e-5);
            }
        }
    }

    #[test]
    fn test_ehvi_gradients() {
        for n_obj in [2, 3] {
            let (models, objs) = models(n_obj);
            let front_rows = crate::moo::pareto::non_dominated_indices(&objs);
            let ehvi = EhviCriterion::new(&models, &objs, &front_rows);
            for x in [0.1, 0.4, 0.62, 0.9] {
                let (v, g) = ehvi.value_grad(&[x]);
                assert!(v >= 0., "EHVI({x}) = {v}");
                let h = 1e-6;
                let fd = (ehvi.value(&[x + h]) - ehvi.value(&[x - h])) / (2. * h);
                assert_abs_diff_eq!(g[0], fd, epsilon = 1e-4 * (1. + fd.abs()));
            }
        }
    }

    #[test]
    fn test_ehvi_vanishes_at_front_points() {
        let (models, objs) = models(2);
        let front_rows = crate::moo::pareto::non_dominated_indices(&objs);
        let ehvi = EhviCriterion::new(&models, &objs, &front_rows);
        assert_eq!(ehvi.front_size(), front_rows.len());
        // training points are exact predictions: no improvement over the front
        for &i in &front_rows {
            let x = i as f64 / 4.;
            assert_abs_diff_eq!(ehvi.value(&[x]), 0., epsilon = 1e-4);
        }
        // between training points there is some expected improvement
        assert!(ehvi.value(&[0.62]) > 1e-6);
    }
}
