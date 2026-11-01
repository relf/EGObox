//! Multistart strategies: initial points of the infill criterion optimization

use crate::solver::coego;
use crate::utils;

use egobox_doe::{Lhs, LhsKind, SamplingMethod};
use log::debug;
use ndarray::{Array1, Array2, Axis, Zip, aview1, concatenate, s, stack};
use ndarray_rand::rand::Rng;
use ndarray_rand::rand::seq::SliceRandom;

/// A trait for multi start initial points computation
pub(crate) trait MultiStarter {
    /// Return initial points for optimization multistart
    /// taking into account active components given as a set of indices
    fn multistart(&mut self, n_start: usize, active: &[usize]) -> Array2<f64>;

    /// Return the bounds of the design space
    /// Compared to xlimits which are given by the user for all x components,
    /// xbounds are the actual bounds used during optimization taking into account
    /// active components and possibly trust region restrictions
    fn xbounds(&self, active: &[usize]) -> Array2<f64>;
}

/// MiddlePickerMultiStarter is a multistart strategy where starting points
/// are picked in the area in between the training data points where
/// infill criterion is expected to be high
pub(crate) struct MiddlePickerMultiStarter<'a, 'b, R: Rng + Clone> {
    xlimits: &'a Array2<f64>,
    xtrain: &'b Array2<f64>,
    rng: R,
}

impl<R: Rng + Clone> MultiStarter for MiddlePickerMultiStarter<'_, '_, R> {
    fn multistart(&mut self, n_start: usize, active: &[usize]) -> Array2<f64> {
        let xlimits = coego::get_active_x(Axis(0), self.xlimits, active);

        let nt = self.xtrain.nrows();
        // Compute the maximum number of points n to consider to generate midpoints
        // to avoid too much computation when large training set
        // Consider one tenth of training points as midpoints nb will grow as (n * (n-1) / 2)
        // and multistart rarely exceeds 50. We do not want to pick only midpoints
        // but keep some diversity by adding LHS points (at least when nt is small < 50).
        let n = (nt / 10).max(2);

        let xt = self.xtrain;
        let mut indices: Vec<usize> = (0..xt.nrows()).collect();
        indices.shuffle(&mut self.rng);
        let selected = indices
            .iter()
            .take(n)
            .map(|&i| xt.slice(s![i, ..]))
            .collect::<Vec<_>>();
        let xt = stack(Axis(0), &selected).unwrap();

        let xt = coego::get_active_x(Axis(1), &xt, active);
        let midpoints =
            utils::start_points(&xt, &xlimits.column(0), &xlimits.column(1), Some(n_start));
        let n_midpoints = midpoints.nrows();
        let missing_points: i32 = n_start as i32 - n_midpoints as i32;
        if missing_points <= 0 {
            debug!("MiddlePickerMultiStarter: pick {n_midpoints} pts");
            midpoints
        } else {
            debug!(
                "MiddlePickerMultiStarter: pick {n_midpoints} pt(s), add {missing_points} LHS pt(s)"
            );
            // complete with LHS
            let sampling = Lhs::new(&xlimits)
                .kind(LhsKind::Maximin)
                .with_rng(&mut self.rng);
            let missings = sampling.sample((missing_points as usize).max(3)); // sampling with at least 3 points
            let missings = missings.slice(s![0..(missing_points as usize), ..]);
            concatenate(Axis(0), &[midpoints.view(), missings]).unwrap()
        }
    }

    fn xbounds(&self, active: &[usize]) -> Array2<f64> {
        coego::get_active_x(Axis(0), self.xlimits, active)
    }
}

impl<'a, 'b, R: Rng + Clone> MiddlePickerMultiStarter<'a, 'b, R> {
    pub fn new(xlimits: &'a Array2<f64>, xtrain: &'b Array2<f64>, rng: R) -> Self {
        MiddlePickerMultiStarter {
            xlimits,
            xtrain,
            rng,
        }
    }
}

/// LocalMultiStarter is a multistart strategy that samples points in the local area
/// defined by the trust region and the xlimits.
pub(crate) struct LocalLhsMultiStarter<R: Rng + Clone> {
    xlimits: Array2<f64>,
    origin: Array1<f64>,
    max_dist: f64,
    rng: R,
}

impl<R: Rng + Clone> LocalLhsMultiStarter<R> {
    pub(crate) fn new(xlimits: Array2<f64>, origin: Array1<f64>, max_dist: f64, rng: R) -> Self {
        Self {
            xlimits,
            origin,
            max_dist,
            rng,
        }
    }
}

impl<R: Rng + Clone> MultiStarter for LocalLhsMultiStarter<R> {
    fn multistart(&mut self, n_start: usize, active: &[usize]) -> Array2<f64> {
        // Draw n_start initial points (multistart optim) in the local_area
        // xbounds = intersection(trust_region, xlimits)
        let xbounds = self.xbounds(active);
        let lhs = Lhs::new(&xbounds)
            .kind(LhsKind::Maximin)
            .with_rng(&mut self.rng);
        lhs.sample(n_start)
    }

    fn xbounds(&self, active: &[usize]) -> Array2<f64> {
        // local_area = intersection(trust_region, xlimits[active])
        let xlimits = coego::get_active_x(Axis(0), &self.xlimits, active);
        let origin = coego::get_active_x(Axis(0), &self.origin, active);
        let mut local_area = Array2::zeros(xlimits.dim());
        Zip::from(local_area.rows_mut())
            .and(&origin)
            .and(xlimits.rows())
            .for_each(|mut row, xb, xlims| {
                let (lo, up) = (
                    xlims[0].max(xb - self.max_dist),
                    xlims[1].min(xb + self.max_dist),
                );
                row.assign(&aview1(&[lo, up]))
            });
        local_area
    }
}
