use crate::SamplingMethod;
use crate::utils::{cdist, pdist};
use linfa::Float;
use ndarray::{Array, Array2, ArrayBase, Axis, Data, Ix2, ShapeBuilder, s};
use ndarray_rand::{
    RandomExt, rand::Rng, rand::SeedableRng, rand::seq::SliceRandom, rand_distr::Uniform,
};
use ndarray_stats::QuantileExt;
use rand_xoshiro::Xoshiro256Plus;
use std::cmp;
use std::sync::{Arc, RwLock};

#[cfg(feature = "serializable")]
use serde::{Deserialize, Serialize};

/// Kinds of Latin Hypercube Design
#[derive(Clone, Debug, Default, Copy)]
#[cfg_attr(feature = "serializable", derive(Serialize, Deserialize))]
pub enum LhsKind {
    /// sample is choosen randomly within its latin hypercube intervals
    Classic,
    /// sample is the middle of its latin hypercube intervals
    Centered,
    /// distance between points is maximized
    Maximin,
    /// sample is the middle of its latin hypercube intervals and distance between points is maximized
    CenteredMaximin,
    /// samples locations is optimized using the Enhanced Stochastic Evolutionary algorithm (ESE)
    /// See Jin, R. and Chen, W. and Sudjianto, A. (2005), “An efficient algorithm for constructing
    /// optimal design of computer experiments.” Journal of Statistical Planning and Inference, 134:268-287.
    #[default]
    Optimized,
}

type RngRef<R> = Arc<RwLock<R>>;

/// The LHS design is built as follows: each dimension space is divided into `ns` sections
/// where `ns` is the number of sampling points, and one point is selected in each section.
/// The selection method corresponds to different kinds of LHS: see [`LhsKind`]
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serializable", derive(Serialize, Deserialize))]
pub struct Lhs<F: Float, R: Rng> {
    /// Sampling space definition as a (nx, 2) matrix
    /// The ith row is the [lower_bound, upper_bound] of xi, the ith component of x
    xlimits: Array2<F>,
    /// The requested kind of LHS
    kind: LhsKind,
    /// Random generator used for reproducibility (not used in case of Centered LHS)
    rng: RngRef<R>,
}

/// LHS with default random generator
impl<F: Float> Lhs<F, Xoshiro256Plus> {
    /// Constructor given a design space given a (nx, 2) matrix \[\[lower bound, upper bound\], ...\]
    ///
    /// ```
    /// use egobox_doe::Lhs;
    /// use ndarray::arr2;
    ///
    /// let doe = Lhs::new(&arr2(&[[0.0, 1.0], [5.0, 10.0]]));
    /// ```
    pub fn new(xlimits: &ArrayBase<impl Data<Elem = F>, Ix2>) -> Self {
        Self::new_with_rng(xlimits, Xoshiro256Plus::from_entropy())
    }
}

impl<F: Float, R: Rng> SamplingMethod<F> for Lhs<F, R> {
    fn sampling_space(&self) -> &Array2<F> {
        &self.xlimits
    }

    fn normalized_sample(&self, ns: usize) -> Array2<F> {
        match &self.kind {
            LhsKind::Classic => self._classic_lhs(ns),
            LhsKind::Centered => self._centered_lhs(ns),
            LhsKind::Maximin => self._maximin_lhs(ns, false, 5),
            LhsKind::CenteredMaximin => self._maximin_lhs(ns, true, 5),
            LhsKind::Optimized => {
                let doe = self._classic_lhs(ns);
                let nx = self.xlimits.nrows();
                let outer_loop = cmp::min((1.5 * nx as f64) as usize, 30);
                let inner_loop = cmp::min(20 * nx, 100);
                self._maximin_ese(&doe, outer_loop, inner_loop)
            }
        }
    }
}

impl<F: Float, R: Rng> Lhs<F, R> {
    /// Constructor with given design space and random generator.
    /// * `xlimits`: (nx, 2) matrix where nx is the dimension of the samples and the ith row
    ///   is the definition interval of the ith component of x.
    /// * `rng`: random generator used for [LhsKind::Classic] and [LhsKind::Optimized] LHS
    pub fn new_with_rng(xlimits: &ArrayBase<impl Data<Elem = F>, Ix2>, rng: R) -> Self {
        if xlimits.ncols() != 2 {
            panic!("xlimits must have 2 columns (lower, upper)");
        }
        Lhs {
            xlimits: xlimits.to_owned(),
            kind: LhsKind::default(),
            rng: Arc::new(RwLock::new(rng)),
        }
    }

    /// Sets the kind of LHS
    pub fn kind(mut self, kind: LhsKind) -> Self {
        self.kind = kind;
        self
    }

    /// Sets the random generator
    pub fn with_rng<R2: Rng>(self, rng: R2) -> Lhs<F, R2> {
        Lhs {
            xlimits: self.xlimits,
            kind: self.kind,
            rng: Arc::new(RwLock::new(rng)),
        }
    }

    fn _maximin_ese(&self, lhs: &Array2<F>, outer_loop: usize, inner_loop: usize) -> Array2<F> {
        // hard-coded params
        let j_range = 20;
        let p = F::cast(10.);
        let t0 = F::cast(0.005) * self._phip(lhs, p);
        let tol = F::cast(1e-3);

        let mut t = t0;
        let mut lhs_own = lhs.to_owned();
        let mut lhs_best = lhs.to_owned();
        let nx = lhs.ncols();
        let mut phip = self._phip(&lhs_best, p);
        let mut phip_best = phip;

        for _ in 0..outer_loop {
            let mut n_acpt = 0.;
            let mut n_imp = 0.;

            for i in 0..inner_loop {
                let modulo = (i + 1) % nx;
                let mut l_x: Vec<Array2<F>> = Vec::with_capacity(j_range);
                let mut l_phip: Vec<F> = Vec::with_capacity(j_range);

                // Build j different plans with a single swap procedure
                // See description of phip_swap procedure
                let mut rng = self.rng.write().unwrap();
                for j in 0..j_range {
                    l_x.push(lhs_own.to_owned());
                    let php = self._phip_swap(&mut l_x[j], modulo, phip, p, &mut *rng);
                    l_phip.push(php);
                }
                let lphip = Array::from_shape_vec(j_range, l_phip).unwrap();
                let k = lphip.argmin().unwrap();
                let phip_try = lphip[k];
                // Threshold of acceptance
                if phip_try - phip <= t * F::cast(rng.r#gen::<f64>()) {
                    phip = phip_try;
                    n_acpt += 1.;
                    lhs_own = l_x.swap_remove(k);

                    // best plan retained
                    if phip < phip_best {
                        lhs_best = lhs_own.to_owned();
                        phip_best = phip;
                        n_imp += 1.;
                    }
                }
            }
            let p_accpt = n_acpt / (inner_loop as f64); // probability of acceptance
            let p_imp = n_imp / (inner_loop as f64); // probability of improvement

            if phip - phip_best > tol {
                if p_accpt >= 0.1 && p_imp < p_accpt {
                    t *= F::cast(0.8)
                } else if p_accpt >= 0.1 && (p_imp - p_accpt).abs() < f64::EPSILON {
                } else {
                    t /= F::cast(0.8)
                }
            } else if p_accpt <= 0.1 {
                t /= F::cast(0.7)
            } else {
                t *= F::cast(0.9)
            }
        }
        lhs_best
    }

    fn _phip(&self, lhs: &ArrayBase<impl Data<Elem = F> + Sync, Ix2>, p: F) -> F {
        F::powf(pdist(lhs).mapv(|v| F::powf(v, -p)).sum(), F::one() / p)
    }

    fn _phip_swap(&self, x: &mut Array2<F>, k: usize, phip: F, p: F, rng: &mut R) -> F {
        // Choose two random rows
        let i1 = rng.gen_range(0..x.nrows());
        let mut i2 = rng.gen_range(0..x.nrows());
        while i2 == i1 {
            i2 = rng.gen_range(0..x.nrows());
        }
        // Compute new phip
        let mut x_rest = Array2::zeros((x.nrows() - 2, x.ncols()));
        let mut row_i = 0;
        for (i, row) in x.axis_iter(Axis(0)).enumerate() {
            if i != i1 && i != i2 {
                x_rest.row_mut(row_i).assign(&row);
                row_i += 1;
            }
        }

        let mut dist1 = cdist(&x.slice(s![i1..i1 + 1, ..]), &x_rest);
        let mut dist2 = cdist(&x.slice(s![i2..i2 + 1, ..]), &x_rest);

        let m1 = x_rest.column(k).mapv(|v| {
            let diff = v - x[[i1, k]];
            diff * diff
        });
        let m2 = x_rest.column(k).mapv(|v| {
            let diff = v - x[[i2, k]];
            diff * diff
        });

        let two = F::cast(2.);
        let mut d1 = dist1.mapv(|v| v * v) - &m1 + &m2;
        d1.mapv_inplace(|v| F::powf(v, -p / two));
        let mut d2 = dist2.mapv(|v| v * v) + &m1 - &m2;
        d2.mapv_inplace(|v| F::powf(v, -p / two));

        dist1.mapv_inplace(|v| F::powf(v, -p));
        dist2.mapv_inplace(|v| F::powf(v, -p));
        let mut res = (d1 - dist1).sum() + (d2 - dist2).sum();
        res = F::powf(F::powf(phip, p) + res, F::one() / p);

        // swap points
        x.swap([i1, k], [i2, k]);
        res
    }

    fn _classic_lhs(&self, ns: usize) -> Array2<F> {
        let nx = self.xlimits.nrows();
        let cut = Array::linspace(0., 1., ns + 1);

        let mut rng = self.rng.write().unwrap();
        let rnd = Array::random_using((ns, nx).f(), Uniform::new(0., 1.), &mut *rng);
        let a = cut.slice(s![..ns]).to_owned();
        let b = cut.slice(s![1..(ns + 1)]);
        let c = &b - &a;
        let mut rdpoints = Array::zeros((ns, nx).f());
        for j in 0..nx {
            let d = rnd.column(j).to_owned() * &c + &a;
            rdpoints.column_mut(j).assign(&d)
        }
        let mut lhs = Array::zeros((ns, nx).f());
        for j in 0..nx {
            let mut colj = rdpoints.column_mut(j);
            colj.as_slice_mut().unwrap().shuffle(&mut *rng);
            lhs.column_mut(j).assign(&colj);
        }
        lhs.mapv_into_any(F::cast)
    }

    fn _centered_lhs(&self, ns: usize) -> Array2<F> {
        let nx = self.xlimits.nrows();
        let cut = Array::linspace(0., 1., ns + 1);

        let a = cut.slice(s![..ns]).to_owned();
        let b = cut.slice(s![1..(ns + 1)]);
        let mut c = (a + b) / 2.;
        let mut lhs = Array::zeros((ns, nx).f());

        let mut rng = self.rng.write().unwrap();
        for j in 0..nx {
            c.as_slice_mut().unwrap().shuffle(&mut *rng);
            lhs.column_mut(j).assign(&c);
        }
        lhs.mapv_into_any(F::cast)
    }

    fn _maximin_lhs(&self, ns: usize, centered: bool, max_iters: usize) -> Array2<F> {
        let mut lhs = if centered {
            self._centered_lhs(ns)
        } else {
            self._classic_lhs(ns)
        };
        let mut max_dist = *pdist(&lhs).min().unwrap();
        let mut lhs_maximin = lhs;
        for _ in 0..max_iters - 1 {
            if centered {
                lhs = self._centered_lhs(ns);
            } else {
                lhs = self._classic_lhs(ns);
            }
            let d_min = *pdist(&lhs).min().unwrap();
            if max_dist < d_min {
                max_dist = d_min;
                std::mem::swap(&mut lhs_maximin, &mut lhs)
            }
        }
        lhs_maximin
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::{assert_abs_diff_eq, assert_abs_diff_ne};
    use ndarray::{arr2, array};
    use std::time::Instant;

    #[test]
    fn test_lhs() {
        let xlimits = arr2(&[[5., 10.], [0., 1.]]);
        let expected = array![
            [9.000042958859238, 0.2175219214807449],
            [5.085755595295461, 0.7725590934255249],
            [7.062569781563214, 0.44540674774531397],
            [8.306461322653673, 0.9046507902710129],
            [6.310411395727105, 0.0606130622609971]
        ];
        let actual = Lhs::new(&xlimits)
            .with_rng(Xoshiro256Plus::seed_from_u64(42))
            .sample(5);
        assert_abs_diff_eq!(expected, actual, epsilon = 1e-6);
    }

    #[test]
    fn test_lhs_speed() {
        let start = Instant::now();
        let xlimits = arr2(&[[0., 1.], [0., 1.]]);
        let n = 10;
        let _actual = Lhs::new(&xlimits).sample(n);
        let duration = start.elapsed();
        println!("Time elapsed in optimized LHS is: {duration:?}");
    }

    #[test]
    fn test_classic_lhs() {
        let xlimits = arr2(&[[5., 10.], [0., 1.]]);
        let expected = array![
            [9.000042958859238, 0.44540674774531397],
            [5.085755595295461, 0.7725590934255249],
            [7.062569781563214, 0.2175219214807449],
            [8.306461322653673, 0.9046507902710129],
            [6.310411395727105, 0.0606130622609971]
        ];
        let actual = Lhs::new(&xlimits)
            .with_rng(Xoshiro256Plus::seed_from_u64(42))
            .kind(LhsKind::Classic)
            .sample(5);
        assert_abs_diff_eq!(expected, actual, epsilon = 1e-6);
    }

    #[test]
    fn test_centered_lhs() {
        let xlimits = arr2(&[[5., 10.], [0., 1.]]);
        let expected = array![[7.5, 0.9], [8.5, 0.1], [5.5, 0.7], [6.5, 0.3], [9.5, 0.5]];
        let actual = Lhs::new(&xlimits)
            .with_rng(Xoshiro256Plus::seed_from_u64(0))
            .kind(LhsKind::Centered)
            .sample(5);
        assert_abs_diff_eq!(expected, actual, epsilon = 1e-6);
    }

    #[test]
    fn test_centered_maximin_lhs() {
        let xlimits = arr2(&[[5., 10.], [0., 1.]]);
        let expected = array![[7.5, 0.9], [8.5, 0.1], [5.5, 0.7], [6.5, 0.3], [9.5, 0.5]];
        let actual = Lhs::new(&xlimits)
            .with_rng(Xoshiro256Plus::seed_from_u64(0))
            .kind(LhsKind::CenteredMaximin)
            .sample(5);
        assert_abs_diff_eq!(expected, actual, epsilon = 1e-6);
    }

    #[test]
    fn test_phip_swap() {
        let xlimits = arr2(&[[0., 1.], [0., 1.]]);
        let k = 1;
        let phip = 7.290525742903316;
        let mut p0 = array![
            [0.45, 0.75],
            [0.75, 0.95],
            [0.05, 0.45],
            [0.55, 0.15000000000000002],
            [0.35000000000000003, 0.25],
            [0.95, 0.8500000000000001],
            [0.15000000000000002, 0.55],
            [0.25, 0.05],
            [0.8500000000000001, 0.35000000000000003],
            [0.6500000000000001, 0.6500000000000001]
        ];
        let p = 10.;
        let mut rng = Xoshiro256Plus::seed_from_u64(42);
        let _res = Lhs::new(&xlimits)._phip_swap(&mut p0, k, phip, p, &mut rng);
    }

    #[test]
    fn test_no_duplicate() {
        let xlimits = arr2(&[[5., 10.], [0., 1.]]);
        let lhs = Lhs::new(&xlimits).with_rng(Xoshiro256Plus::seed_from_u64(42));

        let sample1 = lhs.sample(5);
        let sample2 = lhs.sample(5);
        assert_abs_diff_ne!(sample1, sample2);
    }

    #[test]
    fn test_lhs_clone_different() {
        let xlimits = array![[-1., 1.]];
        let rng = Xoshiro256Plus::seed_from_u64(42);
        let lhs = Lhs::new(&xlimits)
            .kind(LhsKind::Classic)
            .with_rng(rng.clone());
        let lhs1 = lhs.clone();
        let s1 = lhs1.sample(10);
        let lhs2 = lhs.clone();
        let s2 = lhs2.sample(10);
        assert_abs_diff_ne!(s1, s2);
    }
}
