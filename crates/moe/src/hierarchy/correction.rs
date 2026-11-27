//! Search for the nearest valid configuration, and count of valid configurations
//!
//! Both walk the variables in topological order: when a variable is reached,
//! the variables its inclusion and its allowed values depend on already have
//! their values.
//!
//! The values of a discrete variable that no condition and no value
//! constraint mentions all have the same effect on the other variables. Only
//! one of them, the nearest one, needs to be tried, and they can be counted
//! together. This keeps both walks cheap for integer variables with large
//! ranges.

use super::{DesignSpace, domain_size};
use crate::mixint::space::unfolded_width;
use crate::xtypes::XType;
use std::cmp::Ordering;

impl DesignSpace {
    /// Nearest valid configuration to a point of the relaxed space.
    ///
    /// Returns the folded value and the inclusion of each variable, or `None`
    /// when there is no valid configuration. `starts` gives the first column
    /// of each variable in `relaxed`.
    pub(super) fn correct_row(
        &self,
        relaxed: &[f64],
        starts: &[usize],
    ) -> Option<(Vec<f64>, Vec<bool>)> {
        let n = self.xtypes.len();
        let mut folded = vec![0.; n];
        let mut included = vec![false; n];
        self.search(0, relaxed, starts, &mut folded, &mut included)
            .then_some((folded, included))
    }

    /// Whether at least one valid configuration exists
    pub(super) fn has_valid_configuration(&self) -> bool {
        let starts = self.unfolded_starts();
        let dim = self.xtypes.iter().map(unfolded_width).sum();
        let mut relaxed = vec![0.; dim];
        for (xtype, &start) in self.xtypes.iter().zip(&starts) {
            match xtype {
                XType::Enum(_) => relaxed[start] = 1.,
                _ => relaxed[start] = default_value(xtype),
            }
        }
        self.correct_row(&relaxed, &starts).is_some()
    }

    /// Number of valid configurations of the discrete variables
    pub(super) fn count_valid(&self) -> u128 {
        let n = self.xtypes.len();
        self.count(0, &mut vec![0.; n], &mut vec![false; n])
    }

    /// Depth-first search from the `k`-th variable in topological order,
    /// trying the values of each variable nearest first. Returns whether a
    /// valid configuration was found, which `folded` and `included` then hold.
    fn search(
        &self,
        k: usize,
        relaxed: &[f64],
        starts: &[usize],
        folded: &mut [f64],
        included: &mut [bool],
    ) -> bool {
        let Some(&i) = self.order.get(k) else {
            return true;
        };
        included[i] = self.is_included(i, folded, included);
        let block = &relaxed[starts[i]..starts[i] + unfolded_width(&self.xtypes[i])];
        for v in self.candidates(i, block, folded, included) {
            folded[i] = v;
            if self.search(k + 1, relaxed, starts, folded, included) {
                return true;
            }
        }
        false
    }

    /// Number of valid configurations of the variables from the `k`-th one in
    /// topological order, given the values of the variables before it
    fn count(&self, k: usize, folded: &mut [f64], included: &mut [bool]) -> u128 {
        let Some(&i) = self.order.get(k) else {
            return 1;
        };
        included[i] = self.is_included(i, folded, included);
        let xtype = &self.xtypes[i];
        let Some(size) = domain_size(xtype).filter(|_| included[i]) else {
            // Continuous or excluded: no choice to count
            folded[i] = default_value(xtype);
            return self.count(k + 1, folded, included);
        };
        let forbidden = self.forbidden(i, folded, included);
        let mut total: u128 = 0;
        for &v in &self.mentioned[i] {
            if !forbidden.contains(&v) {
                folded[i] = v;
                total = total.saturating_add(self.count(k + 1, folded, included));
            }
        }
        let others = size.saturating_sub(self.mentioned[i].len() as u128);
        if others > 0
            && let Some(&v) = self.unmentioned(i, default_value(xtype)).first()
        {
            folded[i] = v;
            total =
                total.saturating_add(others.saturating_mul(self.count(k + 1, folded, included)));
        }
        total
    }

    /// Whether variable `i` is included, given the variables before it
    fn is_included(&self, i: usize, folded: &[f64], included: &[bool]) -> bool {
        self.conditions[i]
            .as_ref()
            .is_none_or(|c| c.holds(folded, included))
    }

    /// Values forbidden to variable `i` by the variables before it
    fn forbidden(&self, i: usize, folded: &[f64], included: &[bool]) -> Vec<f64> {
        self.restrictions[i]
            .iter()
            .filter(|r| included[r.source] && r.source_values.contains(&folded[r.source]))
            .flat_map(|r| r.forbidden.iter().copied())
            .collect()
    }

    /// Values to try for variable `i`, nearest to its relaxed columns `block`
    /// first: its default when excluded, its relaxed value when continuous,
    /// and otherwise its allowed mentioned values and the nearest unmentioned
    /// one
    fn candidates(&self, i: usize, block: &[f64], folded: &[f64], included: &[bool]) -> Vec<f64> {
        let xtype = &self.xtypes[i];
        if !included[i] {
            return vec![default_value(xtype)];
        }
        if let XType::Float(_, _) = xtype {
            return vec![block[0]];
        }
        let forbidden = self.forbidden(i, folded, included);
        let nearness = |a: &f64, b: &f64| nearness(xtype, block, *a, *b);
        let mut values: Vec<f64> = self.mentioned[i]
            .iter()
            .copied()
            .filter(|v| !forbidden.contains(v))
            .collect();
        values.extend(self.unmentioned(i, block[0]).into_iter().min_by(nearness));
        values.sort_by(nearness);
        values
    }

    /// Values of discrete variable `i` that no condition or value constraint
    /// mentions. For an Int variable, only the ones that can be nearest to
    /// `around`: its rounded value, the bounds, and the neighbours of the
    /// mentioned values.
    fn unmentioned(&self, i: usize, around: f64) -> Vec<f64> {
        let mentioned = &self.mentioned[i];
        let values: Vec<f64> = match &self.xtypes[i] {
            XType::Float(_, _) => vec![],
            XType::Int(lb, ub) => {
                let (lb, ub) = (*lb as f64, *ub as f64);
                let mut values = vec![around.round().clamp(lb, ub), lb, ub];
                for m in mentioned {
                    values.extend([m - 1., m + 1.]);
                }
                values.retain(|v| (lb..=ub).contains(v));
                values
            }
            XType::Ord(levels) => levels.clone(),
            XType::Enum(n) => (0..*n).map(|l| l as f64).collect(),
        };
        values
            .into_iter()
            .filter(|v| !mentioned.contains(v))
            .collect()
    }
}

/// Default value of an excluded variable, in folded space: the lower bound,
/// the smallest level, or the first level of an Enum
fn default_value(xtype: &XType) -> f64 {
    match xtype {
        XType::Float(lb, _) => *lb,
        XType::Int(lb, _) => *lb as f64,
        XType::Ord(levels) => levels.iter().copied().fold(f64::INFINITY, f64::min),
        XType::Enum(_) => 0.,
    }
}

/// Order of two values of a variable, nearest to its relaxed columns `block`
/// first, with the tie-breaks of the mixed-integer cast: away from zero for
/// Int (as rounding does), first listed level for Ord, lowest level for Enum
fn nearness(xtype: &XType, block: &[f64], a: f64, b: f64) -> Ordering {
    let distance = |v: f64| (v - block[0]).abs();
    match xtype {
        XType::Float(_, _) => Ordering::Equal,
        XType::Int(_, _) => distance(a)
            .total_cmp(&distance(b))
            .then(b.abs().total_cmp(&a.abs())),
        XType::Ord(levels) => {
            let position = |v: f64| levels.iter().position(|&l| l == v);
            distance(a)
                .total_cmp(&distance(b))
                .then(position(a).cmp(&position(b)))
        }
        XType::Enum(_) => block[b as usize]
            .total_cmp(&block[a as usize])
            .then(a.total_cmp(&b)),
    }
}

#[cfg(test)]
mod tests {
    use crate::{DesignSpace, Role, XType};
    use ndarray::array;

    const ASGD: f64 = 0.;
    const ADAM: f64 = 1.;

    /// The MLP problem of Table 1 of the reference, with the decrees and value
    /// constraints of its Fig. 14: no 3 layers with Adam, and no 40 or 45
    /// units in a layer with ASGD
    fn mlp() -> DesignSpace {
        let units = XType::Ord(vec![25., 30., 35., 40., 45.]);
        let mut ds = DesignSpace::named(&[
            ("learning_rate", XType::Float(0., 1.)),
            ("activation", XType::Enum(3)),
            ("optimizer", XType::Enum(2)),
            ("decay", XType::Float(0., 1.)),
            ("power_update", XType::Float(0., 1.)),
            ("average_start", XType::Float(1e3, 1e8)),
            ("running_average_1", XType::Float(0., 1.)),
            ("running_average_2", XType::Float(0., 1.)),
            ("numerical_stability", XType::Float(0., 1.)),
            ("layers", XType::Int(1, 3)),
            ("units_1", units.clone()),
            ("units_2", units.clone()),
            ("units_3", units),
        ])
        .unwrap();
        for var in ["decay", "power_update", "average_start"] {
            ds.decreed_var(var, "optimizer", &[ASGD]).unwrap();
        }
        for var in [
            "running_average_1",
            "running_average_2",
            "numerical_stability",
        ] {
            ds.decreed_var(var, "optimizer", &[ADAM]).unwrap();
        }
        ds.decreed_var("units_1", "layers", &[1., 2., 3.])
            .unwrap()
            .decreed_var("units_2", "layers", &[2., 3.])
            .unwrap()
            .decreed_var("units_3", "layers", &[3.])
            .unwrap()
            .forbid("layers", &[3.], "optimizer", &[ADAM])
            .unwrap();
        for var in ["units_1", "units_2", "units_3"] {
            ds.forbid(var, &[40., 45.], "optimizer", &[ASGD]).unwrap();
        }
        ds
    }

    /// DRAGON's propulsive architecture (Sect. 4.2 of the reference): the
    /// number of motors is a multiple of 4, 8 or 12 for 2, 4 or 6 cores.
    /// Table 3 lists 4 cores for architectures 15 to 17, but Table 4 gives
    /// cores in {2, 4, 6} and the text multiples of 4, 8 or 12, which this
    /// follows.
    fn dragon() -> DesignSpace {
        let motors = (2..=10).map(|k| 4. * k as f64).collect();
        let mut ds = DesignSpace::named(&[
            ("cores", XType::Ord(vec![2., 4., 6.])),
            ("motors", XType::Ord(motors)),
        ])
        .unwrap();
        ds.forbid("cores", &[4.], "motors", &[12., 20., 28., 36.])
            .unwrap()
            // Variables in either order: motors is restricted either way
            .forbid("motors", &[8., 16., 20., 28., 32., 40.], "cores", &[6.])
            .unwrap();
        ds
    }

    #[test]
    fn test_mlp_counts() {
        // Table 6 of the reference
        let ds = mlp();
        assert_eq!(ds.n_valid(), 207);
        assert_eq!(ds.n_declared(), 2250);
    }

    #[test]
    fn test_mlp_roles() {
        let ds = mlp();
        assert_eq!(ds.role("activation"), Role::empty());
        assert_eq!(ds.role("optimizer"), Role::META);
        assert_eq!(ds.role("layers"), Role::META | Role::PARTIALLY_DECREED);
        assert_eq!(ds.role("units_1"), Role::DECREED | Role::PARTIALLY_DECREED);
        assert_eq!(ds.role("decay"), Role::DECREED);
        assert_eq!(ds.parents("layers"), vec![2]);
        assert_eq!(ds.parents("units_2"), vec![2, 9]);
        assert_eq!(ds.value_constraints().len(), 4);
    }

    #[test]
    fn test_mlp_correction() {
        let ds = mlp();
        let x = array![
            // Adam with 2.6 layers: 3 layers are forbidden, so 2
            [
                0.5, 2., ADAM, 0.5, 0.5, 5e3, 0.5, 0.5, 0.5, 2.6, 42., 42., 42.
            ],
            // ASGD: 40 and 45 units are forbidden, so 35
            [
                0.5, 0., ASGD, 0.2, 0.3, 5e3, 0.5, 0.5, 0.5, 2.8, 42., 38., 27.
            ],
        ];
        let (xcorr, included) = ds.correct_get_included(&x).unwrap();
        assert_eq!(
            xcorr,
            array![
                [0.5, 2., ADAM, 0., 0., 1e3, 0.5, 0.5, 0.5, 2., 40., 40., 25.],
                [0.5, 0., ASGD, 0.2, 0.3, 5e3, 0., 0., 0., 3., 35., 35., 25.],
            ]
        );
        let (t, f) = (true, false);
        assert_eq!(
            included,
            array![
                [t, t, t, f, f, f, t, t, t, t, t, t, f],
                [t, t, t, t, t, t, f, f, f, t, t, t, t],
            ]
        );
    }

    #[test]
    fn test_dragon() {
        let ds = dragon();
        // The 17 architectures of Table 3, out of 3 x 9 combinations
        assert_eq!(ds.n_valid(), 17);
        assert_eq!(ds.n_declared(), 27);
        assert_eq!(ds.role("cores"), Role::META);
        assert_eq!(ds.role("motors"), Role::PARTIALLY_DECREED);

        let x = array![[6., 21.], [4., 13.], [2., 13.], [4.5, 36.]];
        let (xcorr, included) = ds.correct_get_included(&x).unwrap();
        // 32 and 40 are equally near 36: the first listed level wins
        assert_eq!(xcorr, array![[6., 24.], [4., 16.], [2., 12.], [4., 32.]]);
        assert!(included.iter().all(|&inc| inc));
    }

    #[test]
    fn test_backtracking() {
        // x0 = 2 leaves no level to x1: the search goes back to x0 and takes
        // its next nearest value
        let mut ds = DesignSpace::new(&[XType::Int(0, 2), XType::Enum(2)]);
        ds.forbid(0, &[2.], 1, &[0., 1.]).unwrap();
        let (xcorr, _) = ds.correct_get_included(&array![[1.8, 1.]]).unwrap();
        assert_eq!(xcorr, array![[1., 1.]]);
        assert_eq!(ds.n_valid(), 4);
        assert_eq!(ds.role(0), Role::META);
        assert_eq!(ds.role(1), Role::PARTIALLY_DECREED);
    }

    #[test]
    fn test_order_relation() {
        // l >= m + 1 between two integers, as one constraint per value of m
        let mut ds =
            DesignSpace::named(&[("m", XType::Int(0, 3)), ("l", XType::Int(1, 4))]).unwrap();
        for m in 1..=3 {
            let forbidden: Vec<f64> = (1..=m).map(f64::from).collect();
            ds.forbid("m", &[f64::from(m)], "l", &forbidden).unwrap();
        }
        assert_eq!(ds.n_valid(), 4 + 3 + 2 + 1);
        assert_eq!(ds.n_declared(), 16);
        let (xcorr, _) = ds
            .correct_get_included(&array![[2., 1.2], [2.6, 4.4], [0., 1.]])
            .unwrap();
        assert_eq!(xcorr, array![[2., 3.], [3., 4.], [0., 1.]]);
    }

    #[test]
    fn test_large_integer_range() {
        // Unmentioned values are tried and counted as one class
        let n = 1_000_000_000;
        let mut ds = DesignSpace::new(&[XType::Int(0, 1), XType::Int(0, n)]);
        ds.forbid(0, &[1.], 1, &[4., 5., 6.]).unwrap();
        let size = n as u128 + 1;
        assert_eq!(ds.n_declared(), 2 * size);
        assert_eq!(ds.n_valid(), 2 * size - 3);
        let x = array![[1., 5.2], [0., 5.2], [1., -3.], [0.2, 2e9]];
        let (xcorr, _) = ds.correct_get_included(&x).unwrap();
        // 4, 5 and 6 are forbidden: 7 is nearer to 5.2 than 3. Out of range
        // values are clamped to the bounds.
        assert_eq!(xcorr, array![[1., 7.], [0., 5.], [1., 0.], [0., n as f64]]);
    }

    #[test]
    fn test_excluded_variables_restrict_nothing() {
        // x1 is decreed by x0 and restricts x2, but only when it is included,
        // even though its default value is the one that forbids x2 = 0
        let mut ds = DesignSpace::new(&[XType::Enum(2), XType::Int(0, 3), XType::Enum(2)]);
        ds.decreed_var(1, 0, &[1.])
            .unwrap()
            .forbid(1, &[0.], 2, &[0.])
            .unwrap();
        let x = array![[0., 2., 0.], [1., 0.2, 0.]];
        let (xcorr, included) = ds.correct_get_included(&x).unwrap();
        assert_eq!(xcorr, array![[0., 0., 0.], [1., 0., 1.]]);
        assert_eq!(included, array![[true, false, true], [true, true, true]]);
        // x0 = 0: 2 values of x2; x0 = 1: 1 with x1 = 0, 2 for each other x1
        assert_eq!(ds.n_valid(), 2 + 1 + 3 * 2);
    }

    #[test]
    fn test_no_valid_configuration_is_rejected() {
        let mut ds = DesignSpace::new(&[XType::Int(0, 1), XType::Enum(2)]);
        ds.forbid(0, &[0.], 1, &[0., 1.]).unwrap();
        // x0 = 0 is left without any x1, so x0 = 1
        let (xcorr, _) = ds.correct_get_included(&array![[0.2, 0.]]).unwrap();
        assert_eq!(xcorr, array![[1., 0.]]);

        let before = serde_json::to_string(&ds).unwrap();
        let err = ds.forbid(0, &[1.], 1, &[0., 1.]).unwrap_err();
        assert!(err.to_string().contains("no valid configuration"), "{err}");
        assert_eq!(serde_json::to_string(&ds).unwrap(), before);
    }

    #[test]
    fn test_invalid_value_constraints() {
        let mut ds = mlp();
        let before = serde_json::to_string(&ds).unwrap();
        // Same variable twice, continuous variable, unknown value or variable
        assert!(ds.forbid("layers", &[1.], "layers", &[2.]).is_err());
        let err = ds
            .forbid("decay", &[0.5], "optimizer", &[ASGD])
            .unwrap_err();
        assert!(err.to_string().contains("3 (decay) is continuous"), "{err}");
        assert!(ds.forbid("layers", &[4.], "optimizer", &[ASGD]).is_err());
        assert!(ds.forbid("units_1", &[42.], "optimizer", &[ASGD]).is_err());
        assert!(ds.forbid("layers", &[], "optimizer", &[ASGD]).is_err());
        assert!(ds.forbid("depth", &[1.], "optimizer", &[ASGD]).is_err());
        assert_eq!(serde_json::to_string(&ds).unwrap(), before);
    }

    #[test]
    fn test_serde_with_value_constraints() {
        let ds = mlp();
        let json = serde_json::to_string(&ds).unwrap();
        let loaded: DesignSpace = serde_json::from_str(&json).unwrap();
        assert_eq!(serde_json::to_string(&loaded).unwrap(), json);
        assert_eq!(loaded.value_constraints(), ds.value_constraints());
        assert_eq!(loaded.n_valid(), 207);
        assert_eq!(loaded.role("layers"), ds.role("layers"));

        let bytes = postcard::to_allocvec(&ds).unwrap();
        let loaded: DesignSpace = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(serde_json::to_string(&loaded).unwrap(), json);

        // Loading checks the constraints again
        let none_left = r#"{"xtypes":[{"Int":[0,1]},{"Enum":2}],"conditions":[null,null],
            "constraints":[{"var1":0,"values1":[0.0,1.0],"var2":1,"values2":[0.0,1.0]}]}"#;
        assert!(serde_json::from_str::<DesignSpace>(none_left).is_err());
        let bad_value = r#"{"xtypes":[{"Int":[0,1]},{"Enum":2}],"conditions":[null,null],
            "constraints":[{"var1":0,"values1":[2.0],"var2":1,"values2":[0.0]}]}"#;
        assert!(serde_json::from_str::<DesignSpace>(bad_value).is_err());
    }
}
