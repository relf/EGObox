use std::collections::HashMap;

use linfa::traits::Transformer;

use linfa_clustering::Dbscan;
use ndarray::{Array1, Array2, ArrayBase, Data, Ix2};

const PORTFOLIO_MAX_CANDIDATES: usize = 3;
const PORTFOLIO_DBSCAN_MIN_POINTS: usize = 2;
const PORTFOLIO_DBSCAN_TOLERANCE_FACTOR: f64 = 0.1;

/// Generate `num` points spaced evenly on a log scale between `start` and `end`.
#[allow(dead_code)]
pub fn logspace(start: f64, end: f64, num: usize) -> Array1<f64> {
    assert!(start > 0.0, "logspace requires start > 0");
    assert!(end > 0.0, "logspace requires end > 0");
    assert!(num >= 2, "logspace requires at least 2 points");

    let log_start = start.log10();
    let log_end = end.log10();
    Array1::from_iter((0..num).map(|i| {
        let t = i as f64 / (num as f64 - 1.0);
        10f64.powf(log_start + t * (log_end - log_start))
    }))
}

pub fn cluster_as_indices(
    xdat: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    xlimits: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    infill_values: &[f64],
) -> Vec<usize> {
    let mut normalized_xdat = xdat.to_owned();
    for (column_index, xlimits_row) in xlimits.rows().into_iter().enumerate() {
        let lower = xlimits_row[0];
        let upper = xlimits_row[1];
        let scale = upper - lower;
        if scale > 0.0 {
            let mut column = normalized_xdat.column_mut(column_index);
            column.mapv_inplace(|value| (value - lower) / scale);
        }
    }

    // Cluster the x information
    let clusters = Dbscan::params(PORTFOLIO_DBSCAN_MIN_POINTS)
        .tolerance(PORTFOLIO_DBSCAN_TOLERANCE_FACTOR * (xdat.ncols() as f64).sqrt())
        .transform(&normalized_xdat)
        .unwrap();

    let mut dict = HashMap::new();
    let mut singleton_indices = Vec::new();
    for (i, c) in clusters.iter().enumerate() {
        match c {
            None => singleton_indices.push(i),
            Some(label) => dict.entry(*label).or_insert(vec![]).push(i),
        }
    }

    let mut representatives = dict
        .values()
        .map(|members| {
            *members
                .iter()
                .max_by(|left, right| {
                    infill_values[**left]
                        .partial_cmp(&infill_values[**right])
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap()
        })
        .collect::<Vec<_>>();
    representatives.extend(singleton_indices);
    representatives.sort_by(|left, right| {
        infill_values[*right]
            .partial_cmp(&infill_values[*left])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    representatives.truncate(PORTFOLIO_MAX_CANDIDATES);
    representatives
}

/// This function clusters portfolio information wrt x values then pick one member of each cluster
#[allow(clippy::type_complexity)]
pub fn select_from_portfolio(
    portfolio: Vec<(Array2<f64>, Array2<f64>, Array2<f64>, Array2<f64>, f64)>,
    xlimits: &ArrayBase<impl Data<Elem = f64>, Ix2>,
) -> (Array2<f64>, Array2<f64>, Array2<f64>, Array2<f64>, f64) {
    let n = portfolio.len();
    let mut xdat = Array2::zeros((n, portfolio[0].0.ncols()));

    for (i, info) in portfolio.iter().enumerate() {
        // Pick the first x (row(0)) as representative of q points batch
        xdat.row_mut(i).assign(&info.0.row(0).to_owned());
    }

    let infill_values = portfolio.iter().map(|info| info.4).collect::<Vec<_>>();

    // Indices of representative of a cluster
    let indices = cluster_as_indices(&xdat, xlimits, &infill_values);

    log::info!("Detect {} clusters", indices.len());

    // Pick information from portfolio of given indices and concatenate
    let nclusters = indices.len();

    let mut xdat = Array2::zeros((nclusters.max(1), portfolio[0].0.ncols()));
    let mut ydat = Array2::zeros((nclusters.max(1), portfolio[0].1.ncols()));
    let mut cdat = Array2::zeros((nclusters.max(1), portfolio[0].2.ncols()));
    let mut ypen = Array2::zeros((nclusters.max(1), portfolio[0].3.ncols()));

    if nclusters > 1 {
        for (i, index) in indices.iter().enumerate() {
            xdat.row_mut(i).assign(&portfolio[*index].0.row(0));
            ydat.row_mut(i).assign(&portfolio[*index].1.row(0));
            cdat.row_mut(i).assign(&portfolio[*index].2.row(0));
            ypen.row_mut(i).assign(&portfolio[*index].3.row(0));
        }
    } else {
        xdat.row_mut(0).assign(&portfolio[0].0.row(0));
        ydat.row_mut(0).assign(&portfolio[0].1.row(0));
        cdat.row_mut(0).assign(&portfolio[0].2.row(0));
        ypen.row_mut(0).assign(&portfolio[0].3.row(0));
    }
    let best_infill_value = indices
        .first()
        .map(|index| portfolio[*index].4)
        .unwrap_or(portfolio[0].4);
    (xdat, ydat, cdat, ypen, best_infill_value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_logspace_endpoints() {
        let vals = logspace(0.1, 100.0, 5);
        assert!((vals[0] - 0.1).abs() < 1e-12);
        assert!((vals[4] - 100.0).abs() < 1e-12);
    }

    #[test]
    fn test_logspace_length() {
        let vals = logspace(0.1, 100.0, 13);
        println!("{vals}");
        assert_eq!(vals.len(), 13);
    }

    #[test]
    fn test_logspace_monotonic_increasing() {
        let vals = logspace(1e-3, 1e3, 20);
        for i in 1..vals.len() {
            assert!(vals[i] > vals[i - 1], "Values must be strictly increasing");
        }
    }

    #[test]
    fn test_logspace_known_values() {
        // logspace(1, 100, 3) should give [1, 10, 100]
        let vals = logspace(1.0, 100.0, 3);
        let expected = [1.0, 10.0, 100.0];
        for (v, e) in vals.iter().zip(expected.iter()) {
            assert!((*v - *e).abs() < 1e-12);
        }
    }

    #[test]
    fn test_clustering() {
        let x = array![
            [0.13],
            [0.70],
            [0.72],
            [0.14],
            [0.15],
            [0.71],
            [0.16],
            [0.73]
        ];
        let xlimits = array![[0.0, 1.0]];
        let cluster_memberships =
            cluster_as_indices(&x, &xlimits, &[1.0, 2.0, 5.0, 3.0, 4.0, 6.0, 0.5, 7.0]);
        assert_eq!(cluster_memberships, vec![7, 4]);
    }

    #[test]
    fn test_clustering_keeps_top_three_noise_points() {
        let x = array![[0.0], [0.3], [0.6], [0.9]];
        let xlimits = array![[0.0, 1.0]];
        let cluster_memberships = cluster_as_indices(&x, &xlimits, &[1.0, 4.0, 3.0, 2.0]);
        assert_eq!(cluster_memberships, vec![1, 2, 3]);
    }

    #[test]
    fn test_clustering_is_scale_invariant_with_xlimits() {
        let x_small = array![[0.10], [0.11], [0.90], [0.91]];
        let x_large = array![[10.0], [11.0], [90.0], [91.0]];
        let unit_limits = array![[0.0, 1.0]];
        let large_limits = array![[0.0, 100.0]];
        let infill_values = [1.0, 5.0, 2.0, 4.0];

        let small = cluster_as_indices(&x_small, &unit_limits, &infill_values);
        let large = cluster_as_indices(&x_large, &large_limits, &infill_values);

        assert_eq!(small, vec![1, 3]);
        assert_eq!(large, small);
    }

    #[test]
    fn test_select_from_portfolio_uses_best_infill_representatives() {
        let portfolio = vec![
            (
                array![[0.13]],
                array![[13.0]],
                array![[1.3]],
                array![[130.0]],
                1.0,
            ),
            (
                array![[0.70]],
                array![[70.0]],
                array![[7.0]],
                array![[700.0]],
                2.0,
            ),
            (
                array![[0.72]],
                array![[72.0]],
                array![[7.2]],
                array![[720.0]],
                5.0,
            ),
            (
                array![[0.14]],
                array![[14.0]],
                array![[1.4]],
                array![[140.0]],
                3.0,
            ),
            (
                array![[0.15]],
                array![[15.0]],
                array![[1.5]],
                array![[150.0]],
                4.0,
            ),
            (
                array![[0.71]],
                array![[71.0]],
                array![[7.1]],
                array![[710.0]],
                6.0,
            ),
            (
                array![[0.16]],
                array![[16.0]],
                array![[1.6]],
                array![[160.0]],
                0.5,
            ),
            (
                array![[0.73]],
                array![[73.0]],
                array![[7.3]],
                array![[730.0]],
                7.0,
            ),
        ];
        let xlimits = array![[0.0, 1.0]];

        let (xdat, ydat, cdat, ypen, infill_value) = select_from_portfolio(portfolio, &xlimits);

        assert_eq!(xdat.column(0).to_vec(), vec![0.73, 0.15]);
        assert_eq!(ydat.column(0).to_vec(), vec![73.0, 15.0]);
        assert_eq!(cdat.column(0).to_vec(), vec![7.3, 1.5]);
        assert_eq!(ypen.column(0).to_vec(), vec![730.0, 150.0]);
        assert_eq!(infill_value, 7.0);
    }
}
