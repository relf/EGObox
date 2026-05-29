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

fn squared_distance(
    xdat: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    left: usize,
    right: usize,
) -> f64 {
    xdat.row(left)
        .iter()
        .zip(xdat.row(right).iter())
        .map(|(left_value, right_value)| {
            let diff = left_value - right_value;
            diff * diff
        })
        .sum()
}

fn summarize_distances(distances: &[f64]) -> Option<(f64, f64, f64)> {
    if distances.is_empty() {
        return None;
    }

    let mut sorted = distances.to_vec();
    sorted.sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));

    let median = if sorted.len().is_multiple_of(2) {
        let upper = sorted.len() / 2;
        (sorted[upper - 1] + sorted[upper]) / 2.0
    } else {
        sorted[sorted.len() / 2]
    };

    Some((*sorted.first().unwrap(), median, *sorted.last().unwrap()))
}

fn pairwise_distance_diagnostics(
    xdat: &ArrayBase<impl Data<Elem = f64>, Ix2>,
) -> (Vec<f64>, Vec<(usize, usize, f64)>) {
    if xdat.nrows() <= 1 {
        return (Vec::new(), Vec::new());
    }

    let mut nearest_neighbor_distances = vec![f64::INFINITY; xdat.nrows()];
    let mut pairwise_distances = Vec::new();
    for left in 0..xdat.nrows() {
        for right in (left + 1)..xdat.nrows() {
            let distance = squared_distance(xdat, left, right).sqrt();
            nearest_neighbor_distances[left] = nearest_neighbor_distances[left].min(distance);
            nearest_neighbor_distances[right] = nearest_neighbor_distances[right].min(distance);
            pairwise_distances.push((left, right, distance));
        }
    }
    (nearest_neighbor_distances, pairwise_distances)
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

    let raw_candidates = xdat
        .rows()
        .into_iter()
        .map(|row| row.to_vec())
        .collect::<Vec<_>>();
    let normalized_candidates = normalized_xdat
        .rows()
        .into_iter()
        .map(|row| row.to_vec())
        .collect::<Vec<_>>();
    let (nearest_neighbor_distances, pairwise_distances) =
        pairwise_distance_diagnostics(&normalized_xdat);
    let nearest_neighbor_summary = summarize_distances(&nearest_neighbor_distances);
    let pairwise_values = pairwise_distances
        .iter()
        .map(|(_, _, distance)| *distance)
        .collect::<Vec<_>>();
    let pairwise_summary = summarize_distances(&pairwise_values);
    let pairwise_within_tolerance = pairwise_values
        .iter()
        .filter(|distance| **distance <= PORTFOLIO_DBSCAN_TOLERANCE_FACTOR)
        .count();

    log::debug!(
        "Portfolio DBSCAN params: tolerance={} min_points={} max_candidates={}",
        PORTFOLIO_DBSCAN_TOLERANCE_FACTOR,
        PORTFOLIO_DBSCAN_MIN_POINTS,
        PORTFOLIO_MAX_CANDIDATES
    );
    log::debug!("Portfolio raw x candidates: {raw_candidates:?}");
    log::debug!("Portfolio normalized x candidates: {normalized_candidates:?}");
    log::debug!("Portfolio nearest-neighbor distances: {nearest_neighbor_distances:?}");
    if let Some((min_distance, median_distance, max_distance)) = nearest_neighbor_summary {
        log::info!(
            "Portfolio nearest-neighbor summary: min={min_distance:.6} median={median_distance:.6} max={max_distance:.6}"
        );
    }
    if let Some((min_distance, median_distance, max_distance)) = pairwise_summary {
        log::info!(
            "Portfolio pairwise summary: min={min_distance:.6} median={median_distance:.6} max={max_distance:.6} within_tolerance={pairwise_within_tolerance}/{}",
            pairwise_values.len()
        );
    }

    // Cluster the x information
    let clusters = Dbscan::params(PORTFOLIO_DBSCAN_MIN_POINTS)
        .tolerance(PORTFOLIO_DBSCAN_TOLERANCE_FACTOR)
        .transform(&normalized_xdat)
        .unwrap();

    let cluster_labels = clusters.iter().copied().collect::<Vec<_>>();
    log::debug!("Portfolio DBSCAN labels: {cluster_labels:?}");

    let mut dict = HashMap::new();
    let mut singleton_indices = Vec::new();
    for (i, c) in clusters.iter().enumerate() {
        match c {
            None => singleton_indices.push(i),
            Some(label) => dict.entry(*label).or_insert(vec![]).push(i),
        }
    }

    let mut cluster_sizes = dict
        .iter()
        .map(|(label, members)| (*label, members.len()))
        .collect::<Vec<_>>();
    cluster_sizes.sort_by_key(|(label, _)| *label);
    log::debug!("Portfolio cluster sizes: {cluster_sizes:?}");
    log::debug!("Portfolio singleton indices: {singleton_indices:?}");

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
    let representative_diagnostics = representatives
        .iter()
        .map(|index| {
            (
                *index,
                infill_values[*index],
                raw_candidates[*index].clone(),
            )
        })
        .collect::<Vec<_>>();
    log::debug!("Portfolio representatives before truncation: {representative_diagnostics:?}");
    representatives.truncate(PORTFOLIO_MAX_CANDIDATES);
    let selected_diagnostics = representatives
        .iter()
        .map(|index| {
            (
                *index,
                infill_values[*index],
                raw_candidates[*index].clone(),
            )
        })
        .collect::<Vec<_>>();
    log::info!("Portfolio selected representatives: {selected_diagnostics:?}");
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
