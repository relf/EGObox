use std::collections::BTreeMap;

use linfa::traits::Transformer;

use linfa_clustering::Dbscan;

use crate::types::Candidates;
use ndarray::{Array1, Array2, ArrayBase, Data, Ix2};

/// Maximum number of points selected from the GP variance portfolio
pub(crate) const GP_VAR_PORTFOLIO_MAX_POINTS: usize = 3;
/// Minimum number of points of a DBSCAN cluster
const DBSCAN_MIN_POINTS: usize = 2;
/// DBSCAN tolerance when it can not be derived from distances (less than 2 points)
const DBSCAN_TOLERANCE_FALLBACK: f64 = 0.2;
/// DBSCAN tolerance as a multiple of the median nearest neighbor distance...
const DBSCAN_TOLERANCE_MULTIPLIER: f64 = 1.1;
/// ... clamped within these bounds (in the unit hypercube)
const DBSCAN_TOLERANCE_MIN: f64 = 0.15;
const DBSCAN_TOLERANCE_MAX: f64 = 0.45;
/// Distance (in the unit hypercube) below which points are duplicates when filling a batch
const DUPLICATE_TOLERANCE: f64 = 1e-3;

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

/// `xdat` rows mapped to the unit hypercube wrt `xlimits`
fn normalize(
    xdat: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    xlimits: &ArrayBase<impl Data<Elem = f64>, Ix2>,
) -> Array2<f64> {
    let mut xnorm = xdat.to_owned();
    for (mut col, limits) in xnorm.columns_mut().into_iter().zip(xlimits.rows()) {
        let (lower, width) = (limits[0], limits[1] - limits[0]);
        if width > 0.0 {
            col.mapv_inplace(|v| (v - lower) / width);
        }
    }
    xnorm
}

/// Distance of each row of `x` to its nearest other row
fn nearest_neighbor_distances(x: &Array2<f64>) -> Vec<f64> {
    let n = x.nrows();
    let mut dists = vec![f64::INFINITY; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let d = (&x.row(i) - &x.row(j)).mapv(|v| v * v).sum().sqrt();
            dists[i] = dists[i].min(d);
            dists[j] = dists[j].min(d);
        }
    }
    dists
}

/// DBSCAN tolerance adapted to the spreading of the points: a multiple of the median
/// nearest neighbor distance, clamped
fn dbscan_tolerance(nn_dists: &[f64]) -> f64 {
    let mut sorted: Vec<f64> = nn_dists.iter().copied().filter(|d| d.is_finite()).collect();
    if sorted.is_empty() {
        return DBSCAN_TOLERANCE_FALLBACK;
    }
    sorted.sort_by(f64::total_cmp);
    let mid = sorted.len() / 2;
    let median = if sorted.len().is_multiple_of(2) {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    } else {
        sorted[mid]
    };
    (median * DBSCAN_TOLERANCE_MULTIPLIER).clamp(DBSCAN_TOLERANCE_MIN, DBSCAN_TOLERANCE_MAX)
}

/// Indices of the best ranked `xdat` rows, one per cluster of rows: rows are clustered
/// in the unit hypercube wrt `xlimits`, each cluster (or isolated row) being represented
/// by its best (highest `ranks` value) row. Representatives are sorted by decreasing rank,
/// at most `max_points` are returned.
pub(crate) fn cluster_as_indices(
    xdat: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    xlimits: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    ranks: &[f64],
    max_points: usize,
) -> Vec<usize> {
    let xnorm = normalize(xdat, xlimits);
    let tolerance = dbscan_tolerance(&nearest_neighbor_distances(&xnorm));
    let clusters = Dbscan::params(DBSCAN_MIN_POINTS)
        .tolerance(tolerance)
        .transform(&xnorm)
        .unwrap();

    // Ordered map for representatives not to depend on hash map iteration order
    let mut members: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    let mut representatives = vec![];
    for (i, c) in clusters.iter().enumerate() {
        match c {
            Some(label) => members.entry(*label).or_default().push(i),
            None => representatives.push(i),
        }
    }
    log::debug!(
        "Portfolio DBSCAN tolerance = {tolerance}, {} clusters, {} isolated points",
        members.len(),
        representatives.len()
    );
    // Best ranked member of each cluster, the first one on ties
    representatives.extend(members.values().map(|indices| {
        indices.iter().copied().fold(
            indices[0],
            |best, i| {
                if ranks[i] > ranks[best] { i } else { best }
            },
        )
    }));

    // Decreasing rank, increasing index on ties
    representatives.sort_by(|&a, &b| ranks[b].total_cmp(&ranks[a]).then(a.cmp(&b)));
    representatives.truncate(max_points.max(1));
    representatives
}

/// Whether `x` is closer than `DUPLICATE_TOLERANCE` to one of the `points` rows
/// (all in the unit hypercube)
fn is_duplicate(x: ndarray::ArrayView1<f64>, points: &Array2<f64>) -> bool {
    points
        .rows()
        .into_iter()
        .any(|p| (&p - &x).mapv(|v| v * v).sum() < DUPLICATE_TOLERANCE * DUPLICATE_TOLERANCE)
}

/// Whether each `xdat` row duplicates one of the `points` rows: closer than
/// `DUPLICATE_TOLERANCE` once mapped to the unit hypercube wrt `xlimits`
pub(crate) fn duplicates(
    xdat: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    xlimits: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    points: &ArrayBase<impl Data<Elem = f64>, Ix2>,
) -> Vec<bool> {
    let points_norm = normalize(points, xlimits);
    normalize(xdat, xlimits)
        .rows()
        .into_iter()
        .map(|x| is_duplicate(x, &points_norm))
        .collect()
}

/// Indices of `xdat` rows making a batch of `max_points` points: the representatives of
/// clusters of rows (see [`cluster_as_indices`]), then the other rows by decreasing rank.
/// Rows duplicating an `x_data` point or a row already in the batch are skipped,
/// unless all rows duplicate `x_data` points.
pub(crate) fn batch_indices(
    xdat: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    xlimits: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    ranks: &[f64],
    max_points: usize,
    x_data: &ArrayBase<impl Data<Elem = f64>, Ix2>,
) -> Vec<usize> {
    let xnorm = normalize(xdat, xlimits);
    let data_norm = normalize(x_data, xlimits);
    let mut usable: Vec<usize> = (0..xnorm.nrows())
        .filter(|&i| !is_duplicate(xnorm.row(i), &data_norm))
        .collect();
    if usable.is_empty() {
        log::debug!("All portfolio points duplicate evaluated points");
        usable = (0..xnorm.nrows()).collect();
    }
    let usable_ranks: Vec<f64> = usable.iter().map(|&i| ranks[i]).collect();
    let mut selected: Vec<usize> = cluster_as_indices(
        &xdat.select(ndarray::Axis(0), &usable),
        xlimits,
        &usable_ranks,
        max_points,
    )
    .into_iter()
    .map(|k| usable[k])
    .collect();

    // Fill the batch with the other rows by decreasing rank
    let mut others: Vec<usize> = usable
        .into_iter()
        .filter(|i| !selected.contains(i))
        .collect();
    others.sort_by(|&a, &b| ranks[b].total_cmp(&ranks[a]).then(a.cmp(&b)));
    for i in others {
        if selected.len() >= max_points {
            break;
        }
        if !is_duplicate(xnorm.row(i), &xnorm.select(ndarray::Axis(0), &selected)) {
            selected.push(i);
        }
    }
    selected
}

/// Select at most `max_points` points from the portfolio: the first points of the portfolio
/// members are clustered and the best member (highest `ranks` value) of each cluster is picked
/// (see [`cluster_as_indices`]), best first. When `fill_wrt` evaluated points are given,
/// the batch is filled up to `max_points` with the other members, skipping duplicates
/// (see [`batch_indices`]).
pub(crate) fn select_from_portfolio(
    portfolio: Vec<Candidates>,
    xlimits: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    ranks: &[f64],
    max_points: usize,
    fill_wrt: Option<&ndarray::ArrayView2<f64>>,
) -> Candidates {
    let mut xdat = Array2::zeros((portfolio.len(), portfolio[0].x.ncols()));
    for (i, member) in portfolio.iter().enumerate() {
        // Pick the first x (row(0)) as representative of q points batch
        xdat.row_mut(i).assign(&member.x.row(0));
    }
    let indices = match fill_wrt {
        Some(x_data) => batch_indices(&xdat, xlimits, ranks, max_points, x_data),
        None => cluster_as_indices(&xdat, xlimits, ranks, max_points),
    };
    log::info!("Select {} points from portfolio", indices.len());

    let mut selected = Candidates::empty(
        portfolio[0].x.ncols(),
        portfolio[0].y.ncols(),
        portfolio[0].c.ncols(),
    );
    for &index in &indices {
        selected.append(portfolio[index].select(&[0]));
    }
    // Infill value of the best selected point
    selected.infill_value = portfolio[indices[0]].infill_value;
    selected
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn member(x: f64, infill_value: f64) -> Candidates {
        Candidates {
            x: array![[x]],
            y: array![[100. * x]],
            c: array![[10. * x]],
            y_penalized: array![[1000. * x]],
            infill_value,
        }
    }

    #[test]
    fn test_cluster_as_indices_is_deterministic() {
        // three groups of points far apart
        let xdat = array![
            [0.],
            [10.],
            [20.],
            [0.1],
            [10.1],
            [20.1],
            [0.2],
            [10.2],
            [20.2]
        ];
        let xlimits = array![[0., 20.2]];
        let ranks = [1., 1., 1., 2., 2., 2., 1., 1., 1.];
        for _ in 0..20 {
            assert_eq!(
                cluster_as_indices(&xdat, &xlimits, &ranks, 3),
                vec![3, 4, 5]
            );
        }
    }

    #[test]
    fn test_clustering_best_ranked_representatives() {
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
        let indices =
            cluster_as_indices(&x, &xlimits, &[1.0, 2.0, 5.0, 3.0, 4.0, 6.0, 0.5, 7.0], 3);
        assert_eq!(indices, vec![7, 4]);
    }

    #[test]
    fn test_clustering_keeps_best_three_isolated_points() {
        let x = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let xlimits = array![[0.0, 1.0], [0.0, 1.0]];
        let indices = cluster_as_indices(&x, &xlimits, &[1.0, 4.0, 3.0, 2.0], 3);
        assert_eq!(indices, vec![1, 2, 3]);
    }

    #[test]
    fn test_clustering_is_scale_invariant() {
        let ranks = [1.0, 5.0, 2.0, 4.0];
        let small = cluster_as_indices(
            &array![[0.10], [0.11], [0.90], [0.91]],
            &array![[0.0, 1.0]],
            &ranks,
            3,
        );
        let large = cluster_as_indices(
            &array![[10.0], [11.0], [90.0], [91.0]],
            &array![[0.0, 100.0]],
            &ranks,
            3,
        );
        assert_eq!(small, vec![1, 3]);
        assert_eq!(large, small);
    }

    #[test]
    fn test_dbscan_tolerance() {
        let tol = dbscan_tolerance(&[0.10, 0.12, 0.14, 0.16, 0.18]);
        assert!((tol - 0.154).abs() < 1e-12);
        assert_eq!(dbscan_tolerance(&[]), DBSCAN_TOLERANCE_FALLBACK);
        assert_eq!(
            dbscan_tolerance(&[f64::INFINITY]),
            DBSCAN_TOLERANCE_FALLBACK
        );
        assert_eq!(dbscan_tolerance(&[0.01, 0.02, 0.03]), DBSCAN_TOLERANCE_MIN);
        assert_eq!(dbscan_tolerance(&[1.0, 2.0, 3.0]), DBSCAN_TOLERANCE_MAX);
    }

    #[test]
    fn test_select_from_portfolio_best_first() {
        let portfolio = vec![
            member(0.13, 1.0),
            member(0.70, 2.0),
            member(0.72, 5.0),
            member(0.14, 3.0),
            member(0.15, 4.0),
            member(0.71, 6.0),
            member(0.16, 0.5),
            member(0.73, 7.0),
        ];
        let ranks: Vec<f64> = portfolio.iter().map(|m| m.infill_value).collect();
        let selected = select_from_portfolio(portfolio, &array![[0.0, 1.0]], &ranks, 3, None);

        assert_eq!(selected.x.column(0).to_vec(), vec![0.73, 0.15]);
        assert_eq!(selected.y.column(0).to_vec(), vec![73.0, 15.0]);
        assert_eq!(selected.c.column(0).to_vec(), vec![7.3, 1.5]);
        assert_eq!(selected.y_penalized.column(0).to_vec(), vec![730.0, 150.0]);
        assert_eq!(selected.infill_value, 7.0);
    }

    #[test]
    fn test_clustering_max_points() {
        let x = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let xlimits = array![[0.0, 1.0], [0.0, 1.0]];
        let ranks = [1.0, 4.0, 3.0, 2.0];
        assert_eq!(cluster_as_indices(&x, &xlimits, &ranks, 1), vec![1]);
        assert_eq!(
            cluster_as_indices(&x, &xlimits, &ranks, 10),
            vec![1, 2, 3, 0]
        );
        // at least one point
        assert_eq!(cluster_as_indices(&x, &xlimits, &ranks, 0), vec![1]);
    }

    #[test]
    fn test_batch_indices_fill_and_skip_duplicates() {
        // two groups in one cluster each, ranked in index order
        let x = array![[0.10], [0.11], [0.12], [0.70], [0.71], [0.700_000_1]];
        let xlimits = array![[0.0, 1.0]];
        let ranks = [-0., -1., -2., -3., -4., -5.];
        let x_data = array![[0.5]];
        // representatives 0 and 3, then filled with 1 and 2 (5 duplicates 3)
        assert_eq!(
            batch_indices(&x, &xlimits, &ranks, 4, &x_data),
            vec![0, 3, 1, 2]
        );
        assert_eq!(
            batch_indices(&x, &xlimits, &ranks, 10, &x_data),
            vec![0, 3, 1, 2, 4]
        );
        // the first point duplicates an evaluated point
        let x_data = array![[0.100_000_1]];
        assert_eq!(batch_indices(&x, &xlimits, &ranks, 1, &x_data), vec![1]);
        // all points duplicate evaluated points: kept
        let x_data = x.clone();
        assert_eq!(batch_indices(&x, &xlimits, &ranks, 1, &x_data), vec![0]);
    }

    #[test]
    fn test_logspace_endpoints() {
        let vals = logspace(0.1, 100.0, 5);
        assert!((vals[0] - 0.1).abs() < 1e-12);
        assert!((vals[4] - 100.0).abs() < 1e-12);
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
}
