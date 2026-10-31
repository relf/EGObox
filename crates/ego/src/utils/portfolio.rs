use std::collections::HashMap;

use linfa::traits::Transformer;

use linfa_clustering::Dbscan;

use crate::types::Candidates;
use ndarray::{Array1, Array2, ArrayBase, Data, Ix2};

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

pub fn cluster_as_indices(xdat: &ArrayBase<impl Data<Elem = f64>, Ix2>) -> Vec<usize> {
    // Cluster the x information
    let clusters = Dbscan::params(2)
        .tolerance((xdat.ncols() as f64).sqrt())
        .transform(&xdat.to_owned())
        .unwrap();
    let mut dict = HashMap::new();
    for (i, c) in clusters.iter().enumerate() {
        match c {
            None => (),
            Some(label) => dict.entry(*label).or_insert(vec![]).push(i),
        }
    }

    // Pick the first element of each cluster as representative, in increasing index
    // order (hash map iteration order is random)
    let mut representatives: Vec<usize> = dict
        .values()
        .map(|members| *members.first().unwrap())
        .collect();
    representatives.sort_unstable();
    representatives
}

/// This function clusters portfolio information wrt x values then pick one member of each cluster
pub(crate) fn select_from_portfolio(portfolio: Vec<Candidates>) -> Candidates {
    let n = portfolio.len();
    let mut xdat = Array2::zeros((n, portfolio[0].x.ncols()));

    for (i, info) in portfolio.iter().enumerate() {
        // Pick the first x (row(0)) as representative of q points batch
        xdat.row_mut(i).assign(&info.x.row(0).to_owned());
    }

    // Indices of representative of a cluster
    let indices = cluster_as_indices(&xdat);

    log::info!("Detect {} clusters", indices.len());

    // Pick information from portfolio of given indices and concatenate
    let nclusters = indices.len();

    let mut xdat = Array2::zeros((nclusters.max(1), portfolio[0].x.ncols()));
    let mut ydat = Array2::zeros((nclusters.max(1), portfolio[0].y.ncols()));
    let mut cdat = Array2::zeros((nclusters.max(1), portfolio[0].c.ncols()));
    let mut ypen = Array2::zeros((nclusters.max(1), portfolio[0].y_penalized.ncols()));

    if nclusters > 1 {
        for (i, index) in indices.iter().enumerate() {
            xdat.row_mut(i).assign(&portfolio[*index].x.row(0));
            ydat.row_mut(i).assign(&portfolio[*index].y.row(0));
            cdat.row_mut(i).assign(&portfolio[*index].c.row(0));
            ypen.row_mut(i)
                .assign(&portfolio[*index].y_penalized.row(0));
        }
    } else {
        xdat.row_mut(0).assign(&portfolio[0].x.row(0));
        ydat.row_mut(0).assign(&portfolio[0].y.row(0));
        cdat.row_mut(0).assign(&portfolio[0].c.row(0));
        ypen.row_mut(0).assign(&portfolio[0].y_penalized.row(0));
    }
    // FIXME: Have to check if fmin and infill_data values are relevant
    // At the moment just pick the first values of portfolio
    Candidates {
        x: xdat,
        y: ydat,
        c: cdat,
        y_penalized: ypen,
        infill_value: portfolio[0].infill_value,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

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
        for _ in 0..20 {
            assert_eq!(cluster_as_indices(&xdat), vec![0, 1, 2]);
        }
    }

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
        let cluster_memberships = cluster_as_indices(&x);
        println!("cluster_memberships = {cluster_memberships :?}");
    }
}
