//! Exact reference of the random MDA on the aerostructures problem (REMAL paper, section 4.2):
//! the distributions of the coupling variables given by GP posterior sample paths, compared
//! with the simplified random MDA of the EGMDO paper (rank one, solved), its linearization
//! (decision D37), and the probability of being the minimizer with the perfect and GP
//! posterior correlations across design points (decision D38).
//!
//! ```text
//! cargo run -p egobox-egmdo --release --example egmdo_aerostructures -- [n_seeds]
//! ```
//!
//! The lift GP has the inputs `(B, φ)` and the pitch GP the input `L`: at a given span `B`,
//! both are one-dimensional, and their posterior sample paths are drawn exactly on fine grids.
//! Each pair of paths is a realization of the disciplinary uncertainty, whose fixed points are
//! found on the grids. The results are printed as Markdown tables.

use egobox_egmdo::criteria::{Feasibility, SampleMinima, qoi_joint_samples};
use egobox_egmdo::mda::{SurrogateCouplings, solve_random};
use egobox_egmdo::pce::normal_samples;
use egobox_egmdo::problems::{self, TestProblem};
use egobox_egmdo::uq::{
    Correlation, JointSamples, RandomMda, UqSettings, linearized_couplings, point_uq,
};
use egobox_egmdo::{DisciplinaryModels, GpSettings, initial_doe};
use linfa_linalg::eigh::EighInto;
use ndarray::{Array1, Array2, ArrayView1, Axis, array, s};

/// Grid of the pitch angle `φ` (rad) for the lift paths
const PHI_GRID: (f64, f64, usize) = (-0.6, 2.2, 301);
/// Grid of the lift `L` (kN) for the pitch paths
const L_GRID: (f64, f64, usize) = (-10., 60., 701);
/// Number of samples of the distributions at a design point
const N_SAMPLES: usize = 2000;
/// Number of samples of the probability of being the minimizer
const N_PMIN: usize = 4000;
/// Spans `B` (cm) where the distributions are compared
const SPANS: [f64; 6] = [100., 200., 225., 700. / 3., 240., 300.];
/// `DoE_UQ` around the optimum of [problems::aerostructures_steep] for `P_min`
const DOE_UQ: [f64; 5] = [200., 220., 700. / 3., 240., 260.];

fn grid((lo, hi, n): (f64, f64, usize)) -> Array1<f64> {
    Array1::linspace(lo, hi, n)
}

/// Linear interpolation of `values` on `grid` at `x`, `None` outside the grid
fn interp((lo, hi, n): (f64, f64, usize), values: &ArrayView1<f64>, x: f64) -> Option<f64> {
    if !(lo..=hi).contains(&x) {
        return None;
    }
    let t = (x - lo) / (hi - lo) * (n - 1) as f64;
    let k = (t.floor() as usize).min(n - 2);
    let w = t - k as f64;
    Some(values[k] * (1. - w) + values[k + 1] * w)
}

/// `n` sample paths `(n, m)` of the GP with posterior mean `mean` `(m)` and covariance `cov`
fn sample_paths(mean: &Array1<f64>, cov: Array2<f64>, n: usize, seed: u64) -> Array2<f64> {
    let (values, vectors) = cov.eigh_into().expect("eigendecomposition");
    let tol = 1e-12 * values.iter().fold(0f64, |m, &v| m.max(v));
    let root = values.mapv(|v| if v > tol { v.sqrt() } else { 0. });
    let sqrt = (&vectors * &root).dot(&vectors.t());
    normal_samples(n, mean.len(), seed).dot(&sqrt) + mean
}

/// The fixed points `L*` of `L = lift(pitch(L))` for one pair of paths, found by sign changes
/// on the lift grid, and the one reached by successive substitution from `φ = 0`
struct PathFixedPoints {
    n_roots: usize,
    reached: Option<f64>,
}

fn fixed_points(lift: &ArrayView1<f64>, pitch: &ArrayView1<f64>) -> PathFixedPoints {
    let l_grid = grid(L_GRID);
    let h: Vec<Option<f64>> = l_grid
        .iter()
        .zip(pitch.iter())
        .map(|(&l, &phi)| interp(PHI_GRID, lift, phi).map(|v| l - v))
        .collect();
    let n_roots = h
        .windows(2)
        .filter(|w| matches!((w[0], w[1]), (Some(a), Some(b)) if a == 0. || a * b < 0.))
        .count();
    // Successive substitution L <- lift(pitch(L)) from φ = 0, as the MDA from the centre of
    // the pitch box
    let mut reached = None;
    if let Some(mut l) = interp(PHI_GRID, lift, 0.) {
        for _ in 0..20_000 {
            let Some(next) = interp(L_GRID, pitch, l).and_then(|phi| interp(PHI_GRID, lift, phi))
            else {
                break;
            };
            if (next - l).abs() < 1e-10 * l.abs().max(1.) {
                reached = Some(next);
                break;
            }
            l = next;
        }
    }
    PathFixedPoints { n_roots, reached }
}

fn train(tp: &TestProblem, n: usize, seed: u64) -> DisciplinaryModels {
    let data = (0..2)
        .map(|i| {
            let x = initial_doe(&tp.problem, i, n, 100 * seed + i as u64);
            let y = tp.eval_discipline(i, &x);
            (x, y)
        })
        .collect();
    DisciplinaryModels::fit(&tp.problem, data, GpSettings::default()).expect("GPs")
}

/// Pitch paths on the lift grid (independent of the span)
fn pitch_paths(models: &DisciplinaryModels, n: usize, seed: u64) -> Array2<f64> {
    let x = grid(L_GRID).insert_axis(Axis(1));
    let gp = &models.gps(1)[0];
    sample_paths(
        &gp.predict(&x).expect("mean"),
        gp.predict_covariance(&x).expect("covariance"),
        n,
        seed,
    )
}

/// Lift paths at the spans `spans`, jointly, on the pitch grid: `(n, spans.len() * m)`
fn lift_paths(models: &DisciplinaryModels, spans: &[f64], n: usize, seed: u64) -> Array2<f64> {
    let phi = grid(PHI_GRID);
    let m = phi.len();
    let mut x = Array2::zeros((spans.len() * m, 2));
    for (i, &b) in spans.iter().enumerate() {
        for (j, &p) in phi.iter().enumerate() {
            x[[i * m + j, 0]] = b;
            x[[i * m + j, 1]] = p;
        }
    }
    let gp = &models.gps(0)[0];
    sample_paths(
        &gp.predict(&x).expect("mean"),
        gp.predict_covariance(&x).expect("covariance"),
        n,
        seed,
    )
}

fn mean_std(v: &[f64]) -> (f64, f64) {
    let a = Array1::from_vec(v.to_vec());
    (a.mean().unwrap_or(f64::NAN), a.std(0.))
}

fn median(v: &[f64]) -> f64 {
    let mut v: Vec<f64> = v.iter().copied().filter(|x| x.is_finite()).collect();
    v.sort_by(|a, b| a.total_cmp(b));
    v.get(v.len() / 2).copied().unwrap_or(f64::NAN)
}

/// Two-sample Kolmogorov–Smirnov distance
fn ks(a: &[f64], b: &[f64]) -> f64 {
    if a.is_empty() || b.is_empty() {
        return f64::NAN;
    }
    let (mut a, mut b) = (a.to_vec(), b.to_vec());
    a.sort_by(|x, y| x.total_cmp(y));
    b.sort_by(|x, y| x.total_cmp(y));
    let (mut i, mut j, mut d) = (0, 0, 0f64);
    while i < a.len() && j < b.len() {
        if a[i] <= b[j] {
            i += 1;
        } else {
            j += 1;
        }
        d = d.max((i as f64 / a.len() as f64 - j as f64 / b.len() as f64).abs());
    }
    d
}

/// Statistics of the distributions of `L*` at one span, over the seeds
#[derive(Default)]
struct SpanStats {
    std_ref: Vec<f64>,
    ratio_solved: Vec<f64>,
    ratio_linearized: Vec<f64>,
    ks_solved: Vec<f64>,
    ks_linearized: Vec<f64>,
    multi_roots: usize,
    ref_failed: usize,
    solved_failed: usize,
    linearized_fallbacks: usize,
    n_paths: usize,
}

fn distributions(n_seeds: u64) {
    println!(
        "Distribution of the lift L* (kN) at fixed spans, over {n_seeds} seeds: exact GP sample \
         paths (reference) against the rank-one solved and the linearized random MDAs, \
         {N_SAMPLES} samples each. Ratios and KS distances: medians over the seeds.\n"
    );
    println!(
        "| Initial DoE per discipline | Span B | L* std, reference | Std ratio solved/ref. | Std ratio lin./ref. | KS solved | KS lin. | Paths with several fixed points | Paths without reached fixed point | Failed solved random MDAs | Linearization fallbacks |"
    );
    println!("|---|---|---|---|---|---|---|---|---|---|---|");
    let tp = problems::aerostructures_steep();
    for n in [3, 4, 6, 10] {
        let mut stats: Vec<SpanStats> = SPANS.iter().map(|_| SpanStats::default()).collect();
        for seed in 0..n_seeds {
            let models = train(&tp, n, seed);
            let pitch = pitch_paths(&models, N_SAMPLES, 1000 + seed);
            let xi = normal_samples(N_SAMPLES, 2, 2000 + seed);
            for (k, &b) in SPANS.iter().enumerate() {
                let st = &mut stats[k];
                let lift = lift_paths(&models, &[b], N_SAMPLES, 3000 + seed);
                let mut reference = vec![];
                for p in 0..N_SAMPLES {
                    let fp = fixed_points(&lift.row(p), &pitch.row(p));
                    st.n_paths += 1;
                    st.multi_roots += (fp.n_roots > 1) as usize;
                    match fp.reached {
                        Some(l) => reference.push(l),
                        None => st.ref_failed += 1,
                    }
                }
                let z = array![b];
                let couplings =
                    SurrogateCouplings::random(&tp.problem, &models, xi.view()).expect("couplings");
                let sol = solve_random(&couplings, &z.view(), &tp.mda).expect("random MDA");
                let solved: Vec<f64> = sol.converged().iter().map(|&r| sol.y[[r, 0]]).collect();
                st.solved_failed += sol.n_failed();
                let linearized: Vec<f64> = match linearized_couplings(
                    &tp.problem,
                    &models,
                    &z.view(),
                    &xi.view(),
                    &tp.mda,
                )
                .expect("linearization")
                {
                    Some(y) => y.column(0).to_vec(),
                    None => {
                        st.linearized_fallbacks += 1;
                        vec![]
                    }
                };
                let (_, s_ref) = mean_std(&reference);
                st.std_ref.push(s_ref);
                if s_ref > 0. {
                    st.ratio_solved.push(mean_std(&solved).1 / s_ref);
                    if !linearized.is_empty() {
                        st.ratio_linearized.push(mean_std(&linearized).1 / s_ref);
                    }
                }
                st.ks_solved.push(ks(&reference, &solved));
                st.ks_linearized.push(ks(&reference, &linearized));
            }
        }
        for (k, &b) in SPANS.iter().enumerate() {
            let st = &stats[k];
            println!(
                "| {n} | {b:.2} | {:.3} | {:.2} | {:.2} | {:.3} | {:.3} | {:.1}% | {:.1}% | {:.1}% | {} |",
                median(&st.std_ref),
                median(&st.ratio_solved),
                median(&st.ratio_linearized),
                median(&st.ks_solved),
                median(&st.ks_linearized),
                100. * st.multi_roots as f64 / st.n_paths as f64,
                100. * st.ref_failed as f64 / st.n_paths as f64,
                100. * st.solved_failed as f64 / st.n_paths as f64,
                st.linearized_fallbacks,
            );
        }
    }
}

fn argmax(p: &Array1<f64>) -> usize {
    p.iter()
        .enumerate()
        .fold(
            (0, f64::NEG_INFINITY),
            |(k, m), (i, &v)| {
                if v > m { (i, v) } else { (k, m) }
            },
        )
        .0
}

fn tv(p: &Array1<f64>, q: &Array1<f64>) -> f64 {
    0.5 * (p - q).mapv(f64::abs).sum()
}

fn probability_of_minimizer(n_seeds: u64) {
    println!(
        "\nProbability P_min of being the minimizer of aerostructures_steep over DoE_UQ = \
         {DOE_UQ:.2?}: exact GP sample paths, jointly over the design points (reference), \
         against the PCEs with the perfect and the GP posterior correlations, solved random \
         MDA. Total variation distances: medians over {n_seeds} seeds.\n"
    );
    println!(
        "| Initial DoE per discipline | Reference P_min (median over seeds) | TV perfect/ref. | TV GP posterior/ref. | Same most probable minimizer, perfect | Same most probable minimizer, GP posterior |"
    );
    println!("|---|---|---|---|---|---|");
    let tp = problems::aerostructures_steep();
    let z_uq = Array2::from_shape_vec((DOE_UQ.len(), 1), DOE_UQ.to_vec()).expect("DoE_UQ");
    let feasibility = Feasibility::unconstrained();
    let settings = UqSettings::default();
    let basis = settings.basis(2);
    for n in [3, 4, 6, 10] {
        let (mut tv_p, mut tv_g, mut same_p, mut same_g) = (vec![], vec![], 0, 0);
        let mut p_refs: Vec<Array1<f64>> = vec![];
        for seed in 0..n_seeds {
            let models = train(&tp, n, seed);
            // Reference: joint lift paths over the DoE_UQ spans and one pitch path per sample
            let pitch = pitch_paths(&models, N_PMIN, 4000 + seed);
            let lift = lift_paths(&models, &DOE_UQ, N_PMIN, 5000 + seed);
            let m = PHI_GRID.2;
            let mut counts = Array1::<f64>::zeros(DOE_UQ.len());
            let mut n_valid = 0;
            for p in 0..N_PMIN {
                let mut best: Option<(usize, f64)> = None;
                let mut valid = true;
                for i in 0..DOE_UQ.len() {
                    let path = lift.slice(s![p, i * m..(i + 1) * m]);
                    let Some(l) = fixed_points(&path, &pitch.row(p)).reached else {
                        valid = false;
                        break;
                    };
                    let phi = interp(L_GRID, &pitch.row(p), l).expect("pitch on grid");
                    let y = array![[l, phi]];
                    let f = tp
                        .problem
                        .eval_qoi(&z_uq.slice(s![i..i + 1, ..]), &y.view())
                        .expect("objective")[[0, 0]];
                    if best.is_none_or(|(_, v)| f < v) {
                        best = Some((i, f));
                    }
                }
                if valid {
                    counts[best.expect("points").0] += 1.;
                    n_valid += 1;
                }
            }
            let p_ref = counts / n_valid.max(1) as f64;

            // PCEs of the solved random MDA at the DoE_UQ points
            let xi = settings.regression_samples(&basis, seed);
            let uq: Vec<_> = (0..DOE_UQ.len())
                .map(|i| {
                    point_uq(
                        &tp.problem,
                        &models,
                        &z_uq.row(i),
                        &xi.view(),
                        &basis,
                        &tp.mda,
                        RandomMda::Solved,
                    )
                    .expect("PCE")
                })
                .collect();
            let p_of = |correlation: Correlation| {
                let samples = JointSamples::draw(
                    correlation,
                    &tp.problem,
                    &models,
                    &z_uq,
                    &uq,
                    N_PMIN,
                    6000 + seed,
                )
                .expect("joint samples");
                SampleMinima::from_values(&qoi_joint_samples(&uq, &samples), &feasibility)
                    .probability_of_solution()
            };
            let (p_perfect, p_gp) = (p_of(Correlation::Perfect), p_of(Correlation::GpPosterior));
            tv_p.push(tv(&p_perfect, &p_ref));
            tv_g.push(tv(&p_gp, &p_ref));
            same_p += (argmax(&p_perfect) == argmax(&p_ref)) as usize;
            same_g += (argmax(&p_gp) == argmax(&p_ref)) as usize;
            p_refs.push(p_ref);
        }
        let p_median: Vec<String> = (0..DOE_UQ.len())
            .map(|i| {
                format!(
                    "{:.2}",
                    median(&p_refs.iter().map(|p| p[i]).collect::<Vec<_>>())
                )
            })
            .collect();
        println!(
            "| {n} | [{}] | {:.3} | {:.3} | {same_p}/{n_seeds} | {same_g}/{n_seeds} |",
            p_median.join(", "),
            median(&tv_p),
            median(&tv_g),
        );
    }
}

fn main() {
    let n_seeds = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(10);
    distributions(n_seeds);
    probability_of_minimizer(n_seeds);
}
