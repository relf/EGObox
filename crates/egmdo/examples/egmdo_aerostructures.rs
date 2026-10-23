//! Exact reference of the random MDA on the aerostructures problem (REMAL paper, section 4.2):
//! the distributions of the coupling variables given by GP posterior sample paths, compared
//! with the simplified random MDA of the EGMDO paper (rank one, solved), its linearization
//! (decision D37), and the probability of being the minimizer with the perfect and GP
//! posterior correlations across design points (decision D38).
//!
//! ```text
//! cargo run -p egobox-egmdo --release --example egmdo_aerostructures -- [n_seeds]
//! cargo run -p egobox-egmdo --release --example egmdo_aerostructures -- local-kl [n_seeds]
//! ```
//!
//! The `local-kl` mode prototypes the local Karhunen–Loève expansion of the disciplinary GPs
//! of issue #493: at each span, each GP posterior is expanded on a trust region
//! `y_0 ± k σ_y` around the mean MDA solution, `σ_y` being the standard deviation of the
//! linearized random MDA, and extended to the whole grid by Nyström. Its fixed points are
//! compared with those of the exact sample paths and of the rank-one model.
//!
//! The lift GP has the inputs `(B, φ)` and the pitch GP the input `L`: at a given span `B`,
//! both are one-dimensional, and their posterior sample paths are drawn exactly on fine grids.
//! Each pair of paths is a realization of the disciplinary uncertainty, whose fixed points are
//! found on the grids. The results are printed as Markdown tables.

use egobox_egmdo::criteria::{Feasibility, SampleMinima, qoi_joint_samples};
use egobox_egmdo::mda::{SurrogateCouplings, solve_one, solve_random};
use egobox_egmdo::pce::normal_samples;
use egobox_egmdo::problems::{self, TestProblem};
use egobox_egmdo::uq::{
    Correlation, JointSamples, RandomMda, UqSettings, coupling_jacobian, linearized_couplings,
    point_uq,
};
use egobox_egmdo::{DisciplinaryModels, GpSettings, initial_doe};
use linfa_linalg::eigh::EighInto;
use ndarray::{Array1, Array2, ArrayView1, Axis, array, s};
use rayon::prelude::*;

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

/// A uniform grid `(lower, upper, number of points)`
type Grid = (f64, f64, usize);

/// Posterior mean and covariance of the pitch GP on the lift grid `l_grid` (independent of
/// the span)
fn pitch_posterior(models: &DisciplinaryModels, l_grid: Grid) -> (Array1<f64>, Array2<f64>) {
    let x = grid(l_grid).insert_axis(Axis(1));
    let gp = &models.gps(1)[0];
    (
        gp.predict(&x).expect("mean"),
        gp.predict_covariance(&x).expect("covariance"),
    )
}

/// Pitch paths on the lift grid (independent of the span)
fn pitch_paths(models: &DisciplinaryModels, n: usize, seed: u64) -> Array2<f64> {
    let (mean, cov) = pitch_posterior(models, L_GRID);
    sample_paths(&mean, cov, n, seed)
}

/// Lift paths at the spans `spans`, jointly, on the pitch grid: `(n, spans.len() * m)`
fn lift_paths(models: &DisciplinaryModels, spans: &[f64], n: usize, seed: u64) -> Array2<f64> {
    let (mean, cov) = lift_posterior(models, spans, PHI_GRID);
    sample_paths(&mean, cov, n, seed)
}

/// Posterior mean and covariance of the lift GP at the spans `spans`, jointly, on the pitch
/// grid `phi_grid`
fn lift_posterior(
    models: &DisciplinaryModels,
    spans: &[f64],
    phi_grid: Grid,
) -> (Array1<f64>, Array2<f64>) {
    let phi = grid(phi_grid);
    let m = phi.len();
    let mut x = Array2::zeros((spans.len() * m, 2));
    for (i, &b) in spans.iter().enumerate() {
        for (j, &p) in phi.iter().enumerate() {
            x[[i * m + j, 0]] = b;
            x[[i * m + j, 1]] = p;
        }
    }
    let gp = &models.gps(0)[0];
    (
        gp.predict(&x).expect("mean"),
        gp.predict_covariance(&x).expect("covariance"),
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

/// Grid of the pitch angle `φ` (rad) for the lift paths of the local KL study
const KL_PHI_GRID: Grid = (-1., 4., 501);
/// Grid of the lift `L` (kN) for the pitch paths of the local KL study
const KL_L_GRID: Grid = (-20., 120., 1401);
/// Number of samples of the local KL study (KS noise level about 0.02)
const N_KL: usize = 8000;
/// Number of samples of the solved rank-one random MDA in the local KL study
const N_KL_SOLVED: usize = 2000;

/// Paths `(n, grid)` of a Gaussian field `mean + basis ξ`, `ξ ~ N(0, I)`, with `basis`
/// `(grid, r)`
fn basis_paths(mean: &Array1<f64>, basis: &Array2<f64>, n: usize, seed: u64) -> Array2<f64> {
    normal_samples(n, basis.ncols(), seed).dot(&basis.t()) + mean
}

/// The grid indices in `[center - half, center + half]`, at least the 3 nearest to `center`
fn trust_region((lo, hi, n): (f64, f64, usize), center: f64, half: f64) -> Vec<usize> {
    let values = grid((lo, hi, n));
    let inside: Vec<usize> = (0..n)
        .filter(|&i| (values[i] - center).abs() <= half)
        .collect();
    if inside.len() >= 3 {
        return inside;
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| {
        (values[i] - center)
            .abs()
            .total_cmp(&(values[j] - center).abs())
    });
    let mut nearest: Vec<usize> = order.into_iter().take(3).collect();
    nearest.sort();
    nearest
}

/// The local KL of a GP posterior of covariance `cov` on a grid: the `m` leading eigenpairs
/// `(λ_i, v_i)` of the covariance on the trust region `tr`, extended to the whole grid by
/// Nyström, `ψ_i = C(·, tr) v_i / λ_i`. Returns the basis `sqrt(λ_i) ψ_i` `(grid, m)` and the
/// fraction of the trust-region variance it captures. With `rescale`, each row is scaled so
/// that the pointwise variance is the posterior variance.
fn local_kl(cov: &Array2<f64>, tr: &[usize], m: usize, rescale: bool) -> (Array2<f64>, f64) {
    let ctt = cov.select(Axis(0), tr).select(Axis(1), tr);
    let (values, vectors) = ctt.eigh_into().expect("eigendecomposition");
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|&i, &j| values[j].total_cmp(&values[i]));
    let total: f64 = values.iter().filter(|&&v| v > 0.).sum();
    let tol = 1e-12 * values[order[0]].max(0.);
    let kept: Vec<usize> = order
        .into_iter()
        .take(m)
        .filter(|&i| values[i] > tol)
        .collect();
    let lambda = values.select(Axis(0), &kept);
    let mut basis =
        cov.select(Axis(1), tr).dot(&vectors.select(Axis(1), &kept)) / &lambda.mapv(f64::sqrt);
    if rescale {
        let sigma = cov.diag().mapv(|v| v.max(0.).sqrt());
        let kl_std = basis.mapv(|b| b * b).sum_axis(Axis(1)).mapv(f64::sqrt);
        let scale = Array1::from_iter(
            sigma
                .iter()
                .zip(kl_std.iter())
                .map(|(&s, &k)| if k > 1e-150 { s / k } else { 0. }),
        );
        basis *= &scale.insert_axis(Axis(1));
    }
    let energy = if total > 0. { lambda.sum() / total } else { 1. };
    (basis, energy)
}

/// For each pair of paths on the local KL grids, the fixed point `(L*, φ*)` of
/// `L = lift(pitch(L))` closest to `l0`, among all those found by sign changes on the lift
/// grid (`None` if there is none on the grids). This rule does not depend on a fixed-point
/// solver, and is the same for every model.
fn nearest_fixed_points(
    lift: &Array2<f64>,
    pitch: &Array2<f64>,
    l0: f64,
) -> Vec<Option<(f64, f64)>> {
    let l_grid = grid(KL_L_GRID);
    (0..lift.nrows())
        .into_par_iter()
        .map(|p| {
            let (lift, pitch) = (lift.row(p), pitch.row(p));
            let h: Vec<Option<f64>> = l_grid
                .iter()
                .zip(pitch.iter())
                .map(|(&l, &phi)| interp(KL_PHI_GRID, &lift, phi).map(|v| l - v))
                .collect();
            let mut best: Option<(f64, f64)> = None;
            for j in 0..h.len() - 1 {
                if let (Some(a), Some(b)) = (h[j], h[j + 1])
                    && (a == 0. || a * b < 0.)
                {
                    let t = a / (a - b);
                    let l = l_grid[j] + t * (l_grid[j + 1] - l_grid[j]);
                    let phi = pitch[j] + t * (pitch[j + 1] - pitch[j]);
                    if best.is_none_or(|(bl, _)| (l - l0).abs() < (bl - l0).abs()) {
                        best = Some((l, phi));
                    }
                }
            }
            best
        })
        .collect()
}

/// A model of the disciplinary uncertainty in the local KL study
#[derive(Clone, Copy)]
enum Model {
    /// The rank-one random MDA solved by Jacobi iterations (EGMDO)
    Solved,
    /// The rank-one fields `μ + σ ξ` on the grids
    RankOne,
    /// The linearized random MDA (D37)
    Linearized,
    /// The local KL with `m` modes on the trust region `y_0 ± k σ_y`, rescaled or not
    LocalKl { m: usize, k: f64, rescale: bool },
}

impl Model {
    fn label(&self) -> String {
        match self {
            Model::Solved => "Rank-one, solved random MDA (EGMDO)".into(),
            Model::RankOne => "Rank-one fields on the grids".into(),
            Model::Linearized => "Linearized random MDA".into(),
            Model::LocalKl { m, k, rescale } => format!(
                "Local KL{}, m = {m}, k = {k}",
                if *rescale { " rescaled" } else { "" }
            ),
        }
    }
}

fn local_kl_models() -> Vec<Model> {
    let mut models = vec![Model::Solved, Model::RankOne, Model::Linearized];
    for rescale in [false, true] {
        for m in [1, 2, 4, 8] {
            models.push(Model::LocalKl { m, k: 3., rescale });
        }
        for k in [2., 5.] {
            models.push(Model::LocalKl { m: 4, k, rescale });
        }
    }
    models
}

/// Statistics of a model against the exact reference, over the seeds and spans
#[derive(Default, Clone)]
struct ModelStats {
    ks: Vec<f64>,
    std_ratio: Vec<f64>,
    energy_lift: Vec<f64>,
    energy_pitch: Vec<f64>,
    coverage: Vec<f64>,
    not_reached: usize,
    n_paths: usize,
}

fn local_kl_study(n_seeds: u64) {
    println!(
        "Local KL prototype: distribution of the lift L* (kN) at the spans {SPANS:.2?}, against \
         the exact GP sample paths, {N_KL} samples ({N_KL_SOLVED} for the solved random MDA), \
         {n_seeds} seeds. For the exact paths, the rank-one fields and the local KL, L* is the \
         fixed point closest to the mean MDA solution, found on the grids; the solved random \
         MDA uses the Jacobi iterations of the solver. KS distance and std ratio: median [90% or 10%-90%] over the seeds \
         and spans. Variance captured: median fraction of the trust-region variance. Coverage: \
         median fraction of the exact fixed points inside the trust regions of both \
         couplings.\n"
    );
    println!(
        "| Initial DoE per discipline | Model | KS vs exact | Std ratio vs exact | Variance captured, lift / pitch | Coverage | Paths without reached fixed point |"
    );
    println!("|---|---|---|---|---|---|---|");
    let tp = problems::aerostructures_steep();
    let models_list = local_kl_models();
    for n in [3, 4, 6, 10] {
        let mut stats = vec![ModelStats::default(); models_list.len()];
        let mut consistency = vec![];
        let (mut ref_paths, mut ref_not_reached) = (0, 0);
        for seed in 0..n_seeds {
            let models = train(&tp, n, seed);
            let (pitch_mean, pitch_cov) = pitch_posterior(&models, KL_L_GRID);
            let pitch_exact = sample_paths(&pitch_mean, pitch_cov.clone(), N_KL, 7000 + seed);
            for &b in SPANS.iter() {
                let (lift_mean, lift_cov) = lift_posterior(&models, &[b], KL_PHI_GRID);
                let lift_exact = sample_paths(&lift_mean, lift_cov.clone(), N_KL, 8000 + seed);
                // Mean MDA solution and standard deviation of the linearized random MDA
                let z = array![b];
                let mean_couplings = SurrogateCouplings::mean(&tp.problem, &models);
                let (y0, _) = solve_one(&mean_couplings, &z.view(), &tp.mda).expect("mean MDA");

                let reference_all = nearest_fixed_points(&lift_exact, &pitch_exact, y0[0]);
                ref_paths += reference_all.len();
                ref_not_reached += reference_all.iter().filter(|r| r.is_none()).count();
                let reference: Vec<(f64, f64)> = reference_all.into_iter().flatten().collect();
                let ref_l: Vec<f64> = reference.iter().map(|r| r.0).collect();
                let (_, ref_std) = mean_std(&ref_l);
                let (jac, sigma) = coupling_jacobian(&tp.problem, &models, &z.view(), &y0.view())
                    .expect("Jacobian");
                let (a, bb, c, d) = (
                    1. - jac[[0, 0]],
                    -jac[[0, 1]],
                    -jac[[1, 0]],
                    1. - jac[[1, 1]],
                );
                let det = a * d - bb * c;
                let gain = array![[d, -bb], [-c, a]] / det * &sigma;
                let sigma_y = gain.mapv(|v| v * v).sum_axis(Axis(1)).mapv(f64::sqrt);

                let rank_one = |cov: &Array2<f64>| {
                    cov.diag()
                        .mapv(|v| v.max(0.).sqrt())
                        .insert_axis(Axis(1))
                        .to_owned()
                };
                let (mut rank_one_l, mut solved_l) = (vec![], vec![]);
                for (mi, model) in models_list.iter().enumerate() {
                    let st = &mut stats[mi];
                    let approx: Vec<Option<(f64, f64)>> = match *model {
                        Model::Solved => {
                            let xi = normal_samples(N_KL_SOLVED, 2, 9000 + seed);
                            let couplings =
                                SurrogateCouplings::random(&tp.problem, &models, xi.view())
                                    .expect("couplings");
                            let sol =
                                solve_random(&couplings, &z.view(), &tp.mda).expect("random MDA");
                            (0..sol.y.nrows())
                                .map(|r| {
                                    sol.status[r]
                                        .is_converged()
                                        .then(|| (sol.y[[r, 0]], sol.y[[r, 1]]))
                                })
                                .collect()
                        }
                        Model::Linearized => {
                            let xi = normal_samples(N_KL, 2, 9100 + seed);
                            match linearized_couplings(
                                &tp.problem,
                                &models,
                                &z.view(),
                                &xi.view(),
                                &tp.mda,
                            )
                            .expect("linearization")
                            {
                                Some(y) => {
                                    y.rows().into_iter().map(|r| Some((r[0], r[1]))).collect()
                                }
                                None => vec![None; N_KL],
                            }
                        }
                        Model::RankOne => nearest_fixed_points(
                            &basis_paths(&lift_mean, &rank_one(&lift_cov), N_KL, 9200 + seed),
                            &basis_paths(&pitch_mean, &rank_one(&pitch_cov), N_KL, 9300 + seed),
                            y0[0],
                        ),
                        Model::LocalKl { m, k, rescale } => {
                            let tr_phi = trust_region(KL_PHI_GRID, y0[1], k * sigma_y[1]);
                            let tr_l = trust_region(KL_L_GRID, y0[0], k * sigma_y[0]);
                            let (lift_basis, e_lift) = local_kl(&lift_cov, &tr_phi, m, rescale);
                            let (pitch_basis, e_pitch) = local_kl(&pitch_cov, &tr_l, m, rescale);
                            st.energy_lift.push(e_lift);
                            st.energy_pitch.push(e_pitch);
                            let inside = reference
                                .iter()
                                .filter(|(l, phi)| {
                                    (l - y0[0]).abs() <= k * sigma_y[0]
                                        && (phi - y0[1]).abs() <= k * sigma_y[1]
                                })
                                .count();
                            st.coverage
                                .push(inside as f64 / reference.len().max(1) as f64);
                            nearest_fixed_points(
                                &basis_paths(&lift_mean, &lift_basis, N_KL, 9400 + seed),
                                &basis_paths(&pitch_mean, &pitch_basis, N_KL, 9500 + seed),
                                y0[0],
                            )
                        }
                    };
                    st.n_paths += approx.len();
                    st.not_reached += approx.iter().filter(|a| a.is_none()).count();
                    let approx_l: Vec<f64> = approx.iter().flatten().map(|a| a.0).collect();
                    st.ks.push(ks(&ref_l, &approx_l));
                    if ref_std > 0. && !approx_l.is_empty() {
                        st.std_ratio.push(mean_std(&approx_l).1 / ref_std);
                    }
                    match model {
                        Model::Solved => solved_l = approx_l,
                        Model::RankOne => rank_one_l = approx_l,
                        _ => {}
                    }
                }
                // Rank-one on the grids against the solved random MDA (same model)
                consistency.push(ks(&solved_l, &rank_one_l));
            }
        }
        let quantile = |v: &[f64], q: f64| {
            let mut v: Vec<f64> = v.iter().copied().filter(|x| x.is_finite()).collect();
            v.sort_by(|a, b| a.total_cmp(b));
            if v.is_empty() {
                f64::NAN
            } else {
                v[((v.len() - 1) as f64 * q).round() as usize]
            }
        };
        println!(
            "| {n} | Exact sample paths (reference) | | | | | {:.1}% |",
            100. * ref_not_reached as f64 / ref_paths.max(1) as f64
        );
        for (mi, model) in models_list.iter().enumerate() {
            let st = &stats[mi];
            let energy = if st.energy_lift.is_empty() {
                "—".to_string()
            } else {
                format!(
                    "{:.3} / {:.3}",
                    median(&st.energy_lift),
                    median(&st.energy_pitch)
                )
            };
            let coverage = if st.coverage.is_empty() {
                "—".to_string()
            } else {
                format!("{:.2}", median(&st.coverage))
            };
            println!(
                "| {n} | {} | {:.3} [{:.3}] | {:.2} [{:.2}, {:.2}] | {energy} | {coverage} | {:.1}% |",
                model.label(),
                median(&st.ks),
                quantile(&st.ks, 0.9),
                median(&st.std_ratio),
                quantile(&st.std_ratio, 0.1),
                quantile(&st.std_ratio, 0.9),
                100. * st.not_reached as f64 / st.n_paths.max(1) as f64,
            );
        }

        println!(
            "| {n} | *Check: rank-one on the grids vs solved random MDA (KS)* | {:.3} [{:.3}] | | | | |",
            median(&consistency),
            quantile(&consistency, 0.9),
        );
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let local = args.first().is_some_and(|a| a == "local-kl");
    let n_seeds = args.iter().find_map(|a| a.parse().ok()).unwrap_or(10);
    if local {
        local_kl_study(n_seeds);
    } else {
        distributions(n_seeds);
        probability_of_minimizer(n_seeds);
    }
}
