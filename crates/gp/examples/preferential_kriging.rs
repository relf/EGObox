use egobox_gp::{PreferenceComparison, PreferentialKriging, ThetaTuning};
use ndarray::{Array, Axis, array};

fn main() {
    // Discrete 1D candidate grid in [0, 1] with latent human utility maximized at x = 0.72
    let candidates = Array::linspace(0.0, 1.0, 26).insert_axis(Axis(1));
    let latent_utility = |x: f64| -((x - 0.72) * 3.0).powi(2) + 0.25 * (6.0 * x).sin();

    let mut observations = Vec::new();
    let mut compared_pairs = Vec::new();

    // Start from prior (0 comparisons) and run 8 active EUBO pairwise queries
    let mut model = PreferentialKriging::<f64>::params()
        .theta_tuning(ThetaTuning::Fixed(array![0.5]))
        .fit(&candidates, &observations)
        .expect("prior preferential GP initialized");

    for step in 1..=8 {
        let (idx_a, idx_b, eubo_val) = model
            .select_next_pair_eubo(&candidates, &compared_pairs)
            .expect("next EUBO pair selected");
        compared_pairs.push((idx_a, idx_b));

        let ua = latent_utility(candidates[[idx_a, 0]]);
        let ub = latent_utility(candidates[[idx_b, 0]]);
        let comp = if (ua - ub).abs() < 0.02 {
            PreferenceComparison::tie(idx_a, idx_b)
        } else if ua > ub {
            PreferenceComparison::prefer(idx_a, idx_b)
        } else {
            PreferenceComparison::prefer(idx_b, idx_a)
        };
        observations.push(comp);

        model = PreferentialKriging::<f64>::params()
            .n_start(3)
            .fit(&candidates, &observations)
            .expect("preferential GP updated");

        println!(
            "Step {step}: queried x_a={:.2} vs x_b={:.2} (EUBO={eubo_val:.4}) -> {model}",
            candidates[[idx_a, 0]],
            candidates[[idx_b, 0]]
        );
    }

    let utilities = model.predict(&candidates).unwrap();
    let best_idx = utilities
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;
    println!(
        "Recommended best option: x = {:.2} (true peak near 0.72)",
        candidates[[best_idx, 0]]
    );
}
