//! Branin-Hoo optimization example with hidden constraints
//!
//! This example demonstrates Bayesian optimization with hidden constraints
//! using the Branin-Hoo test function from the paper:
//! "Bayesian optimization with hidden constraints for aircraft design"
//! by Tfaily et al. (2024).
//!
//! The Branin-Hoo function has a hidden simulation failure region where
//! the function returns NaN values. The example compares two acquisition
//! functions:
//! - Standard EI with viability handling (EFI_P approach)
//! - Feasibility Enhanced Expected Improvement (EFI_FE approach)
//!
//! See: https://doi.org/10.1007/s00158-024-03833-8

use egobox_ego::{
    EgorBuilder, FeasibleInfillStrategy, InfillOptimizer, InfillStrategy, RuntimeFlags,
};
use ndarray::{Array2, ArrayView2, Zip, array};

/// Scaled Branin-Hoo function
///
/// The original Branin-Hoo function is defined as:
/// f(x1, x2) = (x2 - 5.1/(4*pi^2) * x1^2 + 5/pi * x1 - 6)^2
///             + 10 * (1 - 1/(8*pi)) * cos(x1) + 10
///
/// This is the scaled version with x1, x2 in [0, 1]:
/// x1_bar = 15*x1 - 5
/// x2_bar = 15*x2
#[allow(dead_code)]
fn branin_hoo(x: &ArrayView2<f64>) -> Array2<f64> {
    let mut y = Array2::zeros((x.nrows(), 1));
    Zip::from(y.rows_mut())
        .and(x.rows())
        .for_each(|mut yi, xi| {
            let x1 = xi[0];
            let x2 = xi[1];

            // Scale to original domain
            let x1_bar = 15.0 * x1 - 5.0;
            let x2_bar = 15.0 * x2;

            // Branin-Hoo function
            let term1 = x2_bar - 5.1 * x1_bar.powi(2) / (4.0 * std::f64::consts::PI.powi(2))
                + 5.0 * x1_bar / std::f64::consts::PI
                - 6.0;
            let term2 = 10.0 * (1.0 - 1.0 / (8.0 * std::f64::consts::PI)) * x1_bar.cos() + 1.0;

            yi[0] = term1.powi(2) + 10.0 * term2;
        });
    y
}

/// Branin-Hoo with hidden constraint region
///
/// Hidden region H: returns NaN when x1 < 0.4 AND x2 > 0.5
/// except for the inner feasible region where the minimum lies:
/// 0.05 < x1 < 0.2 AND 0.7 < x2 < 0.9
fn branin_hoo_hidden(x: &ArrayView2<f64>) -> Array2<f64> {
    let mut y = Array2::zeros((x.nrows(), 1));
    Zip::from(y.rows_mut())
        .and(x.rows())
        .for_each(|mut yi, xi| {
            let x1 = xi[0];
            let x2 = xi[1];

            // Check if in hidden constraint region
            // Hidden region: x1 < 0.4 AND x2 > 0.5
            // Exception (feasible): 0.05 < x1 < 0.2 AND 0.7 < x2 < 0.9
            let in_hidden = x1 < 0.4 && x2 > 0.5;
            let in_feasible_exception = x1 > 0.05 && x1 < 0.2 && x2 > 0.7 && x2 < 0.9;

            if in_hidden && !in_feasible_exception {
                yi[0] = f64::NAN;
            } else {
                // Scale to original domain
                let x1_bar = 15.0 * x1 - 5.0;
                let x2_bar = 15.0 * x2;

                // Branin-Hoo function
                let term1 = x2_bar - 5.1 * x1_bar.powi(2) / (4.0 * std::f64::consts::PI.powi(2))
                    + 5.0 * x1_bar / std::f64::consts::PI
                    - 6.0;
                let term2 = 10.0 * (1.0 - 1.0 / (8.0 * std::f64::consts::PI)) * x1_bar.cos() + 1.0;

                yi[0] = term1.powi(2) + 10.0 * term2 + 3.0 * x1_bar;
            }
        });
    y
}

/// Check if a point is in the hidden constraint region
fn is_hidden_constraint(x: &[f64]) -> bool {
    let x1 = x[0];
    let x2 = x[1];

    // Hidden region: x1 < 0.4 AND x2 > 0.5
    // Exception (feasible): 0.05 < x1 < 0.2 AND 0.7 < x2 < 0.9
    let in_hidden = x1 < 0.4 && x2 > 0.5;
    let in_feasible_exception = x1 > 0.05 && x1 < 0.2 && x2 > 0.7 && x2 < 0.9;

    in_hidden && !in_feasible_exception
}

fn main() -> egobox_ego::Result<()> {
    println!("Branin-Hoo Optimization with Hidden Constraints");
    println!("===============================================");
    println!("DOE size: 5");
    println!("Max iterations: 50");
    println!("Hidden constraint: x1 < 0.4 AND x2 > 0.5 (except feasible island)");
    println!("Feasible island: 0.05 < x1 < 0.2 AND 0.7 < x2 < 0.9");
    println!();

    let xlimits = array![[0.0, 1.0], [0.0, 1.0]]; // x1 in [0, 1], x2 in [0, 1]

    // Run with standard viability handling (similar to EFI_P)
    println!("Run 1: Standard viability handling (EFI_P-like)");
    let result1 = EgorBuilder::optimize(branin_hoo_hidden)
        .configure(|config| {
            config
                .n_doe(5)
                .infill_strategy(InfillStrategy::EI)
                .infill_optimizer(InfillOptimizer::Slsqp)
                .n_start(50)
                .outdir("./branin_hoo_run1")
                .max_iters(50)
                .seed(42)
                .feasibility_infill(FeasibleInfillStrategy::ViabilityWeighted)
                .runtime_flags(RuntimeFlags::default().use_run_recorder(true))
        })
        .verbose(log::LevelFilter::Info)
        .min_within(&xlimits)
        .expect("Egor configured")
        .run()?;

    println!(
        "  Result: y = {:.6} at x = [{:.6}, {:.6}]",
        result1.y_opt[0], result1.x_opt[0], result1.x_opt[1]
    );

    // Run with feasibility enhanced acquisition (EFI_FE)
    println!("\nRun 2: Feasibility Enhanced (EFI_FE)");
    let result2 = EgorBuilder::optimize(branin_hoo_hidden)
        .configure(|config| {
            config
                .n_doe(5)
                .infill_strategy(InfillStrategy::EI)
                .infill_optimizer(InfillOptimizer::Slsqp)
                .n_start(50)
                .outdir("./branin_hoo_run2")
                .max_iters(50)
                .seed(42)
                .feasibility_infill(FeasibleInfillStrategy::AlphaPoweredViabilityWeighted(0.5))
                .runtime_flags(RuntimeFlags::default().use_run_recorder(true))
        })
        .verbose(log::LevelFilter::Info)
        .min_within(&xlimits)
        .expect("Egor configured")
        .run()?;

    println!(
        "  Result: y = {:.6} at x = [{:.6}, {:.6}]",
        result2.y_opt[0], result2.x_opt[0], result2.x_opt[1]
    );

    // Check feasibility of solutions
    println!("\nFeasibility check:");
    println!(
        "  Solution 1 feasible: {}",
        !is_hidden_constraint(&result1.x_opt.to_vec())
    );
    println!(
        "  Solution 2 feasible: {}",
        !is_hidden_constraint(&result2.x_opt.to_vec())
    );

    println!("\nDone!");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_branin_hoo_known_minimum() {
        // Test at known minimum point (in scaled domain)
        // Original minimum at x1 = pi, x2 = 2.475
        // Scaled: x1 = (pi + 5) / 15 ≈ 0.547, x2 = 2.475 / 15 ≈ 0.165
        let x = array![[0.547, 0.165]];
        let y = branin_hoo(&x.view());

        // Known minimum value is approximately 0.397887
        assert!(y[0] < 0.45);
    }

    #[test]
    fn test_hidden_constraint_detection() {
        // Test points in hidden region
        assert!(is_hidden_constraint(&[0.2, 0.6])); // x1 < 0.4, x2 > 0.5
        assert!(is_hidden_constraint(&[0.3, 0.8])); // x1 < 0.4, x2 > 0.5

        // Test points in feasible exception region
        assert!(!is_hidden_constraint(&[0.1, 0.8])); // 0.05 < x1 < 0.2, 0.7 < x2 < 0.9

        // Test points outside hidden region
        assert!(!is_hidden_constraint(&[0.5, 0.3])); // x1 > 0.4
        assert!(!is_hidden_constraint(&[0.3, 0.4])); // x2 < 0.5
    }

    #[test]
    fn test_branin_hoo_hidden_feasible() {
        // Test that feasible points return valid values
        let x = array![[0.1, 0.8]]; // In feasible exception region
        let y = branin_hoo_hidden(&x.view());
        assert!(y[0].is_finite());
    }

    #[test]
    fn test_branin_hoo_hidden_nan() {
        // Test that hidden constraint points return NaN
        let x = array![[0.2, 0.6]]; // In hidden region
        let y = branin_hoo_hidden(&x.view());
        assert!(y[0].is_nan());
    }
}
