//! Same problem as `g24.rs`, but using the new `InfillOptimizer::Ipopt` backend
//! (interior-point method, via the pure-Rust `pounce` crate) instead of Cobyla.
//!
//! Requires the `pounce` cargo feature:
//!
//! ```sh
//! cargo run --example g24_ipopt --features pounce
//! ```

use egobox_doe::{Lhs, SamplingMethod};

use egobox_ego::{EgorBuilder, InfillOptimizer};

use ndarray::{Array2, ArrayBase, ArrayView2, Data, Ix1, Zip, array};

// Objective

fn g24(x: &ArrayBase<impl Data<Elem = f64>, Ix1>) -> f64 {
    // Function G24: 1 global optimum y_opt = -5.5080 at x_opt =(2.3295, 3.1785)
    -x[0] - x[1]
}

// Constraints < 0

fn g24_c1(x: &ArrayBase<impl Data<Elem = f64>, Ix1>) -> f64 {
    -2.0 * x[0].powf(4.0) + 8.0 * x[0].powf(3.0) - 8.0 * x[0].powf(2.0) + x[1] - 2.0
}

fn g24_c2(x: &ArrayBase<impl Data<Elem = f64>, Ix1>) -> f64 {
    -4.0 * x[0].powf(4.0) + 32.0 * x[0].powf(3.0) - 88.0 * x[0].powf(2.0) + 96.0 * x[0] + x[1]
        - 36.0
}

fn f_g24(x: &ArrayView2<f64>) -> Array2<f64> {
    let mut y = Array2::zeros((x.nrows(), 3));
    Zip::from(y.rows_mut())
        .and(x.rows())
        .for_each(|mut yi, xi| {
            yi.assign(&array![g24(&xi), g24_c1(&xi), g24_c2(&xi)]);
        });
    y
}

fn main() {
    let xlimits = array![[0., 3.], [0., 4.]];
    let doe = Lhs::new(&xlimits).sample(3);

    let res = EgorBuilder::optimize(f_g24)
        .configure(|config| {
            config
                .n_cstr(2)
                .doe(&doe)
                .max_iters(100)
                .infill_optimizer(InfillOptimizer::Ipopt)
                .seed(42)
        })
        .min_within(&xlimits)
        .expect("Egor configured")
        .run()
        .expect("Minimize failure");
    println!("G24 optim result (IPOPT infill optimizer) = {}", res.y_opt);
}
