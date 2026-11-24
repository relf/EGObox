use egobox_doe::{Lhs, SamplingMethod};
use egobox_ego::{Cstr, EgorServiceFactory};
use ndarray::{Array2, ArrayBase, ArrayView2, Axis, Data, Ix1, Zip, array, concatenate};

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
    let mut doe = Lhs::new(&xlimits).sample(3);

    // We use Egor optimizer as a service
    let egor = EgorServiceFactory::<Cstr>::optimize()
        .configure(|config| config.n_cstr(2).seed(42))
        .min_within(&xlimits)
        .expect("Egor configured");

    let mut y_doe = f_g24(&doe.view());
    for _i in 0..10 {
        // We tell function values and ask for next x location
        let x_suggested = egor.suggest(&doe, &y_doe).expect("suggestion");

        doe = concatenate![Axis(0), doe, x_suggested];
        y_doe = f_g24(&doe.view());
    }

    println!("G24 optim x suggestion history = {doe:?}");
}
