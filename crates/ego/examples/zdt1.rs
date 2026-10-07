use egobox_ego::EgorBuilder;
use ndarray::{Array2, ArrayView2, Zip, array};

/// ZDT1 bi-objective test function: the Pareto front is f2 = 1 - sqrt(f1)
/// with x1 in [0, 1] and the other components equal to 0
fn zdt1(x: &ArrayView2<f64>) -> Array2<f64> {
    let mut y = Array2::zeros((x.nrows(), 2));
    Zip::from(y.rows_mut())
        .and(x.rows())
        .for_each(|mut yi, xi| {
            let f1 = xi[0];
            let g = 1. + 9. * xi.slice(ndarray::s![1..]).mean().unwrap();
            yi.assign(&array![f1, g * (1. - (f1 / g).sqrt())]);
        });
    y
}

fn main() {
    let res = EgorBuilder::optimize(zdt1)
        .configure(|config| config.n_obj(2).n_doe(10).max_iters(30).seed(42))
        .min_within(&array![[0., 1.], [0., 1.], [0., 1.]])
        .expect("Egor configured")
        .run_pareto()
        .expect("ZDT1 optimization");
    println!(
        "Pareto front approximation ({} points out of {} evaluations):",
        res.y_pareto.nrows(),
        res.y_doe.nrows()
    );
    for (x, y) in res.x_pareto.rows().into_iter().zip(res.y_pareto.rows()) {
        println!("f(x) = {y} at x = {x}");
    }
}
