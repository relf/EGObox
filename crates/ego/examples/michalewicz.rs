use clap::Parser;

use egobox_ego::{EgorBuilder, InfillOptimizer, InfillStrategy, OptimResult, Result, RunInfo};
use egobox_moe::{CorrelationSpec, RegressionSpec};
use ndarray::{Array, Array2, ArrayView2, Zip};

/// Michalewicz test function:
/// min D=2 f(x)=-1.8013 at x=[2.20, 1.57]
/// min D=5 f(x)=-4.687658 at x=[2.20, 1.57, 1.28, 2.31, 1.38]
/// min D=10 f(x)=-9.66015 at x=[2.20, 1.57, 1.28, 2.31, 1.38, 1.87, 1.32, 1.75, 1.46, 1.55]
fn michalewicz(x: &ArrayView2<f64>) -> Array2<f64> {
    let m = 10.0;
    let n_points = x.nrows();

    let mut y = Array2::zeros((n_points, 1));

    Zip::from(y.rows_mut())
        .and(x.rows())
        .for_each(|mut result_row, point| {
            let sum: f64 = point
                .iter()
                .enumerate()
                .map(|(j, &x_j)| {
                    let j_f = (j + 1) as f64;
                    let term = (j_f * x_j * x_j / std::f64::consts::PI).sin().powf(2.0 * m);
                    x_j.sin() * term
                })
                .sum();

            result_row[0] = -sum;
        });
    y
}

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long, default_value_t = 2)]
    dim: usize,
    #[arg(short, long, default_value = "./michalewicz")]
    outdir: String,
    #[arg(short, long, default_value_t = 1)]
    rep: usize,
}

const BUDGET: usize = 300;

fn run_egor(dim: usize, outdir: &String, num: usize) -> Result<OptimResult<f64>> {
    let n_doe = dim + 1;
    let max_iters = BUDGET - n_doe;

    let data = [0., std::f64::consts::PI].repeat(dim);
    let xlimits = Array::from_shape_vec((dim, 2), data).unwrap();

    EgorBuilder::optimize(michalewicz)
        .configure(|config| {
            config
                .n_doe(n_doe)
                .configure_gp(|gp| {
                    gp.regression_spec(RegressionSpec::CONSTANT)
                        .correlation_spec(CorrelationSpec::ALL)
                })
                .infill_strategy(InfillStrategy::LogEI)
                .infill_optimizer(InfillOptimizer::Slsqp)
                .trego(true)
                // for dim=10
                //.coego(egobox_ego::CoegoStatus::Enabled(2))
                .max_iters(max_iters)
                .n_start(400)
                .outdir(outdir)
        })
        .min_within(&xlimits)
        .expect("Egor configured")
        .run_info(RunInfo {
            fname: "michalewicz".to_string(),
            num,
        })
        .run()
}

fn main() -> Result<()> {
    let args = Args::parse();

    let dim = args.dim;
    let outdir = args.outdir;
    let rep = args.rep;

    for num in 1..=rep {
        println!(">>>> Run {} of {}", num, rep);
        let outdir = format!("{}/run{:0>2}", outdir, num);
        let res = run_egor(dim, &outdir, num)?;

        println!("Michalewicz minimum y = {} at x = {}", res.y_opt, res.x_opt);
    }

    Ok(())
}
