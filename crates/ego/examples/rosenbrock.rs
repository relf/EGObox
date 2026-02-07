use clap::Parser;

use egobox_ego::{EgorBuilder, InfillOptimizer, InfillStrategy, OptimResult, Result, RunInfo};
use egobox_moe::{CorrelationSpec, RegressionSpec};
use ndarray::{Array, Array2, ArrayView2, Zip, array};

/// Rosenbrock test function: min f(x)=0 at x=(1, 1)
fn rosenbrock(x: &ArrayView2<f64>) -> Array2<f64> {
    let mut y: Array2<f64> = Array2::zeros((x.nrows(), 1));
    Zip::from(y.rows_mut())
        .and(x.rows())
        .par_for_each(|mut yi, xi| {
            yi.assign(&array![argmin_testfunctions::rosenbrock(&xi.to_vec())])
        });
    y
}

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long, default_value_t = 2)]
    dim: usize,
    #[arg(short, long, default_value = "./rosenbrock")]
    outdir: String,
    #[arg(short, long, default_value_t = 1)]
    rep: usize,
}

const BUDGET: usize = 100;

fn run_egor(dim: usize, outdir: &String, num: usize) -> Result<OptimResult<f64>> {
    let n_doe = dim + 1;
    let max_iters = BUDGET - n_doe;

    let data = [-2., 2.].repeat(dim);
    let xlimits = Array::from_shape_vec((dim, 2), data).unwrap();

    EgorBuilder::optimize(rosenbrock)
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
                .max_iters(max_iters)
                .n_start(300)
                // .target(1e-2)
                .outdir(outdir)
        })
        .min_within(&xlimits)
        .expect("Egor configured")
        .run_info(RunInfo {
            fname: "rosenbrock".to_string(),
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

        println!("Rosenbrock minimum y = {} at x = {}", res.y_opt, res.x_opt);
    }

    Ok(())
}
