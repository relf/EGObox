use crate::types::Verbose;
use egobox_ego::EGOBOX_LOG;
use env_logger::{Builder, Env};
use pyo3::prelude::*;

// Set filter from env variable or default to error level,
// then override with verbose argument if any
pub(crate) fn init_logger(py: Python, verbose: Option<Py<PyAny>>) {
    let env = Env::default().filter_or(EGOBOX_LOG, "error");
    let mut builder = Builder::from_env(env);
    let verbose = match verbose {
        Some(v) => {
            if let Ok(v) = v.extract::<Verbose>(py) {
                v.into()
            } else if let Ok(n) = v.extract::<u64>(py) {
                match n {
                    0 => log::LevelFilter::Error,
                    1 => log::LevelFilter::Warn,
                    2 => log::LevelFilter::Info,
                    3 => log::LevelFilter::Debug,
                    _ => log::LevelFilter::Trace,
                }
            } else {
                log::LevelFilter::Off
            }
        }
        None => log::LevelFilter::Off,
    };
    if verbose != log::LevelFilter::Off {
        builder.filter_level(verbose);
    }
    builder.target(env_logger::Target::Stdout).try_init().ok();
}
