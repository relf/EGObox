use crate::utils::EGOBOX_LOG;
use env_logger::{Builder, Env};

// Set filter from env variable or default to error level,
// then override with verbose argument if any
pub(crate) fn init_logger(verbose: Option<log::LevelFilter>) {
    let env = Env::default().filter_or(EGOBOX_LOG, "error");
    let mut builder = Builder::from_env(env);
    let verbose = match verbose {
        Some(v) => v,
        None => log::LevelFilter::Off,
    };
    if verbose != log::LevelFilter::Off {
        builder.filter_level(verbose);
    }
    builder.target(env_logger::Target::Stdout).try_init().ok();
}
