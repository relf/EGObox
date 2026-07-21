use pyo3_stub_gen::Result;

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().filter_or("RUST_LOG", "info")).init();
    let mut stub = egobox::stub_info()?;

    // pyo3_stub_gen heuristic fails to find the correct package root with egobox project layout.
    // We have a pure Rust project with a Python package in a subdirectory,
    // so we need to adjust the python_root to point to the egobox package
    // instead of the root of the python source tree.
    stub.python_root = stub.python_root.join("egobox");
    stub.generate()?;

    Ok(())
}
