use pyo3_stub_gen::Result;

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().filter_or("RUST_LOG", "info")).init();
    let stub = _rust::stub_info()?;
    stub.generate()?;

    // The Rust stub generator generates a stub file in the python/egobox/_rust directory,
    // As Python API is the same as core Rust API, we move the stub file to python/egobox/egobox.pyi
    log::info!("Moving stub file to egobox/egobox.pyi");
    let from = std::path::Path::new("python/egobox/_rust/__init__.pyi");
    let to = std::path::Path::new("python/egobox/egobox.pyi");
    std::fs::rename(from, to).expect(
        "Should move stub file to egobox/egobox.pyi. Ensure to run this command from the root of the repository, and that the stub file exists.",
    );
    Ok(())
}
