use crate::errors::Result;
#[cfg(feature = "persistent")]
use std::fs;
#[cfg(feature = "persistent")]
use std::io::Write;
use std::path::Path;

/// Save models in a bincode file
pub fn save_gp_models<P: AsRef<Path>>(
    path: P,
    models: &[Box<dyn egobox_moe::MixtureGpSurrogate>],
) -> Result<()> {
    let mut file = fs::File::create(path).unwrap();

    let bytes = bincode::serde::encode_to_vec(models, bincode::config::standard())?;
    file.write_all(&bytes)?;

    Ok(())
}

/// Load models from a bincode file
pub fn load_gp_models<P: AsRef<Path>>(
    path: P,
) -> Result<Vec<Box<dyn egobox_moe::MixtureGpSurrogate>>> {
    let data: Vec<u8> = std::fs::read(path)?;
    let models: Vec<Box<dyn egobox_moe::MixtureGpSurrogate>> =
        bincode::serde::decode_from_slice(&data, bincode::config::standard())
            .map(|(res, _)| res)
            .unwrap_or_default();

    Ok(models)
}
