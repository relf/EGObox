use crate::errors::Result;
use std::fs;
use std::io::Write;
use std::path::Path;

/// Save models in a bincode file
pub fn save_gp_models<P: AsRef<Path>>(
    path: P,
    models: &[Box<dyn egobox_moe::MixtureGpSurrogate>],
) -> Result<()> {
    let mut file = fs::File::create(path)?;

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
            .map(|(res, _)| res)?;

    Ok(models)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_gp_models_from_invalid_file() {
        let path = "target/test_gp_recorder_invalid.bin";
        std::fs::write(path, b"not gp models").unwrap();
        let res = load_gp_models(path);
        let _ = std::fs::remove_file(path);
        assert!(res.is_err(), "decoding error expected");
    }

    #[test]
    fn test_save_gp_models_in_missing_dir() {
        let res = save_gp_models("target/test_gp_recorder_missing_dir/gp.bin", &[]);
        assert!(res.is_err(), "file creation error expected");
    }
}
