//! Saving and loading of surrogate models in files

use crate::errors::{MoeError, Result};
use crate::types::GpFileFormat;
use crate::{GpMixture, MixintGpMixture, MixtureGpSurrogate};
use serde::Serialize;
use serde::de::DeserializeOwned;
use std::fs;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

/// Write a file through a temporary file of the same directory renamed into place.
///
/// A process stopped while writing (e.g. killed on Ctrl-C) leaves the previous file if any,
/// never a truncated one. The temporary file is removed when `write` fails. This protects from
/// process termination, not from a system crash (no `fsync`).
///
/// # Errors
///
/// The error returned by `write`, or an I/O error creating, writing or renaming the file
/// (e.g. when its directory does not exist).
pub fn write_atomically<P, E>(
    path: P,
    write: impl FnOnce(&mut BufWriter<fs::File>) -> std::result::Result<(), E>,
) -> std::result::Result<(), E>
where
    P: AsRef<Path>,
    E: From<std::io::Error>,
{
    let path = path.as_ref();
    let temporary = temporary_sibling(path)?;
    let result = fs::File::create(&temporary)
        .map_err(E::from)
        .and_then(|file| {
            let mut writer = BufWriter::new(file);
            write(&mut writer)?;
            writer.into_inner().map_err(|e| E::from(e.into_error()))?;
            fs::rename(&temporary, path).map_err(E::from)
        });
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

/// Hidden temporary file next to `path`, unique within the process and among processes
fn temporary_sibling(path: &Path) -> std::io::Result<PathBuf> {
    static COUNT: AtomicU64 = AtomicU64::new(0);
    let name = path.file_name().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("not a file path: {path:?}"),
        )
    })?;
    let mut temporary = std::ffi::OsString::from(".");
    temporary.push(name);
    temporary.push(format!(
        ".{}.{}.tmp",
        std::process::id(),
        COUNT.fetch_add(1, Ordering::Relaxed)
    ));
    Ok(path.with_file_name(temporary))
}

/// Encode the model in the given format (compact JSON or bincode)
fn encode<T: Serialize + ?Sized>(model: &T, format: GpFileFormat) -> Result<Vec<u8>> {
    match format {
        GpFileFormat::Json => serde_json::to_vec(model).map_err(MoeError::SaveJsonError),
        GpFileFormat::Binary => bincode::serde::encode_to_vec(model, bincode::config::standard())
            .map_err(MoeError::SaveBinaryError),
    }
}

/// Write the encoded bytes in the file (see [write_atomically]), encoded beforehand
/// so that an encoding error leaves the file untouched
fn write_file<P: AsRef<Path>>(path: P, encode: impl FnOnce() -> Result<Vec<u8>>) -> Result<()> {
    let bytes = encode()?;
    write_atomically(path, |file| file.write_all(&bytes).map_err(MoeError::from))
}

/// Save the model in the given file
pub(crate) fn write_model<T: Serialize + ?Sized>(
    path: &str,
    model: &T,
    format: GpFileFormat,
) -> Result<()> {
    write_file(path, || encode(model, format))
}

/// Load the model saved in the given file.
///
/// A JSON decoding error is a [MoeError::LoadError], a binary one a [MoeError::LoadBinaryError].
pub(crate) fn read_model<T: DeserializeOwned>(path: &str, format: GpFileFormat) -> Result<T> {
    let data = fs::read(path)?;
    match format {
        GpFileFormat::Json => serde_json::from_slice(&data).map_err(|err| {
            MoeError::LoadError(format!("Error while loading from {path}: ({err})"))
        }),
        GpFileFormat::Binary => {
            bincode::serde::decode_from_slice(&data, bincode::config::standard())
                .map(|(model, _)| model)
                .map_err(MoeError::LoadBinaryError)
        }
    }
}

/// Save several models in one file, as a vector of models.
///
/// JSON is written pretty-printed. The file is loaded back with [load_models].
pub fn save_models<P: AsRef<Path>>(
    path: P,
    models: &[Box<dyn MixtureGpSurrogate>],
    format: GpFileFormat,
) -> Result<()> {
    write_file(path, || match format {
        GpFileFormat::Json => serde_json::to_vec_pretty(models).map_err(MoeError::SaveJsonError),
        GpFileFormat::Binary => encode(models, format),
    })
}

/// Decode `data` as a `T` with bincode, all bytes being used: a saved
/// [MixintGpMixture] must not be decoded as its leading [GpMixture] field
fn decode_binary<T: DeserializeOwned>(data: &[u8]) -> Option<T> {
    bincode::serde::decode_from_slice(data, bincode::config::standard())
        .ok()
        .filter(|(_, read)| *read == data.len())
        .map(|(model, _)| model)
}

fn decode_json<T: DeserializeOwned>(data: &[u8]) -> Option<T> {
    serde_json::from_slice(data).ok()
}

type Models = Vec<Box<dyn MixtureGpSurrogate>>;

fn single<T: MixtureGpSurrogate + 'static>(model: T) -> Models {
    vec![Box::new(model)]
}

/// Decode a vector of models, then a single [GpMixture], then a single [MixintGpMixture]
fn decode_binary_models(data: &[u8]) -> Option<Models> {
    decode_binary::<Models>(data)
        .or_else(|| decode_binary::<GpMixture>(data).map(single))
        .or_else(|| decode_binary::<MixintGpMixture>(data).map(single))
}

/// JSON counterpart of [decode_binary_models]
fn decode_json_models(data: &[u8]) -> Option<Models> {
    decode_json::<Models>(data)
        .or_else(|| decode_json::<GpMixture>(data).map(single))
        .or_else(|| decode_json::<MixintGpMixture>(data).map(single))
}

/// Load the models saved in the given file.
///
/// The file format (binary or JSON) is detected, and the file may hold either
/// a vector of models ([save_models], `Egor` GP files, `gpx fit`) or a single
/// [GpMixture] or [MixintGpMixture] (`save()`, Python `Gpx.save()`).
///
/// # Errors
///
/// [MoeError::LoadIoError] when the file cannot be read,
/// [MoeError::LoadError] when its content cannot be decoded.
pub fn load_models<P: AsRef<Path>>(path: P) -> Result<Vec<Box<dyn MixtureGpSurrogate>>> {
    let data = fs::read(path.as_ref())?;
    decode_binary_models(&data)
        .or_else(|| decode_json_models(&data))
        .ok_or_else(|| {
            MoeError::LoadError(format!(
                "cannot decode models from {}: neither binary nor JSON models",
                path.as_ref().display()
            ))
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GpMixtureParams, GpSurrogate, MixintGpMixtureParams, XType};
    use linfa::ParamGuard;
    use linfa::traits::Fit;
    use ndarray::{Array1, array};

    fn tmp_path(name: &str) -> String {
        std::env::temp_dir()
            .join(format!("egobox_moe_persistence_{name}"))
            .to_str()
            .unwrap()
            .to_string()
    }

    fn gp_mixture() -> GpMixture {
        let xt = array![[0.], [1.], [2.], [3.], [4.]];
        let yt = array![0., 1., 4., 9., 16.];
        GpMixtureParams::new()
            .check()
            .unwrap()
            .fit(&linfa::Dataset::new(xt, yt))
            .unwrap()
    }

    fn mixint_gp_mixture() -> MixintGpMixture {
        let xt = array![
            [0., 0.],
            [1., 1.],
            [2., 0.],
            [0., 1.],
            [1., 0.],
            [2., 1.],
            [0.5, 0.],
            [1.5, 1.]
        ];
        let yt: Array1<f64> = xt.column(0).mapv(|v| v * v) + xt.column(1);
        let xtypes = [XType::Float(0., 2.), XType::Enum(2)];
        let mut params = MixintGpMixtureParams::new(Some(&xtypes), &GpMixtureParams::new());
        params.work_in_folded_space(true);
        params
            .check()
            .unwrap()
            .fit(&linfa::Dataset::new(xt, yt))
            .unwrap()
    }

    /// Predictions of the loaded models match (JSON floats are not round-tripped exactly)
    fn assert_same_predictions(
        loaded: &[Box<dyn MixtureGpSurrogate>],
        expected: &[&dyn MixtureGpSurrogate],
        x: &ndarray::Array2<f64>,
        name: &str,
    ) {
        assert_eq!(loaded.len(), expected.len(), "{name}");
        for (gp, ref_gp) in loaded.iter().zip(expected) {
            let err = gp.predict(&x.view()).unwrap() - ref_gp.predict(&x.view()).unwrap();
            assert!(err.iter().all(|d| d.abs() < 1e-4), "{name}: {err}");
        }
    }

    #[test]
    fn test_save_load_models() {
        let (gp, mixint) = (gp_mixture(), mixint_gp_mixture());
        let models: Vec<Box<dyn MixtureGpSurrogate>> =
            vec![Box::new(gp.clone()), Box::new(gp.clone())];
        for (name, format) in [
            ("vec.json", GpFileFormat::Json),
            ("vec.bin", GpFileFormat::Binary),
        ] {
            let path = tmp_path(name);
            save_models(&path, &models, format).unwrap();
            let loaded = load_models(&path);
            let _ = fs::remove_file(&path);
            assert_same_predictions(&loaded.unwrap(), &[&gp, &gp], &array![[0.5], [2.5]], name);
        }
        // single models, mixint ones must not be decoded as their inner GpMixture
        for (name, format) in [
            ("gp.json", GpFileFormat::Json),
            ("gp.bin", GpFileFormat::Binary),
        ] {
            let path = tmp_path(name);
            gp.save(&path, format).unwrap();
            let loaded = load_models(&path);
            let _ = fs::remove_file(&path);
            assert_same_predictions(&loaded.unwrap(), &[&gp], &array![[0.5], [2.5]], name);
        }
        for (name, format) in [
            ("mixint.json", GpFileFormat::Json),
            ("mixint.bin", GpFileFormat::Binary),
        ] {
            let path = tmp_path(name);
            mixint.save(&path, format).unwrap();
            let loaded = load_models(&path);
            let _ = fs::remove_file(&path);
            assert_same_predictions(
                &loaded.unwrap(),
                &[&mixint],
                &array![[1.2, 1.], [0.3, 0.]],
                name,
            );
        }
    }

    #[test]
    fn test_save_models_binary_is_bincode_vector() {
        // same bytes as the vector encoding used so far by Egor and gpx
        let models: Vec<Box<dyn MixtureGpSurrogate>> = vec![Box::new(gp_mixture())];
        let path = tmp_path("bytes.bin");
        save_models(&path, &models, GpFileFormat::Binary).unwrap();
        let bytes = fs::read(&path);
        let _ = fs::remove_file(&path);
        assert_eq!(
            bytes.unwrap(),
            bincode::serde::encode_to_vec(&models, bincode::config::standard()).unwrap()
        );
    }

    #[test]
    fn test_load_models_errors() {
        let path = tmp_path("invalid.bin");
        fs::write(&path, b"not a model").unwrap();
        let res = load_models(&path);
        let _ = fs::remove_file(&path);
        assert!(matches!(res, Err(MoeError::LoadError(_))));
        assert!(matches!(
            load_models(tmp_path("missing.bin")),
            Err(MoeError::LoadIoError(_))
        ));
        assert!(save_models(tmp_path("missing_dir/gp.bin"), &[], GpFileFormat::Binary).is_err());
    }

    #[test]
    fn test_write_atomically() {
        let dir = std::path::PathBuf::from(tmp_path(&format!("atomic_{}", std::process::id())));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("out.txt");
        fs::write(&path, b"old").unwrap();
        write_atomically(&path, |file| file.write_all(b"new")).unwrap();
        assert_eq!(fs::read(&path).unwrap(), b"new");

        // stopped while writing: the previous file is kept, no temporary file is left
        let res: std::io::Result<()> = write_atomically(&path, |file| {
            file.write_all(b"partial")?;
            Err(std::io::Error::other("stopped"))
        });
        assert!(res.is_err());
        assert_eq!(fs::read(&path).unwrap(), b"new");
        assert_eq!(fs::read_dir(&dir).unwrap().count(), 1);

        // a model encoding error leaves the file untouched
        let res = write_file(&path, || Err(MoeError::InvalidValueError("encode".into())));
        assert!(res.is_err());
        assert_eq!(fs::read(&path).unwrap(), b"new");
        fs::remove_dir_all(dir).unwrap();
    }
}
