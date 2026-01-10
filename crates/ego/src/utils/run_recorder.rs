use ndarray::{Array2, Zip};
use serde::{Deserialize, Serialize};
use std::{fs::File, io::BufReader, path::Path};

use egobox_moe::CorrelationSpec;

use crate::{EgorState, Result, ValidEgorConfig};

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct ProblemMetadata {
    pub test_function: String,
    pub dimensionality: usize,
    pub lower_bounds: Vec<f64>,
    pub upper_bounds: Vec<f64>,
    pub replication_number: usize,
}

#[derive(Debug, Clone, Default)]
pub struct OtherParams;

impl Serialize for OtherParams {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        // Always serialize as an empty object
        let empty = serde_json::Map::new();
        empty.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for OtherParams {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        // Deserialize into a serde_json::Value and ignore the content
        let _ = serde_json::Value::deserialize(deserializer)?;
        Ok(OtherParams)
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct AlgorithmParameters {
    pub acquisition_function: String,
    pub kernel: String,
    pub initial_samples: usize,
    #[serde(rename = "BO_iterations")]
    pub bo_iterations: u64,
    pub total_samples: usize,
    pub batch_size: usize,
    pub other_params: OtherParams,
    pub seed: i32,
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct ExtraInfo {
    pub team_notes: String,
    pub code_reference: String,
    pub other_files: Vec<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct OrdSample {
    pub iterations: u64,
    pub locations: Vec<f64>,
    pub evaluations: f64,
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct InitialSamples {
    pub sampled_locations: Vec<OrdSample>,
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct SearchIteration {
    pub iterations: u64,
    pub batch_size: usize,
    pub sampled_locations: Vec<Sample>,
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct Sample {
    pub locations: Vec<f64>,
    pub evaluations: f64,
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct EgorRunData {
    pub problem_metadata: ProblemMetadata,
    pub algorithm_parameters: AlgorithmParameters,
    pub extra_info: ExtraInfo,
    pub initial_samples: InitialSamples,
    pub search_iterations: Vec<SearchIteration>,
}

pub(crate) fn init_run_info(
    xlimits: Array2<f64>,
    config: ValidEgorConfig,
    state: &EgorState<f64>,
) -> EgorRunData {
    let data = state.data.clone().unwrap();
    let version = env!("CARGO_PKG_VERSION");
    let name = env!("CARGO_PKG_NAME");

    let (xdata, ydata, _) = data;
    let mut sampled_locations = vec![];
    Zip::indexed(xdata.rows())
        .and(ydata.rows())
        .for_each(|i, x, y| {
            sampled_locations.push(OrdSample {
                iterations: i as u64 + 1,
                locations: x.to_vec(),
                evaluations: y[0],
            })
        });

    EgorRunData {
        problem_metadata: ProblemMetadata {
            dimensionality: xlimits.nrows(),
            lower_bounds: xlimits.column(0).to_vec(),
            upper_bounds: xlimits.column(1).to_vec(),
            ..Default::default()
        },
        algorithm_parameters: AlgorithmParameters {
            acquisition_function: config.infill_criterion.name().to_string(),
            kernel: match config.gp.correlation_spec {
                CorrelationSpec::ABSOLUTEEXPONENTIAL => "Absolute Exponential".to_string(),
                CorrelationSpec::SQUAREDEXPONENTIAL => "Squared Exponential".to_string(),
                CorrelationSpec::MATERN32 => "Matern 3/2".to_string(),
                CorrelationSpec::MATERN52 => "Matern 5/2".to_string(),
                _ => "Mixed".to_string(),
            },
            initial_samples: sampled_locations.len(),
            bo_iterations: config.max_iters as u64,
            total_samples: xdata.nrows(),
            batch_size: config.qei_config.batch,
            seed: config.seed.map_or(-1, |v| v as i32),
            ..Default::default()
        },
        initial_samples: InitialSamples { sampled_locations },
        extra_info: ExtraInfo {
            team_notes: format!("Native configuration info: {:?}", config),
            code_reference: format!("{name} {version}"),
            ..ExtraInfo::default()
        },
        ..EgorRunData::default()
    }
}

pub(crate) fn update_run_info(
    run_data: &mut EgorRunData,
    n_iter: u64,
    xdata: &Array2<f64>,
    ydata: &Array2<f64>,
) {
    let mut sampled_locations: Vec<Sample> = vec![];
    Zip::from(xdata.rows()).and(ydata.rows()).for_each(|x, y| {
        sampled_locations.push(Sample {
            locations: x.to_vec(),
            evaluations: y[0],
        })
    });

    run_data.search_iterations.push(SearchIteration {
        iterations: run_data.search_iterations.len() as u64 + 1,
        batch_size: xdata.nrows(),
        sampled_locations,
    });

    run_data.algorithm_parameters.bo_iterations = n_iter;
    run_data.algorithm_parameters.total_samples += xdata.nrows();
}

pub(crate) fn save_run<P: AsRef<Path>>(path: P, run_data: &EgorRunData) -> Result<()> {
    let out_json = serde_json::to_string_pretty(run_data)?;
    std::fs::write(path, out_json)?;

    Ok(())
}

#[allow(dead_code)]
pub(crate) fn load_run<P: AsRef<Path>>(path: P) -> Result<EgorRunData> {
    // Open the file in read-only mode with buffer.
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    // Read the JSON contents of the file as an instance of `User`.
    let run_data = serde_json::from_reader(reader)?;

    // Return the `User`.
    Ok(run_data)
}
