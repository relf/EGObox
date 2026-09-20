use argmin::core::{TerminationReason, TerminationStatus};
use egobox_ego::EgorBuilder;
use ndarray::{Array2, ArrayView2, array};

fn xsinx(x: &ArrayView2<f64>) -> Array2<f64> {
    (x - 3.5) * ((x - 3.5) / std::f64::consts::PI).mapv(f64::sin)
}

#[test]
fn target_already_reached_stops_before_first_iteration() {
    let result = EgorBuilder::optimize(xsinx)
        .configure(|cfg| {
            cfg.doe(&array![[0.], [7.], [25.]])
                .seed(42)
                .max_iters(1)
                .target(100.)
        })
        .min_within(&array![[0., 25.]])
        .unwrap()
        .run()
        .unwrap();
    assert!(result.y_opt[0] < 100.);
    assert_eq!(result.state.iter, 0);
}

#[test]
fn iteration_limit_is_reported_in_returned_state() {
    let result = EgorBuilder::optimize(xsinx)
        .configure(|cfg| cfg.seed(42).max_iters(0))
        .min_within(&array![[0., 25.]])
        .unwrap()
        .run()
        .unwrap();
    assert_eq!(
        result.state.termination_status,
        TerminationStatus::Terminated(TerminationReason::MaxItersReached)
    );
}

#[test]
fn returned_state_records_elapsed_time() {
    let result = EgorBuilder::optimize(xsinx)
        .configure(|cfg| cfg.seed(42).max_iters(1))
        .min_within(&array![[0., 25.]])
        .unwrap()
        .run()
        .unwrap();
    assert!(result.state.time.unwrap() > std::time::Duration::ZERO);
}

#[test]
fn compare_seeded_target_workload() {
    let start = std::time::Instant::now();
    let result = EgorBuilder::optimize(xsinx)
        .configure(|cfg| {
            cfg.doe(&array![[0.], [7.], [25.]])
                .seed(42)
                .max_iters(30)
                .infill_strategy(egobox_ego::InfillStrategy::EI)
                .target(-15.1)
        })
        .min_within(&array![[0., 25.]])
        .unwrap()
        .run()
        .unwrap();
    println!(
        "seeded xsinx: iterations={}, best={}, observations={}, elapsed={:?}",
        result.state.iter,
        result.y_opt[0],
        result.x_doe.nrows(),
        start.elapsed()
    );
    assert!(result.y_opt[0] <= -15.1);
}

#[test]
fn negative_infinity_target_is_disabled() {
    let result = EgorBuilder::optimize(xsinx)
        .configure(|cfg| cfg.seed(42).max_iters(2).target(f64::NEG_INFINITY))
        .min_within(&array![[0., 25.]])
        .unwrap()
        .run()
        .unwrap();
    assert_eq!(result.state.iter, 2);
    assert_eq!(
        result.state.termination_status,
        TerminationStatus::Terminated(TerminationReason::MaxItersReached)
    );
}

#[test]
fn timeout_is_reported() {
    let result = EgorBuilder::optimize(xsinx)
        .configure(|cfg| cfg.seed(42).max_iters(10).timeout(0.0))
        .min_within(&array![[0., 25.]])
        .unwrap()
        .run()
        .unwrap();
    assert_eq!(result.state.iter, 1);
    assert_eq!(
        result.state.termination_status,
        TerminationStatus::Terminated(TerminationReason::Timeout)
    );
}

#[cfg(feature = "basin")]
#[test]
fn checkpoint_continuation_matches_uninterrupted_run() {
    use egobox_ego::HotStartMode;
    let dir = std::env::temp_dir().join(format!("egobox-basin-resume-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let checkpoint = dir.join(egobox_ego::CHECKPOINT_FILE);
    let _ = std::fs::remove_file(&checkpoint);
    let run = |iters, mode, save| {
        EgorBuilder::optimize(xsinx)
            .configure(|cfg| {
                let cfg = cfg.seed(42).max_iters(iters).hot_start(mode);
                if save {
                    cfg.outdir(dir.to_str().unwrap())
                } else {
                    cfg
                }
            })
            .min_within(&array![[0., 25.]])
            .unwrap()
            .run()
            .unwrap()
    };
    let expected = run(4, HotStartMode::Disabled, false);
    run(1, HotStartMode::Enabled, true);
    let resumed = run(99, HotStartMode::ExtendedIters(3), true);
    assert_eq!(resumed.state.iter, 4);
    assert_eq!(resumed.x_doe, expected.x_doe);
    assert_eq!(resumed.y_doe, expected.y_doe);
    assert_eq!(resumed.state.last_best_iter, expected.state.last_best_iter);
    assert_eq!(resumed.state.counts, expected.state.counts);
    let again = run(99, HotStartMode::Enabled, true);
    assert_eq!(again.x_doe, resumed.x_doe);
    assert_eq!(again.state.iter, 4);
    std::fs::remove_dir_all(dir).unwrap();
}

#[cfg(feature = "basin")]
#[test]
fn checkpoint_restores_target_before_any_iteration() {
    use egobox_ego::HotStartMode;
    let dir = std::env::temp_dir().join(format!("egobox-basin-target-{}", std::process::id()));
    let checkpoint = dir.join(egobox_ego::CHECKPOINT_FILE);
    let _ = std::fs::remove_file(checkpoint);
    for target in [100., -100.] {
        let result = EgorBuilder::optimize(xsinx)
            .configure(|cfg| {
                cfg.seed(42)
                    .max_iters(5)
                    .target(target)
                    .hot_start(HotStartMode::Enabled)
                    .outdir(dir.to_str().unwrap())
            })
            .min_within(&array![[0., 25.]])
            .unwrap()
            .run()
            .unwrap();
        assert_eq!(result.state.iter, 0);
        assert_eq!(
            result.state.termination_status,
            TerminationStatus::Terminated(TerminationReason::TargetCostReached)
        );
    }
    std::fs::remove_dir_all(dir).unwrap();
}

#[cfg(feature = "persistent")]
#[test]
fn recorder_checkpoint_round_trip_preserves_history() {
    use egobox_ego::HotStartMode;
    let dir = std::env::temp_dir().join(format!("egobox-recorder-resume-{}", std::process::id()));
    let checkpoint = dir.join(egobox_ego::CHECKPOINT_FILE);
    let _ = std::fs::remove_file(&checkpoint);
    let run = |mode| {
        EgorBuilder::optimize(xsinx)
            .configure(|cfg| {
                cfg.seed(42)
                    .max_iters(1)
                    .configure_runtime_flags(|flags| flags.use_run_recorder(true))
                    .hot_start(mode)
                    .outdir(dir.to_str().unwrap())
            })
            .min_within(&array![[0., 25.]])
            .unwrap()
            .run()
            .unwrap()
    };

    let initial = run(HotStartMode::Enabled);
    let recorded = serde_json::to_value(initial.state.run_data.as_ref().unwrap()).unwrap();
    assert!(checkpoint.exists());
    assert_eq!(recorded["search_iterations"].as_array().unwrap().len(), 1);
    assert_eq!(
        recorded["algorithm_parameters"]["other_params"],
        serde_json::json!({})
    );

    let restored = run(HotStartMode::Enabled);
    assert_eq!(restored.state.iter, 1);
    assert_eq!(
        serde_json::to_value(restored.state.run_data.as_ref().unwrap()).unwrap(),
        recorded
    );

    let resumed = run(HotStartMode::ExtendedIters(2));
    assert_eq!(resumed.state.iter, 3);
    let run_data = resumed.state.run_data.as_ref().unwrap();
    assert_eq!(run_data.algorithm_parameters.bo_iterations, 3);
    assert_eq!(
        run_data.algorithm_parameters.total_samples,
        resumed.x_doe.nrows()
    );
    assert_eq!(run_data.search_iterations.len(), 3);
    let continued = serde_json::to_value(run_data).unwrap();
    assert_eq!(continued["initial_samples"], recorded["initial_samples"]);
    assert_eq!(
        continued["search_iterations"][0],
        recorded["search_iterations"][0]
    );
    std::fs::remove_dir_all(dir).unwrap();
}

#[cfg(feature = "basin")]
#[test]
fn corrupt_checkpoint_returns_an_error() {
    use egobox_ego::HotStartMode;
    let dir = std::env::temp_dir().join(format!("egobox-basin-corrupt-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let checkpoint = dir.join(egobox_ego::CHECKPOINT_FILE);
    std::fs::write(&checkpoint, b"invalid checkpoint").unwrap();
    let result = EgorBuilder::optimize(xsinx)
        .configure(|cfg| {
            cfg.seed(42)
                .max_iters(0)
                .hot_start(HotStartMode::Enabled)
                .outdir(dir.to_str().unwrap())
        })
        .min_within(&array![[0., 25.]])
        .unwrap()
        .run();
    assert!(result.is_err());
    assert_eq!(std::fs::read(checkpoint).unwrap(), b"invalid checkpoint");
    std::fs::remove_dir_all(dir).unwrap();
}

#[cfg(all(feature = "basin", unix))]
#[test]
fn interruption_works_for_repeated_runs() {
    if std::env::var_os("EGOBOX_INTERRUPT_TEST_CHILD").is_some() {
        for _ in 0..2 {
            let result = EgorBuilder::optimize(|x: &ArrayView2<f64>| {
                assert!(
                    std::process::Command::new("kill")
                        .args(["-INT", &std::process::id().to_string()])
                        .status()
                        .unwrap()
                        .success()
                );
                // The signal handler runs on ctrlc's background thread.
                std::thread::sleep(std::time::Duration::from_millis(20));
                xsinx(x)
            })
            .configure(|cfg| cfg.seed(42).max_iters(3))
            .min_within(&array![[0., 25.]])
            .unwrap()
            .run()
            .unwrap();
            assert_eq!(
                result.state.termination_status,
                TerminationStatus::Terminated(TerminationReason::Interrupt)
            );
            assert_eq!(result.state.iter, 0);
        }
    } else {
        // Isolate process-wide signal handlers from the other integration tests.
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "interruption_works_for_repeated_runs",
                "--nocapture",
            ])
            .env("EGOBOX_INTERRUPT_TEST_CHILD", "1")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stdout)
        );
    }
}
