use egobox_ego::{EgorBuilder, TerminationReason, TerminationStatus};
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

/// Run the test `name` in a child process to isolate process-wide signal handling from the
/// other integration tests: the output of the child, or `None` within the child which runs
/// the test body
#[cfg(unix)]
fn in_child(name: &str) -> Option<std::process::Output> {
    if std::env::var_os("EGOBOX_INTERRUPT_TEST_CHILD").is_some() {
        return None;
    }
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", name, "--nocapture"])
        .env("EGOBOX_INTERRUPT_TEST_CHILD", "1")
        .output()
        .unwrap();
    Some(output)
}

/// Send `sig` to the calling thread: its handler runs before returning
#[cfg(unix)]
fn raise(sig: libc::c_int) {
    assert_eq!(unsafe { libc::raise(sig) }, 0);
}

/// Run xsinx optimization sending `sig` (`n` times) at the given objective call
#[cfg(unix)]
fn run_signaled(sig: libc::c_int, n: usize, call: usize) -> egobox_ego::OptimResult<f64> {
    let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    EgorBuilder::optimize(move |x: &ArrayView2<f64>| {
        if calls.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1 == call {
            (0..n).for_each(|_| raise(sig));
        }
        xsinx(x)
    })
    .configure(|cfg| cfg.seed(42).max_iters(3))
    .min_within(&array![[0., 25.]])
    .unwrap()
    .run()
    .unwrap()
}

#[cfg(unix)]
fn assert_success(output: &std::process::Output) {
    assert!(
        output.status.success(),
        "{:?}\n{}\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[cfg(unix)]
#[test]
fn interruption_works_for_repeated_runs() {
    match in_child("interruption_works_for_repeated_runs") {
        Some(output) => assert_success(&output),
        None => {
            for _ in 0..2 {
                // interrupted while evaluating the initial DOE: stops after initialization
                let result = run_signaled(libc::SIGINT, 1, 1);
                assert_eq!(
                    result.state.termination_status,
                    TerminationStatus::Terminated(TerminationReason::Interrupt)
                );
                assert_eq!(result.state.iter, 0);
            }
        }
    }
}

#[cfg(unix)]
#[test]
fn termination_signals_stop_run_gracefully() {
    match in_child("termination_signals_stop_run_gracefully") {
        Some(output) => assert_success(&output),
        None => {
            for sig in [libc::SIGTERM, libc::SIGHUP, libc::SIGUSR1, libc::SIGUSR2] {
                let result = run_signaled(sig, 1, 2);
                assert_eq!(
                    result.state.termination_status,
                    TerminationStatus::Terminated(TerminationReason::Interrupt),
                    "signal {sig}"
                );
                assert_eq!(result.state.iter, 1);
            }
        }
    }
}

#[cfg(unix)]
#[test]
fn signal_handlers_are_restored_after_run() {
    use std::os::unix::process::ExitStatusExt;
    match in_child("signal_handlers_are_restored_after_run") {
        Some(output) => {
            // Ctrl-C after the run terminates the process as usual
            assert_eq!(output.status.signal(), Some(libc::SIGINT), "{output:?}");
            assert!(String::from_utf8_lossy(&output.stdout).contains("RUN DONE"));
            assert!(!String::from_utf8_lossy(&output.stdout).contains("ALIVE"));
        }
        None => {
            run_signaled(libc::SIGINT, 0, 0);
            println!("RUN DONE");
            raise(libc::SIGINT);
            println!("ALIVE");
        }
    }
}

#[cfg(unix)]
#[test]
fn second_signal_terminates() {
    use std::os::unix::process::ExitStatusExt;
    match in_child("second_signal_terminates") {
        Some(output) => {
            assert_eq!(output.status.signal(), Some(libc::SIGINT), "{output:?}");
            assert!(!String::from_utf8_lossy(&output.stdout).contains("RETURNED"));
        }
        None => {
            run_signaled(libc::SIGINT, 2, 2);
            println!("RETURNED");
        }
    }
}

#[cfg(unix)]
static HOST_SIGNALS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

#[cfg(unix)]
extern "C" fn host_handler(_: libc::c_int) {
    HOST_SIGNALS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

#[cfg(unix)]
#[test]
fn host_signal_handler_is_kept() {
    match in_child("host_signal_handler_is_kept") {
        Some(output) => assert_success(&output),
        None => {
            let handler = host_handler as extern "C" fn(libc::c_int) as libc::sighandler_t;
            unsafe { libc::signal(libc::SIGINT, handler) };
            let result = run_signaled(libc::SIGINT, 1, 2);
            // handled by the host only: the optimization is not interrupted
            assert_eq!(
                result.state.termination_status,
                TerminationStatus::Terminated(TerminationReason::MaxItersReached)
            );
            assert_eq!(HOST_SIGNALS.load(std::sync::atomic::Ordering::Relaxed), 1);
            raise(libc::SIGINT);
            assert_eq!(HOST_SIGNALS.load(std::sync::atomic::Ordering::Relaxed), 2);
        }
    }
}

#[test]
fn interrupt_check_stops_repeated_runs() {
    for _ in 0..2 {
        let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let counter = calls.clone();
        let (objective, flag) = interrupted_in_objective(3);
        let result = EgorBuilder::optimize(objective)
            .configure(|cfg| cfg.seed(42).max_iters(10))
            .min_within(&array![[0., 25.]])
            .unwrap()
            .interrupt_check(move || {
                counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                flag.load(std::sync::atomic::Ordering::Relaxed)
            })
            .run()
            .unwrap();
        assert_eq!(
            result.state.termination_status,
            TerminationStatus::Terminated(TerminationReason::Interrupt)
        );
        assert_eq!(result.state.iter, 2);
        // checked at iteration boundaries and during long computations
        assert!(calls.load(std::sync::atomic::Ordering::Relaxed) >= 3);
    }
}

/// xsinx objective setting the returned flag once the `call`-th evaluation is done
/// (call 1 is the initial DOE)
fn interrupted_in_objective(
    call: usize,
) -> (
    impl Fn(&ArrayView2<f64>) -> Array2<f64> + Clone,
    std::sync::Arc<std::sync::atomic::AtomicBool>,
) {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    let calls = std::sync::Arc::new(AtomicUsize::new(0));
    let flag = std::sync::Arc::new(AtomicBool::new(false));
    let set = flag.clone();
    let objective = move |x: &ArrayView2<f64>| {
        let y = xsinx(x);
        if calls.fetch_add(1, Ordering::Relaxed) + 1 == call {
            set.store(true, Ordering::Relaxed);
        }
        y
    };
    (objective, flag)
}

/// Uninterrupted optimization of xsinx with the given number of iterations
fn uninterrupted(trego: bool, max_iters: usize) -> egobox_ego::OptimResult<f64> {
    EgorBuilder::optimize(xsinx)
        .configure(|cfg| cfg.seed(42).trego(trego).max_iters(max_iters))
        .min_within(&array![[0., 25.]])
        .unwrap()
        .run()
        .unwrap()
}

#[test]
fn interruption_after_evaluation_defers_models_update() {
    use egobox_ego::HotStartMode;
    for trego in [false, true] {
        let dir = std::env::temp_dir().join(format!(
            "egobox-deferred-update-{}-{trego}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        let run = |obj, flag: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>| {
            let egor = EgorBuilder::optimize(obj)
                .configure(|cfg| {
                    cfg.seed(42)
                        .trego(trego)
                        .max_iters(4)
                        .hot_start(HotStartMode::Enabled)
                        .outdir(dir.to_str().unwrap())
                })
                .min_within(&array![[0., 25.]])
                .unwrap();
            match flag {
                Some(flag) => {
                    egor.interrupt_check(move || flag.load(std::sync::atomic::Ordering::Relaxed))
                }
                None => egor,
            }
            .run()
            .unwrap()
        };
        // interrupted while the models are updated with the point of iteration 2
        let (objective, flag) = interrupted_in_objective(3);
        let interrupted = run(objective, Some(flag));
        assert_eq!(
            interrupted.state.termination_status,
            TerminationStatus::Terminated(TerminationReason::Interrupt)
        );
        assert_eq!(interrupted.state.iter, 2, "trego={trego}");
        assert!(interrupted.state.surrogate.pending_update.is_some());
        let expected = uninterrupted(trego, 2);
        assert_eq!(interrupted.x_doe, expected.x_doe);
        assert_eq!(interrupted.y_doe, expected.y_doe);

        // the evaluated point is kept: the hot start goes on as if not interrupted
        let (objective, _) = interrupted_in_objective(0);
        let resumed = run(objective, None);
        let expected = uninterrupted(trego, 4);
        assert_eq!(resumed.state.iter, 4);
        assert!(resumed.state.surrogate.pending_update.is_none());
        assert_eq!(resumed.x_doe, expected.x_doe, "trego={trego}");
        assert_eq!(resumed.y_doe, expected.y_doe);
        assert_eq!(resumed.state.last_best_iter, expected.state.last_best_iter);
        assert_eq!(resumed.state.counts, expected.state.counts);
        std::fs::remove_dir_all(dir).unwrap();
    }
}

// Interruption in the infill optimization of iteration 2: function constraints are called
// once on the point evaluated at iteration 1, then by the infill optimizer
static INFILL_OBJ_CALLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static INFILL_CSTR_CALLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static INFILL_ARMED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static INFILL_FLAG: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

fn infill_objective(x: &ArrayView2<f64>) -> Array2<f64> {
    INFILL_OBJ_CALLS.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    xsinx(x)
}

fn infill_cstr(x: &[f64], g: Option<&mut [f64]>, _u: &mut egobox_ego::InfillObjData<f64>) -> f64 {
    use std::sync::atomic::Ordering;
    if INFILL_OBJ_CALLS.load(Ordering::SeqCst) == 2
        && INFILL_CSTR_CALLS.fetch_add(1, Ordering::SeqCst) == 1
        && INFILL_ARMED.load(Ordering::SeqCst)
    {
        INFILL_FLAG.store(true, Ordering::SeqCst);
        #[cfg(unix)]
        if std::env::var_os("EGOBOX_INTERRUPT_TEST_CHILD").is_some() {
            raise(libc::SIGINT);
        }
    }
    if let Some(g) = g {
        g[0] = 1.;
    }
    x[0] - 30. // always satisfied
}

fn run_infill_interrupted(
    armed: bool,
    max_iters: usize,
    hot_start: Option<&std::path::Path>,
    with_check: bool,
) -> egobox_ego::OptimResult<f64> {
    use std::sync::atomic::Ordering;
    INFILL_OBJ_CALLS.store(0, Ordering::SeqCst);
    INFILL_CSTR_CALLS.store(0, Ordering::SeqCst);
    INFILL_ARMED.store(armed, Ordering::SeqCst);
    INFILL_FLAG.store(false, Ordering::SeqCst);
    let cstr: egobox_ego::Cstr = infill_cstr;
    let egor = EgorBuilder::optimize(infill_objective)
        .subject_to(vec![cstr])
        .configure(|cfg| {
            let cfg = cfg.seed(42).max_iters(max_iters);
            match hot_start {
                Some(dir) => cfg
                    .hot_start(egobox_ego::HotStartMode::Enabled)
                    .outdir(dir.to_str().unwrap()),
                None => cfg,
            }
        })
        .min_within(&array![[0., 25.]])
        .unwrap();
    if with_check {
        egor.interrupt_check(|| INFILL_FLAG.load(Ordering::SeqCst))
    } else {
        egor
    }
    .run()
    .unwrap()
}

#[test]
#[serial_test::serial(infill_interrupted)]
fn interruption_before_evaluation_drops_iteration() {
    let dir = std::env::temp_dir().join(format!("egobox-dropped-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let interrupted = run_infill_interrupted(true, 4, Some(&dir), true);
    assert_eq!(
        interrupted.state.termination_status,
        TerminationStatus::Terminated(TerminationReason::Interrupt)
    );
    assert_eq!(interrupted.state.iter, 1);
    assert!(interrupted.state.surrogate.pending_update.is_none());
    let expected = run_infill_interrupted(false, 1, None, false);
    assert_eq!(interrupted.x_doe, expected.x_doe);

    // nothing evaluated is lost: the hot start goes on as if not interrupted
    let resumed = run_infill_interrupted(false, 4, Some(&dir), false);
    let expected = run_infill_interrupted(false, 4, None, false);
    assert_eq!(resumed.state.iter, 4);
    assert_eq!(resumed.x_doe, expected.x_doe);
    assert_eq!(resumed.y_doe, expected.y_doe);
    assert_eq!(resumed.state.last_best_iter, expected.state.last_best_iter);
    assert_eq!(resumed.state.counts, expected.state.counts);
    std::fs::remove_dir_all(dir).unwrap();
}

#[cfg(unix)]
#[test]
fn signal_during_infill_optimization_drops_iteration() {
    match in_child("signal_during_infill_optimization_drops_iteration") {
        Some(output) => assert_success(&output),
        None => {
            // SIGINT raised by the function constraint, handled by the default handlers
            let result = run_infill_interrupted(true, 4, None, false);
            assert_eq!(
                result.state.termination_status,
                TerminationStatus::Terminated(TerminationReason::Interrupt)
            );
            assert_eq!(result.state.iter, 1);
        }
    }
}

#[test]
fn current_param_and_cost_are_an_evaluated_point() {
    let result = EgorBuilder::optimize(xsinx)
        .configure(|cfg| cfg.doe(&array![[0.], [7.], [25.]]).seed(42).max_iters(3))
        .min_within(&array![[0., 25.]])
        .unwrap()
        .run()
        .unwrap();
    let param = result.state.param.as_ref().expect("current param");
    let cost = result.state.cost.as_ref().expect("current cost");
    let index = result
        .x_doe
        .rows()
        .into_iter()
        .position(|x| x == param)
        .expect("current param is an evaluated point");
    // evaluated value, not the surrogate prediction
    assert_eq!(result.y_doe.row(index), cost);
}

/// xsinx with a constraint feasible only in [20, 21], away from the initial DOE
fn xsinx_cstr(x: &ArrayView2<f64>) -> Array2<f64> {
    let c = (x - 20.) * (x - 21.);
    ndarray::concatenate![ndarray::Axis(1), xsinx(x), c]
}

#[test]
fn checkpoint_continuation_uses_run_feasibility_first() {
    use egobox_ego::HotStartMode;
    let dir = std::env::temp_dir().join(format!("egobox-basin-ff-{}", std::process::id()));
    let run = |iters, feasibility_first, mode, save| {
        EgorBuilder::optimize(xsinx_cstr)
            .configure(|cfg| {
                let cfg = cfg
                    .n_cstr(1)
                    .doe(&array![[0.], [7.], [12.]])
                    .cstr_infill(true)
                    .feasibility_first(feasibility_first)
                    .seed(42)
                    .max_iters(iters)
                    .hot_start(mode);
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
    let expected = run(4, true, HotStartMode::Disabled, false);

    // feasibility_first is not saved in the checkpoint: the resumed run uses its own setting
    let _ = std::fs::remove_dir_all(&dir);
    run(1, true, HotStartMode::Enabled, true);
    let resumed = run(99, true, HotStartMode::ExtendedIters(3), true);
    assert_eq!(resumed.state.iter, 4);
    assert_eq!(resumed.x_doe, expected.x_doe);
    assert_eq!(resumed.y_doe, expected.y_doe);

    // the setting matters after resumption
    let _ = std::fs::remove_dir_all(&dir);
    run(1, true, HotStartMode::Enabled, true);
    let switched = run(99, false, HotStartMode::ExtendedIters(3), true);
    assert_eq!(switched.state.iter, 4);
    assert_ne!(switched.x_doe, expected.x_doe);
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn feasible_initial_doe_disables_feasibility_first_search() {
    // the initial DOE has a feasible point (x=20.5): the feasibility-first search,
    // used only while no feasible point is known, never applies
    let run = |feasibility_first| {
        EgorBuilder::optimize(xsinx_cstr)
            .configure(|cfg| {
                cfg.n_cstr(1)
                    .doe(&array![[0.], [7.], [20.5]])
                    .cstr_infill(true)
                    .feasibility_first(feasibility_first)
                    .seed(42)
                    .max_iters(2)
            })
            .min_within(&array![[0., 25.]])
            .unwrap()
            .run()
            .unwrap()
    };
    assert_eq!(run(true).x_doe, run(false).x_doe);
}

/// xsinx objective returning `EgoError::Interrupted` at the given call (1 is the initial DOE)
#[derive(Clone)]
struct InterruptedAt {
    call: usize,
    calls: std::sync::Arc<std::sync::atomic::AtomicUsize>,
}

impl InterruptedAt {
    fn new(call: usize) -> Self {
        Self {
            call,
            calls: Default::default(),
        }
    }
}

impl egobox_ego::ObjFn for InterruptedAt {
    fn eval(&self, x: &ArrayView2<f64>) -> egobox_ego::Result<Array2<f64>> {
        let call = self
            .calls
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            + 1;
        if call == self.call {
            Err(egobox_ego::EgoError::Interrupted)
        } else {
            Ok(xsinx(x))
        }
    }
}

#[test]
fn interrupted_evaluation_drops_unfinished_iteration() {
    let expected = EgorBuilder::optimize(xsinx)
        .configure(|cfg| cfg.seed(42).max_iters(2))
        .min_within(&array![[0., 25.]])
        .unwrap()
        .run()
        .unwrap();
    // initial DOE, iterations 1 and 2, then interrupted in iteration 3
    let result = EgorBuilder::optimize(InterruptedAt::new(4))
        .configure(|cfg| cfg.seed(42).max_iters(10))
        .min_within(&array![[0., 25.]])
        .unwrap()
        .run()
        .unwrap();
    assert_eq!(
        result.state.termination_status,
        TerminationStatus::Terminated(TerminationReason::Interrupt)
    );
    assert_eq!(result.state.iter, 2);
    assert_eq!(
        result.state.surrogate.x_fail,
        expected.state.surrogate.x_fail
    );
    assert_eq!(result.x_doe, expected.x_doe);
    assert_eq!(result.y_doe, expected.y_doe);
}

#[test]
fn interrupted_initial_doe_is_an_error() {
    let result = EgorBuilder::optimize(InterruptedAt::new(1))
        .configure(|cfg| cfg.seed(42).max_iters(10))
        .min_within(&array![[0., 25.]])
        .unwrap()
        .run();
    assert!(matches!(result, Err(egobox_ego::EgoError::Interrupted)));
}

#[test]
fn hot_start_does_interrupted_iteration_again() {
    use egobox_ego::HotStartMode;
    let expected = EgorBuilder::optimize(xsinx)
        .configure(|cfg| cfg.seed(42).max_iters(4))
        .min_within(&array![[0., 25.]])
        .unwrap()
        .run()
        .unwrap();
    let run = |obj: InterruptedAt, dir: &std::path::Path| {
        EgorBuilder::optimize(obj)
            .configure(|cfg| {
                cfg.seed(42)
                    .max_iters(4)
                    .hot_start(HotStartMode::Enabled)
                    .outdir(dir.to_str().unwrap())
            })
            .min_within(&array![[0., 25.]])
            .unwrap()
            .run()
            .unwrap()
    };
    // interrupted in iteration 1 (no checkpoint yet) and in iteration 2
    for call in [2, 3] {
        let dir =
            std::env::temp_dir().join(format!("egobox-interrupted-{}-{call}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let interrupted = run(InterruptedAt::new(call), &dir);
        assert_eq!(interrupted.state.iter, call as u64 - 2);
        // no failed point saved (no file before the first completed iteration)
        let failed = dir.join(egobox_ego::FAILED_POINTS_FILE);
        if failed.exists() {
            let x_fail: Array2<f64> = ndarray_npy::read_npy(failed).unwrap();
            assert_eq!(x_fail.nrows(), 0);
        }
        // never interrupted again
        let resumed = run(InterruptedAt::new(0), &dir);
        assert_eq!(resumed.state.iter, 4);
        assert_eq!(
            resumed.state.surrogate.x_fail,
            expected.state.surrogate.x_fail
        );
        assert_eq!(resumed.x_doe, expected.x_doe);
        assert_eq!(resumed.y_doe, expected.y_doe);
        assert_eq!(resumed.state.last_best_iter, expected.state.last_best_iter);
        // the interrupted evaluation attempt is counted
        assert_eq!(
            resumed.state.counts["cost_count"],
            expected.state.counts["cost_count"] + 1
        );
        std::fs::remove_dir_all(dir).unwrap();
    }
}
