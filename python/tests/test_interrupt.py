"""Ctrl-C (SIGINT) and signal handling: graceful optimization interruption, Python behavior
elsewhere.

Each case runs in a subprocess to isolate process-wide signal handling.
"""

import os
import subprocess
import sys
import textwrap

import pytest

PRELUDE = """
import signal
import sys
import threading
import time

import numpy as np

import egobox as egx


def xsinx(x):
    return (x - 3.5) * np.sin((x - 3.5) / np.pi)


def interrupting(fun, at_call):
    calls = 0

    def wrapped(*args):
        nonlocal calls
        calls += 1
        if calls == at_call:
            signal.raise_signal(signal.SIGINT)
        return fun(*args)

    return wrapped
"""


# Ctrl-C acted on at once while Rust code runs (SIGINT hook): on Windows, raise_signal() calls
# Python's handler directly, without going through the console control handlers
PROMPT = pytest.mark.skipif(
    sys.platform == "win32", reason="raise_signal bypasses the console handler"
)

CTRL_C_LATER = """
signal_time = {}


def ctrl_c_later(delay=0.5):
    # Ctrl-C while the main thread runs Rust code
    def send():
        time.sleep(delay)
        signal_time["t"] = time.perf_counter()
        signal.raise_signal(signal.SIGINT)

    threading.Thread(target=send, daemon=True).start()


def since_signal():
    return time.perf_counter() - signal_time["t"]
"""

# Training with many short optimization starts, on few threads as on CI runners: Ctrl-C is
# acted on within a start even with optimizers which cannot be stopped during a start
# (c-cobyla, c-slsqp features), while the whole training lasts several seconds
FEW_THREADS = {**os.environ, "RAYON_NUM_THREADS": "2"}


def run_script(body, env=None):
    script = PRELUDE + CTRL_C_LATER + textwrap.dedent(body)
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
        env=env,
    )
    assert proc.returncode == 0, (
        f"exit code {proc.returncode}\n{proc.stdout}\n{proc.stderr}"
    )
    return proc.stdout


def test_interrupt_in_objective_after_fit():
    out = run_script(
        """
        egx.GpMix().fit(np.array([[0.0], [1.0], [2.0]]), np.array([0.0, 1.0, 4.0]))
        res = egx.Egor([[0.0, 25.0]]).minimize(
            interrupting(xsinx, at_call=2), max_iters=10, seed=42
        )
        print("EXIT", res.status.exit, res.status.total_iters < 10)
        """
    )
    assert "EXIT ExitStatus.INTERRUPT True" in out


def test_interrupted_iteration_done_again_by_hot_start(tmp_path):
    # the iteration interrupted in the objective is dropped, not recorded as a failed point
    out = run_script(
        f"""
        outdir = {str(tmp_path)!r}
        expected = egx.Egor([[0.0, 25.0]]).minimize(xsinx, max_iters=4, seed=42)
        res = egx.Egor([[0.0, 25.0]]).minimize(
            interrupting(xsinx, at_call=3),
            max_iters=4,
            seed=42,
            outdir=outdir,
            hot_start=True,
        )
        print("EXIT", res.status.exit, res.status.total_iters)
        print("FAILED", len(np.load(outdir + "/egor_failed_points.npy")))
        resumed = egx.Egor([[0.0, 25.0]]).minimize(
            xsinx, max_iters=4, seed=42, outdir=outdir, hot_start=True
        )
        print("RESUMED", resumed.status.total_iters)
        print("SAME DOE", np.array_equal(resumed.x_doe, expected.x_doe))
        """
    )
    # initial DOE, iteration 1, then interrupted in iteration 2
    assert "EXIT ExitStatus.INTERRUPT 1" in out
    assert "FAILED 0" in out
    assert "RESUMED 4" in out
    assert "SAME DOE True" in out


def test_repeated_interrupted_runs():
    out = run_script(
        """
        for _ in range(2):
            res = egx.Egor([[0.0, 25.0]]).minimize(
                interrupting(xsinx, at_call=2), max_iters=10, seed=42
            )
            print("EXIT", res.status.exit)
        """
    )
    assert out.count("EXIT ExitStatus.INTERRUPT") == 2


def test_interrupt_in_initial_doe():
    # no result yet: KeyboardInterrupt is raised
    out = run_script(
        """
        for stop_on_error in (False, True):
            try:
                egx.Egor([[0.0, 25.0]]).minimize(
                    interrupting(xsinx, at_call=1),
                    max_iters=10,
                    seed=42,
                    stop_on_error=stop_on_error,
                )
                print("NO INTERRUPT")
            except KeyboardInterrupt:
                print("KEYBOARD INTERRUPT")
        """
    )
    assert out.count("KEYBOARD INTERRUPT") == 2


def test_repeated_interrupt_keeps_result():
    # Ctrl-C again while the optimizer ends the interrupted iteration (Rust code):
    # the result is returned, no KeyboardInterrupt is raised once the optimization is over
    out = run_script(
        """
        def ctrl_c_later():
            time.sleep(0.05)
            signal.raise_signal(signal.SIGINT)

        calls = 0

        def fun(x):
            global calls
            calls += 1
            if calls == 2:
                threading.Thread(target=ctrl_c_later, daemon=True).start()
                signal.raise_signal(signal.SIGINT)
            return np.sum(np.sin(6.0 * x), axis=1, keepdims=True)

        try:
            res = egx.Egor([[0.0, 1.0]] * 4, n_doe=300).minimize(
                fun, max_iters=10, seed=42
            )
            print("EXIT", res.status.exit)
        except KeyboardInterrupt:
            print("KEYBOARD INTERRUPT")
        try:
            time.sleep(1.0)  # if the 2nd Ctrl-C comes once the optimization is over
        except KeyboardInterrupt:
            pass
        """
    )
    assert "EXIT ExitStatus.INTERRUPT" in out


def test_signal_handler_exception_in_objective():
    # e.g. SystemExit raised by a SIGTERM handler: not an evaluation error,
    # the optimization stops after the current iteration then it is raised
    out = run_script(
        """
        signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(3))
        calls = 0

        def fun(x):
            global calls
            calls += 1
            if calls == 2:
                signal.raise_signal(signal.SIGTERM)
            return xsinx(x)

        try:
            egx.Egor([[0.0, 25.0]]).minimize(fun, max_iters=10, seed=42)
            print("NO EXIT")
        except SystemExit as exc:
            print("SYSTEM EXIT", exc.code, calls)
        """
    )
    assert "SYSTEM EXIT 3 2" in out


def test_signal_handler_exception_while_optimizer_runs():
    out = run_script(
        """
        signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(3))

        def send():
            time.sleep(0.3)
            signal.raise_signal(signal.SIGTERM)

        threading.Thread(target=send, daemon=True).start()
        try:
            egx.Egor([[0.0, 25.0]]).minimize(xsinx, max_iters=1000, seed=42)
            print("NO EXIT")
        except SystemExit as exc:
            print("SYSTEM EXIT", exc.code)
        """
    )
    assert "SYSTEM EXIT 3" in out


def test_interrupt_in_function_constraint():
    out = run_script(
        """
        res = egx.Egor([[0.0, 25.0]]).minimize(
            xsinx,
            fcstrs=[(interrupting(lambda x: -1.0, at_call=3), np.zeros_like)],
            max_iters=10,
            seed=42,
        )
        print("EXIT", res.status.exit)
        """
    )
    assert "EXIT ExitStatus.INTERRUPT" in out


def test_interrupt_while_optimizer_runs():
    # the signal arrives while Rust code runs (no Python code to raise into)
    out = run_script(
        """
        def send():
            time.sleep(0.3)
            signal.raise_signal(signal.SIGINT)

        threading.Thread(target=send, daemon=True).start()
        res = egx.Egor([[0.0, 25.0]]).minimize(xsinx, max_iters=1000, seed=42)
        print("EXIT", res.status.exit, res.status.total_iters < 1000)
        """
    )
    assert "EXIT ExitStatus.INTERRUPT True" in out


def test_keyboard_interrupt_after_minimize():
    out = run_script(
        """
        egx.Egor([[0.0, 25.0]]).minimize(xsinx, max_iters=1, seed=42)
        egx.GpMix().fit(np.array([[0.0], [1.0], [2.0]]), np.array([0.0, 1.0, 4.0]))
        try:
            signal.raise_signal(signal.SIGINT)
            time.sleep(0.1)
            print("NO INTERRUPT")
        except KeyboardInterrupt:
            print("KEYBOARD INTERRUPT")
        """
    )
    assert "KEYBOARD INTERRUPT" in out


def test_keyboard_interrupt_after_fit():
    # the signal arrives during the fit (Rust code), raised when the fit returns
    out = run_script(
        """
        x = np.linspace(0.0, 25.0, 800).reshape(-1, 1)

        def send():
            time.sleep(0.3)
            signal.raise_signal(signal.SIGINT)

        threading.Thread(target=send, daemon=True).start()
        try:
            egx.GpMix(seed=42).fit(x, xsinx(x))
            time.sleep(0.1)
            print("NO INTERRUPT")
        except KeyboardInterrupt:
            print("KEYBOARD INTERRUPT")
        """
    )
    assert "KEYBOARD INTERRUPT" in out


@PROMPT
def test_ctrl_c_stops_minimize_training_at_once():
    # Ctrl-C while the surrogates are trained after the initial DOE: the iteration is dropped
    out = run_script(
        """
        calls = 0

        def fun(x):
            global calls
            calls += 1
            if calls == 1:
                ctrl_c_later()
            return np.sum(np.sin(6.0 * x), axis=1, keepdims=True)

        # training of about 4s after the initial DOE (stopped after it without the hook)
        gp_config = egx.GpConfig(theta_n_start=1500)
        res = egx.Egor([[0.0, 1.0]] * 4, n_doe=100, gp_config=gp_config).minimize(
            fun, max_iters=10, seed=42
        )
        print("EXIT", res.status.exit, res.status.total_iters, calls, since_signal() < 1.0)
        """,
        env=FEW_THREADS,
    )
    assert "EXIT ExitStatus.INTERRUPT 0 1 True" in out


@PROMPT
def test_ctrl_c_interrupts_fit_at_once():
    out = run_script(
        """
        x = egx.lhs(np.array([[0.0, 1.0]] * 4), 100, seed=1)
        y = np.sum(np.sin(6.0 * x), axis=1)
        # fits of about 4s
        for model in (
            egx.GpMix(theta_n_start=1500, seed=1),
            egx.SparseGpMix(n_inducing=20, theta_n_start=3000, seed=1),
        ):
            ctrl_c_later()
            try:
                model.fit(x, y)
                print("NO INTERRUPT")
            except KeyboardInterrupt:
                print("KEYBOARD INTERRUPT", since_signal() < 2.0)
        """,
        env=FEW_THREADS,
    )
    assert out.count("KEYBOARD INTERRUPT True") == 2


@PROMPT
def test_ctrl_c_interrupts_suggest_at_once():
    out = run_script(
        """
        # suggestion of about 4s
        x = egx.lhs(np.array([[0.0, 1.0]] * 4), 100, seed=1)
        y = np.sum(np.sin(6.0 * x), axis=1, keepdims=True)
        gp_config = egx.GpConfig(theta_n_start=1500)
        ctrl_c_later()
        try:
            egx.Egor([[0.0, 1.0]] * 4, gp_config=gp_config, seed=42).suggest(x, y)
            print("NO INTERRUPT")
        except KeyboardInterrupt:
            print("KEYBOARD INTERRUPT", since_signal() < 2.0)
        try:
            time.sleep(0.3)
            print("NO LATE INTERRUPT")
        except KeyboardInterrupt:
            print("LATE INTERRUPT")
        """,
        env=FEW_THREADS,
    )
    assert "KEYBOARD INTERRUPT True" in out
    assert "NO LATE INTERRUPT" in out


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX sigaction")
def test_sigint_hook_installed_then_restored():
    out = run_script(
        """
        import ctypes

        libc = ctypes.CDLL(None)

        def sigint_handler():
            # address of the system SIGINT handler (first field of struct sigaction)
            action = ctypes.create_string_buffer(512)
            assert libc.sigaction(signal.SIGINT, None, action) == 0
            return ctypes.c_void_p.from_buffer(action).value

        python_handler = sigint_handler()
        hooked = []

        def fun(x):
            hooked.append(sigint_handler() != python_handler)
            return xsinx(x)

        egx.Egor([[0.0, 25.0]]).minimize(fun, max_iters=2, seed=42)
        egx.GpMix(seed=1).fit(np.array([[0.0], [1.0], [2.0]]), np.array([0.0, 1.0, 4.0]))
        print("HOOKED", all(hooked), "RESTORED", sigint_handler() == python_handler)
        try:
            signal.raise_signal(signal.SIGINT)
            time.sleep(0.1)
            print("NO INTERRUPT")
        except KeyboardInterrupt:
            print("KEYBOARD INTERRUPT")
        """
    )
    assert "HOOKED True RESTORED True" in out
    assert "KEYBOARD INTERRUPT" in out


def test_custom_sigint_handler_is_respected():
    # a custom SIGINT handler which does not raise: the optimization is not interrupted
    out = run_script(
        """
        received = []
        signal.signal(signal.SIGINT, lambda signum, frame: received.append(signum))
        calls = 0

        def fun(x):
            global calls
            calls += 1
            if calls == 1:
                ctrl_c_later()
            return np.sum(np.sin(6.0 * x), axis=1, keepdims=True)

        res = egx.Egor([[0.0, 1.0]] * 4, n_doe=600).minimize(fun, max_iters=1, seed=42)
        print("EXIT", res.status.exit, res.status.total_iters, len(received))
        """
    )
    assert "EXIT ExitStatus.MAX_ITERS_REACHED 1 1" in out


# InterruptFlag: interruption from Python code (no signal, all platforms)


def test_interrupt_flag():
    import threading

    import egobox as egx

    flag = egx.InterruptFlag()
    assert not flag.is_set()
    assert repr(flag) == "InterruptFlag(set=False)"
    thread = threading.Thread(target=flag.set)
    thread.start()
    thread.join()
    assert flag.is_set()
    flag.clear()
    assert not flag.is_set()
    assert issubclass(egx.Interrupted, Exception)


def test_interrupt_flag_stops_minimize_from_another_thread():
    # set while the surrogates are trained after the initial DOE: the iteration is dropped
    out = run_script(
        """
        flag = egx.InterruptFlag()
        calls = 0
        set_time = {}

        def set_later():
            set_time["t"] = time.perf_counter()
            flag.set()

        def fun(x):
            global calls
            calls += 1
            if calls == 1:
                threading.Timer(0.5, set_later).start()
            return np.sum(np.sin(6.0 * x), axis=1, keepdims=True)

        gp_config = egx.GpConfig(theta_n_start=1500)
        res = egx.Egor([[0.0, 1.0]] * 4, n_doe=100, gp_config=gp_config).minimize(
            fun, max_iters=10, seed=42, interrupt=flag
        )
        delay = time.perf_counter() - set_time["t"]
        print("EXIT", res.status.exit, res.status.total_iters, calls, delay < 1.0)
        """,
        env=FEW_THREADS,
    )
    assert "EXIT ExitStatus.INTERRUPT 0 1 True" in out


def test_interrupt_flag_set_within_objective():
    # the flag set when evaluating the point of iteration 2: this point is kept
    import numpy as np

    import egobox as egx

    def xsinx(x):
        return (x - 3.5) * np.sin((x - 3.5) / np.pi)

    flag = egx.InterruptFlag()
    calls = 0

    def fun(x):
        nonlocal calls
        calls += 1
        if calls == 3:  # initial DOE, iterations 1 and 2
            flag.set()
        return xsinx(x)

    res = egx.Egor([[0.0, 25.0]]).minimize(fun, max_iters=10, seed=42, interrupt=flag)
    assert res.status.exit == egx.ExitStatus.INTERRUPT
    assert res.status.total_iters == 2
    expected = egx.Egor([[0.0, 25.0]]).minimize(xsinx, max_iters=2, seed=42)
    np.testing.assert_array_equal(res.x_doe, expected.x_doe)


def test_interrupt_flag_already_set():
    import numpy as np

    import egobox as egx

    flag = egx.InterruptFlag()
    flag.set()
    with pytest.raises(egx.Interrupted):
        egx.GpMix(seed=1).fit(
            np.array([[0.0], [1.0], [2.0]]), np.array([0.0, 1.0, 4.0]), interrupt=flag
        )
    # the initial DOE is evaluated, then the optimization stops
    res = egx.Egor([[0.0, 25.0]]).minimize(
        lambda x: (x - 3.5) * np.sin((x - 3.5) / np.pi),
        max_iters=10,
        seed=42,
        interrupt=flag,
    )
    assert res.status.exit == egx.ExitStatus.INTERRUPT
    assert res.status.total_iters == 0


def test_interrupt_flag_stops_fit_and_suggest_from_another_thread():
    out = run_script(
        """
        x = egx.lhs(np.array([[0.0, 1.0]] * 4), 100, seed=1)
        y = np.sum(np.sin(6.0 * x), axis=1)
        gp_config = egx.GpConfig(theta_n_start=1500)
        egor = egx.Egor([[0.0, 1.0]] * 4, gp_config=gp_config, seed=42)
        # calls of about 4s
        for call in (
            lambda flag: egx.GpMix(theta_n_start=1500, seed=1).fit(x, y, interrupt=flag),
            lambda flag: egx.SparseGpMix(n_inducing=20, theta_n_start=3000, seed=1).fit(
                x, y, interrupt=flag
            ),
            lambda flag: egor.suggest(x, y.reshape(-1, 1), interrupt=flag),
        ):
            flag = egx.InterruptFlag()
            set_time = {}

            def set_later():
                set_time["t"] = time.perf_counter()
                flag.set()

            threading.Timer(0.5, set_later).start()
            try:
                call(flag)
                print("NO INTERRUPT")
            except egx.Interrupted:
                print("INTERRUPTED", time.perf_counter() - set_time["t"] < 2.0)
        """,
        env=FEW_THREADS,
    )
    assert out.count("INTERRUPTED True") == 3


@PROMPT
def test_ctrl_c_with_interrupt_flag_raises_keyboard_interrupt():
    # Ctrl-C keeps raising KeyboardInterrupt when an (unset) interrupt flag is given
    out = run_script(
        """
        x = egx.lhs(np.array([[0.0, 1.0]] * 4), 100, seed=1)
        y = np.sum(np.sin(6.0 * x), axis=1)
        ctrl_c_later()
        try:
            egx.GpMix(theta_n_start=1500, seed=1).fit(x, y, interrupt=egx.InterruptFlag())
            print("NO INTERRUPT")
        except KeyboardInterrupt:
            print("KEYBOARD INTERRUPT", since_signal() < 2.0)
        """,
        env=FEW_THREADS,
    )
    assert "KEYBOARD INTERRUPT True" in out
