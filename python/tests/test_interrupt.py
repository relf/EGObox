"""Ctrl-C (SIGINT) and signal handling: graceful optimization interruption, Python behavior
elsewhere.

Each case runs in a subprocess to isolate process-wide signal handling.
"""

import subprocess
import sys
import textwrap

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


def run_script(body):
    script = PRELUDE + textwrap.dedent(body)
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
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
