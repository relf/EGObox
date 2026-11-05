"""Ctrl-C (SIGINT) handling: graceful optimization interruption, Python behavior elsewhere.

Each case runs in a subprocess to isolate process-wide signal handling.
"""

import subprocess
import sys
import textwrap

PRELUDE = """
import signal
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
