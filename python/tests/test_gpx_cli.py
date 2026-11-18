import signal
import subprocess
import sys
import time

import numpy as np
import pytest

import egobox as egx


def test_gpx_help_exit_code_zero(capfd):
    code = egx._run_gpx_cli(["--help"])

    assert code == 0
    out, err = capfd.readouterr()
    help_text = f"{out}\n{err}"
    assert "Fit GP surrogates from tabular data" in help_text
    assert "predict" in help_text.lower()


def test_gpx_main_restores_python_sigint_handler(monkeypatch, capfd):
    from egobox import gpx_cli

    monkeypatch.setattr("sys.argv", ["gpx", "--help"])
    assert gpx_cli.main() == 0
    capfd.readouterr()
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signal delivery")
def test_gpx_ctrl_c_terminates_fit(tmp_path):
    # Ctrl-C during a long fit terminates gpx at once, no model is written
    rng = np.random.default_rng(1)
    x = rng.random((1200, 4))
    y = np.sum(np.sin(6.0 * x), axis=1, keepdims=True)
    data = tmp_path / "train.csv"
    np.savetxt(
        data, np.hstack([x, y]), delimiter=",", header="x1,x2,x3,x4,y", comments=""
    )
    model = tmp_path / "model.gpx"
    script = (
        "import sys\n"
        "from egobox import gpx_cli\n"
        f"sys.argv = ['gpx', 'fit', {str(data)!r}, '-o', {str(model)!r}]\n"
        "print('READY', flush=True)\n"
        "gpx_cli.main()\n"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", script], stdout=subprocess.PIPE, text=True
    )
    assert proc.stdout.readline().strip() == "READY"
    time.sleep(0.5)  # fitting (several seconds)
    proc.send_signal(signal.SIGINT)
    proc.communicate(timeout=120)
    assert proc.returncode == -signal.SIGINT
    assert not model.exists()
