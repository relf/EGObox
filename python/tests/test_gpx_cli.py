import egobox as egx


def test_gpx_help_exit_code_zero(capfd):
    code = egx._rust._run_gpx_cli(["--help"])

    assert code == 0
    out, err = capfd.readouterr()
    help_text = f"{out}\n{err}"
    assert "Fit GP surrogates from tabular data" in help_text
    assert "predict" in help_text.lower()
