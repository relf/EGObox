from egobox import egobox as native


def test_gpx_help_exit_code_zero(capfd):
    code = native._run_gpx_cli(["--help"])

    assert code == 0
    out, err = capfd.readouterr()
    help_text = f"{out}\n{err}"
    assert "Fit GP surrogates from tabular data" in help_text
    assert "predict" in help_text.lower()
