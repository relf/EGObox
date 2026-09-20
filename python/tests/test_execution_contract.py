"""Execution behavior shared by the default and Basin builds."""

import time

import numpy as np
import pytest

import egobox as egx


def xsinx(x):
    # RunStatus currently rounds elapsed time to milliseconds.
    time.sleep(0.002)
    return (x - 3.5) * np.sin((x - 3.5) / np.pi)


@pytest.mark.parametrize(
    "algorithm", [egx.InfillOptimizer.COBYLA, egx.InfillOptimizer.SLSQP]
)
def test_iteration_status_and_elapsed_time(algorithm):
    result = egx.Egor([[0.0, 25.0]], infill_optimizer=algorithm).minimize(
        xsinx, max_iters=1, seed=42
    )
    assert result.status.exit == egx.ExitStatus.MAX_ITERS_REACHED
    assert result.status.total_iters == 1
    assert result.status.elapsed_time > 0


def test_zero_iteration_budget():
    result = egx.Egor([[0.0, 25.0]]).minimize(xsinx, max_iters=0, seed=42)
    assert result.status.exit == egx.ExitStatus.MAX_ITERS_REACHED
    assert result.status.total_iters == 0


def test_initial_doe_reaches_target():
    result = egx.Egor(
        [[0.0, 25.0]], doe=np.array([[0.0], [7.0], [25.0]]), target=100.0
    ).minimize(xsinx, max_iters=5, seed=42)
    assert result.status.exit == egx.ExitStatus.TARGET_COST_REACHED
    assert result.status.total_iters == 0
    assert result.result.x_doe.shape[0] == 3


def test_timeout_status():
    result = egx.Egor([[0.0, 25.0]]).minimize(xsinx, max_iters=10, timeout=0.0, seed=42)
    assert result.status.exit == egx.ExitStatus.TIMEOUT
    assert result.status.elapsed_time > 0
