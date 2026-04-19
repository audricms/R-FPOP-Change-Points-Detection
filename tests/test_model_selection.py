import numpy as np
import pytest

from src.model_selection import (
    biweight_phi,
    compute_loss_bound_K,
    compute_penalty_beta,
    get_gamma_builder,
    huber_phi,
)


def test_biweight_phi():
    assert biweight_phi(z=1.0, K_std=3.0) == pytest.approx(2.0)
    assert biweight_phi(z=5.0, K_std=3.0) == 0.0


def test_huber_phi():
    assert huber_phi(z=1.0, K_std=1.345) == pytest.approx(2.0)
    assert huber_phi(z=5.0, K_std=1.345) == pytest.approx(2 * 1.345)


def test_compute_penalty_beta_l2_returns_positive():
    rng = np.random.default_rng(42)
    y = rng.standard_normal(50).tolist()
    beta = compute_penalty_beta(y=y, loss="l2")
    assert beta is not None and beta > 0


def test_compute_penalty_beta_invalid_loss_raises():
    with pytest.raises(ValueError):
        compute_penalty_beta(y=[1.0, 2.0, 3.0], loss="unknown")


def test_compute_loss_bound_K_returns_positive():
    rng = np.random.default_rng(0)
    y = rng.standard_normal(50).tolist()
    assert compute_loss_bound_K(y=y, loss="huber") > 0
    assert compute_loss_bound_K(y=y, loss="biweight") > 0


def test_get_gamma_builder_returns_callable():
    y = [1.0, 2.0, 3.0, 2.0, 1.0]
    builder = get_gamma_builder(y=y, loss="l2")
    assert callable(builder)
    pieces = builder(y_t=2.0, t=0)
    assert len(pieces) == 1
