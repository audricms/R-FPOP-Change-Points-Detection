from src.losses import gamma_builder_L2
from src.rfpop_algorithms import (
    add_qstar_and_gamma,
    min_over_theta,
    prune_compare_to_constant,
    rfpop_algorithm,
)
from src.variables import INF


def test_min_over_theta_finds_quadratic_minimum():
    pieces = [(-INF, INF, 1.0, -2.0, 1.0, 0)]
    val, tau = min_over_theta(pieces)
    assert abs(val) < 1e-9
    assert tau == 0


def test_add_qstar_and_gamma_sums_coefficients():
    qstar = [(-INF, INF, 1.0, 0.0, 0.0, 0)]
    gamma = [(-INF, INF, 0.0, 0.0, 5.0, 1)]
    result = add_qstar_and_gamma(qstar, gamma)
    assert len(result) == 1
    _, _, A, B, C, tau = result[0]
    assert A == 1.0 and B == 0.0 and C == 5.0
    assert tau == 0  # tau from Qstar


def test_prune_compare_to_constant_replaces_above_threshold():
    pieces = [(-3.0, 3.0, 1.0, 0.0, 0.0, 0)]
    result = prune_compare_to_constant(pieces, Qt_val=0.0, beta=1.0, t_index_for_new=1)
    assert len(result) == 3
    assert result[1][2] == 1.0
    assert result[0][2] == 0.0 and abs(result[0][4] - 1.0) < 1e-9
    assert result[2][2] == 0.0 and abs(result[2][4] - 1.0) < 1e-9
    assert result[0][5] == 1 and result[2][5] == 1


def test_rfpop_algorithm_detects_changepoint():
    y = [0.0, 0.0, 0.0, 10.0, 10.0, 10.0]

    def builder(y_t, t):
        return gamma_builder_L2(y=y_t, tau_for_new=t)

    cp_tau, Qt_vals, _ = rfpop_algorithm(y=y, gamma_builder=builder, beta=0.1)
    assert len(Qt_vals) == len(y)
    assert len(set(cp_tau)) > 1
