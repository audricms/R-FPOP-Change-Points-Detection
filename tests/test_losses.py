from src.losses import gamma_builder_biweight, gamma_builder_huber, gamma_builder_L2
from src.variables import INF


def test_gamma_builder_L2():
    pieces = gamma_builder_L2(y=3.0, tau_for_new=0)
    assert len(pieces) == 1
    a, b, A, B, C, tau = pieces[0]
    assert a == -INF and b == INF
    assert A == 1.0 and B == -6.0 and C == 9.0
    assert tau == 0


def test_gamma_builder_biweight():
    y, K = 5.0, 2.0
    pieces = gamma_builder_biweight(y=y, K=K, tau_for_new=1)
    assert len(pieces) == 3
    assert pieces[0][2] == 0.0 and pieces[0][4] == K**2
    assert pieces[1][2] == 1.0 and pieces[1][3] == -2.0 * y
    assert pieces[2][2] == 0.0 and pieces[2][4] == K**2
    assert pieces[0][1] == y - K and pieces[1][1] == y + K


def test_gamma_builder_huber():
    y, K = 2.0, 1.0
    pieces = gamma_builder_huber(y=y, K=K, tau_for_new=0)
    assert len(pieces) == 3
    assert pieces[0][2] == 0.0
    assert pieces[1][2] == 1.0
    assert pieces[2][2] == 0.0
