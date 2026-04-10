from typing import List

from src.utils.rfpop_algorithms import QuadPiece
from src.utils.variables import INF


def gamma_builder_L2(y: float, tau_for_new: int) -> List[QuadPiece]:
    """Return a piecewise-quadratic representation of the L2 loss.

    The L2 loss gamma(y, theta) = (y - theta)**2 is a single quadratic
    defined on the whole real line. This builder returns one QuadPiece
    covering (-INF, INF].

    Parameters
    ----------
    y : float
        Observation value y_t.
    tau_for_new : int
        Index to use for the ``tau`` field of the produced QuadPiece.

    Returns
    -------
    List[QuadPiece]
        Single-element list containing the quadratic coefficients for the
        entire real line: (a, b, A, B, C, tau).
    """
    A = 1.0
    B = -2.0 * y
    C = y * y
    return [(-INF, INF, A, B, C, tau_for_new)]


def gamma_builder_biweight(y: float, K: float, tau_for_new: int) -> List[QuadPiece]:
    """Return the biweight/clipped-quadratic loss as piecewise quadratics.

    The biweight-style loss equals (y-theta)**2 for |y-theta| <= K and is
    constant (K**2) outside that interval. The returned representation has
    three QuadPiece entries corresponding to the left constant piece,
    the central quadratic piece and the right constant piece.

    Parameters
    ----------
    y : float
        Observation value y_t.
    K : float
        Tuning constant defining the clipping threshold.
    tau_for_new : int
        Index to use for the ``tau`` field of the produced pieces.

    Returns
    -------
    List[QuadPiece]
        Three QuadPiece entries for (-INF, y-K], (y-K, y+K], (y+K, INF).
    """
    A_q = 1.0
    B_q = -2.0 * y
    C_q = y * y
    constC = float(K * K)
    return [
        (-INF, y - K, 0.0, 0.0, constC, tau_for_new),
        (y - K, y + K, A_q, B_q, C_q, tau_for_new),
        (y + K, INF, 0.0, 0.0, constC, tau_for_new),
    ]


def gamma_builder_huber(y: float, K: float, tau_for_new: int) -> List[QuadPiece]:
    """Return a piecewise representation of the Huber loss.

    The Huber loss with threshold K is quadratic in a central region
    |y-theta| <= K and linear outside. We express the linear pieces as
    quadratics with A=0 and appropriate B, C coefficients so every piece is
    a QuadPiece.

    Parameters
    ----------
    y : float
        Observation value y_t.
    K : float
        Huber threshold.
    tau_for_new : int
        Index to use for the ``tau`` field of the produced pieces.

    Returns
    -------
    List[QuadPiece]
        Three QuadPiece entries for the left linear piece, the central
        quadratic piece, and the right linear piece.
    """
    A_q = 1.0
    B_q = -2.0 * y
    C_q = y * y
    B_left = -2.0 * K
    C_left = 2.0 * K * y - K * K
    B_right = 2.0 * K
    C_right = -2.0 * K * y - K * K
    return [
        (-INF, y - K, 0.0, B_left, C_left, tau_for_new),
        (y - K, y + K, A_q, B_q, C_q, tau_for_new),
        (y + K, INF, 0.0, B_right, C_right, tau_for_new),
    ]


def gamma_builder_L1(y: float, tau_for_new: int) -> List[QuadPiece]:
    """Return a piecewise-quadratic representation for the L1 loss.

    The absolute loss |y - theta| is represented as two linear pieces which
    we express as QuadPiece entries with A=0 (i.e. degenerate quadratics).

    Parameters
    ----------
    y : float
        Observation value y_t.
    tau_for_new : int
        Index to use for the ``tau`` field of the produced pieces.

    Returns
    -------
    List[QuadPiece]
        Two QuadPiece entries covering (-INF,y] and (y,INF] respectively.
    """
    return [
        (-INF, y, 0.0, -1.0, float(y), tau_for_new),
        (y, INF, 0.0, 1.0, float(-y), tau_for_new),
    ]
