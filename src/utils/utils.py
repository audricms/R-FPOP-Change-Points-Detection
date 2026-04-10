from typing import NamedTuple


class QuadPiece(NamedTuple):
    """A named tuple representing a piece of a piecewise quadratic function.

    Fields
    ------
    a: float
        Left bound of the interval (open)
    b: float
        Right bound of the interval (closed)
    A: float
        Quadratic coefficient (theta^2)
    B: float
        Linear coefficient (theta)
    C: float
        Constant term
    tau: int
        Index of the last changepoint associated with this piece
    """

    a: float
    b: float
    A: float
    B: float
    C: float
    tau: int
