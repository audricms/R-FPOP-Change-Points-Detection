from typing import Any, Dict, Optional, Sequence

import numpy as np
from tqdm import tqdm

from src.utils.gamma_builders import (
    gamma_builder_biweight,
    gamma_builder_huber,
    gamma_builder_L2,
)
from src.utils.rfpop_algorithms import rfpop_algorithm
from src.utils.utils import compute_loss_bound_K, compute_penalty_beta


def online_most_recent_changepoint(
    y: Sequence[float],
    loss: str = "biweight",
    beta: Optional[float] = None,
    K: Optional[float] = None,
    step: int = 50,
    min_obs: int = 100,
) -> Dict[str, Any]:
    """Run RFPOP repeatedly on prefixes of `y` and report recent changepoints.

    This convenience routine runs the RFPOP offline algorithm on growing
    prefixes y[:t] for t in range(min_obs, len(y)+1, step) and returns the
    most-recent detected changepoint at each checkpoint together with all
    changepoints and used parameters.

    Parameters
    ----------
    y : sequence of float
        Time series to analyse.
    loss : {'biweight','huber','l2'}
        Loss family to use when building the local cost functions.
    beta : float, optional
        Penalty constant. If None it is estimated from the full series.
    K : float, optional
        Robust tuning constant (only used for 'huber' and 'biweight'). If
        None a data-driven estimate is used.
    step : int
        Evaluate the online estimator every `step` observations.
    min_obs : int
        Minimum number of observations before starting analysis.

    Returns
    -------
    dict
        Dictionary with keys: 't_values', 'most_recent_cp', 'all_changepoints',
        and 'params' containing {'beta','K','loss'}.
    """

    n = len(y)
    y_list = list(y) if not isinstance(y, list) else y

    if beta is None:
        beta = compute_penalty_beta(y=np.array(y_list), loss=loss)
    if K is None and loss in ["huber", "biweight"]:
        K = compute_loss_bound_K(y=np.array(y_list), loss=loss)

    if loss == "huber":

        def gamma_builder(y_t, t, K=K):
            return gamma_builder_huber(y=y_t, K=K, tau_for_new=t)

    elif loss == "biweight":

        def gamma_builder(y_t, t, K=K):
            return gamma_builder_biweight(y=y_t, K=K, tau_for_new=t)

    else:

        def gamma_builder(y_t, t):
            return gamma_builder_L2(y=y_t, tau_for_new=t)

    t_values = []
    most_recent_cp = []
    all_changepoints = []

    for t in tqdm(range(min_obs, n + 1, step), desc=f"Online analysis ({loss})"):
        y_partial = y_list[:t]

        cp_tau, Qt_vals, _ = rfpop_algorithm(
            y=y_partial, gamma_builder=gamma_builder, beta=beta
        )

        changepoints = []
        idx = t - 1
        while idx > 0:
            tau = cp_tau[idx]
            if tau > 0:
                changepoints.append(tau)
            idx = tau
        changepoints = sorted(changepoints)

        recent_cp = changepoints[-1] if changepoints else 0

        t_values.append(t)
        most_recent_cp.append(recent_cp)
        all_changepoints.append(changepoints.copy())

    return {
        "t_values": t_values,
        "most_recent_cp": most_recent_cp,
        "all_changepoints": all_changepoints,
        "params": {"beta": beta, "K": K, "loss": loss},
    }


if __name__ == "__main__":
    # Run a quick test with dummy data to ensure the function executes without error
    _ = online_most_recent_changepoint(
        y=np.random.randn(200), loss="huber", step=20, min_obs=50
    )
