import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils.cv import cross_validate_rfpop
from src.utils.gamma_builders import (
    gamma_builder_biweight,
    gamma_builder_huber,
    gamma_builder_L2,
)
from src.utils.rfpop_algorithms import rfpop_algorithm
from src.utils.simulations import generate_scenarios
from src.utils.utils import compute_loss_bound_K, compute_penalty_beta
from src.utils.vizualization import (
    plot_most_recent_changepoint,
    plot_segments,
    plot_sensitivity_tobeta,
)


def extract_changepoints_backtrack(cp_tau):
    """Extrait les changepoints par backtracking."""
    n = len(cp_tau)
    changepoints = []
    t = n - 1

    while t > 0:
        tau = cp_tau[t]
        if tau > 0:
            changepoints.append(tau)
        t = tau

    changepoints.reverse()
    return changepoints


def get_segments_from_cp_tau(cp_tau, y):
    """Extrait les segments à partir de cp_tau."""
    n = len(cp_tau)
    segments = []
    t = n - 1

    while t > 0:
        t_prev = int(cp_tau[t])
        if isinstance(y, pd.Series):
            seg_mean = y.iloc[t_prev : t + 1].mean()
        else:
            seg_mean = np.mean(y[t_prev : t + 1])
        segments.append((t_prev, t, seg_mean))
        t = t_prev

    segments.reverse()
    return segments


def online_most_recent_changepoint(
    y,
    loss: str = "biweight",
    beta: float = None,
    K: float = None,
    step: int = 50,  # Calculer tous les 'step' points
    min_obs: int = 100,  # Minimum d'observations avant de commencer
):
    """
    Analyse online: pour chaque t, estime le changepoint le plus récent
    étant donné les données y[0:t].

    Returns:
    --------
    dict avec 't_values' et 'most_recent_cp'
    """
    n = len(y)
    y_list = list(y) if not isinstance(y, list) else y

    # Paramètres par défaut
    if beta is None:
        beta = compute_penalty_beta(y=np.array(y_list), loss=loss)
    if K is None and loss in ["huber", "biweight"]:
        K = compute_loss_bound_K(y=np.array(y_list), loss=loss)

    # Gamma builder
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

    # Pour chaque t de min_obs à n
    for t in tqdm(range(min_obs, n + 1, step), desc=f"Online analysis ({loss})"):
        y_partial = y_list[:t]

        cp_tau, Qt_vals, _ = rfpop_algorithm(
            y=y_partial, gamma_builder=gamma_builder, beta=beta
        )

        # Extraire les changepoints
        changepoints = []
        idx = t - 1
        while idx > 0:
            tau = cp_tau[idx]
            if tau > 0:
                changepoints.append(tau)
            idx = tau
        changepoints = sorted(changepoints)

        # Le plus récent = le dernier de la liste (ou 0 si aucun)
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
    # run all the functions with dummy data for pre-commit sanity check to pass
    _, _ = generate_scenarios()
    _ = compute_penalty_beta(y=np.random.randn(100), loss="biweight")
    _ = compute_loss_bound_K(y=np.random.randn(100), loss="huber")
    _ = cross_validate_rfpop(y=np.random.randn(100), loss="biweight", verbose=False)
    _ = plot_sensitivity_tobeta(
        df=pd.DataFrame({"value": np.random.randn(100)}), name="value", verbose=False
    )
    _ = plot_segments(
        df=pd.DataFrame({"value": np.random.randn(100)}),
        name="value",
        scaling_huber=1,
        scaling_biweight=1,
        scaling_l2=1,
    )
    _ = online_most_recent_changepoint(
        y=np.random.randn(200), loss="huber", step=20, min_obs=50
    )
    _ = plot_most_recent_changepoint(
        online_result=_, true_changepoints=[50, 100, 150], title="Dummy Data"
    )
    _ = extract_changepoints_backtrack(cp_tau=[0, 0, 1, 1, 2, 2, 3])
    _ = get_segments_from_cp_tau(
        cp_tau=[0, 0, 1, 1, 2, 2, 3], y=np.array([10, 10, 20, 20, 30, 30, 40])
    )
