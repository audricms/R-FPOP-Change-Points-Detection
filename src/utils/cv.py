import logging
from typing import Literal, Optional

import numpy as np
import pandas as pd
from statsmodels import robust

from src.utils.gamma_builders import (
    gamma_builder_biweight,
    gamma_builder_huber,
    gamma_builder_L1,
    gamma_builder_L2,
)
from src.utils.rfpop_algorithms import rfpop_algorithm
from src.utils.utils import compute_loss_bound_K, compute_penalty_beta


def cross_validate_rfpop(
    y: np.ndarray,
    loss: Literal["huber", "biweight", "l2", "l1"] = "biweight",
    beta_range: Optional[np.ndarray] = None,
    K_range: Optional[np.ndarray] = None,
    criterion: Literal["elbow", "penalized_cost", "bic"] = "elbow",
    verbose: bool = True,
) -> dict:
    """Cross-validate RFPOP over grids of penalty parameters.

    Parameters
    ----------
    y : np.ndarray
        Time series to segment.
    loss : {'huber','biweight','l2','l1'}
        Loss family to use.
    beta_range : array-like, optional
        Grid of beta penalty values to try. If None a default wide grid is used.
    K_range : array-like, optional
        Grid of K tuning constants (only for robust losses). If None a default
        grid around the theoretical value is used.
    criterion : {'elbow','penalized_cost','bic'}
        Selection criterion to pick the best hyperparameters.
    verbose : bool
        Whether to log progress.

    Returns
    -------
    dict
        Dictionary summarising the search and best parameters (best_beta, best_K,
        changepoints, all_results, paper_params, ...).
    """

    n = len(y)
    y_list = list(y)

    ys = pd.Series(y)
    sigma_hat = robust.mad(ys.diff().dropna()) / np.sqrt(2)
    if sigma_hat == 0:
        sigma_hat = np.std(y)

    if loss in ["huber", "biweight"]:
        K_paper = compute_loss_bound_K(y=y, loss=loss)
        beta_paper = compute_penalty_beta(y=y, loss=loss)
    else:  # l2 ou l1
        K_paper = None
        beta_paper = compute_penalty_beta(y=y, loss="l2")

    if beta_range is None:
        beta_mults = np.array([0.5, 1, 1.5, 2, 3, 4, 5, 7, 10, 20, 50, 100, 1000])
        beta_range = beta_mults * beta_paper

    if loss in ["huber", "biweight"]:
        if K_range is None:
            K_mults = np.array([0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0])
            K_range = K_mults * K_paper
    else:
        K_range = [None]

    results = []

    if verbose:
        total = len(beta_range) * len(K_range)
        logging.info(f"Cross-validation ({criterion}) pour {loss} loss")
        logging.info(
            f"Exploration de {len(beta_range)} × {len(K_range)} = {total} combinaisons..."
        )

    for beta in beta_range:
        for K in K_range:
            try:
                if loss == "huber":

                    def make_gamma_huber(K_val):
                        def gamma_inner(y_t, t):
                            return gamma_builder_huber(y=y_t, K=K_val, tau_for_new=t)

                        return gamma_inner

                    gamma_builder = make_gamma_huber(K)
                elif loss == "biweight":

                    def make_gamma_biweight(K_val):
                        def gamma_inner(y_t, t):
                            return gamma_builder_biweight(y=y_t, K=K_val, tau_for_new=t)

                        return gamma_inner

                    gamma_builder = make_gamma_biweight(K)
                elif loss == "l2":

                    def gamma_builder(y_t, t):
                        return gamma_builder_L2(y=y_t, tau_for_new=t)

                else:

                    def gamma_builder(y_t, t):
                        return gamma_builder_L1(y=y_t, tau_for_new=t)

                cp_tau, Qt_vals, _ = rfpop_algorithm(
                    y=y_list, gamma_builder=gamma_builder, beta=beta
                )

                changepoints = list(set(cp_tau))
                if 0 in changepoints:
                    changepoints.remove(0)
                n_cp = len(changepoints)

                penalized_cost = Qt_vals[-1]

                results.append(
                    {
                        "beta": beta,
                        "beta_mult": beta / beta_paper,
                        "K": K,
                        "K_mult": K / K_paper if K else None,
                        "n_changepoints": n_cp,
                        "penalized_cost": penalized_cost,
                        "cp_tau": cp_tau,
                        "changepoints": sorted(changepoints),
                    }
                )

            except Exception:
                if verbose:
                    logging.exception("Erreur durant l'exploration de beta/K")

    if not results:
        raise ValueError("Aucun résultat valide obtenu")

    if criterion == "elbow":
        best = _find_elbow_point(results)
    elif criterion == "bic":
        for r in results:
            r["bic"] = r["penalized_cost"] + np.log(n) * r["n_changepoints"]
        best = min(results, key=lambda x: x["bic"])
    else:
        best = min(results, key=lambda x: x["penalized_cost"])

    return {
        "best_beta": best["beta"],
        "best_K": best["K"],
        "best_ratio": best["beta"] / best["K"] if best["K"] else None,
        "best_n_cp": best["n_changepoints"],
        "changepoints": best["changepoints"],
        "cp_tau": best["cp_tau"],
        "all_results": results,
        "paper_params": {"beta": beta_paper, "K": K_paper, "sigma_hat": sigma_hat},
        "loss": loss,
        "criterion": criterion,
    }


def _find_elbow_point(results):
    """Find the 'elbow' point in a grid search results list.

    The function implements the heuristic used in the original code: it looks
    for the first stable region (two consecutive grid points with equal
    number of changepoints) where the number of changepoints is at most half
    of the observed maximum. Several fallbacks are used if no such region is
    found.
    """
    sorted_results = sorted(results, key=lambda x: x["beta"])
    n_cps = [r["n_changepoints"] for r in sorted_results]

    if not n_cps:
        return results[0]

    max_ncp = max(n_cps)
    threshold = max_ncp / 2

    for i in range(len(n_cps) - 1):
        n_cp = n_cps[i]
        if n_cp > 0 and n_cp <= threshold and n_cps[i + 1] == n_cp:
            return sorted_results[i]

    for i in range(len(n_cps) - 1):
        if n_cps[i] > 0 and n_cps[i + 1] == n_cps[i]:
            return sorted_results[i]

    for i, r in enumerate(sorted_results):
        if r["n_changepoints"] > 0:
            return r

    return sorted_results[0]
