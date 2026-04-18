from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import integrate
from scipy.stats import norm
from statsmodels import robust

from src.losses import gamma_builder_biweight, gamma_builder_huber, gamma_builder_L2
from src.rfpop_algorithms import rfpop_algorithm


def biweight_phi(z, K_std):
    """Biweight influence/psi function used in robust losses.

    This function computes the (derivative-like) influence function for the
    biweight loss scaled by the standard-deviation-normalised threshold
    K_std. It returns 2*z when the standardized residual is within the
    cutoff and 0 otherwise.

    Parameters
    ----------
    z : float
        Standardized residual (usually (y - theta) / sigma).
    K_std : float
        Cutoff value expressed in units of the standardized residual (K / sigma).

    Returns
    -------
    float
        Influence value: 2*z if |z| <= K_std, else 0.0.
    """

    return 2 * z if abs(z) <= K_std else 0.0


def huber_phi(z, K_std):
    """Huber influence/psi function.

    For small residuals (|z| <= K_std) this behaves like 2*z (linear). For
    large residuals it is clipped to +/- 2*K_std (constant slope), which
    reduces the influence of outliers.

    Parameters
    ----------
    z : float
        Standardized residual (usually (y - theta) / sigma).
    K_std : float
        Huber cutoff expressed in units of the standardized residual (K / sigma).

    Returns
    -------
    float
        Influence value: 2*z if |z| <= K_std, else 2*K_std*sign(z).
    """

    return 2 * z if abs(z) <= K_std else 2 * K_std * np.sign(z)


def compute_penalty_beta(y, loss):
    """Compute a penalty constant (beta) for change-point detection.

    The returned penalty depends on the chosen loss function and an
    estimate of the noise scale. For the quadratic (L2) loss this returns
    2*sigma^2*log(n) (up to the scale estimate used here). For robust
    losses (biweight, huber) an extra multiplicative factor E[phi(Z)^2]
    (where Z~N(0,1) and phi is the influence function) is included;
    this factor is computed by numerical integration. For L1 loss the
    function returns log(n).

    Parameters
    ----------
    y : Sequence[float]
        Observed signal (1D sequence) used to estimate the noise scale.
    loss : str
        One of: 'l2', 'biweight', 'huber', 'l1'. Determines which penalty
        form is used.

    Returns
    -------
    float
        Penalty constant to use in the change-point penalised objective.
    """

    ys = pd.Series(y)
    sigma = robust.mad(ys.diff().dropna()) / np.sqrt(2)
    n = len(y)

    if loss == "l2":
        return 2 * sigma**2 * np.log(n)

    elif loss == "biweight":
        K_std = 3.0
        E_phi2, _ = integrate.quad(
            lambda z: (biweight_phi(z=z, K_std=K_std) ** 2) * norm.pdf(z),
            -np.inf,
            np.inf,
        )
        return 2 * sigma**2 * np.log(n) * E_phi2

    elif loss == "huber":
        K_std = 1.345
        E_phi2, _ = integrate.quad(
            lambda z: (huber_phi(z=z, K_std=K_std) ** 2) * norm.pdf(z),
            -np.inf,
            np.inf,
        )
        return 2 * sigma**2 * np.log(n) * E_phi2

    elif loss == "l1":
        return np.log(n)


def compute_loss_bound_K(y, loss: Literal["huber", "biweight"]):
    """Return the tuning constant K (in original units) for robust losses.

    The routine estimates the noise scale using a MAD-based estimator on the
    first differences and then returns a recommended tuning constant K in the
    same units as the data. Supported `loss` values are 'biweight' and
    'huber'. These defaults follow common robust statistics recommendations
    (3*sigma for Tukey's biweight and 1.345*sigma for Huber).

    Parameters
    ----------
    y : Sequence[float]
        Observed signal (1D sequence) used to estimate the noise scale.
    loss : Literal['huber', 'biweight']
        Which loss type to return a K for.

    Returns
    -------
    float
        Recommended tuning constant K (in the same units as `y`).
    """

    ys = pd.Series(y)
    mad = robust.mad(ys.diff().dropna()) / np.sqrt(2)
    if loss == "biweight":
        return 3 * mad
    elif loss == "huber":
        return 1.345 * mad


def plot_sensitivity_tobeta(
    df,
    name,
    loss,
    scaling_list=[0.01, 0.1, 1, 5, 10, 50, 100, 500, 1000, 5000, 10000, 50000],
    progress_bar=None,
):
    """Plot number of detected changepoints as a function of beta scaling for a specific loss.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the series to plot (time series indexed by date).
    name : str
        Column name in ``df`` to analyze.
    loss : str
        The loss function to use ('huber', 'biweight', 'l2').
    scaling_list : Sequence[float], optional
        List of multipliers applied to the theoretical beta.
    """
    valid_losses = ["huber", "biweight", "l2"]
    if loss not in valid_losses:
        raise ValueError(f"Loss '{loss}' not recognized. Must be one of {valid_losses}")

    y = df[name].dropna()

    beta = compute_penalty_beta(y=y, loss=loss)

    # Optimisation : Définition de la fonction de construction du gamma hors de la boucle
    if loss in ["huber", "biweight"]:
        K = compute_loss_bound_K(y=y, loss=loss)

        if loss == "huber":

            def gamma_builder(y_t, t):
                return gamma_builder_huber(y=y_t, K=K, tau_for_new=t)

        else:

            def gamma_builder(y_t, t):
                return gamma_builder_biweight(y=y_t, K=K, tau_for_new=t)

    elif loss == "l2":

        def gamma_builder(y_t, t):
            return gamma_builder_L2(y=y_t, tau_for_new=t)

    list_scaling = np.array(scaling_list) * beta
    nb_changepoints = []
    total_steps = len(list_scaling)

    for idx, scaling in enumerate(list_scaling):
        cp_tau, _, _ = rfpop_algorithm(
            y=y,
            gamma_builder=gamma_builder,
            beta=scaling,
        )
        nb_changepoints.append(len(set(cp_tau)))

        # Mise à jour conditionnelle de la barre de progression Streamlit
        if progress_bar is not None:
            progress_percentage = int(((idx + 1) / total_steps) * 100)
            # Streamlit requiert un entier entre 0 et 100
            progress_percentage = max(0, min(100, progress_percentage))
            progress_bar.progress(progress_percentage)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(scaling_list, nb_changepoints, marker="o", linestyle="-", markersize=4)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Beta scaling factor (logscale)")
    ax.set_ylabel("Number of detected changepoints (logscale)")
    ax.set_title(f"{name} - {loss} loss: Sensitivity to beta")
    ax.grid(True, which="both", ls="--", alpha=0.5)

    plt.tight_layout()
    return fig
