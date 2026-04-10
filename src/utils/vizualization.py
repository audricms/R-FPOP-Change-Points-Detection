from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from src.utils.gamma_builders import (
    gamma_builder_biweight,
    gamma_builder_huber,
    gamma_builder_L2,
)
from src.utils.rfpop_algorithms import rfpop_algorithm
from src.utils.utils import compute_loss_bound_K, compute_penalty_beta


def plot_sensitivity_tobeta(
    df, name, scaling_list=[1, 5, 10, 50, 100, 500, 1000, 5000, 10000, 50000]
):
    """Plot number of detected changepoints as a function of beta scaling.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the series to plot (time series indexed by date).
    name : str
        Column name in ``df`` to analyze.
    scaling_list : Sequence[float]
        List of multipliers applied to the theoretical beta.

    Notes
    -----
    Uses the rfpop algorithm with different losses (huber, biweight, l2)
    and plots the number of detected changepoints as beta varies (log-log
    scale).
    """
    y = df[name].dropna()
    y = y[y.index > "2000"]

    _, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes = axes.flatten()

    for i, loss in tqdm(enumerate(["huber", "biweight", "l2"])):

        beta = compute_penalty_beta(y=y, loss=loss)
        K = compute_loss_bound_K(y=y, loss=loss)

        list_scaling = np.array(scaling_list) * beta
        nb_changepoints = []

        for scaling in list_scaling:
            if loss == "huber":
                cp_tau, _, _ = rfpop_algorithm(
                    y=y,
                    gamma_builder=(
                        lambda y_t, t: gamma_builder_huber(y=y_t, K=K, tau_for_new=t)
                    ),
                    beta=scaling,
                )
            elif loss == "biweight":
                cp_tau, _, _ = rfpop_algorithm(
                    y=y,
                    gamma_builder=(
                        lambda y_t, t: gamma_builder_biweight(y=y_t, K=K, tau_for_new=t)
                    ),
                    beta=scaling,
                )
            elif loss == "l2":
                cp_tau, _, _ = rfpop_algorithm(
                    y=y,
                    gamma_builder=(
                        lambda y_t, t: gamma_builder_L2(y=y_t, tau_for_new=t)
                    ),
                    beta=scaling,
                )

            nb_changepoints.append(len(set(cp_tau)))

        ax = axes[i]
        ax.plot(scaling_list, nb_changepoints)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("beta scaling factor (logscale)")
        ax.set_ylabel("number of detected changepoints (logscale)")
        ax.set_title(f"{name} - {loss} loss: number of changepoints detected")

    plt.tight_layout()
    plt.show()


def plot_segments(df, name, scaling_huber, scaling_biweight, scaling_l2):
    """Plot detected segments for three different losses side-by-side.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the time series.
    name : str
        Column name to process.
    scaling_huber, scaling_biweight, scaling_l2 : float
        Scaling multipliers for beta for each loss.
    """
    y = df[name].dropna()
    y = y[y.index > "2000"]

    _, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes = axes.flatten()

    for i, loss in tqdm(enumerate(["huber", "biweight", "l2"])):

        beta = compute_penalty_beta(y=y, loss=loss)
        K = compute_loss_bound_K(y=y, loss=loss)

        if loss == "huber":
            cp_tau, _, _ = rfpop_algorithm(
                y=y,
                gamma_builder=(
                    lambda y_t, t: gamma_builder_huber(y=y_t, K=K, tau_for_new=t)
                ),
                beta=beta * scaling_huber,
            )
        elif loss == "biweight":
            cp_tau, _, _ = rfpop_algorithm(
                y=y,
                gamma_builder=(
                    lambda y_t, t: gamma_builder_biweight(y=y_t, K=K, tau_for_new=t)
                ),
                beta=beta * scaling_biweight,
            )
        elif loss == "l2":
            cp_tau, _, _ = rfpop_algorithm(
                y=y,
                gamma_builder=(lambda y_t, t: gamma_builder_L2(y=y_t, tau_for_new=t)),
                beta=beta * scaling_l2,
            )

        ax = axes[i]
        ax.plot(y, ".", markersize=2)
        t = len(y) - 1
        segments = []

        while t > 0:
            t_prev = int(cp_tau[t])
            segments.append((t_prev, t))
            t = t_prev

        segments.reverse()

        for start_idx, end_idx in segments:

            segment_data = y.iloc[start_idx : end_idx + 1]
            seg_mean = segment_data.mean()

            date_start = y.index[start_idx]
            date_end = y.index[end_idx]

            ax.plot(
                [
                    date_start,
                    date_end,
                ],
                [seg_mean, seg_mean],
                color="black",
                linewidth=2,
                label="Mean on each segment" if start_idx == 0 else "",
            )

            if end_idx < len(y) - 1:
                ax.axvline(x=date_end, color="r", linestyle="--", alpha=0.5)

        if loss == "l2":
            ax.set_title(f"{name} - {loss} loss\nbeta = {round(beta * scaling_l2, 1)}")
        elif loss == "huber":
            ax.set_title(
                f"{name} - {loss} loss\nK = {round(K, 1)} | beta = {round(beta * scaling_huber, 1)}"
            )
        elif loss == "biweight":
            ax.set_title(
                f"{name} - {loss} loss\nK = {round(K, 1)} | beta = {round(beta * scaling_biweight, 1)}"
            )

        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())

    plt.tight_layout()
    plt.show()


def plot_most_recent_changepoint(
    online_result, true_changepoints: Optional[list] = None, title: str = ""
):
    """Plot the most-recent changepoint vs time as a step function.

    Parameters
    ----------
    online_result : dict
        Result dict produced by ``online_most_recent_changepoint`` with keys
        't_values', 'most_recent_cp', and 'params'.
    true_changepoints : list, optional
        If provided, plot horizontal lines showing the true changepoints.
    title : str, optional
        Title prefix for the plot.
    """
    t_vals = online_result["t_values"]
    recent_cp = online_result["most_recent_cp"]
    recent_cp = np.maximum.accumulate(recent_cp)

    _, ax = plt.subplots(figsize=(12, 6))

    ax.step(
        t_vals,
        recent_cp,
        where="post",
        linewidth=2,
        color="black",
        label="Estimated most recent CP",
    )

    if true_changepoints is not None:
        for cp in true_changepoints:
            ax.axhline(y=cp, color="red", linestyle="--", alpha=0.7, linewidth=1)
        ax.plot([], [], color="red", linestyle="--", label="True changepoints")

    ax.set_xlabel("Time (t)", fontsize=12)
    ax.set_ylabel("Estimated Most Recent Changepoint", fontsize=12)
    ax.set_title(
        f"{title}\nOnline Analysis - {online_result['params']['loss'].upper()} loss",
        fontsize=14,
    )
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    online_result = {
        "t_values": np.arange(50, 200, 10),
        "most_recent_cp": [0, 0, 0, 50, 50, 100, 100, 150, 150],
        "params": {"loss": "huber"},
    }
    plot_most_recent_changepoint(
        online_result=online_result,
        true_changepoints=[50, 100, 150],
        title="Dummy Data",
    )
    plot_segments(
        df=None,
        name="value",
        scaling_huber=1,
        scaling_biweight=1,
        scaling_l2=1,
    )
    plot_sensitivity_tobeta(
        df=None,
        name="value",
        scaling_list=[1, 5, 10],
    )
