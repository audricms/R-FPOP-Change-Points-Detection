import time
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.logger import get_logger
from src.model_selection import (
    compute_loss_bound_K,
    compute_penalty_beta,
    get_gamma_builder,
)
from src.rfpop_algorithms import rfpop_algorithm
from src.variables import VALID_LOSSES

logger = get_logger(__name__)


def plot_segments(
    df: pd.DataFrame, name: str, loss: str, scaling: float = 1.0
) -> go.Figure:
    """Plot detected segments for a specific loss.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the time series.
    name : str
        Column name to process.
    loss : str
        The loss function to use ('huber', 'biweight', 'l2').
    scaling : float, optional
        Scaling multiplier for beta, by default 1.0.
    """
    if loss not in VALID_LOSSES:
        raise ValueError(f"Loss '{loss}' not recognized. Must be one of {VALID_LOSSES}")

    y = df[name].dropna()

    beta = compute_penalty_beta(y=y, loss=loss)
    t0 = time.perf_counter()
    cp_tau, _, _ = rfpop_algorithm(
        y=y,
        gamma_builder=get_gamma_builder(y=y, loss=loss),
        beta=beta * scaling,
    )
    duration_ms = round((time.perf_counter() - t0) * 1000)
    n_changepoints = len(set(cp_tau)) - 1
    logger.info(
        "algorithm_run",
        extra={
            "function": "plot_segments",
            "loss": loss,
            "n_points": len(y),
            "scaling": scaling,
            "beta": round(float(beta * scaling), 6),
            "n_changepoints": n_changepoints,
            "duration_ms": duration_ms,
        },
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=list(y.index),
            y=list(y),
            mode="markers",
            marker=dict(size=3, color="steelblue"),
            name="Data",
        )
    )

    t = len(y) - 1
    segments = []
    while t > 0:
        t_prev = int(cp_tau[t])
        segments.append((t_prev, t))
        t = t_prev
    segments.reverse()

    has_changepoints = False
    for i, (start_idx, end_idx) in enumerate(segments):
        segment_data = y.iloc[start_idx : end_idx + 1]
        seg_mean = segment_data.mean()
        date_start = y.index[start_idx]
        date_end = y.index[end_idx]

        fig.add_trace(
            go.Scatter(
                x=[date_start, date_end],
                y=[seg_mean, seg_mean],
                mode="lines",
                line=dict(color="black", width=2),
                name="Segment mean",
                showlegend=(i == 0),
            )
        )

        if end_idx < len(y) - 1:
            fig.add_vline(
                x=date_end,
                line=dict(color="red", dash="dash", width=1),
                opacity=0.5,
            )
            has_changepoints = True

    if has_changepoints:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                line=dict(color="red", dash="dash", width=1),
                name="Detected changepoints",
            )
        )

    if loss == "l2":
        title = f"{name} — {loss} loss | beta = {round(beta * scaling, 1)}"
    else:
        K = compute_loss_bound_K(y=y, loss=loss)
        title = f"{name} — {loss} loss | K = {round(K, 1)} | beta = {round(beta * scaling, 1)}"

    fig.update_layout(
        title=title,
        xaxis_title="Index",
        yaxis_title=name,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


def plot_sensitivity_to_beta(
    df: pd.DataFrame,
    name: str,
    loss: str,
    scaling_list: list[float] = [
        0.01,
        0.1,
        1,
        5,
        10,
        50,
        100,
        500,
        1000,
        5000,
        10000,
        50000,
    ],
    progress_bar: Any = None,
) -> go.Figure:
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
    if loss not in VALID_LOSSES:
        raise ValueError(f"Loss '{loss}' not recognized. Must be one of {VALID_LOSSES}")

    y = df[name].dropna()

    beta = compute_penalty_beta(y=y, loss=loss)
    gamma_builder = get_gamma_builder(y=y, loss=loss)

    list_scaling = np.array(scaling_list) * beta
    nb_changepoints = []
    total_steps = len(list_scaling)

    t0 = time.perf_counter()
    for idx, scaling in enumerate(list_scaling):
        cp_tau, _, _ = rfpop_algorithm(
            y=y,
            gamma_builder=gamma_builder,
            beta=scaling,
        )
        nb_changepoints.append(len(set(cp_tau)))

        if progress_bar is not None:
            progress_percentage = max(0, min(100, int(((idx + 1) / total_steps) * 100)))
            progress_bar.progress(progress_percentage)

    duration_ms = round((time.perf_counter() - t0) * 1000)
    logger.info(
        "sensitivity_run",
        extra={
            "function": "plot_sensitivity_to_beta",
            "loss": loss,
            "n_points": len(y),
            "n_scaling_steps": total_steps,
            "duration_ms": duration_ms,
        },
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=scaling_list,
            y=nb_changepoints,
            mode="lines+markers",
            marker=dict(size=6),
            line=dict(color="steelblue"),
            name="Changepoints",
        )
    )
    fig.update_layout(
        title=f"{name} — {loss} loss: Sensitivity to beta",
        xaxis=dict(type="log", title="Beta scaling factor (logscale)"),
        yaxis=dict(type="log", title="Number of detected changepoints (logscale)"),
    )

    return fig
