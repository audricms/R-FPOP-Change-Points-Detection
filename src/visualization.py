import matplotlib.pyplot as plt

from src.losses import gamma_builder_biweight, gamma_builder_huber, gamma_builder_L2
from src.model_selection import compute_loss_bound_K, compute_penalty_beta
from src.rfpop_algorithms import rfpop_algorithm


def plot_segments(df, name, loss, scaling=1.0):
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
    valid_losses = ["huber", "biweight", "l2"]
    if loss not in valid_losses:
        raise ValueError(f"Loss '{loss}' not recognized. Must be one of {valid_losses}")

    y = df[name].dropna()

    fig, ax = plt.subplots(figsize=(10, 5))

    beta = compute_penalty_beta(y=y, loss=loss)

    if loss in ["huber", "biweight"]:
        K = compute_loss_bound_K(y=y, loss=loss)

    if loss == "huber":
        cp_tau, _, _ = rfpop_algorithm(
            y=y,
            gamma_builder=(
                lambda y_t, t: gamma_builder_huber(y=y_t, K=K, tau_for_new=t)
            ),
            beta=beta * scaling,
        )
    elif loss == "biweight":
        cp_tau, _, _ = rfpop_algorithm(
            y=y,
            gamma_builder=(
                lambda y_t, t: gamma_builder_biweight(y=y_t, K=K, tau_for_new=t)
            ),
            beta=beta * scaling,
        )
    elif loss == "l2":
        cp_tau, _, _ = rfpop_algorithm(
            y=y,
            gamma_builder=(lambda y_t, t: gamma_builder_L2(y=y_t, tau_for_new=t)),
            beta=beta * scaling,
        )

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
            [date_start, date_end],
            [seg_mean, seg_mean],
            color="black",
            linewidth=2,
            label="Mean on each segment" if start_idx == 0 else "",
        )

        if end_idx < len(y) - 1:
            ax.axvline(
                x=date_end,
                color="r",
                linestyle="--",
                alpha=0.5,
                label="Detected changepoints",
            )

    if loss == "l2":
        ax.set_title(f"{name} - {loss} loss\nbeta = {round(beta * scaling, 1)}")
    else:
        ax.set_title(
            f"{name} - {loss} loss\nK = {round(K, 1)} | beta = {round(beta * scaling, 1)}"
        )

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

    plt.tight_layout()
    return fig
