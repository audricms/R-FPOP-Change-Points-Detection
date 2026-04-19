import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from src.visualization import plot_segments, plot_sensitivity_to_beta


@pytest.fixture
def step_df():
    rng = np.random.default_rng(42)
    y = np.concatenate([rng.standard_normal(20), rng.standard_normal(20) + 5])
    return pd.DataFrame({"signal": y})


def test_plot_segments_returns_figure(step_df):
    fig = plot_segments(df=step_df, name="signal", loss="l2")
    assert isinstance(fig, Figure)


def test_plot_segments_invalid_loss_raises(step_df):
    with pytest.raises(ValueError):
        plot_segments(df=step_df, name="signal", loss="invalid")


def test_plot_sensitivity_to_beta_returns_figure(step_df):
    fig = plot_sensitivity_to_beta(
        df=step_df,
        name="signal",
        loss="l2",
        scaling_list=[0.1, 1.0, 10.0],
    )
    assert isinstance(fig, Figure)
