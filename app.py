import os
import time

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from src.logger import get_logger
from src.utils import list_s3_csv_files, natural_key, read_csv_from_s3
from src.variables import DATA_DIR, S3_ENDPOINT_URL, VALID_LOSSES
from src.visualization import plot_segments, plot_sensitivity_to_beta

load_dotenv()

logger = get_logger(__name__)

st.set_page_config(
    page_title="Changepoint detection in the presence of outliers", layout="wide"
)
st.title("Changepoint Detection for Time Series with Outliers")

st.markdown(
    """
**Overview of the application:**
* This application features a user-friendly interface that allows you to experiment with the RFPOP algorithm (Robust Functional Pruning Optimal Partitioning).
* The educational goal of this application is to demonstrate how this algorithm is affected by hyperparameter choices. Users are encouraged to experiment with different parameter settings and time series to understand under what conditions the algorithm will work and when it will not.
* This algorithm is designed to detect abrupt changes in the mean of a time series while being resistant to the presence of outliers. It was introduced in *Fearnhead, P., & Rigaill, G. (2019). Changepoint Detection in the Presence of Outliers. Journal of the American Statistical Association*. Until now, this algorithm was only available in R, and we have implemented it in Python.
"""
)

with st.expander("ℹ️ Details about the RFPOP algorithm and parameters"):
    st.markdown(
        r"""
    **Details about the RFPOP algorithm and its parameters:**

    **1. Loss functions:**
    * **L2:** Standard quadratic loss. Theoretically more sensitive to outliers (although not always the case).
    * **Huber / Biweight:** Robust loss functions. They limit the influence of extreme values, preventing the algorithm from falsely detecting outliers as structural changepoints.

    **2. Parameters:**
    * **Penalty factor ($\beta$):** This represents the cost of adding a new changepoint to the model. A higher $\beta$ forces the algorithm to detect fewer changepoints. A lower $\beta$ increases sensitivity to outliers.
    * **Robustness threshold ($K$):** This parameter is specific to the Huber and Biweight losses. It defines the boundary beyond which an observation is classified as an outlier. By capping the influence of values exceeding $K$, the algorithm won't detect isolated outliers as false changepoints.

    **3. Parameter selection:**
    * **Schwarz Information Criteria (SIC):** The automated statistical method proposed in the paper, used to penalize the addition of new changepoints and avoid overfitting. The algorithm always runs with a scaling multiplier $\gamma$ applied to $\beta^{SIC}$: by default $\gamma = 1$ (pure SIC). However, in some cases the order of magnitude suggested by SIC is inappropriate and the algorithm detects too few or too many changepoints.
    * **Elbow plot (optional helper):** If the default SIC result is not satisfying, you can generate an elbow plot — a visual heuristic that shows the number of detected changepoints against the order of magnitude of $\beta$. The optimal order of magnitude is typically located just before the "elbow" of the curve, where the drop in changepoints stabilizes. Once you have identified a good order of magnitude $\gamma$ from the plot, enter it as the scaling multiplier and re-run the algorithm. The values of $(K,\beta)$ used will then be $(K^{SIC},\gamma \times \beta^{SIC})$, where $K^{SIC}, \beta^{SIC}$ are the parameters chosen by SIC.

    **4. About the success and failure of the algorithm:**
    * Detecting changepoints in time series with outliers is a very difficult task, and in some cases, even this algorithm fails to solve the problem and produces oversegmentation (detecting too many changepoints) or undersegmentation (detecting too few changepoints).
    * As explained above, the RFPOP algorithm is highly sensitive to the choice of parameters: the goal of this application is to allow the user to experiment with these different parameters to see their impact on the detected changepoints.
    * We have included a set of pre-configured time series that illustrate this: on some series, the algorithm works well; on others, it performs very poorly. For series 2 and 4, the algorithm works well with parameters chosen by SIC, but for series 1 and 3, the results are more or less satisfactory with SIC depending on the loss function, and in some cases the elbow method must be used to obtain better results.
    """
    )


st.markdown("---")


S3_BUCKET = os.getenv("S3_BUCKET", None)
S3_PREFIX = os.getenv("S3_PREFIX", "")


data_source = st.radio(
    "Data",
    ["Upload a time series", "Use one of the pre-configured time series"],
    horizontal=True,
)

df = None

if data_source == "Upload a time series":
    uploaded_file = st.file_uploader(
        "Please drop a time series in the CSV format", type=["csv"]
    )
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        logger.info(
            "dataset_loaded",
            extra={"source": "upload", "dataset_filename": uploaded_file.name},
        )
else:
    try:
        toy_files = list_s3_csv_files(
            bucket=S3_BUCKET, prefix=S3_PREFIX, endpoint_url=S3_ENDPOINT_URL
        )
        logger.info("s3_listing_succeeded", extra={"file_count": len(toy_files)})
    except Exception as list_error:
        logger.warning(
            "s3_listing_failed",
            extra={"error": str(list_error), "fallback": "local"},
        )
        internal_files = []
        if os.path.exists(DATA_DIR):
            internal_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
        toy_files = sorted(internal_files, key=natural_key)

    if not toy_files:
        st.warning("No CSV file found.")
    else:
        selected_filename = st.selectbox("Choose a dataset", toy_files)
        s3_key = (
            f"{S3_PREFIX.rstrip('/')}/{selected_filename}"
            if S3_PREFIX
            else selected_filename
        )

        try:
            t0 = time.perf_counter()
            df = read_csv_from_s3(
                bucket=S3_BUCKET, key=s3_key, endpoint_url=S3_ENDPOINT_URL
            )
            duration_ms = round((time.perf_counter() - t0) * 1000)
            logger.info(
                "dataset_loaded",
                extra={
                    "source": "s3",
                    "dataset_filename": selected_filename,
                    "duration_ms": duration_ms,
                },
            )
            st.caption("Dataset loaded from public S3.")
        except Exception as s3_error:
            local_file_path = os.path.join(DATA_DIR, selected_filename)
            if os.path.exists(local_file_path):
                df = pd.read_csv(local_file_path)
                logger.warning(
                    "s3_load_failed",
                    extra={
                        "dataset_filename": selected_filename,
                        "error": str(s3_error),
                        "fallback": "local",
                    },
                )
                st.warning(
                    "Could not read dataset from S3. Falling back to local file. "
                    f"Reason: {s3_error}"
                )
            else:
                logger.error(
                    "dataset_load_failed",
                    extra={
                        "dataset_filename": selected_filename,
                        "error": str(s3_error),
                    },
                )
                st.error(f"Could not load dataset from S3: {s3_error}")
                st.stop()


if df is not None:
    numerical_columns = sorted(df.select_dtypes(include=["number"]).columns.tolist())
    if not numerical_columns:
        st.error("The CSV does not contain any numerical variable.")
        st.stop()

    def reset_state() -> None:
        if "elbow_done" in st.session_state:
            del st.session_state["elbow_done"]
        if "elbow_fig" in st.session_state:
            del st.session_state["elbow_fig"]

    col_name = st.selectbox(
        "Choose the variable to analyze", numerical_columns, on_change=reset_state
    )

    col1, col2 = st.columns(2)
    with col1:
        loss_choices = sorted(VALID_LOSSES)
        loss = st.selectbox(
            "Choose the loss function", loss_choices, on_change=reset_state
        )
    with col2:
        chosen_scaling = st.number_input(
            "Scaling multiplier for β (1.0 = pure SIC)",
            min_value=0.001,
            value=1.0,
            format="%f",
        )

    with st.expander(
        "Optional: generate the elbow plot to help choose the scaling multiplier"
    ):
        if st.button("Generate the elbow plot") or st.session_state.get(
            "elbow_done", False
        ):
            if "elbow_fig" not in st.session_state:
                bar = st.progress(
                    0, text="Computing results for the grid of parameters..."
                )
                try:
                    fig_elbow = plot_sensitivity_to_beta(
                        df, name=col_name, loss=loss, progress_bar=bar
                    )
                    st.session_state.elbow_fig = fig_elbow
                    st.session_state.elbow_done = True
                except Exception as e:
                    logger.error(
                        "algorithm_error",
                        extra={"method": "elbow", "loss": loss, "error": str(e)},
                    )
                    st.error(f"Error when generating the elbow plot: {e}")
                    st.stop()
                finally:
                    bar.empty()

            st.plotly_chart(st.session_state.elbow_fig, use_container_width=True)

    if st.button("Start computation"):
        bar = st.progress(0, text="Running the RFPOP algorithm...")
        try:
            bar.progress(50, text="Running the algorithm...")
            fig = plot_segments(df, name=col_name, loss=loss, scaling=chosen_scaling)
            bar.progress(100, text="Done.")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            logger.error(
                "algorithm_error",
                extra={"loss": loss, "scaling": chosen_scaling, "error": str(e)},
            )
            st.error(f"Error when running the algorithm: {e}")
        finally:
            bar.empty()
