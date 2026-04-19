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
    * **Schwarz Information Criteria (SIC):** The automated statistical method proposed in the paper, used to penalize the addition of new changepoints and avoid overfitting. However, this method has limitations as the algorithm is highly sensitive to the choice of $\beta$. In some cases, the order of magnitude of $\beta$ suggested by this method is inappropriate and the algorithm will detect either very few changepoints or too many changepoints. In such cases, we propose an alternative method based on the elbow heuristic.
    * **Elbow Method (if no satisfying results with SIC):** A visual heuristic. It plots the number of detected changepoints against the order of magnitude of $\beta$. The optimal order of magnitude is typically located a bit before the "elbow" of the curve, where the drop in the number of changepoints stabilizes. Using the elbow plot, we can iteratively test various orders of magnitude for $\beta$ and see which order of magnitude yield a number of changepoints that suits our needs. For a given order of magnitude $\gamma$ that we choose, the values of $(K,\beta)$ used by the model will be $(K^{SIC},\gamma \times \beta^{SIC})$, where $K^{SIC}, \beta^{SIC}$ are the parameters chosen by the SIC method.

    **4. About the success and failure of the algorithm:**
    * Detecting changepoints in time series with outliers is a very difficult task, and in some cases, even this algorithm fails to solve the problem and produces oversegmentation (detecting too many changepoints) or undersegmentation (detecting too few changepoints).
    * As explained above, the RFPOP algorithm is highly sensitive to the choice of parameters: the goal of this application is to allow the user to experiment with these different parameters to see their impact on the detected changepoints.
    * We have included a set of time series in the application that illustrate this: on some series, the algorithm works well; on others, it performs very poorly. For series 2 and 4, the algorithm works well with parameters chosen by SIC, but for series 1 and 3, the results are more or less satisfactory with SIC depending on the loss function, and in some cases the elbow method must be used to obtain better results.
    """
    )


st.markdown("---")


S3_BUCKET = os.getenv("S3_BUCKET", None)
S3_PREFIX = os.getenv("S3_PREFIX", "")


internal_files = []
if os.path.exists(DATA_DIR):
    internal_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]

data_source = st.radio(
    "Source of data",
    ["Upload a time series", "Use a time series from the application (toy examples)"],
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
            "dataset_loaded", extra={"source": "upload", "filename": uploaded_file.name}
        )
else:
    if internal_files:
        toy_files = sorted(internal_files, key=natural_key)
    else:
        try:
            toy_files = list_s3_csv_files(
                bucket=S3_BUCKET, prefix=S3_PREFIX, endpoint_url=S3_ENDPOINT_URL
            )
        except Exception:
            toy_files = []

    if not toy_files:
        st.warning("No toy CSV file is configured.")
    else:
        selected_filename = st.selectbox("Choose a toy dataset", toy_files)
        s3_key = f"{S3_PREFIX}{selected_filename}"

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
                    "filename": selected_filename,
                    "duration_ms": duration_ms,
                },
            )
            st.caption("Toy dataset loaded from public S3 (SSPCloud MinIO).")
        except Exception as s3_error:
            local_file_path = os.path.join(DATA_DIR, selected_filename)
            if os.path.exists(local_file_path):
                df = pd.read_csv(local_file_path)
                logger.warning(
                    "s3_load_failed",
                    extra={
                        "filename": selected_filename,
                        "error": str(s3_error),
                        "fallback": "local",
                    },
                )
                st.warning(
                    "Could not read toy dataset from public S3. Falling back to local file. "
                    f"Reason: {s3_error}"
                )
            else:
                logger.error(
                    "dataset_load_failed",
                    extra={"filename": selected_filename, "error": str(s3_error)},
                )
                st.error(f"Could not load dataset from public S3: {s3_error}")
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
        loss = st.selectbox("Loss", loss_choices, on_change=reset_state)
    with col2:
        method_choices = sorted(
            [
                "Schwarz Information Criteria",
                "Elbow Method (recommended if no satisfying results with the SIC method)",
            ]
        )
        method = st.selectbox(
            "Parameter selection method", method_choices, on_change=reset_state
        )

    st.markdown("---")

    if method == "Schwarz Information Criteria":
        if st.button("Start computation (Schwarz Information Criteria)"):
            progress_text = (
                "Running the RFPOP algorithm with Schwarz Information Criteria..."
            )
            bar = st.progress(0, text=progress_text)

            try:
                bar.progress(50, text="Running the algorithm...")
                fig = plot_segments(df, name=col_name, loss=loss, scaling=1.0)
                bar.progress(100, text="End.")
                st.pyplot(fig)
            except Exception as e:
                logger.error(
                    "algorithm_error",
                    extra={"method": "sic", "loss": loss, "error": str(e)},
                )
                st.error(f"Error when running the algorithm : {e}")
            finally:
                bar.empty()

    elif (
        method
        == "Elbow Method (recommended if no satisfying results with the SIC method)"
    ):
        if st.button("Generate the elbow plot") or st.session_state.get(
            "elbow_done", False
        ):
            if "elbow_fig" not in st.session_state:
                progress_text = "Computing results for the grid of parameters..."
                bar = st.progress(0, text=progress_text)

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
                    st.error(f"Error when generating the elbow plot : {e}")
                    st.stop()
                finally:
                    bar.empty()

            st.pyplot(st.session_state.elbow_fig)

            st.markdown("### Manually choose the order of magnitude of beta")
            chosen_scaling = st.number_input(
                "Choose the beta order of magnitude that you want from the elbow method plot:",
                min_value=0.001,
                value=1.0,
                format="%f",
            )

            if st.button(
                "Run the RFPOP algorithm with manually chosen beta order of magnitude"
            ):
                progress_text = (
                    f"Running the algorithm for order of magnitude={chosen_scaling}..."
                )
                bar_final = st.progress(0, text=progress_text)

                try:
                    bar_final.progress(50, text="Running the RFPOP algorithm...")
                    fig_final = plot_segments(
                        df, name=col_name, loss=loss, scaling=chosen_scaling
                    )
                    bar_final.progress(100, text="End")
                    st.pyplot(fig_final)
                except Exception as e:
                    logger.error(
                        "algorithm_error",
                        extra={
                            "method": "elbow_manual",
                            "loss": loss,
                            "scaling": chosen_scaling,
                            "error": str(e),
                        },
                    )
                    st.error(f"Error : {e}")
                finally:
                    bar_final.empty()
