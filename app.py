import os

import pandas as pd
import streamlit as st

# Imports basés sur l'architecture du dossier src/
# Vous devez vous assurer que ces fonctions renvoient bien l'objet 'fig' de matplotlib.
from src.model_selection import (
    plot_sensitivity_tobeta,  # Votre code existant pour le BIC
)
from src.visualization import plot_segments

# Configuration de la page
st.set_page_config(
    page_title="Changepoint detection in the presence of outliers", layout="wide"
)
st.title("Changepoint detection for time series with outliers (RFPOP algorithm)")

# --- AJOUT DU TEXTE EXPLICATIF ---
st.markdown(
    """
**Overview of the application:**
* This application features a user-friendly interface that allows you to experiment with the RFPOP algorithm (Robust Functional Pruning Optimal Partitioning).
* The educational goal of this application is to demonstrate how this algorithm is affected by hyperparameter choices. Users are encouraged to experiment with different parameter settings and time series to understand under what conditions the algorithm will work and when it will not.
* This algorithm is designed to detect abrupt changes in the mean of a time series while being resistant to the presence of outliers. It was introduced in *Fearnhead, P., & Rigaill, G. (2019). Changepoint Detection in the Presence of Outliers. Journal of the American Statistical Association*. Until now, this algorithm was only available in R, and we have implemented it in Python.
"""
)

with st.expander("ℹ️ Details about the algorithm and parameters"):
    st.markdown(
        r"""
    **Details about the algorithm and its parameters:**

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


st.markdown("---")  # Ligne horizontale pour séparer l'introduction de l'outil

# --- SECTION CHARGEMENT DES DONNÉES ---

DATA_DIR = "data"

internal_files = []
if os.path.exists(DATA_DIR):
    internal_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]

# MODIFICATION : Remplacement de st.sidebar.radio par st.radio
data_source = st.radio(
    "Source of data",
    ["Upload a time series", "Use a time series from the application"],
    horizontal=True,  # Met les options sur une seule ligne
)

df = None

if data_source == "Upload a time series":
    uploaded_file = st.file_uploader(
        "Please drop a time series in the CSV format", type=["csv"]
    )
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
else:
    if not internal_files:
        st.warning(f"No CSV file found in `{DATA_DIR}/`.")
    else:
        selected_filename = st.selectbox(
            "Choose a dataset from the application", internal_files
        )
        file_path = os.path.join(DATA_DIR, selected_filename)
        df = pd.read_csv(file_path)

# --- FIN SECTION CHARGEMENT ---

if df is not None:
    colonnes_numeriques = df.select_dtypes(include=["number"]).columns.tolist()
    if not colonnes_numeriques:
        st.error("The CSV does not contain any numerical variable.")
        st.stop()

    # --- Fonction pour réinitialiser la mémoire si l'utilisateur change un paramètre ---
    def reset_state():
        if "elbow_done" in st.session_state:
            del st.session_state["elbow_done"]
        if "elbow_fig" in st.session_state:
            del st.session_state["elbow_fig"]

    col_name = st.selectbox(
        "Choose the variable to analyze", colonnes_numeriques, on_change=reset_state
    )

    col1, col2 = st.columns(2)
    with col1:
        loss = st.selectbox("Loss", ["huber", "biweight", "l2"], on_change=reset_state)
    with col2:
        method = st.selectbox(
            "Parameter selection method",
            [
                "Schwarz Information Criteria",
                "Elbow Method (recommended if no statisfying results with the SIC method)",
            ],
            on_change=reset_state,
        )

    st.markdown("---")

    # --- Bloc d'exécution strictement aligné ---
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
                st.error(f"Error when running the algorithm : {e}")
            finally:
                bar.empty()

    elif (
        method
        == "Elbow Method (recommended if no statisfying results with the SIC method)"
    ):
        # Le bouton s'affiche ici. S'il ne s'affiche pas, vérifiez l'indentation de ce 'elif'
        if st.button("Generate the elbow plot") or "elbow_done" in st.session_state:
            if "elbow_fig" not in st.session_state:
                progress_text = "Computing results for the grid of parameters..."
                bar = st.progress(0, text=progress_text)

                try:
                    fig_elbow = plot_sensitivity_tobeta(
                        df, name=col_name, loss=loss, progress_bar=bar
                    )
                    st.session_state.elbow_fig = fig_elbow
                except Exception as e:
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
                    st.error(f"Error : {e}")
                finally:
                    bar_final.empty()
