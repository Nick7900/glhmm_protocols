import streamlit as st
from glhmm import graphics
import numpy as np
from importlib import reload
reload(graphics)



st.set_page_config(page_title="Inspect HMM Model", page_icon="🔍")
st.title("🔍 Inspect HMM Model")

# Check if an HMM model is loaded
model = st.session_state.get("hmm_model", None)

if model is None:
    st.error("❌ No HMM model found in memory.")
    st.info("Please go to the **fitting hmm** page to train or load a model before using this page.")
    st.stop()

st.markdown("### Select plots to display")


col_init, col_means  = st.columns(2)

with col_init:
    show_init = st.checkbox("Initial state probabilities")
with col_means:
    show_means = st.checkbox("State means (activation patterns)")
col_cov, col_trans  = st.columns(2)
with col_cov:
    show_covs = st.checkbox("Covariances matrices")
with col_trans:
    show_trans = st.checkbox("Transition matrix")

# Section: Initial state probabilities
if show_init:
    try:
        label = "init_prob"
        init_prob = model.Pi
        col_width, col_height = st.columns(2)
        with col_width:
            fig_width = st.number_input("Width", min_value=1.0, value=2.0, step=0.5,
                                        help="Width of the figure in inches", key = f'fig_width_{label}')
        with col_height:
            fig_height = st.number_input("Height", min_value=1.0, value=2.0, step=0.5,
                                        help="Height of the figure in inches", key = f'fig_height_{label}')
        figsize = (fig_width, fig_height)
        fig =graphics.plot_initial_state_probabilities(init_prob,figsize=figsize, return_fig=True)
        st.pyplot(fig)
    except Exception as e:
        st.warning("Could not plot initial state probabilities.")
        st.exception(e)

# Section: State Means
if show_means:
    try:
        if not getattr(model, "model_mean", False):
            st.error("ℹ️ This model was trained without state means (`Mean model=No`).")
        else:
            label = "state_means"
            col_width, col_height = st.columns(2)
            with col_width:
                fig_width = st.number_input("Width", min_value=1.0, value=4.0, step=0.5,
                                            help="Figure width (inches)", key=f'fig_width_{label}')
            with col_height:
                fig_height = st.number_input("Height", min_value=1.0, value=3.5, step=0.5,
                                             help="Figure height (inches)", key=f'fig_height_{label}')
            figsize = (fig_width, fig_height)

            col_xlabel, col_ylabel = st.columns(2)
            with col_xlabel:
                xlabel = st.text_input("X label", value="State", key=f'xlabel_{label}')
            with col_ylabel:
                ylabel = st.text_input("Y label", value="Brain region", key=f'ylabel_{label}')

            state_means = model.get_means()
            fig = graphics.plot_state_means_activations(state_means, figsize=figsize, return_fig=True, xlabel=xlabel, ylabel=ylabel)
            st.pyplot(fig)
    except Exception as e:
        st.warning("Could not plot state means.")
        st.exception(e)
# Section: State Covariances
if show_covs:
    try:

        label = "cov_matrix"
        col_width, col_height = st.columns(2)
        with col_width:
            fig_width = st.number_input("Width", min_value=1.0, value=4.0, step=0.5,
                                        help="Width of the figure in inches", key = f'fig_width_{label}')
        with col_height:
            fig_height = st.number_input("Height", min_value=1.0, value=4.0, step=0.5,
                                        help="Height of the figure in inches", key = f'fig_height_{label}')
        figsize_per_plot = (fig_width, fig_height)

        num_cols = st.number_input("Columns", min_value=1, value=3, step=1,
                                        help="Figures per column", key = f'num_col_{label}')
        col_xlabel, col_ylabel = st.columns(2)
        with col_xlabel:
            xlabel = st.text_input(f"X-axis label", value="Brain region", key = f'xlabel_{label}')
        with col_ylabel:
            ylabel = st.text_input(f"Y-axis label", value="Brain region", key = f'ylabel_{label}')

        state_cov = model.get_covariance_matrices()
        fig =graphics.plot_state_covariances(state_cov, num_cols=num_cols,figsize_per_plot=figsize_per_plot, return_fig=True)
        st.pyplot(fig)
    except Exception as e:
        st.warning("Could not plot covariance matrices.")
        st.exception(e)

# Section: Transition Matrix
if show_trans:
    try:
        label = "trans_matrix"
        col_width, col_height = st.columns(2)
        with col_width:
            fig_width = st.number_input("Width", min_value=1.0, value=3.5, step=0.5,
                                        help="Width of the figure in inches", key = f'fig_width_{label}')
        with col_height:
            fig_height = st.number_input("Height", min_value=1.0, value=3.5, step=0.5,
                                        help="Height of the figure in inches", key = f'fig_height_{label}')
        figsize = (fig_width, fig_height)
        with_self_transitions = st.checkbox("Include self transition", help="Show transition probabilities with self-transitions.", key = f'self_{label}')
        fig =graphics.plot_transition_matrix(model.P, figsize=figsize, with_self_transitions =with_self_transitions,return_fig=True)
        st.pyplot(fig)
    except Exception as e:
        st.warning("Could not plot transition matrix.")
        st.exception(e)