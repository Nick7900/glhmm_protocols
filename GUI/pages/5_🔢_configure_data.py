import streamlit as st
import numpy as np
import pickle
from pathlib import Path
from glhmm import glhmm, auxiliary, io, graphics, statistics, utils, preproc
from utils import create_index_table, analysis_from_gamma, detect_state_type
import tempfile
import io, contextlib

# Remove raw data if no longer needed
if "data_raw" in st.session_state:
    st.session_state.pop("data_raw")

st.set_page_config(page_title="Configure data for statistical analysis", page_icon="🔢")
st.title("🔢 Configure data for statistical analysis")
st.session_state.analysis_fig = None

# Prevent running if no data loaded
if st.session_state.get("data_load") is None:
    st.error("❌ No data (D-matrix) was found in memory.")
    st.info("Please go to **load date**  and load the data before using this page.")
    st.stop()
    
if "indices" not in st.session_state:
    st.error("❌ No indices was found in memory.")
    st.info("Please define or load session structure on the **load data** page.")
    st.stop()

# Handle single-session edge case
proceed_with_mode = True
if len(st.session_state.indices) == 1:
    st.warning("⚠️ Only one subject/session found. This may limit group-level analyses.")
    proceed_with_mode = False

    redefine = st.radio(
        "Redefine session structure?",
        ["Load from file (Page 1)", "Define manually", "Keep current"],
        key="redefine_indices_choice"
    )

    if redefine == "Define manually":
        st.markdown("#### Define session/subject structure")
        n_subjects = st.number_input("Number of subjects/sessions", min_value=2, step=1)
        n_timepoints = st.number_input("Number of timepoints per subject/session", min_value=1, step=1)

        if st.button("Generate indices"):
            total_expected = n_subjects * n_timepoints
            gamma_length = st.session_state.gamma.shape[0] if st.session_state.get("gamma") is not None else None

            if gamma_length is not None and total_expected != gamma_length:
                st.error(f"❌ Expected {total_expected} samples, but Gamma has {gamma_length}.")
            else:
                idx_subject = statistics.get_indices_timestamp(n_timepoints, n_subjects)
                st.session_state.indices = idx_subject
                st.success(f"✅ Created new indices with shape: {idx_subject.shape}")
                proceed_with_mode = True

    elif redefine == "Keep current":
        proceed_with_mode = True

    elif redefine == "Load from file (Page 1)":
        st.info("Use **Page 1: Load Data** to upload session indices.")
        st.stop()

# Show mode selector only when ready
st.info(
    """
    Prepare your data for statistical testing by extracting **summary statistics** (e.g., fractional occupancy, switching rate, state lifetime) 
    or restructuring state time courses (Gamma) into **epochs**.  
    If you prefer to use raw or precomputed data directly, you can skip this section.
    """
)
valid_load_keys = [
    k for k in st.session_state.keys()
    if (k.startswith("data_") or k.startswith("gamma") or k.startswith("vpath"))
    and not any(k.startswith(prefix) for prefix in ["data_reshape","data_type_detected", "data_behav", "data_FO", "data_LT", "data_SR"])
]

selected_type = st.radio("Choose state time course type", ["Gamma (state probabilities)", "Viterbi Path (most likely states)"], key="select_data_type")
Y_key = st.selectbox(f"Select {selected_type}", options=valid_load_keys, key="select_data_key")

if st.button("Select data", key="select_data_button"):
    st.session_state.run_analysis = True
    st.session_state.selected_Y_key = Y_key
    st.session_state.selected_state_type = selected_type
    st.session_state.analysis_chosen = False

    selected_data = st.session_state[Y_key]
    if selected_data is not None:
        detected_type = detect_state_type(selected_data)
        st.session_state.selected_state_detected = detected_type
        selected_data = st.session_state[st.session_state.selected_Y_key]
        detected_type = st.session_state.selected_state_detected
        st.write(f"Auto-detected type: `{detected_type}`")
        st.write(f"**Selected data shape:** {selected_data.shape}")

        if detected_type == "Gamma":
            st.session_state.gamma = selected_data
            st.session_state.vpath = None
        elif detected_type == "Viterbi Path":
            st.session_state.vpath = selected_data
            st.session_state.gamma = None
        else:
            st.warning("⚠️ Could not confidently detect if the data is Gamma or Viterbi path.")
            st.stop()
    else:
        st.warning("⚠️ No data found under the selected key. Please check that the data is loaded.")
        st.session_state.run_analysis = False
 

    st.session_state.indices = st.session_state["indices"]
  
# Shared function to configure statistical analysis

if st.session_state.get("run_analysis", False):


    analysis_options = []

    if st.session_state.gamma is not None and st.session_state.gamma.ndim == 2:
        analysis_options.append("Fractional Occupancy")
        analysis_options.append("Switching Rate")

    if st.session_state.vpath is not None and st.session_state.vpath.ndim == 2:
        analysis_options.append("State Lifetime")

    if st.session_state.gamma is not None:
        analysis_options.append("Restructure data into Epochs")
        if st.session_state.get("event_markers") is not None:
            st.session_state.reconstruct_event = True
        else:
            st.session_state.reconstruct_event = False

    if not analysis_options:
        st.warning("⚠️ Could not determine available analysis options. Please ensure Gamma or Viterbi path is loaded.")
    else:
        analysis_from_gamma(
            Gamma=st.session_state.gamma,
            vpath=st.session_state.vpath,
            analysis_options=analysis_options,
            indices=st.session_state.indices,
            data_behav=st.session_state.get("data_behav"),
            fs_default=st.session_state.get("fs", 250),
        )

    if st.button("🔁: Reselect data"):
        st.session_state.run_analysis = False
        st.session_state.analysis_chosen = False
