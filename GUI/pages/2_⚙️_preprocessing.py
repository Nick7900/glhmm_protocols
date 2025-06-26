import streamlit as st
from glhmm import preproc
import numpy as np
import json
from pathlib import Path
import pickle 
from utils import detect_state_type
from importlib import reload
reload(preproc)

st.set_page_config(page_title="Preprocessing", page_icon="⚙️")
st.title("⚙️ Preprocessing")

# Define variables
lags = None
pca = None
ica = None
exact_pca = None
ica_algorithm = None
post_standardise = None
enable_lags = False
autoregressive_order = None
# Check if data is loaded
# Prevent running if no data loaded
if st.session_state.get("data_load") is None:
    st.error("❌ No data was found in memory.")
    st.info("Please go to the **load date** before using this page.")
    st.stop()

# Detect state type only once and store it
if "data_type_detected" not in st.session_state:
    detected_type = detect_state_type(st.session_state["data_load"])
    st.session_state.data_type_detected = detected_type
else:
    detected_type = st.session_state.data_type_detected

if detected_type in ["Gamma", "Viterbi Path"]:
    st.warning(
        "⚠️ State time course data like **Gamma** or **Viterbi Path** cannot be preprocessed.\n\n"
        "To preprocess data, please load raw or processed signal data instead.\n\n"
        "✅ Since you have already decoded state time courses, you can skip the preprocssing step and go directly to the **fitting HMM** page.\n"
        "There, select:\n\n"
        "➡️ **Configure loaded data for statistical analysis** (third option above)."
    )
    st.stop()
preprocess_both = False
if st.session_state.get("D_and_R_same", False):
    st.session_state.preprocess_both = st.checkbox(
        "Preprocess both D and R (for GLHMM)", 
        value=st.session_state.get("preprocess_both", False),
        help="Tick this if you want to preprocess both brain data and behavioural data (they have the same shape)."
    )
else:
    st.session_state.preprocess_both = False
    
st.session_state.fs = None  # Initialize fs in session state
st.subheader("Preprocessing Parameters")
standardise = st.checkbox("Standardise (z-score)", value=True)

fs = st.number_input("Sampling frequency (Hz)", min_value=1.0, value=1000.0, step=1.0)
st.session_state.fs = fs  # Store fs in session state for later use

# Downsampling
enable_downsample = st.checkbox("Enable downsampling", value=False, key="downsample_checkbox", help="Reduce the sampling rate of your data to decrease computational load. Useful for long recordings or high-frequency data.")
if enable_downsample:
    downsample = st.number_input(
        "Target frequency (Hz)",
        min_value=1.0,
        max_value=float(fs),
        value=float(fs/2),
        step=1.0,
        key="downsample_input",
        help="Specify the new sampling frequency. The original data will be resampled to this rate."
    )
    st.info(f"Downsampling from {fs}Hz to {downsample}Hz")
downsample = downsample if enable_downsample else None



# Dampen mode selection (inside the form but controlled via session_state)
st.selectbox(
    "Dampen extreme peaks",
    ["None", "True", "Set value"],
    index=0,
    key="dampen_mode",
    help="Suppress unusually large amplitude values in the signal. This can reduce artefacts and improve model stability."
)

# Conditionally show dampening strength input
if st.session_state.dampen_mode == "Set value":
    dampen_val = st.number_input("Dampening strength", min_value=1, value=20, key="dampen_strength", help="Clamp the signal so that extreme values are reduced. Lower values apply stronger dampening."
    )
elif st.session_state.dampen_mode == "True":
    dampen_val = True
else:
    dampen_val = None

st.markdown("---")
# Filtering
st.markdown("#### Filter Settings")
filter_type = st.radio(
    "Filter type",
    options=["None", "Band-pass", "High-pass", "Low-pass"],
    index=0,
    help="Apply temporal filtering to isolate specific frequency bands of interest."
)

freqs = None
if filter_type == "Band-pass":
    col3, col4 = st.columns(2)
    with col3:
        low = st.number_input("Low cutoff (Hz)", min_value=0.0, max_value=fs / 2, value=1.0, step=0.5, key="band_low", help="Frequencies below this value will be attenuated.")
    with col4:
        high = st.number_input("High cutoff (Hz)", min_value=0.0, max_value=fs / 2, value=250.0, step=0.5, key="band_high",  help="Frequencies above this value will be attenuated.")
    if low < high:
        freqs = [low, high]
elif filter_type == "High-pass":
    low = st.number_input("High-pass cutoff (Hz)", min_value=0.0, max_value=fs / 2, value=1.0, step=0.5, key="hp_cutoff", help="Remove slow drifts and baseline shifts by filtering out low frequencies.")
    freqs = [low, None]
elif filter_type == "Low-pass":
    high = st.number_input("Low-pass cutoff (Hz)", min_value=0.0, max_value=fs / 2, value=50.0, step=0.5, key="lp_cutoff", help="Remove high-frequency noise by attenuating frequencies above this cutoff.")
    freqs = [0.0, high]

st.markdown("---")

# Transformations
st.markdown("#### Transformations")

col5, col6 = st.columns(2)
with col5:
    detrend = st.checkbox("Detrend", value=False, help="Remove linear trends from the signal to focus on fluctuations around the mean.")
with col6:
    onpower = st.checkbox("Extract power (Hilbert)", value=False, help="Compute the instantaneous power of the signal using the Hilbert transform.")

onphase = st.checkbox("Extract phase (Hilbert)", value=False, help="Compute the instantaneous phase using the Hilbert transform. Used in phase-based connectivity and state modelling.")
enable_AR = False
if preprocess_both == False:
    st.markdown("---")
    st.markdown("#### Embedding Options")
    enable_lags = st.checkbox("Enable time lags (for HMM-TDE)", value=False, key="enable_lags")

    enable_AR = st.checkbox("Enable autoregressive coefficient (for HMM-MAR)", value=False, key="enable_AR")


    
    if enable_AR:
        st.markdown("""
    The **HMM-MAR** (Hidden Markov Model with Multivariate Autoregressive observations) models the temporal dynamics of your data by estimating AR coefficients within each state.
    
    ℹ️ A typical choice is AR order 5–10 for high temporal resolution signals like MEG or EEG (See original [MAR-HMM](https://www.sciencedirect.com/science/article/pii/S1053811915010691?via%3Dihub) paper for more information).
    """)
        
        autoregressive_order = st.number_input(
        "Autoregressive Order",
        min_value=1,
        value=5,
        step=1,
        key="Autoregressive_input"
    )
autoregressive_order = autoregressive_order if enable_AR else None


# Dimensionality Reduction (only if NOT using lags)
if not enable_lags:
    st.markdown("---")
    st.markdown("### Dimensionality Reduction")
    st.selectbox(
        "Method",
        ["None", "PCA", "ICA"],
        index=0,
        key="dim_reduction_radio"
    )

    # Conditionally show dimensionality reduction input
    if st.session_state.dim_reduction_radio == "PCA":
        ica = None
        ica_algorithm = None
        pca_input = st.number_input("PCA: Number of components (≥1) or variance (0-1)", 
                                    min_value=0.01, 
                                    value=0.95, 
                                    step=0.05,
                                    format="%.2f",
                                    key="pca_components_input")
        try:
            if pca_input >= 1:
                pca = int(pca_input)
            elif 0 < pca_input < 1:
                pca = float(pca_input)
            else:
                st.warning("PCA must be greater than 0")
                pca = None
        except ValueError:
            st.warning("Invalid PCA value")
            pca = None

        exact_pca = st.checkbox("Exact PCA computation", value=True, key="exact_pca_check")
        post_standardise = st.checkbox("Standardize components after PCA", value=True, key="post_pca_std")

    elif st.session_state.dim_reduction_radio == "ICA":
        pca = None
        ica = st.number_input("Number of ICA components", 
                                min_value=1, 
                                value=20, 
                                step=1,
                                key="ica_components_input")
        
        ica_algorithm = st.selectbox(
        "ICA algorithm:",
        options=["parallel", "deflation"],
        key="ica_algorithm_select"
        )
    

else:

    st.markdown("""
    Specify a window of time lags for **temporal embedding** (used in HMM-TDE). This is optional and should only be set if you plan to apply a Time-Delay Embedded Hidden Markov Model.

    - **Half-window size (L):** How far back and forward in time to look (e.g. L=3 means include 3 steps before and after).
    - **Step size (S):** The spacing between time lags.

    For example, setting **L = 3** and **S = 1** gives the lags:  
    `[-3, -2, -1, 0, 1, 2, 3]`
    """)

    col1, col2 = st.columns(2)
    with col1:
        L = st.number_input("Half-window size (L)", min_value=1, value=14, step=1, key="lag_L")
    with col2:
        S = st.number_input("Step size (S)", min_value=1, value=1, step=1, key="lag_S")
    
    positive_lags = np.arange(0, L + 1, S)
    lags = np.sort(np.unique(np.concatenate([-positive_lags[1:], positive_lags])))

    st.info(f"Computed lags: {lags.tolist()}")

    # PCA recommendation section (after lag selection)
    n_channels = st.session_state.data_load.shape[-1]
    suggested_pca = n_channels * 2

    st.markdown(f"""
    **Number of PCA Components**

    When applying a Time-Delay Embedded HMM (TDE-HMM), we recommend reducing the dimensionality of the embedded data using PCA.

    Based on [Vidaurre et al. (2018)](https://www.nature.com/articles/s41467-018-05316-z), the number of PCA components should be approximately **twice the number of channels**  
    to balance computational efficiency with preserving signal structure.

    ℹ️ Your data has **{n_channels}** features. Suggested number of PCA components: **{suggested_pca}**
    """)
    apply_pca = st.checkbox("Apply PCA", value=True, key="apply_pca")
    if apply_pca:
        pca = st.number_input("Number of PCA components", 
                            min_value=1, 
                            value=suggested_pca, 
                            step=1,
                            key="pca_components_input")
    else:
        pca = None




st.checkbox("💾 Save preprocessed values", value=True, key="save_preproc_values")
if st.session_state.save_preproc_values:
    st.markdown("#### Select which variables to save:")
    col_data_preproc, col_indices, col_preproc_log = st.columns(3)
    with col_data_preproc:
        st.checkbox("Save preprocessed values", value=True, key="save_preproc_data")
    with col_indices:
        st.checkbox("Save preprocessed indices values", value=True, key="save_preproc_indices")
    with col_preproc_log:
        st.checkbox("Save log-file of preprocessed data", value=True, key="save_preproc_log")


    st.text_input("Folder name for saving results", value="Preprocessed_results", key="hmm_preproc_folder")
    st.markdown(
        "> 💡 **Note:** It's a good idea to save selected values to disk. "
        "If your Streamlit app is idle for too long or the page reloads, all unsaved session data will be lost. "
        "You can later reload saved files for further analysis."
    )

# --- Submit button only ---
with st.form("preprocessing_form"):
    submitted = st.form_submit_button("Run Preprocessing")


# Run preprocessing
if submitted:
    with st.spinner("Preprocessing data..."):
        try:
            data_load = st.session_state.data_load
            data_behav = st.session_state.data_behav
            indices = st.session_state.indices

            if preprocess_both:
                data_behav_preproc, idx_behav_preproc, log = preproc.preprocess_data(
                    data=data_behav,
                    indices=indices,
                    fs=fs,
                    dampen_extreme_peaks=dampen_val,
                    standardise=standardise,
                    filter=freqs,
                    detrend=detrend,
                    onpower=onpower,
                    onphase=onphase,
                    pca=pca,
                    exact_pca=exact_pca,
                    ica=ica,
                    ica_algorithm=ica_algorithm,
                    post_standardise=post_standardise,
                    downsample=downsample,
                )

                st.session_state.data_behav_preproc = data_behav_preproc
                st.session_state.indices_behav_preproc = idx_behav_preproc
                st.session_state.preproc_log = log
                st.session_state.lags = lags
                st.session_state.preproc_run = submitted
                st.session_state.autoregressive_order = autoregressive_order
                log_GUI = log.copy()  # make a copy to avoid modifying original log

                log_GUI["pca"] = pca  
                log_GUI["ica"] = ica
                log_GUI["lags"] = lags.tolist() if enable_lags else None
                log_GUI.pop("pcamodel", None)  # safely remove if it exists
                st.session_state.preproc_log_GUI = log_GUI

                st.success("✅ Preprocessing completed!")
                st.write(f"New shape: {data_preproc.shape}")

                model_type = "Gaussian HMM"

            else:
                data_preproc, idx_preproc, log = preproc.preprocess_data(
                    data=data_load,
                    indices=indices,
                    fs=fs,
                    dampen_extreme_peaks=dampen_val,
                    standardise=standardise,
                    filter=freqs,
                    detrend=detrend,
                    onpower=onpower,
                    onphase=onphase,
                    pca=pca,
                    exact_pca=exact_pca,
                    ica=ica,
                    ica_algorithm=ica_algorithm,
                    post_standardise=post_standardise,
                    downsample=downsample,
                    lags=lags,
                    autoregressive_order= autoregressive_order
                )



                st.session_state.data_preproc = data_preproc
                st.session_state.indices_preproc = idx_preproc
                st.session_state.preproc_log = log
                st.session_state.lags = lags
                st.session_state.preproc_run = submitted
                st.session_state.autoregressive_order = autoregressive_order
                log_GUI = log.copy()  # make a copy to avoid modifying original log

                log_GUI["pca"] = pca  
                log_GUI["ica"] = ica
                log_GUI["lags"] = lags.tolist() if enable_lags else None
                log_GUI["autoregressive_order"] = autoregressive_order if enable_AR else None
                log_GUI.pop("pcamodel", None)  # safely remove if it exists
                st.session_state.preproc_log_GUI = log_GUI

                st.success("✅ Preprocessing completed!")


                # --- Determine and display HMM model type ---
                if preprocess_both:
                    model_type = "GLHMM"
                elif enable_lags:
                    model_type = "HMM-TDE"
                elif enable_AR:
                    model_type = "HMM-MAR"
                else:
                    model_type = "Gaussian HMM"

            st.session_state.HMM_model = model_type  # Store for use on other pages


            # Save selected values
            if st.session_state.save_preproc_values:
                save_dir = Path(__file__).resolve().parent.parent / st.session_state.hmm_preproc_folder
                save_dir.mkdir(exist_ok=True)

                if st.session_state.save_preproc_log:
                    with open(save_dir / "preproc_log.pkl", "wb") as f:
                        pickle.dump(st.session_state.preproc_log, f)

                if st.session_state.save_preproc_indices:
                    np.save(save_dir / "prepoc_idx.npy", idx_preproc)

                if st.session_state.save_preproc_data:
                    np.save(save_dir / "preproc_data.npy", data_preproc)


                st.success(f"✅ Saved selected values to: `{save_dir}`")

            st.info(f"✅ The model is prepared for: **{model_type}**")
            st.write(f"New shape: {data_preproc.shape}")

        except Exception as e:
            st.error("❌ Preprocessing failed")
            st.error(str(e))

    # Log download
    if "preproc_log" in st.session_state:
        st.markdown("---")
        st.subheader("📋 Preprocessing Log")
        with st.expander("View Log"):
            st.json(st.session_state.preproc_log_GUI)

        # log_text = json.dumps(st.session_state.preproc_log_GUI, indent=2)
        # st.download_button(
        #     "💾 Download Preprocessing Log",
        #     data=log_text,
        #     file_name="preprocessing_log.json",
        #     mime="application/json"
        # )

