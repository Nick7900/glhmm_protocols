import streamlit as st
import pandas as pd
import numpy as np
import pickle
from glhmm import statistics
from glhmm.utils import get_FO, get_switching_rate, get_life_times
import sys
from pathlib import Path
import tempfile
from importlib import reload
reload(statistics)
sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils import load_data, validate_confounds, _save_object_ui

if "data_raw" in st.session_state:
    st.session_state.pop("data_raw")  # removes it from memory only on this page
    
st.set_page_config(page_title="Statistical Tests", page_icon="📊")
st.title("📊 Statistical Testing")

# Prevent running if no data loaded
if st.session_state.get("data_load") is None:
    st.error("❌ No data (D-matrix) was found in memory.")
    st.info("Please go to **load date**  and load the data before using this page.")
    st.stop()
# Prevent running if no data loaded
if st.session_state.get("data_behav") is None:
    st.error("❌ No behavioral data (R-matrix) was found in memory.")
    st.info("Please go to **load date** and load the data before using this page.")
    st.stop()


# Test selection

st.subheader("Select Test Type")
test_type = st.selectbox("Choose the type of test to perform", [
    "Across-subjects",
    "Across-trials",
    "Across-sessions",
    "Across-state-visits"
])

# Define session keys 
session_keys = list(st.session_state.keys())

# Dynamic filtering of keys based on test type
if test_type == "Across-subjects":
    # Valid D must be one of the aggregated data types
    valid_d_keys = [
        k for k in session_keys if (
            (k.startswith("data_FO") or k.startswith("data_LT") or k.startswith("data_SR"))
            and not k.startswith("data_behav") and not k.startswith("data_raw")
        )
    ]
    valid_r_keys = [k for k in session_keys if k.startswith("data_beh") and not k.endswith("_epoch") or k.startswith("R_") ]
    valid_idx_keys = []

    if not valid_d_keys:
        st.warning(
            "⚠️ No aggregated data found. For across-subjects testing, you need to compute group-level features such as "
            "**Fractional Occupancy (FO)**, **Lifetime (LT)**, or **Switching Rate (SR)**.\n\n"
            "Please go to the HMM Outputs section and compute these before running this test."
        )
        valid_d_keys = [None]  # Force user to select "None" in dropdown
        st.stop()

elif test_type in ["Across-trials", "Across-sessions"]:
    valid_d_keys = [k for k in session_keys if k.startswith("gamma_epoch")]
    valid_r_keys = [k for k in session_keys if k.startswith("data_behav_epoch") or k.startswith("R_")]
    valid_idx_keys = [k for k in session_keys if k.startswith("indices_epoch")]

    if not valid_d_keys:
        st.warning(
            "⚠️ Epoch data not found. To run this test, you must first structure your data into **epochs**.\n\n"
            "➡️ Go to **fitting hmm**, choose **Configure loaded data for statistical analysis** and select **Restructure data into Epochs**."

        )
        valid_d_keys = [None]  # Force user to select "None" in dropdown
        st.stop()
    
elif test_type == "Across-state-visits":
    valid_d_keys = [k for k in session_keys if k.startswith("vpath") and not k.endswith("_epoch")]
    valid_r_keys = [k for k in session_keys if k.startswith("data_behav")and not k.endswith("_epoch") or k.startswith("R_")]
    valid_idx_keys = []  # not required
    if not valid_d_keys or st.session_state.get("vpath") is None:
        st.warning(
            "⚠️ No valid 'vpath' data found. Across-state-visits testing requires decoded state sequences of the Viterbi Path."
        )
        valid_d_keys = [None]
        st.stop()

# Matrix selection 
st.markdown("---")
st.subheader("Select data")

D_key = st.selectbox("Select D (brain data)", options=sorted(valid_d_keys) if valid_d_keys else [None])
R_key = st.selectbox("Select R (behaviour data)", options=sorted(valid_r_keys) if valid_r_keys else [None])


# Load data from selected keys 
D_data = st.session_state.get(D_key)
R_data = st.session_state.get(R_key)

if test_type in ["Across-trials", "Across-sessions"]:
    idx_key = st.selectbox("Select index list (session/trial)", options=sorted(valid_idx_keys) if valid_idx_keys else [None])
    idx_data = st.session_state.get(idx_key)
    # Display shapes
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("**D_data**")
        st.code(f"Shape: {D_data.shape}" if hasattr(D_data, "shape") else "Unavailable")

    with col2:
        st.write("**R_data**")
        if isinstance(R_data, list):
            try:
                R_data = np.concatenate(R_data, axis=0)
                shape_str = f"Shape: {R_data.shape}"
            except Exception:
                shape_str = "Invalid list"
        else:
            shape_str = f"Shape: {R_data.shape}" if hasattr(R_data, "shape") else "Unavailable"
        st.code(shape_str)

    with col3:
        st.write("**Indices**")
        if idx_data is not None:
            try:
                st.code(f"Shape: {np.array(idx_data).shape}")
            except Exception:
                st.code("Invalid format")
        else:
            st.code("None")
else:
    # Display shapes
    col1, col2, = st.columns(2)
    with col1:
        st.write("**D_data**")
        st.code(f"Shape: {D_data.shape}" if hasattr(D_data, "shape") else "Unavailable")

    with col2:
        st.write("**R_data**")
        if isinstance(R_data, list):
            try:
                R_data = np.concatenate(R_data, axis=0)
                shape_str = f"Shape: {R_data.shape}"
            except Exception:
                shape_str = "Invalid list"
        else:
            shape_str = f"Shape: {R_data.shape}" if hasattr(R_data, "shape") else "Unavailable"
        st.code(shape_str)

if D_data is None or R_data is None:
    st.error("❌ Missing data: D and/or R is not loaded. Please check your selections above.")
    st.stop()

if test_type == "Across-state-visits" or test_type == "Across-subjects":
    if len(D_data) != len(R_data):
        st.error(
            f"❌ Mismatched lengths: D has {len(D_data)} entries and R has {len(R_data)}. "
            "For this test, both must have the same number of entries."
        )
        st.stop()
elif test_type == "Across-trials" or test_type == "Across-sessions":
    if D_data.ndim == 3 and D_data.shape[1] !=len(R_data):
        st.error(
            f"❌ The number of trials in D_data ({D_data.shape[1]}) does not match the number of trials in R_data ({len(R_data)}). "
        )
        st.stop()

    elif D_data.ndim == 2 and len(D_data) != len(R_data):
        st.error(
            f"❌ The number of trials in D_data ({D_data.shape[1]}) does not match the number of trials in R_data ({len(R_data)}). "
        )
        st.stop()

confounds_data = st.session_state.get('confounds')
if st.checkbox("Add confounds"):
    D_matrix = st.session_state[D_key]

    is_large_file_confounds= st.toggle("File is larger than 200 MB", key="large_confound")
    if is_large_file_confounds:
        file_path_confound = st.text_input(
                    "Enter full path to confounds file with filename",
                    key="file_path_confound"
                )
    else:
        confound_file = st.file_uploader(
            "Upload indices data file",
            type=["csv", "npy", "mat", "fif", "pkl"],
            help="Supports .fif, .npy, .mat, .csv, .pkl",
            key="confound_uploader"
        )
        if confound_file:
            temp_path_confound = Path(tempfile.gettempdir()) / confound_file.name
            with open(temp_path_confound, "wb") as f:
                f.write(confound_file.getvalue())
            file_path_confound = temp_path_confound



    if st.button("Load confounds", key="load_confounds"):
        with st.spinner("Loading indices..."):
            data_confound = load_data(file_path_confound)
            is_valid, msg = validate_confounds(data_confound, D_matrix)
            if is_valid:
                st.success("✅ Confounds loaded and validated.")
                st.session_state.confounds = data_confound
            else:
                st.error(f"❌ Confounds invalid: {msg}")

dict_fam = None
if test_type == "Across-subjects" and st.checkbox("Load family structure file"):
    eb_file = st.file_uploader("Upload EB.csv", type=["csv"])
    if eb_file:
        dict_fam = {'file_location': eb_file}


# Test configuration
st.markdown("---")
st.subheader("Configure test parameters")
if test_type == "Across-state-visits":
    method = st.selectbox("Statistical method", ["multivariate", "univariate", "cca", "osr","osa"],
                                  help=(
            "Choose the statistical test to apply:\n\n"
            "- **multivariate**: test all variables together\n"
            "- **univariate**: test each variable independently\n"
            "- **cca**: Canonical Correlation Analysis\n"
            "- **osr**: One-state vs the rest\n"
            "- **osa**: One-state vs another"
        )
        )
else:
    method = st.selectbox("Statistical method", ["multivariate", "univariate", "cca"],         help=(
            "Choose the statistical test to apply:\n\n"
            "- **multivariate**: test all variables together\n"
            "- **univariate**: test each variable independently\n"
            "- **cca**: Canonical Correlation Analysis"
        )
        )
    if method =="cca":
        n_cca_components = st.number_input("Number of CC's", min_value=1, value=1, step=1,         help=(
            "Number of Canonical Components to extract during Canonical Correlation Analysis (CCA). "
            "Each component captures a pair of maximally correlated projections between the input (X) and outcome (Y) variables."
        ))

Nnull_samples = st.number_input("Number of samples", min_value=0, value=10000, step=1000,    help=(
               "Number of null samples to generate for statistical testing. "
        "These are drawn using either permutation or Monte Carlo resampling, depending on the test setup. "
        "More samples increase statistical precision but also increase runtime."
    ))
if method != "cca":
    pvalue_comb = st.selectbox("Combine p-values", [False, True, "across_rows", "across_columns"],         help=(
            "Choose how to combine p-values if your test produces a matrix of values:\n"
            "- *False*: Keep individual p-values.\n"
            "- *True*: Combine all p-values into a single summary p-value.\n"
            "- *across_rows*: Combine p-values across rows (e.g., multiple outcomes).\n"
            "- *across_columns*: Combine p-values across columns (e.g., multiple non-imaging features).\n\n"
        ))
else:
    pvalue_comb = None
FWER_correction = st.selectbox("FWER correction (maxT)", [False, True],     help=(
        "Applies Family-Wise Error Rate (FWER) correction using the max-T approach. "
        "This controls for multiple comparisons by comparing the observed test statistic "
        "to the maximum value from each null sample."
    ))

if test_type == "Across-state-visits" and method in ["osr", "osa"]:
    comparison_statistics = st.selectbox("Comparison statistic", ["mean", "median"])
    if method == "osr":
        state_comparison = st.selectbox(
            "Comparison logic:",
            ["larger", "smaller"],
            help="Compare the selected state to the average across other states."
        )

if test_type == "Across-subjects":
    permute_within_blocks = st.selectbox(
        "Permute within blocks", [False, True],
        help="Limit permutations to within predefined groups (e.g., subjects)."
    )
    permute_between_blocks = st.selectbox(
        "Permute between blocks", [False, True],
        help="Allow permutations between different predefined groups."
    )

detect_categorical = st.selectbox(
    "Automatically detect variable types in R_data?",
    options=[False, True],
    index=0,
    help=(
        "If True, each column in R_data will be examined:\n"
        "- Binary categorical → t-test\n"
        "- Multiclass categorical → MANOVA (multivariate) / ANOVA (univariate)\n"
        "- Continuous → F-regression (multivariate) / Pearson’s t-test (univariate)"
    )
)

category_limit = st.number_input(
    "Max unique values for categorical", min_value=1, value=10, step=1,
    help=(
            "Used to avoid misclassifying integer-valued continuous variables (like age) as categorical.\n"
            "If a column has more than this number of unique values, it will be treated as continuous."
        )
) if detect_categorical else 10



predictor_names = [s.strip() for s in st.text_area(
    "Predictor names (comma-separated)", "",
    help="Variables in D_data (match column order)."
).split(",") if s.strip()]

outcome_names = [s.strip() for s in st.text_area(
    "Outcome names (comma-separated)", "",
    help="Variables in R_data (match column order)."
).split(",") if s.strip()]

st.session_state.test_config = {
    # Common across all test types
    "method": method,
    "Nnull_samples": Nnull_samples,
    "combine_tests": pvalue_comb,
    "FWER_correction": FWER_correction,
    "predictor_names": predictor_names,
    "outcome_names": outcome_names,
    "detect_categorical": detect_categorical,
    "category_limit": category_limit,

    # Only for Across-subjects
    "dict_family": dict_fam if test_type == "Across-subjects" else None,
    "permute_within_blocks": permute_within_blocks if test_type == "Across-subjects" else None,
    "permute_between_blocks": permute_between_blocks if test_type == "Across-subjects" else None,

    # For subjects/trials/sessions
    "confounds": confounds_data if test_type in ["Across-subjects", "Across-trials", "Across-sessions"] else None,

    # Only for Across-state-visits
    "state_comparison": state_comparison if test_type == "Across-state-visits" else None,
    "comparison_statistics": comparison_statistics if test_type == "Across-state-visits" else None,

}

# Run test
if st.button("▶ Run Statistical Test"):
    st.session_state.run_test = True


# Guarded block that executes the test
if st.session_state.get("run_test", False):
    st.session_state.run_test = False  # Prevent rerun on next script execution
    with st.spinner("Running statistical test..."):
        try:
            config = {k: v for k, v in st.session_state.test_config.items() if v is not None}

            if test_type == "Across-subjects":
                result = statistics.test_across_subjects(D_data, R_data, **config)
            elif test_type == "Across-trials":
                result = statistics.test_across_trials(D_data, R_data, idx_data, **config)
            elif test_type == "Across-sessions":
                result = statistics.test_across_sessions(D_data, R_data, idx_data, **config)
            elif test_type == "Across-state-visits":
                result = statistics.test_across_state_visits(D_data, R_data, idx_data, **config)

            st.session_state.test_result = result
            st.success("✅ Test completed!")

        except Exception as e:
            st.error("❌ Test failed")
            st.exception(e)


# Always render the figure if it exists
if "test_result" in st.session_state:
    st.markdown("---")
    st.subheader("Save Results")
    _save_object_ui(st.session_state.test_result, default_filename=f"results_{test_type}.pkl".lower().replace("-", "_"), 
                    label="Save Results", 
                    key_prefix="save")

# Save and correct results
if "test_result" in st.session_state:
    st.markdown("---")
    st.subheader("Multiple Testing Correction")

    correction_option = st.radio("Select correction method", [
        "None",
        "Traditional (FDR, Bonferroni, etc.)",
        "MaxT",
        "Cluster-statistics"
    ])

    # Different options for correction  
    if correction_option == "Traditional (FDR, Bonferroni, etc.)":
        traditional_methods = {
            "Bonferroni": "bonferroni",
            "Benjamini–Hochberg": "fdr_bh",
            "Benjamini–Yekutieli": "fdr_by",
            "Two-stage BH": "fdr_tsbh",
            "Two-stage BY": "fdr_tsbky",
            "Sidak": "sidak",
            "Holm–Sidak": "holm-sidak",
            "Holm": "holm",
            "Simes–Hochberg": "simes-hochberg",
            "Hommel": "hommel",
        }
        methods_requiring_alpha = {"bonferroni", "sidak", "holm-sidak", "holm", "simes-hochberg", "hommel"}

        selected_label = st.selectbox("Correction method", list(traditional_methods.keys()))
        selected_method = traditional_methods[selected_label]

        # Conditionally show alpha input
        alpha = 0.05  # default fallback
        if selected_method in methods_requiring_alpha:
            alpha = st.number_input(
                "Alpha level", min_value=0.0, max_value=1.0, value=0.05,
                help="Significance threshold used for this correction method"
        )
    elif correction_option == "Cluster-statistics":
        alpha = st.number_input("Alpha level", min_value=0.0, max_value=1.0, value=0.05)

    if st.button("🔁 Apply Correction"):
        try:
            if correction_option == "Traditional (FDR, Bonferroni, etc.)":
                corrected_p = statistics.pval_correction(
                    result_dic=st.session_state.test_result,
                    method=selected_method,
                    alpha=alpha,
                    include_nan=True
                )
                label = selected_label

            elif correction_option == "MaxT":
                corrected_p = statistics.pval_FWER_correction(st.session_state.test_result)
                label = "MaxT"

            elif correction_option == "Cluster-statistics":
                corrected_p = statistics.pval_cluster_based_correction(
                    st.session_state.test_result,
                    alpha=alpha
                )
                label = "Cluster-statistics"

            else:
                st.info("ℹ️ No correction applied.")
                corrected_p = None
                label = "None"

            st.session_state.corrected_p = corrected_p
            st.session_state.correction_label = label
            st.success(f"✅ Applied correction: {label}")


        except Exception as e:
            st.error("❌ Correction failed")
            st.exception(e)

    # Save corrected p-values
    if "corrected_p" in st.session_state:
        st.markdown("---")
        st.subheader("Save corrected results")

        # Resolve filename from correction label
        filename = st.session_state.get("correction_label", "correction")
        filename = f"results_{filename}".lower().replace(" ", "_").replace("-", "_") + ".pkl"

        _save_object_ui(
            st.session_state.corrected_p,
            filename,
            label="Save Corrected Results",
            key_prefix="save_corrected_p"
    )
