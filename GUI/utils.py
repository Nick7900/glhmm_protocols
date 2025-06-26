import pandas as pd
import streamlit as st
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import scipy.io
from glhmm import statistics, utils, graphics
from importlib import reload
reload(statistics)
import tempfile
import re


@st.cache_resource
def cached_load_data(path_str):
    """Wrapper around load_data that caches the result based on the file path"""
    return load_data(path_str)

def create_index_table(indices):
    """Create styled dataframe for session indices"""
    df = pd.DataFrame(indices, columns=['Start', 'End'])
    df['Duration'] = df['End'] - df['Start']
    df['Session'] = df.index + 1
    # return df[['Session', 'Start', 'End', 'Duration']].style.background_gradient(
    #     subset=['Duration'],
    #     cmap='Blues'
    # )

    return df[['Session', 'Start', 'End', 'Duration']]



def load_data(path, mmap_mode=None):
    """Load a single file with memory mapping support"""
    try:
        path = Path(path)
        ext = path.suffix.lower()
        
        if ext == '.csv':
            return pd.read_csv(path, header=None).to_numpy()
        elif ext == '.npy':
            #return np.load(path, mmap_mode=mmap_mode or 'r')
            return np.load(path, mmap_mode=None)
        elif ext == '.mat':
            return next(iter(scipy.io.loadmat(path).values()))
        elif ext == '.pkl':
            with open(path, "rb") as f:
                return pickle.load(f)
        else:
            st.warning(f"Unsupported file type: {ext}")
            return None
    except Exception as e:
        st.error(f"❌ Failed to load {path.name}: {str(e)}")
        return None
    

def validate_confounds(confounds_array, D_matrix):
    if confounds_array is None:
        return False, "No data found in confounds."
    if not isinstance(confounds_array, np.ndarray):
        confounds_array = np.asarray(confounds_array)

    if D_matrix.ndim == 2:
        expected = D_matrix.shape[0]
        if confounds_array.shape[0] != expected:
            return False, f"Expected shape ({expected}, ...), got {confounds_array.shape}."
    elif D_matrix.ndim == 3:
        expected = D_matrix.shape[1]
        if confounds_array.shape[0] != expected:
            return False, f"Expected shape ({expected}, ...), got {confounds_array.shape}."
    else:
        return False, "Unsupported D_matrix dimension."
    
    return True, "Confounds valid and shape matches."


def same_shape_except_last(a, b):
    # Convert to numpy arrays with dtype=object to preserve nested structure
    a_array = np.array(a, dtype=object)
    b_array = np.array(b, dtype=object)

    # Get shapes
    try:
        shape_a = a_array.shape
        shape_b = b_array.shape
    except Exception:
        return False

    # Must have at least one dimension to compare
    if len(shape_a) < 1 or len(shape_b) < 1:
        return False

    # Compare all dimensions except the last
    return shape_a[:-1] == shape_b[:-1]


def same_shape_and_structure(a, b):
    # Convert lists to numpy arrays (if not already)
    a_array = np.array(a, dtype=object)
    b_array = np.array(b, dtype=object)
    
    # Compare shapes
    if a_array.shape != b_array.shape:
        return False
    
    # Check recursive structure (e.g., if it's a list of lists, etc.)
    def check_structure(x, y):
        if isinstance(x, (list, np.ndarray)) and isinstance(y, (list, np.ndarray)):
            if len(x) != len(y):
                return False
            return all(check_structure(xi, yi) for xi, yi in zip(x, y))
        return True  # Base case: leaf elements, don't care about values
    
    return check_structure(a, b)




def analysis_from_gamma(Gamma, vpath, indices, analysis_options, data_behav=None, fs_default=250):
    
    analysis_type = st.selectbox("Choose analysis type", analysis_options, key="selected_analysis_type")

    # Button to confirm selection
    if st.button("Configure data", key="confirm_analysis_type"):
        st.session_state.analysis_chosen = True

    # Show analysis config only if user confirmed
    if st.session_state.get("analysis_chosen", False):
        analysis_type = st.session_state["selected_analysis_type"]
        if analysis_type == "Fractional Occupancy":
            D_fo = utils.get_FO(Gamma, indices)
            st.session_state.data_FO = D_fo
            st.session_state.analysis_fig = get_cached_plot_fo(D_fo)
            st.write(f"Fractional Occupancy shape: {D_fo.shape}")

        elif analysis_type == "Switching Rate":
            D_SR = utils.get_switching_rate(Gamma, indices)
            st.session_state.data_SR = D_SR
            st.session_state.analysis_fig = get_cached_plot_switching_rate(D_SR)
            st.write(f"Switching Rate shape: {D_SR.shape}")

        elif analysis_type == "State Lifetime":
            LTmean, _, _ = utils.get_life_times(vpath, indices)
            st.session_state.data_LT = LTmean
            st.session_state.analysis_fig = get_cached_plot_lifetimes(LTmean)
            st.write(f"State Lifetime shape: {LTmean.shape}")


        elif analysis_type == "Restructure data into Epochs":
            if st.session_state.data_behav is None:
                st.error("❌ Behavioral data for each trial condition is required to configure epochs. Please load it on the **Load data** page.")
                st.stop()

            st.subheader("Choose method for epoching")

            epoch_method = st.radio(
                "Select how to structure the data into epochs:",
                options=["Use event markers", "Use session-wise trial indices"],
                help="Choose whether to extract epochs using event markers or using trial indices per session."
            )
        
            if epoch_method == "Use event markers":
                file_path_event = None  # Initialize to ensure defined for the warning check
                if "load_event" not in st.session_state:
                    st.session_state.load_event = False
                if "reconstruct_event" not in st.session_state:
                    st.session_state.reconstruct_event = False
                if "event_markers" not in st.session_state:
                    st.session_state.event_markers = None
                st.markdown("#### Upload event marker file")
                st.info(
                    "You need to upload a file that contains **event markers** used to define when each trial begins.\n\n"
                    "This file should contain a **list with one array per session**. Each array must have shape **(n_trials, 3)** and include:\n\n"
                    "- **Start time**: The position (in timepoints) where the trial begins in e.g. brain data. _(Used for epoching)_\n"
                    "- **Condition**: Set to `0` by default in this dataset. _(Not used here)_\n"
                    "- **Event code**: A number that can be used to describe the event type. _(Not used here)_\n\n"
                    "Only the first column (start time) is used for creating epochs, but the other columns are included to ensure compatibility with tools like MNE.\n\n"
                    "**Example:**\n"
                    "```\n"
                    "[array([[   30,    0,   5],\n"
                    "        [  280,    0,   5],\n"
                    "        [  530,    0,   5]])\n"
                    ",\n"
                    " array([[   20,    0,   5],\n"
                    "        [  270,    0,   5],\n"
                    "        [  520,    0,   5]])\n"
                    "]\n"
                    "```\n\n"
                    "**If you don’t have this file, ask the person who helped prepare your data.**"
                )
                is_large_file_event = st.toggle("File is larger than 200 MB", key="large_event")
                if is_large_file_event:
                    file_path_event = st.text_input(
                        "Path to event marker file:",
                        help="Should be a list of arrays, one per session, each with shape (n_trials, 3)",
                        key="file_path_event"
                    )
                else:
                    event_file = st.file_uploader(
                        "Upload event marker file",
                        type=["csv", "npy", "mat", "fif", "pkl"],
                        help="Should be a list of arrays, one per session, each with shape (n_trials, 3)",
                        key="event_uploader"
                    )
                    if event_file:
                        temp_path_event = Path(tempfile.gettempdir()) / event_file.name
                        with open(temp_path_event, "wb") as f:
                            f.write(event_file.getvalue())
                        file_path_event = temp_path_event
                
                if st.button("Load event markers", key="load_events"):
                    if not file_path_event:
                        st.error("❌ Please specify or upload a file with event markers before loading.")
                        st.stop()
                    else:
                
                        with st.spinner("Loading event markers..."):
                            event_markers = load_data(file_path_event)
                            st.session_state.event_markers = event_markers
                            st.session_state.load_event = True  # Prevent rerun on next script execution
                            # st.write(isinstance(event_markers, list) and all(
                            # isinstance(arr, np.ndarray) and arr.shape[1] >= 3 for arr in event_markers))


                if st.session_state.get("load_event", False):
                    if isinstance(st.session_state.get("event_markers"), list) and all(
                        isinstance(arr, np.ndarray) and arr.shape[1] >= 3 for arr in st.session_state.get("event_markers")
                    ):
                        event_markers =st.session_state.get("event_markers")
                        st.session_state.reconstruct_event = True
                        st.success("Event markers loaded successfully.")
                        st.write(f"✅ Loaded {len(event_markers)} sessions.")
                        st.write("📌 Preview of first session:")
                        df = pd.DataFrame(event_markers[0], columns=["Start", "Condition", "Code"]).head()
                        st.dataframe(df, use_container_width=True, hide_index=True)
                    else:
                        st.error("❌ Invalid format. Expecting a list of arrays with shape (n_trials, 3).")
                        st.session_state.reconstruct_event = False

                if st.session_state.get("reconstruct_event", False):
                    st.markdown("#### Reconstructing state time courses (Gamma) into epochs based on event markers.")
                    # Main configuration – compact 3-column layout
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        fs = st.number_input("Sampling frequency", value=1000, help="Original data sampling rate (e.g., 1000 Hz)")
                    with col2:
                        fs_target = st.number_input("Target sampling frequency", value=fs_default, help="Downsampled rate (e.g., 250 Hz)")
                    with col3:
                        epoch_window_tp = st.number_input("Epoch window (timepoints)", value=fs_default, help="Epoch length in timepoints")

                    st.session_state.fs = fs
                    st.session_state.fs_target = fs_target
                    st.session_state.epoch_window_tp = epoch_window_tp
                    # # Optional setting – separate full-width input
                    # ms_before_stimulus = st.number_input(
                    #     "Pre-stimulus offset (ms)",
                    #     value=0,
                    #     help="Time before stimulus to include in the epoch. Use 200 to capture 200 ms before each event."
                    # )
                    # Stop if required data is missing or invalid
                    #  Validate all required inputs and report specific issues 
                    errors = []

                    if Gamma is None or not hasattr(Gamma, "size") or Gamma.size == 0:
                        errors.append("❌ `Gamma` (state probabilities or Viterbi path) is missing or empty.")
                    if indices is None or not isinstance(indices, np.ndarray):
                        errors.append("❌ `Indices` (session or trial-wise boundaries) is missing or not a valid NumPy array.")
                    if data_behav is None:
                        errors.append("❌ `Behavioral data` is missing.")
                    if st.session_state.event_markers is False:
                        errors.append("❌ `Event markers` are missing or invalid.")
                    if errors:
                        st.error("### The following issues were found:\n" + "\n".join(errors))
                        st.stop()  
        
                #st.write("event_markers", st.session_state.get("event_markers", False))       
                #st.write(st.session_state.get("event_markers"))
                if st.session_state.get("event_markers") is not None:
                    if st.button("Configure data", key="configure_data_button"):
                        fs = st.session_state.fs
                        fs_target = st.session_state.fs_target
                        epoch_window_tp = st.session_state.epoch_window_tp
                        event_markers = st.session_state.event_markers 

                        gamma_epoch, idx_epoch, R_epoch = statistics.get_event_epochs(
                            Gamma, 
                            indices, 
                            data_behav, 
                            event_markers,
                            fs=fs, 
                            fs_target=fs_target, 
                            epoch_window_tp=epoch_window_tp,
                            #ms_before_stimulus=int(ms_before_stimulus)
                        )
                        st.session_state.gamma_epoch = gamma_epoch
                        st.session_state.indices_epoch = idx_epoch
                        st.session_state.data_behav_epoch = R_epoch

                        st.success(f"Epochs extracted: {gamma_epoch.shape}")
                        st.session_state.analysis_fig = None  # Clear old figure

                        all_conditions = np.unique(R_epoch)
                        if len(all_conditions) == 2:
                            st.session_state.analysis_fig = get_cached_plot_condition_difference(gamma_epoch, R_epoch)
                            pass

            elif epoch_method == "Use session-wise trial indices":

                st.markdown("### Load session-wise indices for restructuring Gamma data into epochs")
                st.info(
                    "You need to upload a file that tells the app which trials belong to which session.\n\n"
                    "Each row should contain two numbers: the starting trial and the ending trial for one session.\n"
                    "This means that the session includes all trials starting from the first number and up to (but not including) the second number.\n\n"
                    "**Example:**\n"
                    "```\n"
                    "[[   0,  250],\n"
                    " [ 250,  500],\n"
                    " [ 500,  800],\n"
                    " [ 800,  900],\n"
                    " [ 900, 1000]]\n"
                    "```\n"
                    "The last number (e.g., `1000`) must match the total number of trials in your behavioral data.\n\n"
                    "**If you don’t have this file, ask the person who helped prepare your data.**"
                )
                is_large_trial_file = st.toggle("File is larger than 200 MB", key="large_trial_file")
                if is_large_trial_file:
                    file_path_trial_idx = st.text_input("Path to trial indices file:", key="trial_idx_path")
                else:
                    trial_idx_file = st.file_uploader("Upload trial indices file", type=["csv", "npy", "mat", "fif", "pkl"], key="trial_idx_uploader")
                    if trial_idx_file:
                        temp_path_trial = Path(tempfile.gettempdir()) / trial_idx_file.name
                        with open(temp_path_trial, "wb") as f:
                            f.write(trial_idx_file.getvalue())
                        file_path_trial_idx = temp_path_trial

                # Load indices
                if st.button("Load trial indices", key="load_trial_idx"):
                    with st.spinner("Loading trial indices..."):
                        trial_indices = load_data(file_path_trial_idx)
                        if isinstance(trial_indices, list):
                            trial_indices = np.array(trial_indices)
                        if trial_indices.ndim != 2 or trial_indices.shape[1] != 2:
                            st.error("❌ Trial indices must be a 2D array of shape (n_sessions, 2).")
                            st.stop()

                        st.session_state.trial_indices = trial_indices
                        st.session_state.button_trial_indices = True
                        st.success(f"✅ Loaded {trial_indices.shape[0]} sessions.")
                        st.write(trial_indices)

                # Restructure Gamma using trial indices
                if st.session_state.get("button_trial_indices", False):
                    trial_indices = st.session_state.trial_indices
                    total_trials_in_r = len(st.session_state.data_behav)

                    if trial_indices[-1, 1] != total_trials_in_r:
                        st.error(f"❌ Trial indices end at {trial_indices[-1, 1]}, but your R_data has {total_trials_in_r} trials.")
                        st.stop()

                    n_sessions = len(trial_indices)
                    total_timepoints = Gamma.shape[0]
                    n_features = Gamma.shape[1]

                    if total_timepoints % n_sessions != 0:
                        st.error(f"Gamma has {total_timepoints} timepoints which is not divisible by the {n_sessions} sessions.")
                        st.stop()

                    n_timepoints = total_timepoints // n_sessions

                    if st.button("Configure data"):
                        Gamma_epoch = statistics.reconstruct_concatenated_to_3D(
                            Gamma,
                            n_timepoints=n_timepoints,
                            n_entities=n_sessions,
                            n_features=n_features,
                        )
                        st.session_state.gamma_epoch = Gamma_epoch
                        st.success(f"Gamma reshaped to {Gamma_epoch.shape}")

                        st.session_state.analysis_fig = get_cached_plot_condition_difference(gamma_epoch, R_epoch)
                        pass
                    


    # Always render the figure if it exists
    if "analysis_fig" in st.session_state and st.session_state.analysis_fig is not None:
        st.pyplot(st.session_state.analysis_fig)
        _save_figure_ui(st.session_state.analysis_fig)

        

@st.cache_resource
def get_cached_plot_condition_difference(gamma_epoch, R_epoch):
    return graphics.plot_condition_difference(
        gamma_epoch, R_epoch,
        condition_labels=("Condition (0)", "Condition (1)"),
        title="Average Probability and Difference",
        x_tick_min=0,
        x_tick_max=gamma_epoch.shape[0],
        num_x_ticks=5,
        xlabel="Time (ms)",
        figsize=(15, 5),
        return_fig=True
    )

@st.cache_resource
def get_cached_plot_fo(D_fo):
    return graphics.plot_FO(D_fo, xlabel="Sessions", width=1, figsize=(10, 5), return_fig=True)

@st.cache_resource
def get_cached_plot_switching_rate(D_SR):
    return graphics.plot_switching_rates(D_SR, xlabel="Sessions", width=1, figsize=(10, 5), return_fig=True)

@st.cache_resource
def get_cached_plot_lifetimes(LTmean):
    return graphics.plot_state_lifetimes(LTmean, xlabel="Sessions", width=1, figsize=(10, 5), return_fig=True)
    

def detect_state_type(data):
    if data.ndim == 2 and np.all((data >= 0) & (data <= 1)):
        row_sums = np.sum(data, axis=1)
        if np.allclose(row_sums, 1, atol=1e-2):
            return "Gamma"
    # Check if all values are whole numbers
    if np.all(np.mod(data, 1) == 0):
        return "Viterbi Path"
    return "Unknown"


def _save_figure_ui(fig, default_name="figure", key_prefix="fig"):
    with st.expander(f"💾 Save: {default_name}"):
        file_format = st.selectbox(
            "File format", 
            ["jpg", "png", "pdf"], 
            key=f"{key_prefix}_format"
        )
        filename = st.text_input(
            "Filename", 
            value=default_name, 
            key=f"{key_prefix}_filename"
        )
        if st.button("Save figure", key=f"{key_prefix}_button"):
            output_dir = Path.cwd()
            full_path = output_dir / f"{filename}.{file_format}"
            fig.savefig(full_path, format=file_format, bbox_inches="tight")
            st.success(f"✅ Figure saved as: `{full_path}`")


def _save_object_ui(obj, default_filename="results.pkl", label="Save Results", key_prefix="save"):
    with st.expander(label, expanded=False):
        default_dir = str(Path(__file__).resolve().parent.parent)
        save_path = st.text_input(f"{label} - Save directory", value=default_dir, key=f"{key_prefix}_path")
        file_name = st.text_input(f"{label} - File name", value=default_filename, key=f"{key_prefix}_filename")

        if st.button("💾 Save", key=f"{key_prefix}_button"):
            if obj is None:
                st.warning("⚠️ Nothing to save.")
                return
            try:
                path = Path(save_path).expanduser().resolve() / file_name
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, "wb") as f:
                    pickle.dump(obj, f)
                st.success(f"✅ Saved to `{path}`")
            except Exception as e:
                st.error("❌ Could not save")
                st.exception(e)



def check_indices_match_data_streamlit(data, indices):
    if isinstance(indices, np.ndarray) and indices.dtype == object:
        indices = list(indices)

    flat_indices = np.concatenate(indices)
    expected_length = data.shape[0]

    if flat_indices[-1] != expected_length:
        st.error(
            f"❌ Indices mismatch: last index is {flat_indices[-1]}, but expected {expected_length}."
        )
        st.stop()

    if np.any(flat_indices < 0) or np.any(flat_indices > expected_length):
        st.error("❌ Some indices are out of bounds of the data array.")
        st.stop()

    st.success("✅ Indices match the data.")




def ends_with_number(s):
    """
    Check if the input string ends with a number.

    Uses a regular expression to detect if the string ends with one or more digits.

    re.search(r'\d+$', s):

    Uses regular expressions to search for a pattern in string s.

    \d+ means one or more digits.

    $ means the end of the string.

    So \d+$ matches digits at the end of the string.
    """

    return bool(re.search(r'\d+$', s))

def conditional_numeric_sort(items):
    items = list(items)  # ensure it's a list, not numpy array
    if all(ends_with_number(item) for item in items):
        return sorted(items, key=lambda x: int(re.search(r'\d+$', x).group()))
    return items  # return original order,



def get_analysis_options(gamma, vpath, event_markers):
    options = []
    if gamma is not None:
        if gamma.ndim == 2:
            options += ["Fractional Occupancy", "Switching Rate", "Restructure into Trials"]
        elif gamma.ndim == 3 and event_markers is not None:
            options += ["Event-related epoch extraction"]
    if vpath is not None and vpath.ndim == 2:
        options.append("State Lifetime")
    return options