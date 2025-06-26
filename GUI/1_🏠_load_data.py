import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import pickle
import scipy.io
from pathlib import Path
import matplotlib.pyplot as plt
#import psutil
import tempfile
import shutil
from glhmm import glhmm, statistics
from PIL import Image
import tkinter as tk
from tkinter import filedialog
from utils import load_data, create_index_table, same_shape_and_structure, same_shape_except_last, check_indices_match_data_streamlit,cached_load_data

def browse_file():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename()
    return file_path

# Load logo
logo = Image.open(Path(__file__).parent / "logo" / "logo2_full.png")


# Title and logo
st.image(logo, width=250)
# Introduction
st.markdown("""
Welcome to the **GLHMM Streamlit Interface** – a graphical tool for exploring and analysing brain dynamics using the [Gaussian Linear Hidden Markov Model (GLHMM)](https://github.com/vidaurre/glhmm).

This tool is inspired by the setup in [this paper](https://arxiv.org/abs/2505.02541), and guides you through:

1. **Loading neuroimaging and behavioural data**
2. **Preprocessing the time-series data**
3. **Fitting a Hidden Markov Model**
4. **Running statistical tests**
5. **Visualising results**

Use the sidebar to navigate through each step. Start by loading your data below.
""")


# Initialize session state if not already set
if 'data_load' not in st.session_state:
    st.session_state.data_load = None
if 'indices' not in st.session_state:
    st.session_state.indices = None
if 'data_behav' not in st.session_state:
    st.session_state.data_behav = None
if 'event_markers' not in st.session_state:
    st.session_state.event_markers = None

# Clear data_raw after leaving loading page
if st.session_state.get("clear_data_raw", False):
    st.session_state.pop("data_raw", None)
    st.session_state.clear_data_raw = False

# Helper functions

def natural_keys(text):
    return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', text.name)]

# def memory_usage():
#     return psutil.virtual_memory().percent

def clear_temp_files():
    if hasattr(st.session_state, 'temp_dir') and st.session_state.temp_dir.exists():
        shutil.rmtree(st.session_state.temp_dir)
    st.session_state.temp_dir = Path(tempfile.mkdtemp())

@st.cache_resource
def init_temp_dir():
    return Path(tempfile.mkdtemp())


def load_multiple_files_large(folder_path, filter_string=""):
    """Memory-efficient loading of multiple large files"""
    files = [f for f in Path(folder_path).iterdir() if f.is_file()]
    if filter_string:
        files = [f for f in files if filter_string in f.name]
    files = sorted(files, key=natural_keys)
    
    if not files:
        st.error("❌ No matching files found!")
        return None, None
    
    # Initialize progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # First pass: determine total size and dtype
    status_text.text("Scanning files...")
    total_samples = 0
    dtype = None
    n_channels = None
    indices = []
    
    for i, f in enumerate(files):
        data = cached_load_data(str(f))
        if data is None:
            continue
            
        n_samp = data.shape[0]
        if len(data.shape) > 1:
            n_chan = data.shape[1]
        else:
            n_chan = 1
        data_dtype = data.dtype
            
        if dtype is None:
            dtype = data_dtype
            n_channels = n_chan
            
        indices.append([total_samples, total_samples + n_samp - 1])
        total_samples += n_samp
        progress_bar.progress((i + 1) / len(files))
    
    if not indices:
        return None, None
    
    # Create memory-mapped array
    temp_file = st.session_state.temp_dir / "temp_data.npy"
    if temp_file.exists():
        temp_file.unlink()
        
    memmap_shape = (total_samples, n_channels) if n_channels > 1 else (total_samples,)
    memmap_arr = np.lib.format.open_memmap(
        temp_file,
        mode='w+',
        dtype=dtype,
        shape=memmap_shape
    )
    
    # Second pass: load data
    status_text.text("Loading data...")
    current_sample = 0
    
    for i, f in enumerate(files):
        data = load_data(f)
        if data is None:
            continue
    
        chunk = data
            
        if len(chunk.shape) == 1:
            chunk = chunk.reshape(-1, 1)
            
        n_samples = chunk.shape[0]
        memmap_arr[current_sample:current_sample + n_samples] = chunk
        current_sample += n_samples
        progress_bar.progress((i + 1) / len(files))
    
    progress_bar.empty()
    status_text.empty()
    return memmap_arr, np.array(indices)

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


# UI components

def data_loading_section():
    st.header("📁 Data Loading")
    st.session_state.HMM_model = "Gaussian HMM"    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Brain Data")
        st.caption("ℹ️ Used as HMM input (`Y`) and D-matrix for statistical testing.")

        multi_mode = st.checkbox(
            "🔀 Load multiple files from folder", 
            value=False,
            key="multi_mode_brain"
        )

        if multi_mode:
            folder_path = st.text_input(
                "Folder path for brain data files:",
                help="e.g. folder with .fif, .npy, .csv files",
                key="brain_folder_path"
            )
            filter_string = st.text_input(
                "Filename filter (optional):",
                help="Only load files containing this string",
                key="brain_filter"
            )
        else:
            is_large_file = st.toggle("File is larger than 200 MB")
            file_path = None  # define before scope

            if is_large_file:
                file_path = st.text_input(
                    "Path to single brain data file:",
                    help="Supports .fif, .npy, .mat, .csv, .pkl",
                    key="brain_file_path"
                )
            else:
                brain_file = st.file_uploader(
                    "Upload brain data file",
                    type=["csv", "npy", "mat", "pkl"],
                    help="Supports .fif, .npy, .mat, .csv, .pkl"
                )
                if brain_file:
                    temp_path = Path(tempfile.gettempdir()) / brain_file.name
                    with open(temp_path, "wb") as f:
                        f.write(brain_file.getvalue())
                    file_path = temp_path
            

        # Load button
        if st.button("Load Brain Data", key="load_brain"):
            with st.spinner("Loading brain data..."):
                if multi_mode:
                    if folder_path:
                        data_load, indices = load_multiple_files_large(folder_path, filter_string)
                    else:
                        st.warning("Please enter a folder path.")
                        return
                else:
                    if file_path:
                        data_load = cached_load_data(str(file_path))
                        indices = (
                            statistics.get_indices_from_list(data_load)
                            if isinstance(data_load, list)
                            else np.array([[0, data_load.shape[0]]])
                        )
                    else:
                        st.warning("Please select or enter a valid file path.")
                        return

                if data_load is not None:
                    # Flatten if it's a list
                    final_data = np.concatenate(data_load, axis=0) if isinstance(data_load, list) else data_load

                    st.session_state.data_load = final_data
                    st.session_state.indices = indices

                    # Delete the raw version
                    st.session_state.pop("data_raw", None)

                    st.session_state.brain_data_loaded = True
                    st.success("✅ Brain data loaded successfully.")
                else:
                    st.error("❌ Failed to load brain data.")

        # Show persistent success message if already loaded
        elif st.session_state.get("brain_data_loaded"):
            st.success("✅ Brain data already loaded.")
                

    with col2:
        st.subheader("Behavioural Data")
        st.caption("ℹ️ Used as HMM input (`X`) and R-matrix for statistical testing.")

        multi_mode_behav = st.checkbox(
            "🔀 Load multiple files from folder", 
            value=False,
            key="multi_mode_behav"
        )

        file_path_r = None

        if multi_mode_behav:
            folder_path_r = st.text_input(
                "Folder path for behavioural data files:",
                help="e.g. folder with .fif, .npy, .mat, .csv, .pkl",
                key="behav_folder_path"
            )
            filter_string_r = st.text_input(
                "Filename filter (optional):",
                help="Only load files containing this string",
                key="behav_filter"
            )
        else:
            is_large_file_r = st.toggle("File is larger than 200 MB", key="large_behav")
            if is_large_file_r:
                file_path_r = st.text_input(
                    "Path to single behavioural data file:",
                    help="Supports .fif, .npy, .mat, .csv, .pkl",
                    key="behav_file_path"
                )
            else:
                behav_file = st.file_uploader(
                    "Upload behavioural data file",
                    type=["csv", "npy", "mat", "pkl"],
                    help="Supports .fif, .npy, .mat, .csv, .pkl",
                    key="behav_uploader"
                )
                if behav_file:
                    temp_path_r = Path(tempfile.gettempdir()) / behav_file.name
                    with open(temp_path_r, "wb") as f:
                        f.write(behav_file.getvalue())
                    file_path_r = temp_path_r

        if st.button("Load Behavioural Data", key="load_behav"):
            with st.spinner("Loading behavioural data..."):
                try:
                    if multi_mode_behav:
                        if not folder_path_r:
                            st.warning("Please enter a folder path.")
                            st.stop()
                        data_behav, _ = load_multiple_files_large(folder_path_r, filter_string_r)
                    else:
                        if not file_path_r:
                            st.warning("Please select or enter a valid file path.")
                            st.stop()
                        data_behav = load_data(file_path_r)

                    if data_behav is not None:
                        final_behav = np.concatenate(data_behav, axis=0) if isinstance(data_behav, list) else data_behav
                        st.session_state.data_behav = final_behav

                        # Check for shape match (optional)
                        D = st.session_state.get("data_load")
                        st.session_state.D_and_R_same = same_shape_except_last(D, final_behav) if D is not None else False
                        st.session_state.behav_data_loaded = True
                        st.success("✅ Behavioural data loaded successfully.")
                    else:
                        st.error("❌ Failed to load behavioural data.")
                except Exception as e:
                    st.error(f"❌ Loading failed: {str(e)}")

        elif st.session_state.get("behav_data_loaded"):
            st.success("✅ Behavioural data already loaded.")

    st.subheader("Index data (Optional)")
    st.info(
    "This file defines how your data is grouped into sessions or subjects. It's required for training an HMM and for some statistical tests.\n\n"
    "Each row should specify a range of trial indices that belong to the same group. For example:\n"
    "```\n"
    "[[     0,   5000],\n"
    " [   5000,   10000],\n"
    " [   10000,   15000]]\n"
    "```\n"
    "This means the first group includes trials 4999, the second group includes 5000–9999, and so on.\n\n"
    "The final index (e.g., 15000) must match the total number of datapoint in your loaded data."
    )

    use_idx = st.checkbox("📌 Load indices or generate indices", value=False)
    if use_idx:
        
        redefine = st.radio(
            "Do you want to redefine the session structure?",
            ["Load indices", "Manual generate indices"],
            key="get_indices_choice"
        )

        if redefine == "Manual generate indices":
            # Common data check
            if st.session_state.get("data_load") is None:
                st.error("❌ Please load data first from the **Data Loading** page.")
                st.stop()
                                            
            st.markdown("#### Define session/subject structure")
            st.info(
            "ℹ️ When generating indices manually, all subjects/sessions must have the **same number of timepoints**. "
            "If your data has subjects/sessions of unequal length, please generate the indices manually and load them using the 'Load indices' option."
        )
            
            n_subjects = st.number_input("Number of subjects/sessions", min_value=2, step=1, key="define_n_subjects")
            n_timepoints = st.number_input("Number of timepoints per subject/session", min_value=1, step=1, key="define_n_timepoints")

            data_length = st.session_state.data_load.shape[0] if st.session_state.get("data_load") is not None else None
            if st.button("Generate indices"):
                total_expected = n_subjects * n_timepoints
                data_length = st.session_state.data_load.shape[0] if st.session_state.get("data_load") is not None else None
 #

                
                if data_length is not None and total_expected != data_length:
                    if st.session_state.data_load.ndim==3:
                        st.error(
                            "❌ The shape of your data suggests it is structured as a 3D-matrix.\n\n"
                            "➡️ Please click **'Reorder and concatenate data into 2D'** on the **Data Summary** page before defining session indices manually."
                        )
                    else:
                        st.error(
                        f"❌ The expected total number of samples is **{total_expected}** ({n_subjects} × {n_timepoints}), "
                        f"but the actual data length is **{data_length}**.\n\n"
                        "This mismatch suggests that **your sessions may not all have the same number of timepoints**.\n\n"
                        "➡️ In this case, use the **'Load indices from file'** option on the **Data Loading** page to define session boundaries manually."
                        )
                        
                else:
                    idx_subject = statistics.get_indices_timestamp(n_timepoints, n_subjects)
                    st.session_state.indices = idx_subject
                    st.success(f"✅ New indices created with shape: {idx_subject.shape}")

        elif "Load indices":
            file_path_idx = None  # Initialize to ensure defined for the warning check
            is_large_file_idx = st.toggle("File is larger than 200 MB", key="large_idx")
            if is_large_file_idx:
                file_path_idx = st.text_input(
                    "Path to indices file:",
                    help="Should entries with shape (n_subjects or n_sessions, 2)",
                    key="file_path_idx"
                )
            else:
                idx_file = st.file_uploader(
                    "Upload indices data file",
                    type=["csv", "npy", "mat", "pkl"],
                    help="Supports .fif, .npy, .mat, .csv, .pkl",
                    key="idx_uploader"
                )
                if idx_file:
                    temp_path_idx = Path(tempfile.gettempdir()) / idx_file.name
                    with open(temp_path_idx, "wb") as f:
                        f.write(idx_file.getvalue())
                    file_path_idx = temp_path_idx


            if st.button("Load indices", key="load_idx"):
                if not file_path_idx:
                    st.warning("⚠️ Please specify or upload a trial indices file before loading.")
                else:
                    with st.spinner("Loading indices..."):
                        data_idx = load_data(file_path_idx)

                        if isinstance(data_idx, (list, np.ndarray)):
                            # If it's a list, convert to array
                            if isinstance(data_idx, list):
                                data_idx = np.array(data_idx)

                            if data_idx.ndim == 2 and data_idx.shape[1] == 2:
                                st.session_state.indices = data_idx
                                st.success("Indices loaded successfully.")
                                st.write(f"✅ Loaded {data_idx.shape[0]} sessions.")
                                st.write("📌 Preview of five sessions:")
                                df = pd.DataFrame(data_idx[:5, :], columns=["Start", "End"])
                                st.dataframe(df, use_container_width=True, hide_index=True)
                            else:
                                st.error("❌ Invalid format. Expecting array of shape (n_sessions, 2).")
                                st.stop()
                            check_indices_match_data_streamlit(st.session_state.data_load, st.session_state.indices)
                        else:
                            st.error("❌ Invalid file format for indices.")


            



def data_summary_section():
    if st.session_state.get("data_load") is None and st.session_state.get("data_behav") is None:
        return

    st.header("📊 Data Summary")

    if st.session_state.get("data_load") is not None:
        st.subheader("Brain Data")

        data_load = st.session_state.data_load
        indices = st.session_state.get("indices")

        if data_load.ndim == 3:
            st.warning("Your data is 3D. Expected shape is (Timepoints, Sessions, Features).")

            dim_order = st.selectbox(
                "Select the current axis order of your data:",
                options=[
                    "(Timepoints, Sessions, Features)",
                    "(Sessions, Timepoints, Features)",
                    "(Features, Sessions, Timepoints)"
                ],
                index=0
            )

            if not st.session_state.get("data_reshaped", False):
                reorder_button = st.button("Reorder and concatenate data into 2D")
                if reorder_button:
                    if dim_order == "(Sessions, Timepoints, Features)":
                        data_load = np.transpose(data_load, (1, 0, 2))
                    elif dim_order == "(Features, Sessions, Timepoints)":
                        data_load = np.transpose(data_load, (2, 1, 0))

                    if st.session_state.get("D_and_R_same", False):
                        st.session_state.data_load = statistics.get_concatenate_subjects(data_load)
                        st.session_state.data_behav = statistics.get_concatenate_subjects(
                            st.session_state.data_behav
                        )
                    else:
                        st.session_state.data_load = statistics.get_concatenate_subjects(data_load)

                    st.session_state.data_reshaped = True
                    st.success(f"Data has been reshaped to 2D: shape {st.session_state.data_load.shape}")
                else:
                    st.info("No reordering applied yet. Use the button above.")
            else:
                st.success("✅ Data has already been reshaped.")

        data_final = st.session_state.data_load
        indices = st.session_state.get("indices")

        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Shape:** {data_final.shape}")
            st.write(f"**Data type:** {data_final.dtype}")
            st.write(f"**Memory usage:** {data_final.nbytes / 1e9:.2f} GB")
        with col2:
            st.write(f"**Sessions loaded:** {len(indices) if indices is not None else '?'}")
            st.write(f"**Total samples:** {data_final.shape[0]:,}")
            st.write(f"**Channels:** {data_final.shape[1] if data_final.ndim == 2 else '?'}")

        if data_final.ndim == 2 and indices is not None:
            with st.expander("Session Indices", expanded=True):
                st.dataframe(create_index_table(indices), use_container_width=True, hide_index=True)
            if len(indices) == 1:
                st.warning("Only one session loaded. You need to load 'indices data' if you got multiple sessions.")

    if st.session_state.get("data_behav") is not None:
        st.subheader("Behavioural Data")

        if st.session_state.get("D_and_R_same"):
            data_final_behav = st.session_state.data_behav

            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Shape:** {data_final_behav.shape}")
                st.write(f"**Data type:** {data_final_behav.dtype}")
                st.write(f"**Memory usage:** {data_final_behav.nbytes / 1e9:.2f} GB")
            with col2:
                st.write(f"**Sessions loaded:** {len(indices) if indices is not None else '?'}")
                st.write(f"**Total samples:** {data_final_behav.shape[0]:,}")
                st.write(f"**Non-imaging features:** {data_final_behav.shape[1] if data_final_behav.ndim == 2 else '?'}")

            if data_final_behav.ndim == 2 and indices is not None:
                with st.expander("Session Indices", expanded=True):
                    st.dataframe(create_index_table(indices), use_container_width=True, hide_index=True)
        else:
            data_behav = st.session_state.data_behav
            if isinstance(data_behav, list):
                st.write(f"**List with a length of:** {len(data_behav)}")
                st.write("**Data type: List**")
            else:
                st.write(f"**Shape:** {data_behav.shape}")
                st.write(f"**Data type:** {data_behav.dtype}")

            with st.expander("Data Preview"):
                if isinstance(data_behav, (np.ndarray, pd.DataFrame)):
                    st.dataframe(pd.DataFrame(data_behav).head())
                else:
                    st.write(data_behav)

        

def sidebar_controls():
    st.sidebar.header("Controls")
    
    if st.sidebar.button("🧹 Clear Cache"):
        st.cache_data.clear()
        clear_temp_files()
        # Clear all session_state variables
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.toast("All cached data and session files have been cleared.", icon="🧹")

        st.rerun()
    
   # st.sidebar.metric("System Memory Usage", f"{memory_usage()}%")

# Main app

# Initialize temp directory
if 'temp_dir' not in st.session_state:
    st.session_state.temp_dir = init_temp_dir()

# Layout
sidebar_controls()
data_loading_section() # UI components
data_summary_section() # Data summary

# Cleanup on exit
import atexit
atexit.register(lambda: shutil.rmtree(st.session_state.temp_dir, ignore_errors=True))


