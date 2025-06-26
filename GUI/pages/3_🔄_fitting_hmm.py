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

st.set_page_config(page_title="HMM Training", page_icon="🔄")
st.title("🔄 Train Hidden Markov Model")
st.session_state.analysis_fig = None
# Reset training flags on fresh page load
if "ready_to_train" not in st.session_state:
    st.session_state.ready_to_train = False
if "training_triggered" not in st.session_state:
    st.session_state.training_triggered = False
if "already_trained_this_run" not in st.session_state:
    st.session_state.already_trained_this_run = False

# Prevent running if no data loaded
if st.session_state.get("data_load") is None:
    st.error("❌ No data was found in memory.")
    st.info("Please go to the **load date** before using this page.")
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
if proceed_with_mode:
    mode = st.radio(
        "Select a task:",
        (
            "Train a new HMM",
            "Load and decode data from existing HMM",
        )
    )

    # Filter valid data keys
    valid_load_keys = [k for k in st.session_state.keys() if k.startswith("data_") 
                       and not any(k.startswith(prefix) for prefix in ["data_reshape","data_type_detected","data_behav", "data_FO", "data_LT", "data_SR"])]

    ####### Load and decode existing HMM
    if mode == "Load and decode data from existing HMM":
        st.subheader("Load Existing HMM")
        hmm_file = st.file_uploader("Upload trained HMM (.pkl)", type=["pkl"])

        if hmm_file:
            try:
                hmm_obj = pickle.load(hmm_file)
                if isinstance(hmm_obj, glhmm.glhmm):
                    st.session_state.hmm_model = hmm_obj
                    st.write(hmm_obj.get_means())
                    st.warning("⚠️ Only the HMM model was loaded. No state time courses or decoded path.")
                elif isinstance(hmm_obj, dict):
                    st.session_state.hmm_model = hmm_obj.get("hmm")
                    st.session_state.vpath = hmm_obj.get("vpath")
                    st.session_state.gamma = hmm_obj.get("gamma")
                    st.session_state.xi = hmm_obj.get("xi")
                    st.session_state.fe = hmm_obj.get("fe")
                    st.info("Loaded model components:")
                    st.write(f"✅ HMM: {st.session_state.hmm_model is not None}")
                    st.write(f"✅ Gamma: {st.session_state.gamma is not None}")
                    st.write(f"✅ Viterbi path: {st.session_state.vpath is not None}")
                else:
                    st.error("Unsupported format.")
            except Exception as e:
                st.exception(e)
        if st.session_state.get("hmm_model") is not None:
            st.markdown("#### Select data to decode")

            valid_behav_keys = [k for k in st.session_state if k.startswith("data_behav")]
            valid_indices_keys = [k for k in st.session_state if k.startswith("indices")]

            D_and_R_same = st.session_state.get("D_and_R_same", False)
            if D_and_R_same:
                col1, col2, col3 = st.columns(3)
            else:
                col1, col2 = st.columns(2)

            with col1:
                Y_key = st.selectbox("Select Y data", valid_load_keys)
                Y = st.session_state[Y_key]
                st.write(f"Y shape: {Y.shape}")

            with col2:
                indices_key = st.selectbox("Select indices", valid_indices_keys)
                indices = st.session_state[indices_key]
                st.write(f"indices shape: {indices.shape}")

            if D_and_R_same:
                with col3:
                    valid_behav_keys.insert(0, None)
                    X_key = st.selectbox("Select X data (optional)", valid_behav_keys)
                    X = st.session_state[X_key] if X_key is not None else None
                    if X is not None:
                        st.write(f"X shape: {X.shape}")
            else:
                X = None

            decode_gamma = st.checkbox("Decode Gamma", value=False)
            decode_vpath = st.checkbox("Decode Viterbi path", value=False)

            ############# Need to update the code so it can pad the gamma values or viterbi path
            # if decode_gamma or decode_vpath:
            #     model = st.session_state.hmm_model
            #     with st.spinner("Decoding..."):
            #         if decode_gamma:
            #             gamma = model.decode(X=X, Y=Y, viterbi=False)
            #             st.session_state.gamma = gamma
            #             st.success("✅ Gamma decoded.")
            #             if st.session_state.HMM_model_pick == "TDE-HMM":
            #                 st.session_state.gamma_pad = auxiliary.padGamma(gamma, indices, {"embeddedlags": lags})
            #             elif st.session_state.HMM_model_pick == "HMM-MAR":
            #                 st.session_state.gamma_pad = auxiliary.padGamma(gamma, indices, {"order": autoregressive_order})
            #         if decode_vpath:
            #             vpath = model.decode(X=X, Y=Y, viterbi=True)
            #             st.session_state.vpath = vpath
            #             st.success("✅ Viterbi path decoded.")
            #             if st.session_state.HMM_model_pick == "TDE-HMM":
            #                 st.session_state.vpath_pad = auxiliary.padGamma(vpath, indices, {"embeddedlags": lags})
            #             elif st.session_state.HMM_model_pick == "HMM-MAR":
            #                 st.session_state.vpath_pad = auxiliary.padGamma(vpath, indices, {"order": autoregressive_order})

    ####### Train a new HMM
    elif mode == "Train a new HMM":
        st.markdown("---")
        st.markdown("#### Select HMM type")

        HMM_model_option = [st.session_state.HMM_model]
        if st.session_state.HMM_model == "Gaussian HMM":
            HMM_model_option += ["HMM-MAR", "TDE-HMM"]

        if len(HMM_model_option) == 1:
            analysis_type = st.selectbox("Selected HMM type", HMM_model_option)
        else:
            analysis_type = st.selectbox("Select HMM type", HMM_model_option)

        st.session_state.HMM_model_pick = analysis_type

        if analysis_type == "HMM-MAR":
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
        elif st.session_state.HMM_model_pick=="TDE-HMM":
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
            autoregressive_order = None
            lags = None
            pca = None



        st.markdown("### HMM Configuration")
        st.number_input("Number of states (K)", min_value=2, value=5, key="cfg_K",  
                        help="Number of hidden states in the HMM. Each state models a distinct pattern in the data.")
        st.selectbox("Covariance type", ["full","shareddiag", "diag", "sharedfull", "identity"], key="cfg_covtype",     
                        help=(
        "**Covariance structure of the Gaussian states:**\n"
        "- **full**: Each state has its own full covariance matrix.\n"
        "- **diag**: Each state has its own diagonal covariance.\n"
        "- **shareddiag**: All states share the same diagonal covariance.\n"
        "- **sharedfull**: All states share the same full covariance.\n"
        "- **identity**: Covariance is fixed to identity for all states."
        ))
        st.selectbox("Mean model", ["no","state", "shared"], key="cfg_model_mean",     
                        help=(
        "How the means of the Gaussian states are modeled:\n"
        "- **no**: No mean is estimated (assumed zero).\n"
        "- **state**: Each state has its own mean.\n"
        "- **shared**: All states share the same mean."
            ))
        st.selectbox("Beta model", ["no","state", "shared"], key="cfg_model_beta",     
                        help=(
        "How regression effects (betas) are modeled:\n"
        "- **no**: No regression effect.\n"
        "- **state**: Each state has its own beta.\n"
        "- **shared**: A single beta is shared across states."
                ))
        st.number_input("Dirichlet diagonal prior", value=10.0, key="cfg_dirichlet_diag", 
                        help="Controls the prior on the transition matrix. Higher values encourage more uniform transitions.")
        st.number_input("Max iterations", min_value=10, value=100, step=10, key="cfg_n_iter", help="Maximum number of EM iterations during HMM training.")


        col_vtick, col_save = st.columns(2)
        with col_vtick:
            st.checkbox("📌 Decode Viterbi path", value=False, key="cfg_decode")
        with col_save:
            st.checkbox("💾 Save outputs after training", value=True, key="save_values")
        if st.session_state.save_values:
            st.markdown("#### Select which variables to save:")
            if st.session_state.cfg_decode:
                col_HMM, col_Gamma, col_XI, col_FE, col_vpath= st.columns(5)
                with col_HMM:
                    st.checkbox("Save HMM model", value=True, key="save_model")
                with col_Gamma:
                    st.checkbox("Save Gamma", value=True, key="save_gamma")
                with col_XI:
                    st.checkbox("Save Xi", value=True, key="save_xi")
                with col_FE:
                    st.checkbox("Save Free Energy", value=True, key="save_fe")
                with col_vpath:
                    st.checkbox("Save Viterbi path", value=st.session_state.cfg_decode, key="save_vpath")
            else:
                col_HMM, col_Gamma, col_XI, col_FE= st.columns(4)
                with col_HMM:
                    st.checkbox("Save HMM model", value=True, key="save_model")
                with col_Gamma:
                    st.checkbox("Save Gamma", value=True, key="save_gamma")
                with col_XI:
                    st.checkbox("Save Xi", value=True, key="save_xi")
                with col_FE:
                    st.checkbox("Save Free Energy", value=True, key="save_fe")

            st.text_input("Folder name for saving results", value="HMM_results", key="hmm_save_folder")
            st.markdown(
                "> 💡 **Note:** It's a good idea to save selected values to disk. "
                "If your Streamlit app is idle for too long or the page reloads, all unsaved session data will be lost. "
                "You can later reload saved files for further analysis."
            )


        st.markdown("#### Select variables")

        if st.session_state.get("D_and_R_same", False):
            col1, col2, col3 = st.columns(3)
            with col1:
                Y_key = st.selectbox(f"Select Y data", options=valid_load_keys)
                Y = st.session_state[Y_key]
                detected_type_D = detect_state_type(Y)
                st.write(f"Auto-detected type: `{detected_type_D}`")
            with col2:
                valid_behav_keys = [k for k in st.session_state.keys() if k.startswith("data_behav")]
                valid_behav_keys.insert(0, None)
                X_key = st.selectbox("Select X data (for GLHMM, optional)", options=valid_behav_keys)
                X = st.session_state[X_key] if X_key is not None else None
                detected_type_B = detect_state_type(X)
                st.write(f"Auto-detected type: `{detected_type_B}`")
            with col3:
                valid_indices_keys = [k for k in st.session_state.keys() if k.startswith("indices")]
                indices_key = st.selectbox("Select indices", options=valid_indices_keys)
                indices = st.session_state[indices_key]

            st.write(f"Y data shape: {Y.shape}")
            if X is not None:
                st.write(f"X data shape: {X.shape}")
                if Y.shape[0] != X.shape[0]:
                    st.error("❌ Y and X must have the same shape to fit a GLHMM model.")
                    st.stop()
            st.write(f"Number of sessions/subjects: {len(indices)}")


        else:
            col1, col2 = st.columns(2)
            X = None
            with col1:
                Y_key = st.selectbox(f"Select Y data", options=valid_load_keys)
                Y = st.session_state[Y_key]
                st.write(f"Y data shape: {Y.shape}")
                
            with col2:
                valid_indices_keys = [k for k in st.session_state.keys() if k.startswith("indices")]
                indices_key = st.selectbox("Select indices", options=valid_indices_keys)
                indices = st.session_state[indices_key]
                st.write(f"indices data shape: {indices.shape}")

            if X is not None:
                st.write(f"X data shape: {X.shape}")
                if Y.shape[0] != X.shape[0]:
                    st.error("❌ Y and X must have the same shape to fit a GLHMM model.")
                    st.stop()
            st.write(f"Number of sessions/subjects: {len(indices)}")

            detected_type_D = detect_state_type(Y)
            st.write(f"Auto-detected type: `{detected_type_D}`")

        # Confirm data
        if st.button("Confirm data selection"):
            if detected_type_D in ["Gamma", "Viterbi Path"]:
                st.error(
                    "❌ Cannot train a new HMM using decoded outputs like Gamma or Viterbi Path.\n\n"
                    "✅ Since you have already decoded these state time courses, you can skip training and directly use the option:\n\n"
                    "➡️ **Configure loaded data for statistical analysis** (third option above)."
                
                )
                st.stop()
            if X is not None:
                st.info("You have selected both X and Y variables. The Gaussian Linear HMM (GLHMM) will be used.")
                # Auto-embed if required
            if analysis_type == "TDE-HMM":
                with st.spinner("Embedding data for TDE-HMM..."):
                    X, Y, _ = preproc.build_data_tde(Y, indices, lags=lags, pca=pca)
            elif analysis_type == "HMM-MAR":
                with st.spinner("Embedding data for HMM-MAR..."):
                    X, Y, indices, _ = preproc.build_data_autoregressive(Y, indices, autoregressive_order)
            st.session_state.embedded_X = X
            st.session_state.embedded_Y = Y
            st.session_state.embedded_indices = indices
            st.session_state.ready_to_train = True


        # Reset training flag on fresh page load
        if "already_trained_this_run" not in st.session_state:
            st.session_state.already_trained_this_run = False

        if st.session_state.ready_to_train:
            st.markdown(f"#### Train a {st.session_state.HMM_model_pick}") 
            if st.button("▶Train HMM"):
                st.session_state.training_triggered = True
                st.session_state.hmm_trained = False
                st.session_state.already_trained_this_run = False
                

            if st.session_state.get("training_triggered", False) and not st.session_state.get("already_trained_this_run", False):
            
                with st.spinner("Training HMM..."):
                    st.session_state.already_trained_this_run = True  # Prevent retraining on script rerun
                    try:
       
                        preproclogY = st.session_state.get("preproc_log", None) if Y_key.startswith("data_preproc") else None
                        model = glhmm.glhmm(
                            K=st.session_state.cfg_K,
                            covtype=st.session_state.cfg_covtype,
                            model_mean=st.session_state.cfg_model_mean,
                            model_beta=st.session_state.cfg_model_beta,
                            dirichlet_diag=st.session_state.cfg_dirichlet_diag,
                            preproclogY=preproclogY
                        )

                        Gamma, Xi, FE = model.train(X=X, Y=np.asarray(Y, dtype=np.float32), indices=indices)

           

                        st.success("✅ Training complete.")
                        st.session_state.hmm_trained = True
                        st.session_state.training_triggered = False
                        st.session_state.hmm_model = model
                        st.session_state.gamma = Gamma
                        st.session_state.xi = Xi
                        st.session_state.fe = FE
                        st.session_state.button_config_figures = True
                        if st.session_state.HMM_model_pick == "TDE-HMM":
                            st.session_state.gamma_pad = auxiliary.padGamma(Gamma, indices, {"embeddedlags": lags})
                        elif st.session_state.HMM_model_pick == "HMM-MAR":
                            st.session_state.gamma_pad = auxiliary.padGamma(Gamma, indices, {"order": autoregressive_order})

                        if st.session_state.cfg_decode:
                            vpath = model.decode(X=X, Y=Y, viterbi=True)
                            st.session_state.vpath = vpath

                            if st.session_state.HMM_model_pick == "TDE-HMM":
                                st.session_state.vpath_pad = auxiliary.padGamma(vpath, indices, {"embeddedlags": lags})
                            elif st.session_state.HMM_model_pick == "HMM-MAR":
                                st.session_state.vpath_pad = auxiliary.padGamma(vpath, indices, {"order": autoregressive_order})
                            st.success("Viterbi path decoded")
                        # Save selected values
                        if st.session_state.save_values:
                            save_dir = Path(__file__).resolve().parent.parent / st.session_state.hmm_save_folder
                            save_dir.mkdir(exist_ok=True)

                            if st.session_state.save_model:
                                with open(save_dir / "hmm_model.pkl", "wb") as f:
                                    pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)

                            if st.session_state.save_gamma:
                                np.save(save_dir / "Gamma.npy", Gamma)
                            if st.session_state.save_xi:
                                np.save(save_dir / "Xi.npy", Xi)
                            if st.session_state.save_fe:
                                np.save(save_dir / "FE.npy", FE)
                            if st.session_state.get("save_vpath", None) and "vpath" in st.session_state:
                                np.save(save_dir / "vpath.npy", st.session_state.vpath)
                            st.success(f"✅ Saved selected values to: `{save_dir}`")
                            
                            
                    except Exception as e:
                        st.error("❌ Failed to train HMM")
                        st.session_state.training_triggered = False
                        st.session_state.hmm_trained = False
                        st.exception(e)

