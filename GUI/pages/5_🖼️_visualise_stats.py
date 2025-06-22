import streamlit as st
import numpy as np
from glhmm import graphics
from utils import _save_figure_ui,ends_with_number,conditional_numeric_sort
from importlib import reload
reload(graphics)

st.set_page_config(page_title="Visualize Test Results", page_icon="🖼️")
st.title("🖼️ Visualize Test Results")

st.markdown("---")
st.subheader("Select P-Value Source")

# Get data
pval_uncorrected = st.session_state.get("test_result", {}).get("pval")
corrected_p = st.session_state.get("corrected_p")

if corrected_p is not None:
    pval_corrected = corrected_p[0]
else:
    st.warning("No corrected p-values found in session state.")
    pval_corrected = None

# If no p-values at all, show an error and stop execution
if pval_uncorrected is None and pval_corrected is None:
    st.error("❌ No p-values available. Please run a statistical test first from the Statistical Tests page.")
    st.stop()

n_timepoints = st.session_state.get("test_result", {}).get("test_summary", {}).get("Timepoints", None)

# Get unique predictors and outcomes
predictors = np.unique(st.session_state.get("test_result", {}).get("test_summary", {}).get("Predictor", []))
outcomes = np.unique(st.session_state.get("test_result", {}).get("test_summary", {}).get("Outcome", []))
predictors = conditional_numeric_sort(predictors)
outcomes = conditional_numeric_sort(outcomes)

# User options
plot_uncorrected = st.checkbox("Show uncorrected p-values", value=True)
plot_corrected = st.checkbox("Show corrected p-values", value=False)

# Sanity check
if not plot_uncorrected and not plot_corrected:
    st.info("Please select at least one p-value type to plot.")
    st.stop()

if plot_uncorrected and pval_uncorrected is None:
    st.warning("⚠️ No uncorrected p-values found.")
    plot_uncorrected = False

if plot_corrected and pval_corrected is None:
    st.warning("⚠️ No corrected p-values found.")
    plot_corrected = False

if not plot_uncorrected and not plot_corrected:
    st.stop()

# Helper
def interpret_shape(pval, n_timepoints):
    shape = np.array(pval).shape
    if len(shape) == 3:
        return "3D"
    elif len(shape) == 2:
        if n_timepoints!=1 and n_timepoints and shape[0] == n_timepoints:
            return "2D_time"
        elif 1 in shape:
            return "2D_row"
        else:
            return "2D_matrix"
    elif len(shape) == 1:
        return "1D"
    return "Unsupported"

# Alpha level
alpha = st.number_input(f"Significance threshold (alpha)", min_value=0.001, value=0.050, step=0.001, max_value=1.0)   


# Unified display logic
def plot_pval(label, pval_data, n_timepoints):

    shape_type = interpret_shape(pval_data, n_timepoints)
    options = []
    if shape_type in ["1D", "2D_row"]:
        options = ["Bar plot", "Matrix"]
    elif shape_type == "2D_matrix":
        options = ["Matrix"]
    elif shape_type == "2D_time":
        options = ["Over time", "Matrix"]
    elif shape_type == "3D":
        options = ["3D: Over time"]
    else:
        st.warning(f"Unsupported shape: {np.array(pval_data).shape}")
        return

    selected_plot = st.selectbox(f"Select plot type for {label}", options)
    fig = None
    x_tick_min = None
    x_tick_max = None
    num_x_ticks = None
    st.markdown(f"### {label} P-values")

    # Common user inputs
    title = st.text_input(f"Title for {label}", value=f"{label} P-values")

    if shape_type in ["2D_time", "3D"]:
        if selected_plot == "Over time":
            xlabel = st.text_input(f"X-axis label", value="Time points", key = f'xlabel_{selected_plot}')

            col_width, col_height = st.columns(2)
            with col_width:
                fig_width = st.number_input("Width", min_value=1.0, value=9.0, step=0.5,
                                            help="Width of the figure in inches")
            with col_height:
                fig_height = st.number_input("Height", min_value=1.0, value=3.0, step=0.5,
                                            help="Height of the figure in inches")
            figsize = (fig_width, fig_height)

            col_tickbox, col_min, col_max, col_x_ticks= st.columns(4)

            with col_tickbox:
                use_tick_bounds = st.checkbox(f"Manually set x-tick min/max for {label}", value=False)

            if use_tick_bounds:
                with col_min:
                    x_tick_min = st.number_input(f"x_tick_min", value=0, step=1, key=f"xmin_{selected_plot}")
                with col_max:
                    x_tick_max = st.number_input(f"x_tick_max", value=1000, step=1, key=f"xmax_{selected_plot}")
                with col_x_ticks:
                    num_x_ticks = st.number_input(f"num_x_ticks", value=5, step=1, key=f"xtick_{selected_plot}")

        else:
            col1, col2 = st.columns(2)
            with col1:
                xlabel = st.text_input(f"X-axis label", value="Time points", key = f'xlabel_{selected_plot}')
            with col2:
                ylabel = st.text_input(f"Y-axis label", value="", key = f'ylabel_{selected_plot}')


            col_width, col_height = st.columns(2)
            with col_width:
                fig_width = st.number_input("Width", min_value=1.0, value=9.0, step=0.5,
                                            help="Width of the figure in inches", key = f'fig_width_{label}')
            with col_height:
                fig_height = st.number_input("Height", min_value=1.0, value=3.0, step=0.5,
                                            help="Height of the figure in inches", key = f'fig_width_{label}')
            figsize = (fig_width, fig_height)

            col_tickbox, col_min, col_max, col_x_ticks = st.columns(4)

            with col_tickbox:
                use_tick_bounds = st.checkbox(f"Manually set x-tick min/max for {label}", value=False)
            if use_tick_bounds:
                with col_min:
                    x_tick_min = st.number_input(f"x_tick_min for", value=0, step=1, key=f"xmin_{selected_plot}")
                with col_max:
                    x_tick_max = st.number_input(f"x_tick_max for", value=1000, step=1, key=f"xmax_{selected_plot}")
                with col_x_ticks:
                    num_x_ticks = st.number_input(f"num_x_ticks", value=5, step=1, key=f"xtick_{selected_plot}")
                    
            
    elif shape_type in ["1D", "2D_row"]:
        if selected_plot == "Bar plot":
            pval_text_height_same = st.checkbox(
                "Align p-value text at same height",
                value=True,
                help=(
                    "If checked, all p-value labels will be plotted at the same height above the bars. "
                    "If unchecked, each label will be positioned based on the height of its corresponding bar."
                ), 
                key = f"pval_height_{label}"
            )
            col_width, col_height = st.columns(2)
            with col_width:
                fig_width = st.number_input("Width", min_value=1.0, value=12.0, step=0.5,
                                            help="Width of the figure in inches", key = f'fig_width_{label}')
            with col_height:
                fig_height = st.number_input("Height", min_value=1.0, value=4.0, step=0.5,
                                            help="Height of the figure in inches", key = f'fig_height_{label}')
            figsize = (fig_width, fig_height)

            if st.session_state.get("test_result")["combine_tests"]=="across_columns":
                xticklabels_input = st.text_area(
                    "X-axis labels", 
                    ", ".join(predictors),
                    help="Comma-separated names of predictors (used for x-axis labels)."
                )
                xticklabels = list(np.array([s.strip() for s in xticklabels_input.split(",") if s.strip()]))
            else:
                xticklabels_input = st.text_area(
                    "X-axis labels", 
                    ", ".join(outcomes),
                    help="Comma-separated names of predictors (used for x-axis labels).", 
                    key=f"xticklabels_input_{label}"
                )
                xticklabels = list(np.array([s.strip() for s in xticklabels_input.split(",") if s.strip()]))
            
        elif selected_plot == "Matrix":  

            col_width, col_height = st.columns(2)
            with col_width:
                fig_width = st.number_input("Width", min_value=1.0, value=9.0, step=0.5,
                                            help="Width of the figure in inches", key = f'fig_width_{label}')
            with col_height:
                fig_height = st.number_input("Height", min_value=1.0, value=4.0, step=0.5,
                                            help="Height of the figure in inches", key = f'fig_height_{label}')
            figsize = (fig_width, fig_height)

            xticklabels_input = st.text_area(
                "X-axis labels", 
                ", ".join(outcomes),
                help="Comma-separated names of predictors (used for x-axis labels).", 
                key=f"xticklabels_input_{label}"
            )
            xticklabels = list(np.array([s.strip() for s in xticklabels_input.split(",") if s.strip()]))
            yticklabels = None
            num_y_ticks = 1
            xlabel= ""
            ylabel =""


    elif shape_type in ["2D_matrix"]:
            col1, col2 = st.columns(2)
            with col1:
                xlabel = st.text_input(f"X-axis label", value="", key = f'xlabel_{selected_plot}')
            with col2:
                ylabel = st.text_input(f"Y-axis label", value="", key = f'ylabel_{selected_plot}')


            col_width, col_height = st.columns(2)
            with col_width:
                fig_width = st.number_input("Width", min_value=1.0, value=9.0, step=0.5,
                                            help="Width of the figure in inches", key = f'fig_width_{label}')
            with col_height:
                fig_height = st.number_input("Height", min_value=1.0, value=5.0, step=0.5,
                                            help="Height of the figure in inches", key = f'fig_height_{label}')
            figsize = (fig_width, fig_height)

            col_tickbox, col_min, col_max, col_x_ticks = st.columns(4)


            # X-axis labels
            xticklabels_input = st.text_area(
                "X-axis labels", 
                ", ".join(outcomes),
                                        help=(
        "Enter a comma-separated list (e.g., 'Alpha, Beta, Gamma'), or a single word (e.g., 'Condition') "
        "to auto-number labels (e.g., 'Condition 1', 'Condition 2', ...). "
        "Leave empty to use default labels.\n\n"
        "**Press Enter or click outside the box to apply changes.**"
    )
            )
            

            if "," in xticklabels_input:
                # Multiple labels: convert to list
                xticklabels = [s.strip() for s in xticklabels_input.split(",") if s.strip()]
                if not xticklabels:
                    xticklabels = None
            else:
                # Single label string (no commas)
                xticklabels = xticklabels_input.strip() if xticklabels_input.strip() else None

            # Y-axis labels
            yticklabels_input = st.text_area(
                "Y-axis labels", 
                ", ".join(predictors),
                        help=(
        "Enter a comma-separated list (e.g., 'Alpha, Beta, Gamma'), or a single word (e.g., 'Condition') "
        "to auto-number labels (e.g., 'Condition 1', 'Condition 2', ...). "
        "Leave empty to use default labels.\n\n"
        "**Press Enter or click outside the box to apply changes.**"
    )
            )

            if "," in yticklabels_input:
                yticklabels = [s.strip() for s in yticklabels_input.split(",") if s.strip()]
                if not yticklabels:
                    yticklabels = None
            else:
                yticklabels = yticklabels_input.strip() if yticklabels_input.strip() else None
            num_y_ticks = None



    if selected_plot == "Bar plot":
        fig = graphics.plot_p_values_bar(pval_data, 
                                         title_text=f"{label} p-values", 
                                         alpha=alpha, 
                                         return_fig=True,
                                         pval_text_height_same=pval_text_height_same,
                                         xticklabels=xticklabels,
                                         figsize=figsize
                                         )

    elif selected_plot == "Matrix":
        annot = st.checkbox(f"Annotate values ({label})", value=False)
        if shape_type in ["2D_time", "3D"]:
            pval = np.array(pval_data).T
        else:
            pval = np.array(pval_data)


        fig = graphics.plot_p_value_matrix(pval,
            title_text=title,
            xticklabels=xticklabels,
            yticklabels =yticklabels,
            xlabel=xlabel,
            ylabel=ylabel, 
            alpha=alpha,
            annot=annot,
            xlabel_rotation=45,
            return_fig=True, 
            x_tick_min=x_tick_min, 
            x_tick_max=x_tick_max, 
            num_y_ticks = num_y_ticks,
            num_x_ticks=num_x_ticks,
            figsize=figsize)

    elif selected_plot == "Over time":
        if pval_data.shape[1] == 1:
            feature_idx = 0
        else:
            feature_idx = st.number_input(f"Select feature index for {label}", 0, pval_data.shape[1] - 1, key=f"feat_idx_{label}")
        fig = graphics.plot_p_values_over_time(
            np.array(pval_data)[:, feature_idx],
            title_text=title,
            xlabel=xlabel,
            alpha=alpha,
            x_tick_min=x_tick_min, 
            x_tick_max=x_tick_max, 
            num_x_ticks=num_x_ticks,
            return_fig=True,
            figsize=figsize
        )
    

    elif selected_plot == "3D: Over time":
        filter_type = st.radio(
            "Plot type",
            options=["Line-plot", "Matrix"],
            index=0
        )

        if filter_type == "Matrix":
            behav_idx = st.number_input(f"Select behavioural index ({label})", value=0, min_value=0, max_value=np.array(pval_data).shape[2]-1)
            annot = st.checkbox(f"Annotate values ({label})", value=False)
            fig = graphics.plot_p_value_matrix(
                np.array(pval_data)[:, :, behav_idx].T,
                title_text=title,
                xlabel=xlabel,
                ylabel=ylabel,
                alpha=alpha,
                return_fig=True,
                x_tick_min=x_tick_min, 
                x_tick_max=x_tick_max, 
                num_x_ticks=num_x_ticks,
                annot=annot,
                xlabel_rotation=45,
                figsize=figsize)
            
        elif filter_type == "Line-plot":
            feature_idx = st.number_input(f"Select feature index for {label}", 0, pval_data.shape[1] - 1, key=f"feat_idx_{label}")
            behav_idx = st.number_input(f"Select behavioural index ({label})", value=0, min_value=0, max_value=np.array(pval_data).shape[2]-1)
            fig = graphics.plot_p_values_over_time(
                np.array(pval_data)[:, feature_idx, behav_idx],
                title_text=title,
                xlabel=xlabel,
                alpha=alpha,
                x_tick_min=x_tick_min, 
                x_tick_max=x_tick_max, 
                num_x_ticks=num_x_ticks,
                return_fig=True,
                figsize=figsize
            )

    if fig is not None:
        st.pyplot(fig)
        _save_figure_ui(fig, default_name=f"{label.lower().replace(' ', '_')}_pvalues", key_prefix=label.lower())

    st.markdown("### Plot Null Sample Distributions")

    # Check dimensionality of p-values
    if pval_data is None or np.ndim(pval_data) != 1:
        st.info(
            "ℹ️ Permutation distribution plots are only available for multivariate tests or where test combination, "
            "where p-values form a single vector (not a matrix). This avoids overwhelming the display with too many plots."
        )
        
    else:
        plot_mode = st.radio(
            "Choose distribution plot mode",
            ["Significant only", "All features"],
            help=(
                "• *Significant only*: Shows permutation distributions for features with p-values below the alpha threshold.\n"
                "• *All features*: Shows the distribution for every tested feature, regardless of significance.\n\n"
                "Use this to visually inspect how your test statistics compare to the null distribution created via permutation."
            ),
            key=f"dist_mode_{label}"
        )

        if plot_mode == "Significant only":
            alpha_dist = st.number_input(
                "Alpha threshold for significance", 
                value=0.05, min_value=0.0, max_value=1.0, step=0.01,
                help="Only features with corrected p-values below this threshold will be shown.", key=f"lol_{label}"
            )

        if st.button("Plot permutation distributions", key= f"dist_bot_{label}"):
            null_distributions = st.session_state.get("test_result", {}).get("null_stat_distribution")

            if null_distributions is None or not xticklabels:
                st.warning("Missing null distributions or feature labels.")
            else:
                if plot_mode == "Significant only":
                    sig_idx = np.where(pval_data < alpha_dist)[0]
                    if len(sig_idx) == 0:
                        st.info("No features were significant at the selected alpha level.")
                    else:
                        for idx in sig_idx:
                            fig = graphics.plot_permutation_distribution(
                                null_distributions[:, idx],
                                title_text=f"Permutation Distribution – {xticklabels[idx]}",
                                xlabel="Test Statistic",
                                return_fig=True
                            )
                            st.pyplot(fig)
                else:  # All features
                    for idx, feature_name in enumerate(xticklabels):
                        fig = graphics.plot_permutation_distribution(
                            null_distributions[:, idx],
                            title_text=f"Permutation Distribution – {feature_name}",
                            xlabel="Test Statistic",
                            return_fig=True
                        )
                        st.pyplot(fig)

if plot_uncorrected and pval_uncorrected is not None:
    st.subheader("📊 Uncorrected P-Values")
    plot_pval("Uncorrected", pval_uncorrected, n_timepoints)

if plot_corrected and pval_corrected is not None:
    st.subheader("📊 Corrected P-Values")
    plot_pval("Corrected", pval_corrected, n_timepoints)
