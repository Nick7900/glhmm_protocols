
# Statistical testing framework for brain dynamics

Welcome to the protocols for the statistical testing methods described in the paper [A comprehensive framework for statistical testing of brain dynamics](https://arxiv.org/abs/2505.02541). This repository provides Jupyter notebooks and resources to implement the statistical framework introduced in the paper. Built on the statistical tools included in the [GLHMM toolbox](https://github.com/vidaurre/glhmm), these protocols are designed for analysing the relationships between brain activity, behaviour, physiological signals, and other variables across different experimental designs.

## What’s Included

This repository provides Jupyter notebooks for four statistical tests:

1. **Across-subjects**: Investigates how brain activity relates to individual differences, such as traits or behaviours.
2. **Across-trials**: Compares brain responses under different experimental conditions or actions within a task.
3. **Across-sessions-within-subject**: Examines how brain activity evolves over multiple sessions for a single individual, such as during learning or long-term changes.
4. **Across-state-visits**: Explores how brain states relate to concurrent signals like pupil size or physiological measurements.

## Data Availability

The notebooks will automatically download the required datasets on first run. Alternatively, you can download the data directly from Zenodo: [https://zenodo.org/records/14756003](https://zenodo.org/records/14756003)

## How to Use

1. Clone this repository to a folder of your choice:
   ```bash
   git clone https://github.com/Nick7900/glhmm_protocols.git
   ```

2. Install the GLHMM package:
   To install the latest development version from the repository:
   ```bash
   pip install git+https://github.com/vidaurre/glhmm
   ```

   Otherwise, to install the latest stable release from PyPI:
   ```bash
   pip install glhmm
   ```


3. Navigate to the Protocols folder and launch the notebook corresponding to the statistical test you want to run:

## GUI Version

In addition to the Python package and Jupyter notebooks, the GLHMM toolbox now includes a graphical user interface (GUI) for users who prefer a code-free experience. To use the GUI:

1. Clone the protocols (if not already done):
   ```bash
   git clone https://github.com/Nick7900/glhmm_protocols.git
   ```
   
2. Create a Python 3.10 environment:
   ```bash
   conda create --name glhmm_GUI python=3.10
   ```
3. Activate conda environment:
   ```bash
   conda activate glhmm_GUI
   ```
4. Install GLHMM and Streamlit:
   ```bash
   pip install glhmm
   pip install streamlit
   ```

5. Navigate to the GUI directory and run the app:
   ```bash
   cd glhmm_protocols/GUI
   streamlit run 1_🏠_load_data.py
   ```
### ▶️ Video Tutorial
A introductory video walkthrough of the GUI is available here:

📺 GLHMM GUI Tutorial – [YouTube](https://www.youtube.com/watch?v=XPcoK5zCPtU&t=1497s)

## Features of this Protocol

* **Statistical Inference**: Permutation-based and Monte Carlo resampling tests without strong distributional assumptions.
* **Visualisation Tools**: Integrated plotting functions for interpreting results.
* **Step-by-Step Workflows**: Jupyter notebooks and GUI both guide you through the process.

## How to Cite

If you use this framework in your research, please cite the accompanying protocol paper:  
[A comprehensive framework for statistical testing of brain dynamics](https://arxiv.org/abs/2505.02541)
