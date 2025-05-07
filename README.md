# Statistical testing framework for brain dynamics
Welcome to the protocols for the statistical testing methods described in the paper [A comprehensive framework for statistical testing of brain dynamics](https://arxiv.org/abs/2505.02541). This repository provides Jupyter notebooks and resources to implement the statistical framework introduced in the paper. Built on the statistical tools included in the [GLHMM toolbox](https://github.com/vidaurre/glhmm), these protocols are designed for analysing the relationships between brain activity, behaviour, physiological signals, and other variables across different experimental designs. 

## What’s Included
This repository provides Jupyter notebooks for four statistical tests:
1. **Across-subjects**: Investigates how brain activity relates to individual differences, such as traits or behaviours.
2. **Across-trials**: Compares brain responses under different experimental conditions or actions within a task.
3. **Across-sessions-within-subject**: Examines how brain activity evolves over multiple sessions for a single individual, such as during learning or long-term changes.
4. **Across-state-visits**: Explores how brain states relate to concurrent signals like pupil size or physiological measurements.

## Data Availability
All datasets required to run the notebooks are hosted on Zenodo (link coming soon). The data is ready to use and compatible with the workflows in this repository.

## How to Use
1. Clone this repository:
   ```bash
   git clone https://github.com/Nick7900/glhmm_protocols.git
   cd glhmm_protocols
  
2. **Install the GLHMM package**: 
Install all the necessary Python libraries by running:
   ```bash
   pip install git+https://github.com/vidaurre/glhmm
   ```
3. **Install the required dependencies**: 
Install all the necessary Python libraries by running:
   ```bash
   pip install -r requirements.txt
   ```
4. **Download the data**: 
The example datasets for the notebooks will be available on Zenodo (DOI link coming soon).

5. **Change the path in the `ZENODO_PATH.txt` file**: To make it easy to load in the correct files in each notebook, change the path according to where the Zenodo data is saved on your computer.

## Features of this protocol
* Statistical Inference: Utilises permutation-based and Monte Carlo resampling testing, which avoids strict assumptions about data distributions.
* Visualisation Tools: Easily interpret results with intuitive plots built into the framework.
* Step-by-Step Workflows: Detailed Jupyter notebooks guide you through each test, making it accessible regardless of your programming experience.

## How to Cite
If you use this framework in your research, please cite the accompanying protocol paper:
[A comprehensive framework for statistical testing of brain dynamics](https://arxiv.org/abs/2505.02541)
