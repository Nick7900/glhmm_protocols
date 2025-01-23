# Statistical testing framework for brain dynamics

Welcome to the **Statistical Testing Framework for Brain Dynamics**, a resource built on the statistical testing tools included in the [GLHMM toolbox](https://github.com/vidaurre/glhmm). This framework is designed for analysing the relationships between brain activity, behaviour, physiological signals, and other variables across different experimental designs. 

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

3. **Install the required dependencies**: 
Install all the necessary Python libraries by running:
   ```bash
   pip install -r requirements.txt

4. **Download the data**: 
The example datasets for the notebooks will be available on Zenodo (DOI link coming soon). Download the data and place it in the appropriate folders as described in each of the notebooks.

## Features of this protocol
* Robust Statistical Inference: Utilises permutation-based testing, which avoids strict assumptions about data distributions.
* Visualisation Tools: Easily interpret results with intuitive plots built into the framework.
* Step-by-Step Workflows: Detailed Jupyter notebooks guide you through each test, making it accessible regardless of your programming experience.

## How to Cite
If you use this framework in your research, please cite the accompanying protocol paper:
A Comprehensive Framework for Statistical Testing of Brain Dynamics
