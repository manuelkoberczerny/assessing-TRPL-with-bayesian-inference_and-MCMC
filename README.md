# Fitting Time-Resolved Photoluminescence Data with Bayesian Inference and a Markov-Chain Monte-Carlo Sampler

This is the code that was used in M. Kober-Czerny, <i>et.al.</i>; <i>PRX Energy</i> <b>2025</b>.
The theory and physical model used is described in the publication in detail.


## Installation and Usage
### Windows and MacOS (non-Python users)
Install the [Pymc5](https://www.pymc.io/projects/docs/en/latest/installation.html) package in a new Python (version 3.10) environment using conda-forge (follow the instructions on the Pymc website).
Activate the environment and install jupyter notebook using
```
conda install -c conda-forge notebook
```
Using the command line, navigate to the cloned repository and run

```
jupyter notebook
```
Then open 1-Bayes_MCMC_algorithm to run an inference.

## Running Inference

## Build instructions
To compile the program into a standalone binary file first follow the 'Installation and Usage' instructions for 'Windows and MacOS (Python users)' or 'Linux' above until you have installed the dependencies from the `requirements.txt` file. Then run:
```
pyinstaller data_analysis.spec
```
This will create two new folders in the current directory called `build` and `dist`. The binary file is in the `dist` folder and will be called `data_analysis.exe` on Windows, `data_analysis.app` on MacOSX, and just `data_analysis` on Linux.

