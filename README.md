# Fitting Time-Resolved Photoluminescence Data with Bayesian Inference and a Markov-Chain Monte-Carlo Sampler

This is the code that was used in M. Kober-Czerny, <i>et.al.</i>; <i>accepted</i> <b>XXXX</b>.
The theory and physical model used is described in the publication in detail.
This code is written for data obtained by a [picoquant](https://www.picoquant.com) setup. There are instructions below on how to adapt the code to your own data format.

## Disclaimer

Information inferred from data through this method is only as good as the model that is used. If the model is flawed, so are the 
inferred parameters. We hence strongly advise against over-interpretation of the results and recommend double-checking parameter
values with another methodology in a first instance.


## Getting Started
### Installation and Usage
Install the [Pymc5](https://www.pymc.io/projects/docs/en/latest/installation.html) package in a new Python (version 3.10) environment using conda-forge (follow the instructions on the Pymc website).
Activate the environment and install jupyter notebook using
```
conda install -c conda-forge notebook
```
Using the command line, navigate to the cloned repository and run

```
jupyter notebook
```
Then open 1-Bayes_MCMC_algorithm/Setup_inference.ipynb to run an inference.

### Running Inference
To run the inference, create a .csv file with the columns:
file name (no .txt)  |  thickness (nm)  |  side (1 or 2)  |  alpha (cm-1)  |  reflectance  |  Intensity  |  max_time (ns)

Add the file path to the jupyter notebook and run the inference.
The MCMC sampler will take a few minutes to start. A typical run is between 1-3 hours.

### Analysis of the Results
After the inference is complete, open 2-Extract_Parameters/Extract_Parameters.ipynb.
Add the path and name of the trace_*.nc file (the file encodes the sample name, date and time).
A plot showing the raw data and median 'fit' curve, as well as histograms for all parameters is shown.


## Trouble Shooting

<table  style="width:100%">
  <tr>
    <th style="width:30%">Issue</th>
    <th>Tip</th>
  </tr>
  <tr>
    <td>Sampler crashes with message 'Bad initial energy...'</td>
    <td>- open Manuel_BayesTRPL_model.py
    - navigate to ```sigma = sigmas * (2+99*pm.Beta('sigma_fact', 3,3))``` and change '3,3' to larger nmbers (like '5,5') </td>
  </tr>
  <tr>
    <td>Sampler continues to crash with message 'Bad initial energy...'</td>
    <td>- open Manuel_BayesTRPL_model.py
    - uncomment ```#trace = pm.sample(step=pm.Metropolis()...)``` and comment ```trace = pm.sample(...)``` </td>
  </tr>
  <tr>
    <td>...</td>
    <td>... </td>
  </tr>
</table>

## Adapting the Code or Model
### Different TRPL setup


