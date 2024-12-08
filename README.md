# Fitting Time-Resolved Photoluminescence Data with Bayesian Inference and a Markov-Chain Monte-Carlo Sampler

This is the code that was used in: M. Kober-Czerny, A. Dasgupta, S. Seo, F.M. Rombach, D.P. McMeekin, H. Jin, H.J. Snaith, "Determining Material Parameters of Metal Halide Perovskites Using Time-resolved Photoluminescence". <br>
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
To run the inference, create a .csv file with the columns:<br>
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
    <td>1. open Manuel_BayesTRPL_model.py <br>
    2. navigate to ```sigma = sigmas * (2+99*pm.Beta('sigma_fact', 3,3))``` <br> 
    3. change '3,3' to larger nmbers (like '5,5') </td>
  </tr>
  <tr>
    <td>Sampler continues to crash with message 'Bad initial energy...'</td>
    <td>1. open Manuel_BayesTRPL_model.py <br>
    2. uncomment ```#trace = pm.sample(step=pm.Metropolis()...)``` <br> 
    3. comment ```trace = pm.sample(...)``` </td>
  </tr>
  <tr>
    <td>...</td>
    <td>... </td>
  </tr>
</table>

## Adapting the Code or Model
### Different TRPL setup
If you have a different TRPL setup, the output data may be different. In our case, most metadata (laser rep-rate, laser wavelength, etc.) is stored in the header of the datafile. This information is extracted in Manuel_BayesTRPL_Utilities.py in he functions ```unpack_info()``` and ```Fluence_Calc()```. Skip these two, if you have other means of obtaining this information.<br>

In our setup, the laser pulse is registered at approximately 30-40 ns on the real-time axis, but for the analysis we want it to be at <i>t = 0</i>. The functions ```unpack_Data()``` and ```make_Dataframe()``` are used to extract the raw data, shift the time-axis and normalize the data. They are then stored in a pandas dataframe.<br>

If you want to write your own Utils-code, the function ```Bayes_TRPL_Utils``` needs to output:<br>
```
return df, pile_up, sample_name, Fluence, Thickness, Surface, Absorption_coeff, amax
```
where df is the dataframe containing the columns 'Time'| 1 | 2 | 3 | ...<br>
sample_name is the name of the sample, Fluence is in cm^2, Thickness is in nm, Surface is either 0 (for one surface) or 1 (for the other surface), Absorption_coeff is in cm^-1, amax is the last timepoint in ns.

### Changing the Model
To change the model, open Manuel_BayesTRPL_model.py and navigate to ```model_in_pytensor()```. To understand, how to add or change parameters, refer to the [Pymc documentation](https://www.pymc.io/projects/docs/en/stable/learn/core_notebooks/pymc_overview.html). <br>
All functions are written in either numpy or [PyTensor](https://pytensor.readthedocs.io/en/latest/library/tensor/basic.html). For functions, such as 'for'-loops, we use PyTensors ```scan``` function instead (see [documentation](https://pytensor.readthedocs.io/en/latest/library/scan.html)). The use of PyTensor throughout the model enables a more straightforward use of gradient-based MCMC samplers, such as NUTS.
