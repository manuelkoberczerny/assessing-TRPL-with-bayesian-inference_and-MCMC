import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymc as pm
import datetime
import pytensor.tensor as at
from pytensor import *
from pytensor import config
config.allow_gc = False


from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit
from scipy.signal import medfilt

import logging
logger = logging.getLogger("pymc")
logger.setLevel(logging.ERROR)

import warnings
warnings.filterwarnings('ignore')

###########################
#### General Functions ####
###########################

def log_time(end_point, no_points):

        fib_max = np.log(end_point-2)
        fib_list = np.logspace(0, fib_max, no_points, base=2.73)
        
        fib_list_sorted = np.unique(np.round(fib_list,0)-1)

        while fib_list_sorted[-1] > end_point:
            fib_list_sorted = fib_list_sorted[:-1]

        return np.array(fib_list_sorted).astype('int')


def logspacing(end_point, number_of_datapoints):

    len_a = 0
    nod = number_of_datapoints
    while len_a < number_of_datapoints:
        
        a = log_time(end_point, nod)
        len_a = len(a)
        nod += 1

    return a
    

def lin_time(end_point):
    return np.arange(0,end_point-2)


def spacing_choose(spacing, end_point):

    if spacing == 'linear':
        a = lin_time(end_point)

    elif 'log' in spacing:
        a = logspacing(end_point, int(spacing[3:]))

    else:
        print('spacing not implemented...')

    return a


def multi_exp_approximation(x,y, A1, A2, A3, tau1, tau2, tau3, beta2, beta3, y0):

    decay = 10**A1*np.exp(-x/tau1) + 10**A2*np.exp(-(x/tau2)**beta2) + 10**A3*np.exp(-(x/tau3)**beta3) + 10**(-y0)
    return np.sqrt(decay)

    
def setup_data_for_inference(Data_fit, a, Surface):
    #### Setting up the Data and Timeframe
    time = np.array(Data_fit['Time'])[a]*1e-9    #s

    y_combined = np.zeros((len(time), len(Surface)))
    sigmas = np.zeros((len(time), len(Surface)))
    bckg_list = ()
    
    #BSpline to determine sigma and smooth data
    for s in range(len(Surface)):
        data = np.array(Data_fit[str(s)])

        try:
            popt, _ = curve_fit(multi_exp_approximation, Data_fit['Time'], np.sqrt(data), maxfev=100000)
            spline_fit = multi_exp_approximation(Data_fit['Time'], *popt)**2
        except:
            spline_fit = data
        else:
            popt, _ = curve_fit(multi_exp_approximation, Data_fit['Time'], np.sqrt(data), maxfev=100000)
            spline_fit = multi_exp_approximation(Data_fit['Time'], *popt)**2
        
        y_combined[:,s] = spline_fit[a]
        grad = -np.round(np.median(np.gradient(spline_fit[a], time)[-5:]),0)
        
        
        sigma_calcs = medfilt(np.sqrt((np.sqrt(spline_fit)-np.sqrt(data))**2)/np.sqrt(data),51)
        spline = UnivariateSpline(Data_fit['Time'], np.log(sigma_calcs))
        spline_fit_sigma = np.exp(spline(Data_fit['Time']))
        sigmas[:,s] = spline_fit_sigma[a]

    
    bckg_list = y_combined[-1,:]
    
    return time, y_combined, sigmas, bckg_list


def X_n_maker(d_factor, x_size, dx, D, Sf, Sb):
    
    Xn_1 = at.extra_ops.fill_diagonal_offset(x_size, d_factor, -1)
    
    Xn_2a = at.extra_ops.fill_diagonal_offset(x_size, 1-2.*d_factor, 0)
    Xn_2a1 = at.set_subtensor(Xn_2a[0,0],1-d_factor - (dx/D)*d_factor *Sf)
    Xn_2 = at.set_subtensor(Xn_2a1[-1,-1],1-d_factor - (dx/D)*d_factor *Sb)
    
    Xn_3 = at.extra_ops.fill_diagonal_offset(x_size, d_factor, 1)
    
    return Xn_1 + Xn_2 + Xn_3


### First: Define Rate equations
def rate_equations(n_dens, nt, params):            

    k_c, k_deep, k_e, k_rad, k_aug, p0, _, _, _ = params
    
    p_dens = n_dens + nt

    R_rad = - k_rad*n_dens*p_dens
    
    dnt_dt = k_c*n_dens - k_e*nt
    R_nr = - k_c*n_dens + k_e*nt - k_deep*n_dens
    
    dn_dt = R_rad + R_nr
    
    return dn_dt, dnt_dt


def Runge_Kutta_R4(n_dens, nt, dt, params):

    RuKu1_n, RuKu1_nt = rate_equations(n_dens, nt, params)
    RuKu2_n, RuKu2_nt = rate_equations(n_dens + RuKu1_n*dt/2, nt + RuKu1_nt*dt/2, params)
    RuKu3_n, RuKu3_nt = rate_equations(n_dens + RuKu2_n*dt/2, nt + RuKu2_nt*dt/2, params)
    RuKu4_n, RuKu4_nt = rate_equations(n_dens + RuKu3_n*dt, nt + RuKu3_nt*dt, params)

    Ruku_n = (RuKu1_n + 2*RuKu2_n + 2*RuKu3_n + RuKu4_n)/6
    Ruku_nt = (RuKu1_nt + 2*RuKu2_nt + 2*RuKu3_nt + RuKu4_nt)/6

    return Ruku_n, Ruku_nt



### Looping over time-domain
def total_recombination_rate(dt_current, n_dens, p_dens, ds, x_size, params):

    _, _, _, _, _, _, D, S_f, S_b = params

    # a. Recombination (Runge-Kutta Algorithm)
    nt = p_dens - n_dens
    Ruku_n, Ruku_nt  = Runge_Kutta_R4(n_dens, nt, dt_current, params)
    
    # b. Diffusion
    d_factor = D*dt_current/(2*ds*ds)
    A_n = X_n_maker(-d_factor, x_size, ds, D, S_f, S_b)
    B_n = X_n_maker(d_factor, x_size, ds, D, S_f, S_b)

    Bn_dot_n_dens = at.dot(B_n, n_dens) + Ruku_n*dt_current
    n_dens_new = at.dot(at.linalg.inv(A_n), Bn_dot_n_dens)

    # c. Physical limits
    n_dens_new = at.switch(at.le(n_dens_new, 0), 0, n_dens_new)
    p_dens_new = n_dens_new + nt + Ruku_nt*dt_current
    p_dens_new = at.switch(at.le(p_dens_new, 0), 0, p_dens_new)
    
    return n_dens_new, p_dens_new


def save_trace(trace, folder, config_name, spacing, laserpower_file, _, bckg_list, PN_on_off, diffusion_on_off, sh_defect):

    tracename = config_name[:-4]
    date_time_code = str(datetime.datetime.now()).split('.')
    date_time_code = date_time_code[0].replace(" ","_").replace(":","")

    sample_name = str(tracename + "_trace_" + date_time_code)
   
    trace.to_netcdf(f"{folder}/{sample_name}.nc")
    print("Trace has been saved!")

    text_file = open(f"{folder}/{sample_name}_logfile.txt", 'w')
    my_string = str(spacing  + "\t" + laserpower_file + "\t" + str(sh_defect)+ "\t" + str(PN_on_off) + "\t" + str(diffusion_on_off) + "\t" + str(bckg_list))
    text_file.write(my_string)
    text_file.close()
    print("Logfile has been saved!")


##############################
#### Shallow-Defect Model ####
##############################
    
def loop_over_samples(n_0z, S_f, S_b, pn_eq, Rrad, ds, dt, x_size, params):
    # re-pack parameters
    k_c, k_p, n_em, p0, k_aug, k_rad, Diffusion_coefficient = params
    params = k_c, k_p, n_em, k_rad, k_aug, p0, Diffusion_coefficient, S_f, S_b

    result_one_sample, _ = pytensor.scan(fn=total_recombination_rate,
                                            sequences=[dt],
                                            outputs_info=[n_0z, n_0z+pn_eq],
                                            non_sequences=[ds, x_size, params], strict=True)    

    n_init = (n_0z).dimshuffle('x',0)
    N_calc = at.concatenate([n_init, result_one_sample[0]], axis=0)
    p_init = (n_0z+pn_eq).dimshuffle('x',0)
    P_calc = at.concatenate([p_init, result_one_sample[1]], axis=0)

    return (N_calc)*(P_calc)


def model_in_pytensor(time, Fluence, Surface, thickness, Absorption_coeff, bckg_list, PN_on_off, diffusion_on_off, sh_defect):
    
    ### Define the spacial and temporal domains
    x = np.arange(0,thickness,30)
    z_array_np = x*1e-3
    z_array = at.as_tensor_variable(z_array_np)

    ds = at.as_tensor_variable(z_array_np[1]-z_array_np[0])
    dt = at.extra_ops.diff(time)

    
    ### Define Parameters
    ## 1 - Surface recombination velocities
    SRVs = (1+pm.LogNormal('SRVs', 2, 2, shape=2))*1e4

    S_1_save = pm.Deterministic('S_1', SRVs[0] * 1e-4)
    S_2_save = pm.Deterministic('S_2', SRVs[1] * 1e-4)

    pm.Deterministic('S_upper', at.max([S_1_save, S_2_save]))
    pm.Deterministic('S_lower', at.min([S_1_save, S_2_save]))
       
    S_front_value = at.switch(at.eq(Surface, 1),  SRVs[0],  SRVs[1])
    S_back_value = at.switch(at.eq(Surface, 1),  SRVs[1],  SRVs[0])
    
    ## 2 - Mobility and Diffusion
    Diffusion_coefficient = pm.LogNormal('Diffusion_coefficient', -2, 3)*1e8
    pm.Deterministic('mu_vert', Diffusion_coefficient/(1.380649e-23*292/1.6021766e-19)*1e-8)
     
    
    ## 3 - Radiative and Non-Radiative Recombination 
    k_rad = pm.Deterministic('k_rad', 5e-9/(1+pm.LogNormal('k_rad_model_fact', 2, 2)))*1e12
    
    PN_eq = pm.LogNormal('P_0_list', 5, 2,shape=at.shape(Surface))        
    PN_eq = PN_eq*PN_on_off
    
    # Non-radiative bulk recombination rate (s^-1)
    k_deep = pm.LogNormal("k_deep", 10, 1)
    k_c = pm.Deterministic("k_capture", k_deep * (0.01+pm.LogNormal('k_cd_ratio', 1, 1))) * sh_defect
    k_e = pm.LogNormal("k_emission", 10, 1) * sh_defect

    # Packing parameters
    params = k_c, k_deep, k_e, 0, 0, k_rad, Diffusion_coefficient
    
    ### Calculate initial Charge-Carrier Density
    Absorption_coeff = Absorption_coeff*diffusion_on_off    
    generation = at.exp(-Absorption_coeff*z_array)
    generation_sum = at.sum(((generation[1:] + generation[:-1])/2), axis=0) * ds
    n_0z = at.outer(Fluence/(generation_sum), generation)

    x_size = at.zeros(shape=(at.shape(n_0z)[1],at.shape(n_0z)[1]))
    
    ### Simulate transient, radiative recombination
    result_all_samples, _ = pytensor.scan(fn=loop_over_samples,
                                                sequences=[n_0z, S_front_value, S_back_value, PN_eq],
                                                outputs_info=[at.zeros(shape=(at.shape(time),at.shape(n_0z[0,:])))],
                                                non_sequences=[ds, dt, x_size, params])

    
    ### Turn radiative recombination into PL response
    ## Add PL error parameter
    PL_err = bckg_list*0
    PL_err_2d = at.outer(PL_err, at.ones(shape=at.shape(time)))
    
    Rrad_calc = result_all_samples
    
    PL_calc = at.sum(((Rrad_calc[:,:,1:] + Rrad_calc[:,:,:-1])/2), axis=2) * ds
    
    PL_0 = PL_calc[:,0]/(1-PL_err)

    PL_obs = PL_calc.T/PL_0 + PL_err_2d.T
    no_of_params = 6 + at.shape(Surface)*PN_on_off + diffusion_on_off
    
    return PL_obs, no_of_params
    

def pymc_model(y_combined, sigmas, time, Fluence, Surface, Thickness, Absorption_coeff, tune_no, draws_no, cores_no, bckg_list, PN_on_off, diffusion_on_off, sh_defect):
    
    pymc_model = pm.Model()
    
    with pymc_model:
                       
        #### Simulation of Time-Resolved PL
        N_calc, no_of_params = model_in_pytensor(shared(time), shared(Fluence*1e-8), Surface, Thickness, shared(Absorption_coeff[0]*1e-4), at.as_tensor_variable(bckg_list), PN_on_off, diffusion_on_off, sh_defect)
                        
        pm.Deterministic('N_calc_collect', N_calc)
        
        #sigma = sigmas *(2+9*pm.Beta('sigma_fact', 4,4))
        
        sigma = (y_combined) *(1+10*pm.Beta('sigma_fact', 1,1))
        
        Log_likelihood = (((N_calc) - (y_combined))**2)/(2*sigma**2)
        
        pm.Deterministic('Logp', at.sum(Log_likelihood))        
        
        Y_obs = pm.Normal('Y_obs', mu=(N_calc), sigma = sigma, observed = (y_combined))
        
        #### Draw Samples from the Posterior Distribution
        print("Bayes-MCMC Sampling Running...")
        print(str("(tune: " +str(tune_no)+", draws: " +str(draws_no)+", chains: " +str(cores_no)+ ")"))

        ### Un-comment, if you want to use Metropolis-Hastings algorithm (recommended for testing)
        #trace = pm.sample(step=pm.Metropolis(),  chains=cores_no, draws=draws_no, cores=cores_no, tune=tune_no)
        trace = pm.sample(chains=cores_no, draws=draws_no, cores=cores_no, tune=tune_no, discard_tuned_samples=False)#, 
                          #nuts = {"max_treedepth": 4, "early_max_treedepth": 3})
    
    return trace



###################
#### Main Part ####
###################

def run_bayesian_inference(df, max_arg, spacing, Fluence, Surface, Thickness, Absorption_coeff, tune_no, draws_no, cores_no, folder, config_name, laserpower_file, PN_on_off, diffusion_on_off, sh_defect):


    a = spacing_choose(spacing, max_arg)

    time, y_combined, sigmas, bckg_list = setup_data_for_inference(df, a, Surface)
    
    trace = pymc_model(y_combined, sigmas, time, Fluence, Surface, Thickness, Absorption_coeff, tune_no, draws_no, cores_no, bckg_list, PN_on_off, diffusion_on_off, sh_defect)
    
    print("Bayes-MCMC done!")
    print(" ")

    save_trace(trace, folder, config_name, spacing, laserpower_file, 0 , bckg_list, PN_on_off, diffusion_on_off, sh_defect)
    
    return trace