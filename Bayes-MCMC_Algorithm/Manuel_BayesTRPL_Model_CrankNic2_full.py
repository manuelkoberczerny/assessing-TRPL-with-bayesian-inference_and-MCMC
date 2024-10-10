import os
os.environ['OMP_NUM_THREADS'] = '1'
import datetime

import numpy as np
import pandas as pd
import pymc as pm
from scipy.optimize import shgo
import pytensor.tensor as at
from pytensor import *
from pytensor.graph.op import Op
from pytensor.graph.basic import Apply
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




def golden_ratio(end_point):
        fib_list = [0,1,2]
    
        fib_stack = fib_list[-1]
        i = np.max(fib_stack)
        
        while i < end_point-1:
            fib_next = fib_list[-1]+fib_list[-2]  # fibonacci
            if fib_next <= end_point-1:
                fib_list.append(fib_next)
            
                i = np.max(fib_next)                
            else:
                break

        fib_list = np.array(fib_list)
        fib_list_sorted = np.sort(np.unique(fib_list))
        return fib_list_sorted

def squared_time(end_point):
        fib_list = [0,1,4]
    
        fib_stack = fib_list[-1]
        i = np.max(fib_stack)
        
        while i < end_point-1:
            fib_next = i**2           
            if fib_next <= end_point-1:
                fib_list.append(fib_next)
            
                i += 1   
            else:
                break

        fib_list = np.array(fib_list)
        fib_list_sorted = np.sort(np.unique(fib_list))
        return fib_list_sorted

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
    
    elif spacing == 'squared':
        a = squared_time(end_point)

    elif 'log' in spacing:
        a = logspacing(end_point, int(spacing[3:]))

    else:
        print('spacing not implemented...')

    return a



def X_n_maker(d_factor, size, dx, D, Sf, Sb):
    x_size = at.zeros(shape=(at.shape(size),at.shape(size)))
    Xn_1 = at.extra_ops.fill_diagonal_offset(x_size, d_factor, -1)
    
    Xn_2a = at.extra_ops.fill_diagonal_offset(x_size, 1-2.*d_factor, 0)
    Xn_2a1 = at.set_subtensor(Xn_2a[0,0],1-d_factor - (dx/D)*d_factor *Sf)
    #Xn_2a2 = at.set_subtensor(Xn_2a1[0,1], d_factor *2)
    #Xn_2a3 = at.set_subtensor(Xn_2a2[-1,-2], d_factor *2)
    Xn_2 = at.set_subtensor(Xn_2a1[-1,-1],1-d_factor - (dx/D)*d_factor *Sb)
    
    Xn_3 = at.extra_ops.fill_diagonal_offset(x_size, d_factor, 1)
    
    return Xn_1 + Xn_2 + Xn_3


### First: Define Rate equations
def rate_equations(n_dens, nt_dens, params, p0):            

    kc_n, kc_p, n_em, k_rad, N_t, k_aug, _ = params
    
    p_dens = n_dens + nt_dens
    nt = p_dens-n_dens

    f_t = nt/N_t
    
    f_t = at.switch(at.le(f_t, 0), 0, f_t)
    f_t = at.switch(at.ge(f_t, 1), 1, f_t)  

    R_rad = - p0*k_rad*n_dens*p_dens
    R_aug = - k_aug*(p_dens**2 * n_dens + n_dens**2 * p_dens)
         
    dn_dt = R_rad - kc_n*n_dens*(1-f_t) + kc_n*f_t*n_em + R_aug
    dnt_dt = kc_n*n_dens*(1-f_t) - kc_n*f_t*n_em - kc_p*p_dens*f_t

    dn_dt = at.switch(at.ge(dn_dt, 0),0,dn_dt)
    #dnt_dt = at.switch(at.ge(dnt_dt, 0),0,dnt_dt)
    
    return dn_dt, dnt_dt

def Runge_Kutta_R4(n_dens, nt_dens, dt_current, params, p_0):

    RuKu1_n, RuKu1_t = rate_equations(n_dens, nt_dens, params, p_0)
    RuKu2_n, RuKu2_t = rate_equations(n_dens + RuKu1_n*dt_current/2, nt_dens + RuKu1_t*dt_current/2, params, p_0)
    RuKu3_n, RuKu3_t = rate_equations(n_dens + RuKu2_n*dt_current/2, nt_dens + RuKu2_t*dt_current/2, params, p_0)
    RuKu4_n, RuKu4_t = rate_equations(n_dens + RuKu3_n*dt_current, nt_dens + RuKu3_t*dt_current, params, p_0)

    RuKu_n = (RuKu1_n + 2*RuKu2_n + 2*RuKu3_n + RuKu4_n)/6
    RuKu_t = (RuKu1_t + 2*RuKu2_t + 2*RuKu3_t + RuKu4_t)/6

    return RuKu_n, RuKu_t


### Looping over time-domain
def total_recombination_rate(dt_current, n_dens, p_dens, ds, params, S_f, S_b, p_0):

    # b. Recombination (Runge-Kutta Algorithm)
    nt_dens = p_dens - n_dens

    Ruku_n, Ruku_nt = Runge_Kutta_R4(n_dens, nt_dens, dt_current, params, p_0)

    nt_dens_new = nt_dens + Ruku_nt*dt_current
    
    # a. Diffusion
    _, _, _, _, _, _, D = params
    
    d_factor = D*dt_current/(2*ds*ds)
    A_n = X_n_maker(-d_factor, n_dens, ds, D, S_f, S_b)
    B_n = X_n_maker(d_factor, n_dens, ds, D, S_f, S_b)

    Bn_dot_n_dens = at.dot(B_n, n_dens) + Ruku_n*dt_current/2
    n_dens_new = at.dot(at.linalg.inv(A_n), Bn_dot_n_dens)

    # Physical limits
    n_dens_new = at.switch(at.le(n_dens_new, 0), 0, n_dens_new)
    nt_dens_new = at.switch(at.le(nt_dens_new, 0), 0, nt_dens_new)      
    
    return n_dens_new, n_dens_new + nt_dens_new

def loop_over_pulses(N, P, G, S_f, S_b, p0, ds, dt, params):

    ns = N[-1,:]
    ps = P[-1,:]
    
    kc_n, kc_p, n_em_1, _, N_t, _, _ = params 
    
    peq = ps/(ns + kc_p/kc_n*ps + n_em_1) * N_t
    
    result_one_loop, _ = pytensor.scan(fn=total_recombination_rate,
                                            sequences=[dt],
                                            outputs_info=[G, G+ps],
                                            non_sequences=[ds, params, S_f, S_b, p0], strict=True) 
    
    n_init = (G).dimshuffle('x',0)
    p_init = (G+ps).dimshuffle('x',0)
    N_calc = at.concatenate([n_init, result_one_loop[0]], axis=0)
    P_calc = at.concatenate([p_init, result_one_loop[1]], axis=0)
    return N_calc, P_calc


def loop_over_samples(n_0z, S_f, S_b, p0, Rrad, ds, dt, params):

    N = at.zeros(shape=(at.shape(dt)+1,at.shape(n_0z)))

    result_one_sample, _ = pytensor.scan(fn=loop_over_pulses, outputs_info = [N, N], non_sequences=[n_0z, S_f, S_b, p0, ds, dt, params], n_steps=3)


    #n_init = (n_0z).dimshuffle('x',0)
    #p_init = (n_0z+p0).dimshuffle('x',0)
    N_calc = result_one_sample[0][-1]
    P_calc = result_one_sample[1][-1]

    return N_calc*P_calc*p0



"""This is the Diffusion model in PyTensor

It is mainly based on:  J. Appl. Phys. 116, 123711 (2014).
The important equations are marked.
In addition I took inspiration from:
https://www.nature.com/articles/s41598-019-41716-x#Sec16
https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-017-02670-2/MediaObjects/41467_2017_2670_MOESM1_ESM.pdf
and 
https://www.sciencedirect.com/science/article/pii/S2542435122004202#sec6.2

"""

def diffusion_carrier_density_all(time, Fluence, Surface, thickness, Absorption_coeff, bckg_list):
    
    # To improve reproducability with the NUTS algorithm, I define the surface recombination in terms of their sum (S_sum) and the factor S_b, which
    # relates the product as: S_prod = (S_sum/2)^2*S_b.
    ## Define the spacial domains

    x = np.arange(0,thickness,30)

    z_array_np = x*1e-3

    ds = at.as_tensor_variable(z_array_np[1]-z_array_np[0])
    
    z_array = at.as_tensor_variable(z_array_np)
    
    #z_array_diff = at.as_tensor_variable(np.diff(z_array_np))
    #z_array_diff_3d = z_array_diff.dimshuffle('x','x',0)
    #z_array_diff_3d.broadcastable
    #(True, True, True)

    
    ## Define Parameters
    # Surface recombination velocities are defined as the sum (S_sum), the ratio of sum and product (S_sumprod_ratio) and
    SRVs = pm.LogNormal('SRVs', 2, 2, shape=2)*1e4
    
    S_1_save = pm.Deterministic('S_1', SRVs[0] * 1e-4)
    S_2_save = pm.Deterministic('S_2', SRVs[1] * 1e-4)

    S_upper = pm.Deterministic('S_upper', at.max([S_1_save, S_2_save]))
    S_lower = pm.Deterministic('S_lower', at.min([S_1_save, S_2_save]))
       
    S_front_value = at.switch(at.eq(Surface, 1),  SRVs[0],  SRVs[1])
    S_back_value = at.switch(at.eq(Surface, 1),  SRVs[1],  SRVs[0])

    
    # Diffusion Coeffient in cm2 s-1
    mu_vert = pm.Deterministic('mu_vert', 0.01 + pm.LogNormal('mu_fact', 1, 2))
    Diffusion_coefficient = pm.Deterministic('Diffusion_coefficient', mu_vert*(1.380649e-23*292/1.6021766e-19))*1e8
    

    ########################################################################
    ########################################################################

    alpha_reab = Absorption_coeff * pm.Beta('alpha_reab', 2, 2)
    
    escape_prob_fact = pm.Beta('escape_prob_fact', 2, 2, shape=2)  
    escape_prob_front = at.switch(at.eq(Surface,1), escape_prob_fact[0], escape_prob_fact[1])
    
    Pesc = at.outer(escape_prob_front, at.exp(-alpha_reab * z_array))

    PL_err = pm.Deterministic('PL_err', bckg_list/ (1 + pm.LogNormal('PL_err_fact', 2, 2, shape=at.shape(bckg_list))))
    PL_err_2d = at.outer(PL_err, at.ones(shape=at.shape(time)))

    
    ### II - Recombination Part
    ## Define Parameters
    # Radiative Recombination Rate (um^3 s^-1)
    k_rad = pm.Deterministic('k_rad_model', 5e-9/(1+pm.LogNormal('k_rad_model_fact', 2, 2)))*1e12
    k_rad_potential = pm.Potential('k_rad_potential', pm.math.log((1/k_rad)**2))
    
    # Trap Density (um^-3)
    N_t = pm.LogNormal('N_t_model', 38, 3)*1e-12
    
    effective_e_mass = 0.2*9.1e-31 # kg
    Nc = 2*(2*np.pi*effective_e_mass*1.38e-23*300/6.262e-34**2)**(3/2) * 1e-18 # um^-3
    trap_depth = pm.Deterministic('trap_depth', 8.6e-5*300 + pm.Beta('trap_depth_fact', 2, 2))  # in eV (> kBT)
    n_em_1 = Nc * at.exp(-trap_depth/(8.62e-5 * 300))

    # Non-radiative bulk recombination rate (s^-1)
    kc_n = pm.LogNormal("kc_n_rate", 10, 3)
    kcp_ratio = (2*trap_depth)**5/((2*trap_depth)**5 + (2-2*trap_depth)**5)  # empirically from "What is a deep defect? " https://journals.aps.org/prmaterials/supplemental/10.1103/PhysRevMaterials.4.024602/supp.pdf
    kc_p = pm.Deterministic("kc_p_rate", kc_n*kcp_ratio)
    
    k_aug = pm.Deterministic('k_aug', 5e-27/(1+pm.LogNormal('k_aug_fact', 5, 2)))*1e24

    P_0 = pm.Deterministic('P_0_list', N_t * pm.Beta('P_0_ratio', 1, 10)**2)*at.ones(shape=at.shape(Surface))
    #P_0 = at.switch(at.eq(Surface, 1), P_0[0], P_0[1]) * Fluence/at.max(Fluence)
    
    P_0_save = pm.Deterministic('P_0', P_0*1e12)
    
    


    params = kc_n, kc_p, n_em_1, k_rad, N_t, k_aug, Diffusion_coefficient
    
    
    # Initial Charge-Carrier Density

    limit_mobility = thickness**2/(at.abs(time[1]-time[0])*4)
    Absorption_coeff = at.switch(at.ge(mu_vert, limit_mobility), 0, Absorption_coeff)
    
    generation = at.exp(-Absorption_coeff*z_array)
    generation_sum = at.sum(((generation[1:] + generation[:-1])/2), axis=0) * ds
    n_0z = at.outer(Fluence/(generation_sum), generation)
    
    dt = at.extra_ops.diff(time)
    
    ## Simulate transient, radiative recombination
    result_all_samples, _ = pytensor.scan(fn=loop_over_samples,
                                                sequences=[n_0z, S_front_value, S_back_value, Pesc],
                                                outputs_info=[at.zeros(shape=(at.shape(time),at.shape(n_0z[0,:])))],
                                                non_sequences=[ds, dt, params])
    

    ## Turn radiative recombination into PL response
    Rrad_calc = result_all_samples
    print(at.sqrt(Rrad_calc[:,-1,0]).eval())
    
    PL_calc = at.sum(((Rrad_calc[:,:,1:] + Rrad_calc[:,:,:-1])/2), axis=2) * ds

    PL_0 = PL_calc[:,0]/(1-PL_err)

    PL_obs = PL_calc.T/PL_0 + PL_err_2d.T
    return PL_obs




def multi_exp_approximation(x,y, A1, A2, A3, tau1, tau2, tau3, beta2, beta3, y0):

    decay = 10**A1*np.exp(-x/tau1) + 10**A2*np.exp(-(x/tau2)**beta2) + 10**A3*np.exp(-(x/tau3)**beta3) + 10**(-y0)
    return np.sqrt(decay)
    


def glm_mcmc_inference_diffusion_full(Data_fit, a, Fluence, Surface, Thickness, Absorption_coeff, tune_no, draws_no, cores_no, max_arg, bckg_list):

    
    
    #### Setting up the Data and Timeframe
    time = np.array(Data_fit['Time'])[a]*1e-9    #s

    y_combined = np.zeros((len(time), len(Surface)))
    sigmas = np.zeros((len(time), len(Surface)))

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
        
        
        y_combined[:,s] = spline_fit[a]#np.array(Data_fit[str(s)])[a]
        sigma_calcs = medfilt(np.sqrt((np.sqrt(spline_fit)-np.sqrt(data))**2),51)

        spline = UnivariateSpline(Data_fit['Time'], np.log(sigma_calcs))
        spline_fit_sigma = np.exp(spline(Data_fit['Time']))
        
        sigmas[:,s] = spline_fit_sigma[a]

        import matplotlib.pyplot as plt

        plt.semilogy(time, data[a])
        plt.semilogy(time, y_combined[:,s])
        plt.show()

        plt.semilogy(time, sigma_calcs[a])
        plt.semilogy(time, sigmas[:,s])
        plt.show()
    
    pymc_model = pm.Model()
    
    with pymc_model:
                       
        #### Simulation of Time-Resolved PL
        N_calc = diffusion_carrier_density_all(shared(time), shared(Fluence*1e-8), Surface, Thickness, shared(Absorption_coeff[0]*1e-4), at.as_tensor_variable(bckg_list))
        
       
        sigma_width = pm.Deterministic('sigma_width', 0.1/(1+pm.LogNormal('sigma_width_fact', 2, 2))) 
                

        N_calc_collect = pm.Deterministic('N_calc_collect', N_calc)
        
        LL = pm.Deterministic('Logp', at.sum(1/2*(-2*at.log(sigmas+sigma_width) - at.log(2*3.1415) - ((at.sqrt(N_calc) - at.sqrt(y_combined))**2)/((sigmas+sigma_width)**2)), axis=0)) 
        
        Y_obs = pm.Normal('Y_obs', mu=at.sqrt(N_calc), sigma = sigmas+sigma_width, observed = np.sqrt(y_combined))
        
        #### Draw Samples from the Posterior Distribution
        print("Bayes-MCMC Sampling Running...")
        print(str("(tune: " +str(tune_no)+", draws: " +str(draws_no)+", chains: " +str(cores_no)+ ")"))

        trace = pm.sample(step=pm.Metropolis(),  chains=cores_no, draws=draws_no, cores=cores_no, tune=tune_no)
        #trace = pm.sample(chains=cores_no, draws=draws_no, cores=cores_no, tune=tune_no, nuts={'early_max_treedepth': 3, 'max_treedepth': 4})

    #print('model compiling...')
    #compiled_model = nutpie.compile_pymc_model(pymc_model)
    #print('model compiled. Now sampling...')
    
    #trace = nutpie.sample(compiled_model, chains=4)
    
    return trace





def save_trace(trace, folder, config_name, spacing, laserpower_file):

    tracename = config_name[:-4]
    date_time_code = str(datetime.datetime.now()).split('.')
    date_time_code = date_time_code[0].replace(" ","_").replace(":","")

    sample_name = str(tracename + "_trace_" + date_time_code)

    
    trace.to_netcdf(f"{folder}/{sample_name}.nc")
    print("Trace has been saved!")
    
    
    text_file = open(f"{folder}/{sample_name}_logfile.txt", 'w')
    my_string = str(spacing  + "\t" + laserpower_file)
    text_file.write(my_string)
    text_file.close()
    print("Logfile has been saved!")






def run_bayesian_inference(df, max_arg, spacing, Fluence, Surface, Thickness, Absorption_coeff, tune_no, draws_no, cores_no, folder, config_name, bckg_list, laserpower_file):


    a = spacing_choose(spacing, max_arg)
    print(spacing, len(a))
    
    
    trace = glm_mcmc_inference_diffusion_full(df, a, Fluence, Surface, Thickness, Absorption_coeff, tune_no, draws_no, cores_no, max_arg, bckg_list)
    print("Bayes-MCMC done!")
    print(" ")

    save_trace(trace, folder, config_name, spacing, laserpower_file)
    
    
    
    
    return trace



































