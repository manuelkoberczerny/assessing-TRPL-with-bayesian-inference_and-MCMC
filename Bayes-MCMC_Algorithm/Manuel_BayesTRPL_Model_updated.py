import os
import datetime

import numpy as np
import pandas as pd
import pymc as pm
from scipy.optimize import shgo
import pytensor.tensor as at
from pytensor import *
from pytensor.graph.op import Op
from pytensor.graph.basic import Apply
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit
from scipy.signal import medfilt

import logging
logger = logging.getLogger("pymc")
logger.setLevel(logging.ERROR)

import warnings
warnings.filterwarnings('ignore')


def root_finder_loop_perform(D, a, b, thickness, noroots):
    # Define an Aesara expression for the function to be rooted
    def f(x, D, a, b, thickness):
        #return (np.tan(x*thickness)+(D*(a)*x)/(b-D**2*x**2))**2
        return (np.tan(x*thickness)+(D*(a)*x)/((a/2)**2*b-D**2*x**2))**2

    x_sol1 = np.sort(shgo(f,bounds=[(0,0.8*np.pi/(thickness))], args = (D, a, b, thickness), sampling_method='sobol', n=1000).xl.flatten())
    x_sol2 = np.sort(shgo(f,bounds=[(0,((noroots+1)-0.5)*np.pi/(thickness))], args = (D, a, b, thickness), sampling_method='sobol', n=(noroots+1)*100).xl.flatten())

    x_sol1 = np.delete(x_sol1, np.where(x_sol1 == 0.8*np.pi/(thickness)))
    
    if np.abs(x_sol1[-1]-x_sol2[1])/x_sol2[1] < 1:
        x_sol = np.append(x_sol1[1:], x_sol2[2:])
    else:
        x_sol = np.append(x_sol1[1:], x_sol2[1:])
    
    return x_sol[:noroots]


class RootFinder(Op):
        
    def __init__(self, thickness, noroots):
        
        self.thickness = thickness*1e-7
        self.noroots = noroots
        
    def make_node(self, D, a, b):
        
        outputs = [at.vector(dtype='float64')]
        return Apply(self, [D, a, b], outputs)
    
    def perform(self, node, inputs, outputs_storage):
        D, a, b = inputs
        
        outputs_storage[0][0] = root_finder_loop_perform(D, a, b, self.thickness, self.noroots)
    
    def grad(self, inputs, output_gradients):
        D, a, b = inputs
        x_grad = self(D, a, b)
        
        ## a = sum, b = factor for prod=(sum/2)^2*b
        x_grad_list = a*D/((a**2*b)/4 - D**2*x_grad**2) + 2*a*D**3*x_grad**2/((a**2*b)/4 - D**2*x_grad**2)**2 + (self.thickness)/((at.cos(x_grad*self.thickness))**2)
        D_grad_list = 4*a*(a**2*b*x_grad + 4*D**2*x_grad**3)/(a**2*b-4*D**2*x_grad**2)**2
        a_grad_list = -4*D*x_grad*(a**2*b + 4*D**2*x_grad**2)/(a**2*b-4*D**2*x_grad**2)**2
        b_grad_list = -4*a**3*D*x_grad/(a**2*b - 4*D**2*x_grad**2)**2
        
        Grad_D = at.dot((-D_grad_list/x_grad_list), output_gradients[0])
        Grad_a = at.dot((-a_grad_list/x_grad_list), output_gradients[0])
        Grad_b = at.dot((-b_grad_list/x_grad_list), output_gradients[0])
        
        return Grad_D, Grad_a, Grad_b





def beta_rootfinder(thickness, a, b, Diffusion_value, no_of_roots):
    
    rootfinder = RootFinder(thickness=thickness, noroots = no_of_roots)

    beta = rootfinder(Diffusion_value, a, b)

    return beta



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

        return np.array(fib_list_sorted[1:]).astype('int')

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



def total_recombination_rate(n_tz0_1, n_tz0_2, dt_current, n_dens, p_dens, kc_n, kc_p, ne_1, krad, N_t, p_esc, k_aug, n_eq):
        
    def rate_equations(n_dens0, p_dens0, kc_n, kc_p, ne_1, krad, k_aug, N_t, p_esc, eq_carr):            
                    
        n_dens = n_dens0 + eq_carr[0]
        p_dens = p_dens0 + eq_carr[1]
        
        nt = p_dens-n_dens
        nt = at.switch(at.le(nt, 0), 0, nt)
        nt = at.switch(at.ge(nt, N_t), N_t, nt)
        
        f_t = nt/N_t

        R_rad = - krad*p_esc*(n_dens*p_dens - eq_carr[0]*eq_carr[1])
        R_aug = - k_aug*(p_dens**2 *n_dens + n_dens**2 *p_dens)
                
        dn_dt = R_rad - kc_n*n_dens*(1-f_t) + kc_n*f_t*ne_1 + R_aug
        dp_dt = R_rad - kc_p*p_dens*f_t + R_aug 
                        
        return dn_dt, dp_dt
    
    
    # Runge-Kutta Algorithm (R4)    
    RuKu1_n, RuKu1_t = rate_equations(n_dens, p_dens, kc_n, kc_p, ne_1, krad, k_aug, N_t, p_esc, n_eq)
    RuKu2_n, RuKu2_t = rate_equations(n_dens + RuKu1_n*dt_current/2, p_dens + RuKu1_t*dt_current/2, kc_n, kc_p, ne_1, krad, k_aug, N_t, p_esc, n_eq)
    RuKu3_n, RuKu3_t = rate_equations(n_dens + RuKu2_n*dt_current/2, p_dens + RuKu2_t*dt_current/2,kc_n, kc_p, ne_1, krad, k_aug, N_t, p_esc, n_eq)
    RuKu4_n, RuKu4_t = rate_equations(n_dens + RuKu3_n*dt_current, p_dens + RuKu3_t*dt_current,kc_n, kc_p, ne_1, krad, k_aug, N_t, p_esc, n_eq)

    # diffusion and surface recombination are included via n_ds factor
    n_ds = n_tz0_2/n_tz0_1
    n_next = (n_dens + dt_current/6*(RuKu1_n + 2*RuKu2_n + 2*RuKu3_n + RuKu4_n))*n_ds
    p_next = (p_dens + dt_current/6*(RuKu1_t + 2*RuKu2_t + 2*RuKu3_t + RuKu4_t))*n_ds

    n_next = at.switch(at.le(n_next, 0), 0, n_next)
    p_next = at.switch(at.le(p_next, 0), 0, p_next)
    
    return n_next, p_next


def multiple_pulses(n_tz0, p_tz0, n_tz0_1, n_tz0_2, dt, n_0z, kc_n, kc_p, n_em_1, k_rad_model, k_aug, N_t, p_esc):
        
    result_inner, _ = pytensor.scan(fn=total_recombination_rate,
                                        sequences=[n_tz0_1, n_tz0_2, dt],
                                        outputs_info=[n_0z + n_tz0[-1,:], n_0z + p_tz0[-1,:]],
                                        non_sequences=[kc_n, kc_p, n_em_1, k_rad_model, N_t, p_esc, k_aug, [0, 0]])


    n_tz_init = (n_0z+n_tz0[-1,:]).dimshuffle('x', 0)
    n_tz_new = at.concatenate([n_tz_init, result_inner[0]], axis=0)
    
    p_tz_init = (n_0z+ p_tz0[-1,:]).dimshuffle('x', 0)
    p_tz_new = at.concatenate([p_tz_init, result_inner[1]], axis=0)
    
    return n_tz_new, p_tz_new


def diffusion_carrier_density_all(time, Fluence, Surface, thickness, Absorption_coeff, bckg_list, reabsorption_option):

    """This is the Physical model written in PyTensor
    It is mainly based on:  doi.org/10.1063/1.4896484
    In addition I took inspiration from:
    doi.org/10.1016/0038-1101(92)90228-5
    doi.org/10.1038/s41598-019-41716-x
    doi.org/10.1038/s41467-017-02670-2
    doi.org/10.1016/j.joule.2022.09.002
    """    
    ### I - Diffusion Part
    ## Define the spacial domains
    x = np.arange(0,thickness,5)            # linear spacing with a point every 5 nm               
    z_array_np = x*1e-7                     # in cm

    # as tensor
    z_array = at.as_tensor_variable(z_array_np)
    z_array_4d = z_array.dimshuffle('x','x',0, 'x')
    z_array_4d.broadcastable
    (True, True, False, True)
    
    # change into 4d tensor for later
    z_array_diff = at.as_tensor_variable(np.diff(z_array_np))
    z_array_diff_4d = z_array_diff.dimshuffle('x', 'x', 0, 'x')
    z_array_diff_4d.broadcastable
    (True, True, False, True)
    
    # change into 3d tensor for later
    z_array_diff_3d = z_array_diff.dimshuffle('x', 0, 'x')
    z_array_diff_3d.broadcastable
    (True, False, True)

    ## Define the time-domain as tensor
    time_4d = time.dimshuffle('x', 'x', 'x', 0)
    time_4d.broadcastable
    (True, True, True, False)

    dt = at.extra_ops.diff(time).T


    ## Define Surface Recombination (S is in cm s^-1)
    # Surface recombination velocities are defined as the sum (S_sum), the ratio of sum and product (S_sumprod_ratio) and
    S_sum_model = pm.Deterministic('S_sum_model', 1*(1+pm.LogNormal('S_sum_factor', 0, 3)))
    S_sumprod_ratio_model = pm.Deterministic('S_sumprod_ratio_model', 1e-5+(1-1e-5)*pm.Beta('S_sumprod_ratio_factor', 2, 2))
    
    # S_mix determines, if S_1 is S_front or S_2 is S_front
    S_mix = pm.Bernoulli('S_mix',0.5)

    # Calculate the high and low surface recombination velocities from S_sum and S_b
    S_low = pm.Deterministic('S_low', (S_sum_model - at.sqrt(S_sum_model**2 *(1 - S_sumprod_ratio_model)))/2)
    S_high = pm.Deterministic('S_high', (S_sum_model + at.sqrt(S_sum_model**2 *(1 - S_sumprod_ratio_model)))/2)

    # Calculate S_1 and S_2
    S_1 = pm.Deterministic('S_1', S_low*(1-S_mix) + S_high*S_mix)
    S_2 = pm.Deterministic('S_2', S_low*S_mix + S_high*(1-S_mix))

    # Determine S_front
    S_front_value = at.switch(at.eq(Surface, 1),  S_1,  S_2)
    S_front = S_front_value.dimshuffle(0,'x','x','x')
    S_front.broadcastable
    (False, True, True, True)

        
    ## Define vertical Mobility/Diffusion (Diffusion Coeffient in cm2 s^-1, Mobility in cm2 (Vs)^-1)
    mu_vert = pm.Deterministic('mu_vert', 0.01*(1+pm.LogNormal('mu_fact', 3, 2)))
    Diffusion_coefficient = pm.Deterministic('Diffusion_coefficient', mu_vert*(1.380649e-23*292/1.6021766e-19))


    ## Calculate 'beta'-values of the Eigenvalue function (Equation 6 in paper or see doi.org/10.1063/1.4896484)   
    beta_model = beta_rootfinder(thickness, S_sum_model, S_sumprod_ratio_model, Diffusion_coefficient, 7)    
    beta_0 = pm.Deterministic('beta_0', beta_model[0])
    beta_4d = beta_model.dimshuffle('x', 0 ,'x','x')
    beta_4d.broadcastable
    (True, False, True, True)
    

    ## Calculate U_j for all beta_j (see doi.org/10.1063/1.4896484)
    U_j = at.cos(beta_4d*z_array_4d) + S_front/(Diffusion_coefficient*beta_4d)*at.sin(beta_4d*z_array_4d)
    U_j.broadcastable
    (False, False, False, True)
   

    ## Calculate A_j from U_j and beta_j (see doi.org/10.1063/1.4896484)
    Fluence_3d = (Fluence*Absorption_coeff).dimshuffle(0,'x','x')
    Fluence_3d.broadcastable
    (False,  True, True)

    A_j = Fluence_3d * at.sum((((at.exp(-Absorption_coeff*z_array_4d)*U_j)[:,:,1:,:]+(at.exp(-Absorption_coeff*z_array_4d)*U_j)[:,:,:-1,:])/2*z_array_diff_4d),axis=2)/at.sum(((U_j[:,:,1:,:]**2+(U_j[:,:,:-1,:]**2))/2*z_array_diff_4d),axis=2)
    A_j_4d = A_param.dimshuffle(0, 1, 2, 'x')
    A_j_4d.broadcastable
    (False, False, True, True)

    
    ## Calculate n_tz0 as the 'initial' carrier density distribution from diffusion and surface recombination alone
    # (see doi.org/10.1063/1.4896484)
    n_tz0 = (A_j_4d * U_j * at.exp(-(Diffusion_coefficient*beta_4d**2)*time_4d) ).sum(axis=1)


    
    ### II - Bulk Recombination Part
    ## Define Radiative Recombination (k_rad in cm^3 s^-1)
    k_rad_model = pm.Deterministic('k_rad_model', 1e-9/(1+pm.LogNormal('k_rad_model_fact',2, 2)))

    # Reabsorption via escape probability
    escape_prob_fact = pm.Beta('escape_prob_fact', 2, 2, shape=2)
    escape_prob = pm.Deterministic('escape_prob', escape_prob_fact)
    
    escape_prob_front = at.switch(at.eq(Surface,1), escape_prob[0], escape_prob[1])
    escape_prob_front = at.switch(at.eq(reabsorption_option,0), at.ones(shape=at.shape(escape_prob_front)), escape_prob_front)

    P_esc = at.outer(escape_prob_front, at.ones(shape=at.shape(z_array)))
    P_esc_3d = P_esc.dimshuffle('x', 1, 0)
    P_esc_3d.broadcastable
    (True, False, False)

    ## Define Non-Radiative Recombination
    # Trapping rates of electrons (n) and holes (p) in s^-1
    kc_n = pm.LogNormal("kc_n_rate", 10, 3)
    kc_p = pm.Deterministic("kc_p_rate", kc_n*pm.LogNormal('kc_p_fact', -3, 3))

    # Trap density in cm^-3
    N_t_model = pm.LogNormal('N_t_model', 35, 3) 

    # energetic depth of trap in eV and density of emitted electrons from the trap (in cm^-3)
    trap_depth = pm.Deterministic('trap_depth', pm.Beta('trap_depth_fact', 2, 2)*0.5)  # in eV
    n_em_1 = 1e18 * at.exp(-trap_depth/(8.62e-5 * 292))
       
    # Auger recombination constant
    k_aug = pm.Deterministic('k_aug', 5e-27/(1+pm.LogNormal('k_aug_fact', 5, 2)))
                 
    
    ## loop over time-domain over 10 pulses to estimate n_eq and p_eq
    result_outer, _ = pytensor.scan(fn=multiple_pulses, outputs_info = [at.zeros(shape=at.shape(n_tz0[:,0,:].T)), at.zeros(shape=at.shape(n_tz0[:,0,:].T))], non_sequences=[at.ones(at.shape(dt)), at.ones(at.shape(dt)), dt, at.as_tensor_variable(Fluence/(thickness*1e-7)), kc_n, kc_p, n_em_1, k_rad_model, k_aug, N_t_model, escape_prob_front], n_steps=10)

    N_bckg = result_outer[0][-1,-1,:]
    N_bckg = at.switch(at.le(N_bckg, 0), 0, N_bckg)
    
    P_bckg = result_outer[1][-1,-1,:] #- result_outer[0][-1,-1, 0,:]
    P_bckg = at.switch(at.le(P_bckg, 0), 0, P_bckg)
    N_eq = [N_bckg, P_bckg]
       
    n_eq_models = pm.Deterministic('n_eq_models', N_bckg)
    n_eq_models = pm.Deterministic('p_eq_models', P_bckg)

    ## Loop over time-domain for one pulse while including n_eq and p_eq
    result_PL, _ = pytensor.scan(fn=total_recombination_rate,
                                            sequences=[n_tz0[:,:,:-1].T, n_tz0[:,:,1:].T, dt],
                                            outputs_info=[n_tz0[:,:,0].T, n_tz0[:,:,0].T],
                                            non_sequences=[kc_n, kc_p, n_em_1, k_rad_model, N_t_model, P_esc.T, k_aug, N_eq])

    n_tz_init = (n_tz0[:,:,0].T).dimshuffle('x', 0, 1)
    N_calc = at.concatenate([n_tz_init, result_PL[0]], axis=0)
        
    p_tz_init = (n_tz0[:,:,0].T).dimshuffle('x', 0, 1)
    P_calc = at.concatenate([p_tz_init, result_PL[1]], axis=0)

    ## Calculate R_rad
    N_bckg = N_bckg.dimshuffle('x', 'x', 0)
    N_bckg.broadcastable
    (True, True, False)

    P_bckg = P_bckg.dimshuffle('x', 'x', 0)
    P_bckg.broadcastable
    (True, True, False)
    
    Rrad_calc = (N_calc+N_bckg)*(P_calc+P_bckg)-N_bckg*P_bckg

    ## Calculate PL_calc
    PL_calc = at.sum(((Rrad_calc[:,1:,:] + Rrad_calc[:,:-1,:])/2*z_array_diff_3d), axis=1)

    PL_err = pm.Deterministic('PL_err', bckg_list/ (1 + pm.LogNormal('PL_err_fact', 0, 2, shape=at.shape(bckg_list))))
    PL_err_2d = at.outer(at.ones(shape=at.shape(time)), PL_err)

    ## Normalize PL_calc
    PL_0 = PL_calc[0,:]*(1-PL_err)
    PL_obs = PL_calc/PL_0 + PL_err_2d
    
    return PL_obs


def multi_exp_approximation(x,y, A1, A2, A3, tau1, tau2, tau3, beta2, beta3, y0):
    
    return A1*np.exp(-x/tau1) + A2*np.exp(-(x/tau2)**beta2) + A3*np.exp(-(x/tau3)**beta3) + 10**(-y0)
    

def glm_mcmc_inference_diffusion_full(Data_fit, a, Fluence, Surface, Thickness, Absorption_coeff, tune_no, draws_no, cores_no, max_arg, bckg_list, reabsorption_option):

    ### Setting up the Data and Timeframe
    time = np.array(Data_fit['Time'])[a]*1e-9
    y_combined = np.zeros((len(time),len(Surface)))
    sigmas = np.zeros((len(time),len(Surface)))

    ### Pre-fit to determine a spline of the data and a value for sigma
    for s in range(len(Surface)):
        data = np.array(Data_fit[str(s)])

        popt, _ = curve_fit(multi_exp_approximation, Data_fit['Time'], data, maxfev=100000)
        spline_fit = multi_exp_approximation(Data_fit['Time'], *popt)
        
        y_combined[:,s] = spline_fit[a]
        sigma_calcs = medfilt(np.sqrt((np.sqrt(spline_fit)-np.sqrt(data))**2),51)

        spline = UnivariateSpline(Data_fit['Time'], np.log(sigma_calcs))
        spline_fit_sigma = np.exp(spline(Data_fit['Time']))
        
        sigmas[:,s] = spline_fit_sigma[a]

    
    with pm.Model() as model:

        ## set up tensors
        y_combined = at.as_tensor_variable(y_combined)
                       
        ## Simulate Time-Resolved PL
        N_calc = diffusion_carrier_density_all(shared(time), Fluence, Surface, Thickness, shared(Absorption_coeff[0]), at.as_tensor_variable(bckg_list), shared(reabsorption_option))
        N_calc_collect = pm.Deterministic('N_calc_collect', N_calc)

        ## Likelihood Function
        sigma_width = pm.Deterministic('sigma_width', 0.1/(1+pm.LogNormal('sigma_width_fact', 2, 2))) 
        Logp = pm.Deterministic('Logp', pm.logp(rv=pm.Normal.dist(mu=at.sqrt(N_calc), sigma = sigmas+sigma_width), value=at.sqrt(y_combined)))
        Y_obs = pm.Potential('Y_obs', (1*Logp))
        
        ## Draw Samples from the Posterior Distribution
        print("Bayes-MCMC Sampling Running...")
        print(str("(tune: " +str(tune_no)+", draws: " +str(draws_no)+", chains: " +str(cores_no)+ ")"))
        trace = pm.sample(step=pm.Metropolis(),  chains=cores_no, draws=draws_no, cores=cores_no, tune=tune_no)

    return trace


def save_trace(trace, folder, config_name, spacing, reabsorption, laserpower_file):

    tracename = config_name[:-4]
    date_time_code = str(datetime.datetime.now()).split('.')
    date_time_code = date_time_code[0].replace(" ","_").replace(":","")

    sample_name = str(tracename + "_trace_" + date_time_code)

    
    trace.to_netcdf(f"{folder}/{sample_name}.nc")
    print("Trace has been saved!")
    
    
    text_file = open(f"{folder}/{sample_name}_logfile.txt", 'w')
    my_string = str(spacing +"\t" + reabsorption + "\t" + laserpower_file)
    text_file.write(my_string)
    text_file.close()
    print("Logfile has been saved!")


def run_bayesian_inference(df, max_arg, spacing, Fluence, Surface, Thickness, Absorption_coeff, tune_no, draws_no, cores_no, folder, config_name, bckg_list, reabsorption, laserpower_file):

    a = spacing_choose(spacing, max_arg)
    print(spacing, len(a))
    
    if reabsorption == 'No':
        reabsorption_option = 0
    else:
        reabsorption_option = 1
    
    trace = glm_mcmc_inference_diffusion_full(df, a, Fluence, Surface, Thickness, Absorption_coeff, tune_no, draws_no, cores_no, max_arg, bckg_list, reabsorption_option)
    print("Bayes-MCMC done!")
    print(" ")

    save_trace(trace, folder, config_name, spacing, reabsorption, laserpower_file)
    
    return trace



































