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

import logging
logger = logging.getLogger("pymc")
logger.setLevel(logging.ERROR)

import warnings
warnings.filterwarnings('ignore')


def root_finder_loop_perform(D, a, b, thickness, noroots):
    # Define an Aesara expression for the function to be rooted
    def f(x, D, a, b, thickness):
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



"""This is the Diffusion model in PyTensor

It is mainly based on:  J. Appl. Phys. 116, 123711 (2014).
The important equations are marked.
In addition I took inspiration from:
https://www.nature.com/articles/s41598-019-41716-x#Sec16
https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-017-02670-2/MediaObjects/41467_2017_2670_MOESM1_ESM.pdf
and 
https://www.sciencedirect.com/science/article/pii/S2542435122004202#sec6.2

"""

def diffusion_carrier_density_all(time, Fluence, Surface, thickness, Absorption_coeff, bckg_list, reabsorption_option):
    
    ### I - Diffusion Part
    # I model the diffusion by using an approach described by R.K. Ahrenkiehl (Solid-State Electronics Vol. 35, No. 3, pp. 239-250, 1992)
    # as well as M Maiberg & R. Scheer (Journal of Applied Physics 116, 123711 (2014)).
    # To improve reproducability with the NUTS algorithm, I define the surface recombination in terms of their sum (S_sum) and the factor S_b, which
    # relates the product as: S_prod = (S_sum/2)^2*S_b.
    ## Define the spacial domains

    x = np.arange(0,thickness,5)

    z_array_np = x*1e-7

    z_array = at.as_tensor_variable(z_array_np)
    z_array_4d = z_array.dimshuffle('x','x',0, 'x')
    z_array_4d.broadcastable
    (True, True, False, True)
    
    z_array_diff = at.as_tensor_variable(np.diff(z_array_np))
    z_array_diff_4d = z_array_diff.dimshuffle('x', 'x', 0, 'x')
    z_array_diff_4d.broadcastable
    (True, True, False, True)
 
    z_array_diff_3d = z_array_diff.dimshuffle('x', 0, 'x')
    z_array_diff_3d.broadcastable
    (True, False, True)

    
    ## Define Parameters
    # Surface recombination velocities are defined as the sum (S_sum), the ratio of sum and product (S_sumprod_ratio) and
    S_sum_model = pm.Deterministic('S_sum_model', 1*(1+pm.LogNormal('S_sum_factor', 3, 3)))
   
    S_sumprod_ratio_model = pm.Deterministic('S_sumprod_ratio_model', 1e-5+(1-1e-5)*pm.Beta('S_sumprod_ratio_factor', 2, 2))
    
    S_mix = pm.Bernoulli('S_mix',0.5)
        
    ## Calculate both surface recombination velocities from S_sum and S_b
    S_a = pm.Deterministic('S_low', (S_sum_model - at.sqrt(S_sum_model**2 *(1 - S_sumprod_ratio_model)))/2)
    S_b = pm.Deterministic('S_high', (S_sum_model + at.sqrt(S_sum_model**2 *(1 - S_sumprod_ratio_model)))/2)
    
    # Effective S_front
    S_1 = pm.Deterministic('S_1', S_a*(1-S_mix) + S_b*S_mix)
    S_2 = pm.Deterministic('S_2', S_a*S_mix + S_b*(1-S_mix))
   
    S_front_value = at.switch(at.eq(Surface, 1),  S_1,  S_2)

    S_front = S_front_value.dimshuffle(0,'x','x','x')
    S_front.broadcastable
    (False, True, True, True)

    # Reabsorption

    escape_prob_fact = pm.Beta('escape_prob_fact', 2, 2, shape=2)   
    escape_prob = pm.Deterministic('escape_prob', escape_prob_fact)
    
    escape_prob_front = at.switch(at.eq(Surface,1), escape_prob[0], escape_prob[1])
    escape_prob_front = at.switch(at.eq(reabsorption_option,0), at.ones(shape=at.shape(escape_prob_front)), escape_prob_front)
    P_esc = at.outer(escape_prob_front, at.ones(shape=at.shape(z_array)))
    P_esc_3d = P_esc.dimshuffle('x', 1, 0)
    P_esc_3d.broadcastable
    (True, False, False)
    
    # Diffusion Coeffient in cm2 s-1
    mu_vert = pm.Deterministic('mu_vert', 0.01*(1+pm.LogNormal('mu_fact', 3, 3)))
    
    Diffusion_coefficient = pm.Deterministic('Diffusion_coefficient', mu_vert*(1.380649e-23*292/1.6021766e-19))
    
        
    ## Calculate 'beta'-values of the Eigenvalue function

    a = S_sum_model
    b = S_sumprod_ratio_model
    
    beta_model = beta_rootfinder(thickness, a, b, Diffusion_coefficient, 7)    
    beta_0 = pm.Deterministic('beta_0', beta_model[0])
    beta_4d = beta_model.dimshuffle('x', 0 ,'x','x')
    beta_4d.broadcastable
    (True, False, True, True)

   
    
    ## Here the U_z function is calculates, which will impact the actual diffusion of the charge carriers in the model
    U_z = at.cos(beta_4d*z_array_4d) + S_front/(Diffusion_coefficient*beta_4d)*at.sin(beta_4d*z_array_4d)
    U_z.broadcastable
    (False, False, False, True)
   
    
    ## This part calculates the A_param, which will take care of the ratios of U_z functions, so the total sum ends up in cm-3
    Fluence_3d = (Fluence*Absorption_coeff).dimshuffle(0,'x','x')
    Fluence_3d.broadcastable
    (False,  True, True)
        
    A_param = Fluence_3d * at.sum((((at.exp(-Absorption_coeff*z_array_4d)*U_z)[:,:,1:,:]+(at.exp(-Absorption_coeff*z_array_4d)*U_z)[:,:,:-1,:])/2*z_array_diff_4d),axis=2)/at.sum(((U_z[:,:,1:,:]**2+(U_z[:,:,:-1,:]**2))/2*z_array_diff_4d),axis=2)
    A_param_4d = A_param.dimshuffle(0, 1, 2, 'x')
    A_param_4d.broadcastable
    (False, False, True, True)


    ## Now the time-domain is introduced with the same resolution as the data
    time_4d = time.dimshuffle('x', 'x', 'x', 0)
    time_4d.broadcastable
    (True, True, True, False)
    
    ## n_tz0 is the 'initial' carrier density distribution from diffusion and surface recombination alone
    n_tz0 = (A_param_4d * U_z * at.exp(-(Diffusion_coefficient*beta_4d**2)*time_4d) ).sum(axis=1) # sum over all beta values


    ########################################################################
    ########################################################################
    
    ### II - Recombination Part
    ## Define Parameters
    # Radiative Recombination Rate
    k_rad_model = pm.Deterministic('k_rad_model', 1e-9*pm.Beta('k_rad_model_fact',2, 2))

    #n_eq_model = Fluence*pm.LogNormal('n_eq_model_fact', 2, 1)
    #n_eq_min = pm.Deterministic('n_eq_model', at.min(n_eq_model))
    #n_eq_model = (n_eq_model).dimshuffle(0,'x')
    #n_eq_model.broadcastable
    #(False, True)

    N_t_model = pm.LogNormal('N_t_model', 35, 3) 
    
    kc_n = pm.LogNormal("kc_n_rate", 10, 3)
    kc_p = pm.Deterministic("kc_p_rate", kc_n*pm.LogNormal('kc_p_fact', -3, 3))
    
    k_aug = pm.Deterministic('k_aug', 5e-27*pm.Beta('k_aug_fact', 2, 2))
    
    trap_depth = pm.Deterministic('trap_depth', pm.Beta('trap_depth_fact', 1, 1)*0.5)  # in eV
    n_em_1 = 1e18 * at.exp(-trap_depth/(8.62e-5 * 292))
    
    
    ## turn diffusion/surface recombination into a correctoin factor
    dt = at.extra_ops.diff(time).T
    n_tz0_1 = n_tz0[:,:,:-1]
    n_tz0_2 = n_tz0[:,:,1:]
          
    
    ### Looping over time-domain
    def total_recombination_rate(n_tz0_1, n_tz0_2, dt_current, n_dens, p_dens, kc_n, kc_p, ne_1, krad, N_t, p_esc, k_aug, n_eq):
      
        ### First: Define Rate equations
        def rate_equations(n_dens0, p_dens0, kc_n, kc_p, ne_1, krad, k_aug, N_t, p_esc, eq_carr):            
            ### Pseudo-Equilibrium at each time-point
            
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
      
        # Runge-Kutta Algorithm
        n_ds = n_tz0_2/n_tz0_1


        
        RuKu1_n, RuKu1_t = rate_equations(n_dens, p_dens, kc_n, kc_p, ne_1, krad, k_aug, N_t, p_esc, n_eq)
        RuKu2_n, RuKu2_t = rate_equations(n_dens + RuKu1_n*dt_current/2, p_dens + RuKu1_t*dt_current/2, kc_n, kc_p, ne_1, krad, k_aug, N_t, p_esc, n_eq)
        RuKu3_n, RuKu3_t = rate_equations(n_dens + RuKu2_n*dt_current/2, p_dens + RuKu2_t*dt_current/2,kc_n, kc_p, ne_1, krad, k_aug, N_t, p_esc, n_eq)
        RuKu4_n, RuKu4_t = rate_equations(n_dens + RuKu3_n*dt_current, p_dens + RuKu3_t*dt_current,kc_n, kc_p, ne_1, krad, k_aug, N_t, p_esc, n_eq)


        n_next = (n_dens + dt_current/6*(RuKu1_n + 2*RuKu2_n + 2*RuKu3_n + RuKu4_n))*n_ds
        p_next = (p_dens + dt_current/6*(RuKu1_t + 2*RuKu2_t + 2*RuKu3_t + RuKu4_t))*n_ds

        n_next = at.switch(at.le(n_next, 0), 0, n_next)
        p_next = at.switch(at.le(p_next, 0), 0, p_next)
        
        return n_next, p_next

    
    n_0z = n_tz0[:,:,0]
    
    PL_err = pm.Deterministic('PL_err', bckg_list/ (1 + pm.LogNormal('PL_err_fact', 0, 2, shape=at.shape(bckg_list))))
    PL_err_2d = at.outer(at.ones(shape=at.shape(time)), PL_err)
       
    
    def multiple_pulses(n_tz0, p_tz0, n_tz0_1, n_tz0_2, dt, n_0z, kc_n, kc_p, n_em_1, k_rad_model, k_aug, N_t, p_esc):
    
        #n_tz_init = n_0z + n_tz0[-1,:,:]
        #p_tz_init = p_0z p_tz0[-1,:,:]
        
        
        result_inner, _ = pytensor.scan(fn=total_recombination_rate,
                                            sequences=[n_tz0_1, n_tz0_2, dt],
                                            outputs_info=[n_0z + n_tz0[-1,:], n_0z + p_tz0[-1,:]],
                                            non_sequences=[kc_n, kc_p, n_em_1, k_rad_model, N_t, p_esc, k_aug, [0, 0]])
    
    
        n_tz_init = (n_0z+n_tz0[-1,:]).dimshuffle('x', 0)
        n_tz_new = at.concatenate([n_tz_init, result_inner[0]], axis=0)
        
        p_tz_init = (n_0z+ p_tz0[-1,:]).dimshuffle('x', 0)
        p_tz_new = at.concatenate([p_tz_init, result_inner[1]], axis=0)
        
        return n_tz_new, p_tz_new

    result_outer, _ = pytensor.scan(fn=multiple_pulses, outputs_info = [at.zeros(shape=at.shape(n_tz0[:,0,:].T)), at.zeros(shape=at.shape(n_tz0[:,0,:].T))], non_sequences=[at.ones(at.shape(dt)), at.ones(at.shape(dt)), dt, at.as_tensor_variable(Fluence/(thickness*1e-7)), kc_n, kc_p, n_em_1, k_rad_model, k_aug, N_t_model, escape_prob_front], n_steps=10)


    N_bckg = result_outer[0][-1,-1,:]
    N_bckg = at.switch(at.le(N_bckg, 0), 0, N_bckg)
    
    P_bckg = result_outer[1][-1,-1,:] #- result_outer[0][-1,-1, 0,:]
    P_bckg = at.switch(at.le(P_bckg, 0), 0, P_bckg)

    N_eq = [N_bckg, P_bckg]
        
    result_PL, _ = pytensor.scan(fn=total_recombination_rate,
                                            sequences=[n_tz0[:,:,:-1].T, n_tz0[:,:,1:].T, dt],
                                            outputs_info=[n_0z.T, n_0z.T],
                                            non_sequences=[kc_n, kc_p, n_em_1, k_rad_model, N_t_model, P_esc.T, k_aug, N_eq])


    n_tz_init = (n_0z.T).dimshuffle('x', 0, 1)
    N_calc = at.concatenate([n_tz_init, result_PL[0]], axis=0)
        
    p_tz_init = (n_0z.T).dimshuffle('x', 0, 1)
    P_calc = at.concatenate([p_tz_init, result_PL[1]], axis=0)

    N_bckg = N_bckg.dimshuffle('x', 'x', 0)
    N_bckg.broadcastable
    (True, True, False)

    P_bckg = P_bckg.dimshuffle('x', 'x', 0)
    P_bckg.broadcastable
    (True, True, False)
        
    
    Rrad_calc = (N_calc+N_bckg)*(P_calc+P_bckg)-N_bckg*P_bckg#-(N_bckg*(N_bckg+N_trapped))

    
    PL_calc = at.sum(((Rrad_calc[:,1:,:] + Rrad_calc[:,:-1,:])/2*z_array_diff_3d), axis=1)

    PL_0 = PL_calc[0,:]*(1-PL_err)

    PL_obs = PL_calc/PL_0 + PL_err_2d
    
    return PL_obs




def glm_mcmc_inference_diffusion_full(Data_fit, a, Fluence, Surface, Thickness, Absorption_coeff, tune_no, draws_no, cores_no, max_arg, bckg_list, reabsorption_option):

   
    
    #### Setting up the Data and Timeframe
    time = np.array(Data_fit['Time'])[a]*1e-9    #s
        
    y_combined = np.zeros((len(time),len(Surface)))
    
    for s in range(len(Surface)):
        y_combined[:,s] = np.array(Data_fit[str(s)])[a]
    
    with pm.Model() as model:
        
        
                        
        #### Simulation of Time-Resolved PL
        N_calc = diffusion_carrier_density_all(shared(time), Fluence, Surface, Thickness, shared(Absorption_coeff[0]), at.as_tensor_variable(bckg_list), shared(reabsorption_option))
        
        ## Likelihood Function Student-T Distribution)
       
        sigma_width = pm.Deterministic('sigma_width', 0.05/(1+pm.LogNormal('sigma_width_fact', 1, 2, shape=len(Surface))))            
        #potential = pm.Potential('sigma_potential', sigma_width)
        
        sigma_width = at.outer(at.ones(shape=len(time)), sigma_width)
        
        y_combined = at.as_tensor_variable(y_combined)        

        

        N_calc_collect = pm.Deterministic('N_calc_collect', N_calc)
                                          
        Logp = pm.Deterministic('Logp', pm.logp(rv=pm.Normal.dist(mu=at.sqrt(N_calc), sigma=sigma_width), value=at.sqrt(y_combined)))        
        Y_obs = pm.Potential('Y_obs', (at.sqrt(sigma_width)*Logp))
        
        
        #### Draw Samples from the Posterior Distribution
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



































