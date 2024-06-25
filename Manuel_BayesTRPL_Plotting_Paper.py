import os
folder = os.getcwd()

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import random
from scipy import stats
from scipy.optimize import root

import warnings
warnings.filterwarnings('ignore')



def plot_kernel_1D(a):

    x_a = np.logspace(np.log10(np.min(a)),np.log10(np.max(a)),1000)
    kernel_1d = stats.gaussian_kde(a)(x_a)
    return x_a, kernel_1d/np.max(kernel_1d)

def plot_loghist(a, ax_a, log, colour):
    a = a.ravel()
    
    if log == 'log':
        bins = np.logspace(np.log10(a.min()/5), np.log10(a.max()*5),50)
    
    else:
        bins = np.linspace(a.min(), a.max(),50)
    
    x,_,_ = ax_a.hist(a, bins=bins, color=colour, alpha=0.6)
    ax_a.set_ylim(bottom=0, top=2*x.max())
    ax_a.set_yticks([])
    

def print_function(a_values, resol):
    
    a_test = a_values.ravel()
    a_median = np.median(a_test)
    
    if np.round(np.log10(a_median),0)-np.round(np.log10(a_median),1) > 0.05: 
        order_of_magn = np.round(np.log10(a_median),0)-1
    else:
        order_of_magn = np.round(np.log10(a_median),0)

    quantile_1 = np.quantile(a_test, 0.25)
    quantile_3 = np.quantile(a_test, 0.75)

    if resol == 'lin':
        text = str("{:.2f}".format(a_median) +" ({:.2f}".format(quantile_1) + " — {:.2f}".format(quantile_3)+ ")")
    if resol == 'log':
        text = str("{:.1f}".format(a_median/10**order_of_magn) +" ({:.1f}".format(quantile_1/10**order_of_magn)+" — {:.1f}".format(quantile_3/10**order_of_magn)+ r") $\times 10^{\mathrm{" + "{:.0f}".format(order_of_magn) + "}}$")   
    
    return text



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



def plot_and_save(trace, a, df, Fluence, Surface, Thickness, scaling, sample_name, data_folder_trpl, one_sun_carrier_density, max_arg, pile_up, side_1, side_2, reabs_option):
    
    # Plotting Parameters
    centimeters = 1/2.54
    fontsize_base = 11
    color1 = iter(cm.Set2(np.arange(7)))#iter(cm.Set2(np.linspace(0, 1, (len(Surface)+1))))
    color2 = iter(cm.Dark2(np.arange(7)))

    color_prism = ['#1D6996','#38A6A5','#0F8554','#73AF48','#EDAD08','#E17C05','#CC503E','#94346E','#6F4070','#994E95','#666666']


    time_plot = np.array(df['Time'])[a]   #s

    # DataFrame to save all draws
    df_save = pd.DataFrame()
    
    final_sample_no = 9000

    # Setup of Figure
    fig_final = plt.figure(figsize=(24*centimeters, 32*centimeters))
    #plt.subplots_adjust(wspace=0.2, hspace=0.1)
    
    gs_final_combined = fig_final.add_gridspec(2,1, height_ratios=[1,3.3])
    gs_final_main = gs_final_combined[0].subgridspec(1,6)

    gs_final = gs_final_combined[1].subgridspec(4,6, hspace=0.2, wspace=0.2)

    ax_main = fig_final.add_subplot(gs_final_main[0, 1:5])
    ax_main.set_title('Normalized TRPL Decays + Median of Bayes-MCMC', fontsize=fontsize_base+2)
    ax_main.set_xlabel('Time after Pulse [ns]')
    ax_main.set_ylabel('Normalized PL Intensity [a.u.]')
    ax_main.annotate(f'(a) {sample_name}' ,[0.05, 0.9] , xycoords='axes fraction', fontsize=fontsize_base+1)
    ax_main.annotate(str('pile-up $\leq$' + '{:.1f}'.format(np.max(pile_up)) + "%") ,[0.75, 0.75] , xycoords='axes fraction', fontsize=fontsize_base)

    ax_ylabel = fig_final.add_subplot(gs_final[0:4, 0:6])
    ax_ylabel.set_title('Parameter Histograms', fontsize=fontsize_base+2)
    ax_ylabel.set_ylabel('Sample Frequency [a.u.]')
    ax_ylabel.set_xlabel('', fontsize=fontsize_base+1)
    ax_ylabel.set_xticklabels("")
    ax_ylabel.spines['top'].set_color('none')
    ax_ylabel.spines['bottom'].set_color('none')
    ax_ylabel.spines['left'].set_color('none')
    ax_ylabel.spines['right'].set_color('none')
    ax_ylabel.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)

    ax_mob = fig_final.add_subplot(gs_final[2, 2:4])
    #ax_mob.set_xlabel('Mobility [cm² (Vs)⁻¹]')
    ax_mob.annotate('(i) $\Sigma\mu_{\mathrm{vert}}$ [cm$^2$ (Vs)$^{-1}$]' ,[0.05, 0.88] , xycoords='axes fraction', fontsize=fontsize_base+1)

    ax_tau = fig_final.add_subplot(gs_final[1, 0:2])
    #ax_tau.set_xlabel('$k_{\mathrm{tr}}$ [s$^{-1}$]')
    ax_tau.annotate('(e) $k_{\mathrm{tr}}$ [s$^{-1}$]' ,[0.05, 0.88] , xycoords='axes fraction', fontsize=fontsize_base+1)

    ax_krad = fig_final.add_subplot(gs_final[0, 0:2])
    #ax_krad.set_xlabel('$k_{\mathrm{rad}}$ (cm$^{3}$s$^{-1}$)')
    #ax_krad.annotate('(b) $k_{\mathrm{rad}}$ [cm$^{3}$s$^{-1}$]' ,[0.05, 0.88] , xycoords='axes fraction', fontsize=fontsize_base+1)

    ax_pesc = fig_final.add_subplot(gs_final[0, 4:6])
    #ax_pesc.set_xlabel('$p_{\mathrm{esc}}$ [-]')
    ax_pesc.annotate('(d) $P_{\mathrm{esc}}$ [-]' ,[0.05, 0.88] , xycoords='axes fraction', fontsize=fontsize_base+1)

    ax_Ssub = fig_final.add_subplot(gs_final[1, 4:6])
    #ax_Ssub.set_xlabel('SRV [cm s$^{-1}$]')
    ax_Ssub.annotate('(g) SRV [cm s$^{-1}$]' ,[0.05, 0.88] , xycoords='axes fraction', fontsize=fontsize_base+1)

    ax_neq = fig_final.add_subplot(gs_final[0, 2:4])
    #ax_neq.set_xlabel('$n_{\mathrm{eq}}$ [cm$^{-3}$]')
    ax_neq.annotate('(c) $N_{\mathrm{t}}$ [cm$^{-3}$]' ,[0.05, 0.88] , xycoords='axes fraction', fontsize=fontsize_base+1)

    ax_td = fig_final.add_subplot(gs_final[1, 2:4])
    #ax_td.set_xlabel('$\Delta E_{\mathrm{t}}$ [eV]')
    #ax_td.set_xlim(0,0.5)
    ax_td.annotate('(f) $\Delta E_{\mathrm{t}}$ [meV]' ,[0.05, 0.88] , xycoords='axes fraction', fontsize=fontsize_base+1)

    ax_diffl = fig_final.add_subplot(gs_final[2, 4:6])
    #ax_diffl.set_xlabel('$L_{\mathrm{D}}$ [$\mu m$]')
    ax_diffl.annotate('(j) $L_{\mathrm{D,1sun}}$ [$\mu m$]' ,[0.05, 0.88] , xycoords='axes fraction', fontsize=fontsize_base+1)

    ax_LL = fig_final.add_subplot(gs_final[3, 0:2])
    #ax_LL.set_xlabel('-log(likelihood) [-]')
    ax_LL.annotate('(k) -log($LL$)$_{\mathrm{av}}$ [-]' ,[0.05, 0.88] , xycoords='axes fraction', fontsize=fontsize_base+1)

    ax_fluerr = fig_final.add_subplot(gs_final[2, 0:2])
    #ax_fluerr.set_xlabel('$\epsilon_{\mathrm{F}}$ [%]')
    ax_fluerr.annotate('(h) $k_{\mathrm{aug}}$ [cm$^6$ s$^{-1}$]' ,[0.05, 0.88] , xycoords='axes fraction', fontsize=fontsize_base+1)

    ax_sigmaLL = fig_final.add_subplot(gs_final[3, 2:4])
    #ax_sigmaLL.set_xlabel('$\sigma_{\mathrm{LL}}$ [a.u]')
    ax_sigmaLL.annotate('(l) $\sigma_{\mathrm{LL,med.}}$ [a.u]' ,[0.05, 0.88] , xycoords='axes fraction', fontsize=fontsize_base+1)
    
    ax_PL_err = fig_final.add_subplot(gs_final[3, 4:6])
    #ax_PL_err.set_xlabel('$\epsilon_{\mathrm{PL}}$ [a.u.]')
    ax_PL_err.annotate('(m) $\epsilon_{\mathrm{PL,med.}}$ [a.u.]' ,[0.05, 0.88] , xycoords='axes fraction', fontsize=fontsize_base+1)
  
    


    # Bulk Recombination
    k_rad_model_values = df_save['k_rad(cm3s-1)'] = trace.posterior.k_rad_model.values[:,-final_sample_no:].ravel()       
    #p_D_calc = np.nanmin(np.abs(trace.posterior.p_eq_models.values[:,-final_sample_no:]), axis=2).ravel()

    #print(np.nanmedian(trace.posterior.p_eq_models.values[:,-final_sample_no:], axis=[0,1]))

    #p_D_min = np.min(p_D_calc[np.where(p_D_calc > 0)])
    #p_D_calc[p_D_calc == 0] = p_D_min
    #p_D_model_values = df_save['p_D_min(cm-3)'] = trace.posterior.N_t_model.values[:,-final_sample_no:].ravel()

    N_t_model_values = df_save['N_t(cm-3)'] = trace.posterior.N_t_model.values[:,-final_sample_no:].ravel()

    k_aug_model_values = df_save['k_aug(cm6s-1)'] = trace.posterior.k_aug.values[:,-final_sample_no:].ravel()  

    #N_t = np.median(trace.posterior.n_eq_models.values[:,-final_sample_no:]/trace.posterior.ft_means.values[:,-final_sample_no:], axis=[0,1])


    kc_n_model_values = df_save['kn_trapping(s-1)'] = trace.posterior.kc_n_rate.values[:,-final_sample_no:].ravel()
    kc_p_model_values = df_save['kp_trapping(s-1)'] = trace.posterior.kc_p_rate.values[:,-final_sample_no:].ravel()

    trap_depth_model_values = df_save['td(eV)'] = trace.posterior.trap_depth[:,:].values[:,-final_sample_no:].ravel()
    n1_est = 1e18* np.exp(-trap_depth_model_values*1.60218e-19/(1.380649e-23 * 292))
    

    k_nr1sun = ()
    k_bulk = ()

    def k_nr1sun_estimate(p, n, kc_n, kc_p, ne1, Nt):
        p += n
        #ft = n/(n + kc_p/kc_n*p + ne1)
        ft = (p-n)/Nt

        return kc_n*n*(1-ft) - kc_n*ft*ne1 - kc_p*p*ft

    for i in np.arange(len(kc_n_model_values)):
        solv = root(k_nr1sun_estimate, one_sun_carrier_density, args=(one_sun_carrier_density, kc_n_model_values[i], kc_p_model_values[i], n1_est[i], N_t_model_values[i]))
        k_nr1sun = np.append(k_nr1sun, kc_p_model_values[i]*(one_sun_carrier_density + solv.x)/(one_sun_carrier_density+kc_p_model_values[i]/kc_n_model_values[i]*(one_sun_carrier_density +solv.x) + n1_est[i]))
        k_bulk = k_nr1sun + k_rad_model_values[i]*(one_sun_carrier_density)

    df_save['k_nr1sun(s-1)'] = k_nr1sun #= kc_p_model_values*one_sun_carrier_density/(one_sun_carrier_density + kc_p_model_values/kc_n_model_values*one_sun_carrier_density + n1_est)
    df_save['k_bulk1sun(s-1)'] = k_bulk #= df_save['k_nr1sun(s-1)']  + k_rad_model_values*(one_sun_carrier_density)

    
    ## Surface Recombination
    S_mix_model_values = df_save['S_mix'] = trace.posterior.S_mix.values[:,-final_sample_no:].ravel()
    
    S_1 = trace.posterior.S_1.values[:,-final_sample_no:].ravel()
    S_2 = trace.posterior.S_2.values[:,-final_sample_no:].ravel()

    S_low = trace.posterior.S_low.values[:,-final_sample_no:].ravel()
    S_high = trace.posterior.S_high.values[:,-final_sample_no:].ravel()

    if np.abs(np.mean(S_mix_model_values)-0.5) < 0.05:
        S_report_1 = df_save['S_low(cm s-1)'] = S_1
        S_plot_1_label = "S$_{low}$"
        S_report_2 = df_save['S_high(cm s-1)'] = S_2
        S_plot_2_label = "S$_{high}$"

    else:
        S_report_1 = df_save['S_1(cm s-1)'] = S_1
        S_plot_1_label = "$S_{\mathrm{" + side_1 + "}}$"
        S_report_2 = df_save['S_2(cm s-1)'] = S_2
        S_plot_2_label = "$S_{\mathrm{" + side_2 + "}}$"


    
    ## Diffusion
    Diffusion_coeff_values = trace.posterior.Diffusion_coefficient.values[:,-final_sample_no:].ravel()
    Mobility_values = df_save['Mobility_values(cm2V-1s-1)'] = 2*trace.posterior.mu_vert.values[:,-final_sample_no:].ravel()
    Diffusion_length = df_save['Diffusion_length(um)'] = np.sqrt(Diffusion_coeff_values/k_bulk)*1e4
    df_save['beta_1(-)'] = trace.posterior.beta_0.values[:,-final_sample_no:].ravel()*Thickness*1e-7


    ## Escape Probability 

    

    ## Logp vals
    logp_vals = np.abs(trace.posterior.Logp.values[:,-final_sample_no:,:,:])
    
    logp_vals[logp_vals == np.inf] = 0
    

    #chains = np.shape(logp_vals)[0]
    points = np.shape(logp_vals)[2]
    sets = np.shape(logp_vals)[3]
    N = points*sets

    LL = np.nansum(logp_vals,axis=(2,3))
    LL_norm = LL/N

    df_save['LL'] = LL.ravel()
    df_save['LL_av'] = LL_norm.ravel()
    
    # sigmas and PL_err
    for i in np.arange(len(Surface)):
        df_save[f'sigma_LL_{i}'] = trace.posterior.sigma_width.values[:,-final_sample_no:,i].ravel()
        df_save[f'PL_err_{i}'] =  trace.posterior.PL_err.values[:,-final_sample_no:,i].ravel()



    ### Plotting the Probability Densities
    #ax_mob.plot(plot_kernel_1D(Mobility_values)[0], plot_kernel_1D(Mobility_values)[1], c=color_prism[0], linewidth=2, alpha=0.8)
    plot_loghist(Mobility_values, ax_mob, 'log', color_prism[0])
    ax_mob.annotate(print_function(Mobility_values, 'log'),[0.1, 0.76] , xycoords='axes fraction', fontsize=fontsize_base-2, c=color_prism[0])

    #ax_krad.plot(plot_kernel_1D(k_rad_model_values)[0], plot_kernel_1D(k_rad_model_values)[1], c=color_prism[0], linewidth=2, alpha=0.8)
    plot_loghist(k_rad_model_values, ax_krad, 'log', color_prism[0])
    #ax_neq.plot(plot_kernel_1D(N_t_model_values)[0], plot_kernel_1D(N_t_model_values)[1], c='dimgrey', linewidth=2, alpha=0.8)
    #ax_neq.plot(plot_kernel_1D(p_D_model_values)[0], plot_kernel_1D(p_D_model_values)[1], c=color_prism[2], linewidth=2, alpha=0.8)
    plot_loghist(N_t_model_values, ax_neq, 'log', color_prism[0])
    
    ax_neq.annotate(print_function(N_t_model_values, 'log'),[0.1, 0.76] , xycoords='axes fraction', fontsize=fontsize_base-2, c=color_prism[0])
    #ax_neq.annotate(str('$p_{\mathrm{D,min}}$: ' + print_function(p_D_model_values, 'log')),[0.1, 0.69] , xycoords='axes fraction', fontsize=fontsize_base-2, c=color_prism[2])




    if reabs_option == 'Yes':

        p_esc1 = df_save['p_esc1(-)'] = trace.posterior.escape_prob.values[:,-final_sample_no:,0].ravel()
        p_esc2 = df_save['p_esc2(-)'] = trace.posterior.escape_prob.values[:,-final_sample_no:,1].ravel()  

        plot_loghist(p_esc1, ax_pesc, 'log', color_prism[0])
        plot_loghist(p_esc2, ax_pesc, 'log', color_prism[1])

        #ax_pesc.plot(plot_kernel_1D(p_esc1)[0], plot_kernel_1D(p_esc1)[1], c=color_prism[0], linewidth=2, alpha=0.8)
        #ax_pesc.plot(plot_kernel_1D(p_esc2)[0], plot_kernel_1D(p_esc2)[1], c=color_prism[1], linewidth=2, alpha=0.8)
        ax_pesc.annotate(str("$S_{\mathrm{" + side_1 + "}}:$ " + print_function(p_esc1, 'log')),[0.1, 0.76] , xycoords='axes fraction', fontsize=fontsize_base-2, c=color_prism[0])
        ax_pesc.annotate(str("$S_{\mathrm{" + side_2 + "}}:$ " + print_function(p_esc2, 'log')),[0.1, 0.69] , xycoords='axes fraction', fontsize=fontsize_base-2, c=color_prism[1])
        
        ax_krad.annotate('(b) $k_{\mathrm{rad}}$ [cm$^{3}$s$^{-1}$]' ,[0.05, 0.88] , xycoords='axes fraction', fontsize=fontsize_base+1)
        ax_krad.annotate(str('$k_{\mathrm{rad}}: $ ' + print_function(k_rad_model_values, 'log')),[0.1, 0.76] , xycoords='axes fraction', fontsize=fontsize_base-2, c=color_prism[0])
        #ax_krad.annotate(str('$k_{\mathrm{rad}}$(' + "$S_{\mathrm{" + side_1 + "}}$ " + '): ' + print_function(k_rad_model_values*p_esc1, 'log')),[0.1, 0.69] , xycoords='axes fraction', fontsize=fontsize_base-2, c='dimgrey')
        #ax_krad.annotate(str('$k_{\mathrm{rad}}$(' + "$S_{\mathrm{" + side_2 + "}}$ " + '): ' + print_function(k_rad_model_values*p_esc2, 'log')),[0.1, 0.62] , xycoords='axes fraction', fontsize=fontsize_base-2, c='dimgrey')

    else:
        p_esc1 = df_save['p_esc1(-)'] = np.ones(shape=np.shape(k_rad_model_values))
        p_esc2 = df_save['p_esc2(-)'] = np.ones(shape=np.shape(k_rad_model_values))

        ax_pesc.annotate(str("$S_{\mathrm{" + side_1 + "}}: 1$ "),[0.1, 0.76] , xycoords='axes fraction', fontsize=fontsize_base-2, c=color_prism[0])
        ax_pesc.annotate(str("$S_{\mathrm{" + side_2 + "}}: 1$ "),[0.1, 0.69] , xycoords='axes fraction', fontsize=fontsize_base-2, c=color_prism[1])
        ax_pesc.set_yticks([])

        ax_krad.annotate('(b) $k_{\mathrm{rad}}^*$ [cm$^{3}$s$^{-1}$]' ,[0.05, 0.88] , xycoords='axes fraction', fontsize=fontsize_base+1)
        ax_krad.annotate(str(print_function(k_rad_model_values, 'log')),[0.1, 0.76] , xycoords='axes fraction', fontsize=fontsize_base-2, c=color_prism[0])





    #ax_td.plot(plot_kernel_1D(trap_depth_model_values*1000)[0], plot_kernel_1D(trap_depth_model_values*1000)[1], c=color_prism[0], linewidth=2, alpha=0.8)
    plot_loghist(trap_depth_model_values*1000, ax_td, 'lin', color_prism[0])
    ax_td.annotate(print_function(trap_depth_model_values*1000, 'lin'),[0.1, 0.76] , xycoords='axes fraction', fontsize=fontsize_base-2, c=color_prism[0])
    
    #ax_tau.plot(plot_kernel_1D(kc_n_model_values)[0], plot_kernel_1D(kc_n_model_values)[1], c=color_prism[0], alpha=0.8, linewidth=2)
    #ax_tau.plot(plot_kernel_1D(kc_p_model_values)[0], plot_kernel_1D(kc_p_model_values)[1], c=color_prism[2], alpha=0.8, linewidth=2)
    #ax_tau.plot(plot_kernel_1D(k_nr1sun)[0], plot_kernel_1D(k_nr1sun)[1], c='dimgrey', alpha=0.8, linewidth=2)

    k_combined = np.stack([kc_p_model_values, k_nr1sun, kc_n_model_values]).ravel()
    
    bins = np.logspace(np.log10(k_combined.min()/5), np.log10(k_combined.max()*5),50)
    
    x1,_,_ = ax_tau.hist(kc_p_model_values, bins=bins, color=color_prism[2], alpha=0.6)
    x2,_,_ = ax_tau.hist(k_nr1sun, bins=bins, color='dimgrey', alpha=0.6)
    x3,_,_ = ax_tau.hist(kc_n_model_values, bins=bins, color=color_prism[0], alpha=0.6)
    
    x_array = np.array([x1.max(),x2.max(),x3.max()])
    ax_tau.set_ylim(bottom=0, top=2*x_array.max())
    ax_tau.set_yticks([])


    
    #plot_loghist(kc_p_model_values, ax_tau, 'log', color_prism[2])
    #plot_loghist(k_nr1sun, ax_tau, 'log', 'dimgrey')
    #plot_loghist(kc_n_model_values, ax_tau, 'log', color_prism[0])

    ax_tau.annotate(str('$k_{\mathrm{tr,n}}$: ' + print_function(kc_n_model_values, 'log')),[0.1, 0.76] , xycoords='axes fraction', fontsize=fontsize_base-2, c=color_prism[0])
    ax_tau.annotate(str('$k_{\mathrm{tr,p}}$: ' + print_function(kc_p_model_values, 'log')),[0.1, 0.69] , xycoords='axes fraction', fontsize=fontsize_base-2, c=color_prism[2])
    ax_tau.annotate(str('$k_{\mathrm{nr1s}}$: ' + print_function(k_nr1sun, 'log')),[0.1, 0.62] , xycoords='axes fraction', fontsize=fontsize_base-2, c='dimgrey')

    #ax_diffl.plot(plot_kernel_1D(Diffusion_length)[0], plot_kernel_1D(Diffusion_length)[1], c=color_prism[0], linewidth=2, alpha=0.8)
    plot_loghist(Diffusion_length, ax_diffl, 'lin', color_prism[0])
    ax_diffl.annotate(print_function(Diffusion_length, 'lin'),[0.1, 0.76] , xycoords='axes fraction', fontsize=fontsize_base-2, c=color_prism[0])

    #ax_LL.plot(plot_kernel_1D(LL_norm.ravel())[0], plot_kernel_1D(LL_norm.ravel())[1], c=color_prism[0], linewidth=2, alpha=0.8)
    plot_loghist(LL_norm, ax_LL, 'lin', color_prism[0])
    ax_LL.annotate(print_function(LL_norm.ravel(), 'lin'),[0.1, 0.76] , xycoords='axes fraction', fontsize=fontsize_base-2, c=color_prism[0])

    #ax_fluerr.plot(plot_kernel_1D(k_aug_model_values)[0], plot_kernel_1D(k_aug_model_values)[1], c=color_prism[0], linewidth=2, alpha=0.8)
    plot_loghist(k_aug_model_values, ax_fluerr, 'log', color_prism[0])
    ax_fluerr.annotate(print_function(k_aug_model_values, 'log'),[0.1, 0.76] , xycoords='axes fraction', fontsize=fontsize_base-2, c=color_prism[0])

    
    S_report_1_kernel = S_report_1[S_report_1 != 0]
    S_report_2_kernel = S_report_2[S_report_2 != 0]
    

    S_combined = np.stack([S_report_1_kernel, S_report_2_kernel]).ravel()
    
    bins = np.logspace(np.log10(S_combined.min()/5), np.log10(S_combined.max()*5),50)
    
    x1,_,_ = ax_Ssub.hist(S_report_1_kernel, bins=bins, color=color_prism[0], alpha=0.6)
    x2,_,_ = ax_Ssub.hist(S_report_2_kernel, bins=bins, color=color_prism[1], alpha=0.6)
    
    x_array = np.array([x1.max(),x2.max()])
    ax_Ssub.set_ylim(bottom=0, top=2*x_array.max())
    ax_Ssub.set_yticks([])

    #ax_Ssub.plot(plot_kernel_1D(S_report_1_kernel)[0], plot_kernel_1D(S_report_1_kernel)[1], c=color_prism[0], linewidth=2, alpha=0.8, label=S_plot_1_label)
    #plot_loghist(S_report_1_kernel, ax_Ssub, 'log', color_prism[0])
    ax_Ssub.annotate(str(S_plot_1_label + ': ' + print_function(S_report_1_kernel, 'log')),[0.1, 0.76] , xycoords='axes fraction', fontsize=fontsize_base-2, c=color_prism[0])
    
    #ax_Ssub.plot(plot_kernel_1D(S_report_2_kernel)[0], plot_kernel_1D(S_report_2_kernel)[1], c=color_prism[1], linewidth=2, alpha=0.8,label=S_plot_2_label)
    #plot_loghist(S_report_2_kernel, ax_Ssub, 'log', color_prism[1])
    ax_Ssub.annotate(str(S_plot_2_label + ': ' + print_function(S_report_2_kernel, 'log')),[0.1, 0.69] , xycoords='axes fraction', fontsize=fontsize_base-2, c=color_prism[1])
    ax_Ssub.annotate(str('$S_{\mathrm{mix,av}}$: ' + '{:.2f}'.format(np.mean(S_mix_model_values))),[0.1, 0.62] , xycoords='axes fraction', fontsize=fontsize_base-2, c='dimgrey')
    


    ### Plotting Simulated Curves alongside Data
    N_calc_vals = trace.posterior.N_calc_collect.values[:,-final_sample_no:,:]

    data_no_bckg = pd.DataFrame()
    data_no_bckg['Time (ns)'] = df['Time']
    median_of_simulations = pd.DataFrame()
    median_of_simulations['Time (ns)'] = time_plot

    #for j in range(500):
    #    test_chain = random.randint(0,no_of_chains-1)
    #    test_sample = random.randint(0,len(k_rad_model_values)/(no_of_chains)-1)
    #
    #    ax_main.semilogy(time_plot, N_calc_vals[test_chain,test_sample,:,:], c='pink', linewidth=1, alpha=0.5, zorder=0)

    N_calc_median = np.nanmedian(N_calc_vals,axis=[0, 1])

    sigma_LL_all = trace.posterior.sigma_width.values[:,-final_sample_no:].ravel()
    sigma_LL_logbins = np.logspace(np.log10(sigma_LL_all.min()/5), np.log10(sigma_LL_all.max()*5),50)
    sigma_LL_max = ()
    
    PL_err_all =  trace.posterior.PL_err.values[:,-final_sample_no:].ravel()
    PL_err_logbins = np.logspace(np.log10(PL_err_all.min()/5), np.log10(PL_err_all.max()*5),50)
    PL_err_max = ()

    for n in range(len(Surface)):

        ## Plot Data
        ax_main.scatter(df['Time'], df[str(n)], marker='.',s=10 ,alpha=0.8, color=next(color1))
        line_color = next(color2)
        if Surface[n] == 1:
            S_label = "$S_{\mathrm{" + side_1 + "}}$"
        else:
            S_label = "$S_{\mathrm{" + side_2 + "}}$"
        ax_main.plot(time_plot, N_calc_median[:,n], color=line_color, linewidth=3, label=str(str(S_label) +" {:.1e}".format(Fluence[n])))

        data_no_bckg[str("{:.1e}".format(Fluence[n]) + "[cm-2] S" + str(Surface[n]))] = df[str(n)]
        median_of_simulations[str("{:.1e}".format(Fluence[n]) + "[cm-2] S" + str(Surface[n]))] = N_calc_median[:,n]

        #ax_sigmaLL.plot(plot_kernel_1D(df_save[f'sigma_LL_{n}'])[0], plot_kernel_1D(df_save[f'sigma_LL_{n}'])[1], c=line_color, linewidth=2, alpha=0.8)
        plot_loghist(df_save[f'sigma_LL_{n}'], ax_sigmaLL, 'lin', line_color)
        x,_,_ = ax_sigmaLL.hist(df_save[f'sigma_LL_{n}'], bins=sigma_LL_logbins, color=line_color, alpha=0.6)
        sigma_LL_max = np.append(sigma_LL_max, x.max())
        ax_sigmaLL.annotate('{:.3f}'.format(np.median(df_save[f'sigma_LL_{n}'])),[0.1, 0.76-0.07*n] , xycoords='axes fraction', fontsize=fontsize_base-2, c=line_color)
        
        #ax_PL_err.plot(plot_kernel_1D(df_save[f'PL_err_{n}'])[0], plot_kernel_1D(df_save[f'PL_err_{n}'])[1], c=line_color, linewidth=2, alpha=0.8)
        #plot_loghist(df_save[f'PL_err_{n}'], ax_PL_err, 'lin', line_color)
        x,_,_ = ax_PL_err.hist(df_save[f'PL_err_{n}'], bins=PL_err_logbins, color=line_color, alpha=0.6)
        PL_err_max = np.append(PL_err_max, x.max())
        ax_PL_err.annotate('{:.1e}'.format(np.median(df_save[f'PL_err_{n}'])),[0.1, 0.76-0.07*n] , xycoords='axes fraction', fontsize=fontsize_base-2, c=line_color)


        #print(np.sum(np.abs(np.log10(N_calc_median[:,n]) - np.log10(df[str(n)][a]))))
    ax_sigmaLL.set_ylim(bottom=0, top=2*sigma_LL_max.max())
    ax_sigmaLL.set_yticks([])
    ax_PL_err.set_ylim(bottom=0, top=2*PL_err_max.max())
    ax_PL_err.set_yticks([])


    for plot_name in [ax_main, ax_mob, ax_Ssub, ax_tau, ax_neq, ax_krad, ax_td, ax_pesc, ax_LL, ax_diffl, ax_fluerr, ax_sigmaLL, ax_PL_err]:
        for axis in ['top', 'bottom', 'left', 'right']:
            plot_name.spines[axis].set_linewidth(2)
        plot_name.tick_params(bottom=True,top=True,left=True,right=True,
                       direction='in',width=2, length=3.5, which='major', labelsize=fontsize_base-2, zorder=2000)
        plot_name.tick_params(bottom=True,top=True,left=True,right=True,
                       direction='in',width=0, length=0, which='minor')

        plot_name.set_xscale('log')
        


    #ax_Ssub.set_xlim(1e-1,None)
    ax_main.set_ylim(bottom=np.min(N_calc_median)/2, top=3.5)
    ax_main.set_xlim(np.array(df['Time'])[0], np.array(df['Time'])[-1])
    ax_main.set_yscale('log')
    ax_main.set_xscale(scaling)

    ax_td.set_xscale('linear')
    #ax_mob.set_xscale('linear')
    #ax_pesc.set_xscale('linear')
    #ax_fluerr.set_xscale('linear')
    ax_diffl.set_xscale('linear')
    ax_LL.set_xscale('linear')
    ax_pesc.set_xlim(0,1)

    if scaling == 'log':
        legend_loc = 'lower left'
    else:
        legend_loc = 'center right'
    ax_main.legend(title='Fluences [cm$^{-2}$]', loc=legend_loc, frameon=False)
    #ax_Ssub.legend()
    plt.tight_layout(w_pad=1)



    # Saving stuff
    data_no_bckg.to_csv(f'{data_folder_trpl}/{sample_name}_data_norm.dat', sep='\t', index= True, mode='w')
    median_of_simulations.to_csv(f'{data_folder_trpl}/{sample_name}_median_of_simulations.dat', sep='\t', index= True, mode='w')
    df_save.to_csv(f'{data_folder_trpl}/{sample_name}_drawn_samples.dat', sep='\t', index= True, mode='w')

    #plt.savefig(f'{data_folder_trpl}/{sample_name}_plot.pdf', format='pdf', dpi=300)
    plt.savefig(f'{data_folder_trpl}/{sample_name}_plot.png', format='png', dpi=300, transparent=False)

    
    
    plt.show()

    
    return df_save, trace
    
    
    
    
def make_BayesFigure(trace_name, data_folder_trpl, df,  Fluence, Surface, spacing, reabs_option, max_arg, Thickness, scaling, one_sun_carrier_density, pile_up, side_1, side_2):

    trace = az.from_netcdf(f'{data_folder_trpl}/{trace_name}')
    
    trace_file_split = trace_name.split('_trace_')
    sample_name = str(trace_file_split[0])

    
    a = spacing_choose(spacing, max_arg)
    df_save, trace = plot_and_save(trace, a, df, Fluence, Surface, Thickness, scaling, sample_name, data_folder_trpl, one_sun_carrier_density, max_arg, pile_up, side_1, side_2, reabs_option)
    
    df_save.rename(columns={'k_rad(cm3s-1)': 'k_rad', 'n_eq(cm-3)': 'n_eq', 'k_trapping(s-1)': 'k_trapping','k_nr1sun(s-1)':'k_nr1sun',  'td(eV)': 'td','S_1(cm s-1)':'S_1', 'S_2(cm s-1)':'S_2', 'Reabs_S1(cm-1)': 'Reabs_1', 'Reabs_S2(cm-1)': 'Reabs_2', 'Mobility_values(cm2V-1s-1)': 'Mobility', 'epsilon_fluence(%)': 'epsilon_fluence' }, inplace=True)
 
    
    
    return df_save, trace








def corner_plot_single(a, b, a_label, b_label, cornerlabel):
    centimeters = 1/2.54
    fontsize_base = 11

    fig_corner_1 = plt.figure(figsize=(10.5*centimeters, 10.5*centimeters))
    #plt.subplots_adjust(wspace=0.2, hspace=0.35)
    gs_corner_1 = gridspec.GridSpec(1, 1)
    ax_plot = fig_corner_1.add_subplot(gs_corner_1[0,0])

    
    ## Method 3: Kernel Density Estimation
    def kernel_2D(a, b):
        
        a = np.log10(a)
        b = np.log10(b)

        Matrix_ab = np.vstack([a,b])
    
        x_a1, x_b1 = np.linspace(np.min(a)-0.5,np.max(a)+0.5,50) , np.linspace(np.min(b)-0.5,np.max(b)+0.5,50) 
        x_a, x_b = np.meshgrid(x_a1, x_b1)
        
        kernel_ab = stats.gaussian_kde(Matrix_ab)(np.vstack([x_a.ravel(), x_b.ravel()]))
        kernel_2d = np.reshape(kernel_ab.T, x_a.shape)
    
        return 10**x_a, 10**x_b, kernel_2d
    
    x_a, x_b, test_kernel = kernel_2D(a, b)
    
    
    
    ax_plot.scatter(a,b,s=1,color='black',alpha=0.5)
    ax_plot.contourf(x_a, x_b, test_kernel, levels=10, cmap='bone_r',alpha=0.9)
    ax_plot.contour(x_a, x_b, test_kernel, levels=10, colors='black',alpha=0.25)

    ax_plot.set_xlabel(a_label, fontsize=fontsize_base+1)
    ax_plot.set_ylabel(b_label, fontsize=fontsize_base+1)
    ax_plot.text(0.05, 0.9, cornerlabel, transform=ax_plot.transAxes, fontsize=fontsize_base+1)
    ax_plot.set_yscale('log')
    ax_plot.set_xscale('log')
    
    
    for axis in ['top', 'bottom', 'left', 'right']:
        ax_plot.spines[axis].set_linewidth(2)
    ax_plot.tick_params(bottom=True,top=True,left=True,right=True,
                   direction='in',width=2, length=3.5, which='both', labelsize=fontsize_base, zorder=2000)
 














