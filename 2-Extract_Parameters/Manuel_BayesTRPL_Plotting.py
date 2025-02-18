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
    a_nonnan = ~np.isnan(a)
    a = a[a_nonnan]

    if log == 'log':
        bins = np.logspace(np.log10(np.nanmin(a)/1.5), np.log10(np.nanmax(a)*1.5),30)
    
    else:
        bins = np.linspace(np.nanmin(a), np.nanmax(a),30)
    
    x,_,_ = ax_a.hist(a, bins=bins, color=colour, alpha=0.6)
    ax_a.set_ylim(bottom=0, top=2*x.max())
    ax_a.set_yticks([])
    

def print_function(a_values, resol):
    
    a_test = a_values.ravel()
    a_nonnan = ~np.isnan(a_test)
    a_test = a_test[a_nonnan]
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



def plot_and_save(trace, a, df, Fluence, Surface, Thickness, scaling, sample_name, data_folder_trpl, one_sun_carrier_density, max_arg, pile_up, side_1, side_2, SRV_display, bckg_list, PN_on_off, diffusion_on_off, sh_defect, filter):
    
    # Plotting Parameters
    centimeters = 1/2.54
    fontsize_base = 11
    color1 = iter(cm.Set2(np.arange(7)))
    color2 = iter(cm.Dark2(np.arange(7)))

    color_prism = ['#1D6996','#38A6A5','#0F8554','#73AF48','#EDAD08','#E17C05','#CC503E','#94346E','#6F4070','#994E95','#666666']


    time_plot = np.array(df['Time'])[a]   #s

    # DataFrame to save all draws
    df_save = pd.DataFrame()
    
    # Setup of Figure
    fig_final = plt.figure(figsize=(24*centimeters, 28*centimeters))
    
    gs_final_combined = fig_final.add_gridspec(2,1, height_ratios=[1,3.3])
    gs_final_main = gs_final_combined[0].subgridspec(1,6)

    gs_final = gs_final_combined[1].subgridspec(3,6, hspace=0.2, wspace=0.2)

    ax_main = fig_final.add_subplot(gs_final_main[0, 1:5])
    ax_main.set_title('Normalized TRPL Decays + Median of Bayes-MCMC', fontsize=fontsize_base+2)
    ax_main.set_xlabel('Time after Pulse [ns]')
    ax_main.set_ylabel('Normalized PL Intensity [a.u.]')
    ax_main.annotate(f'(a) {sample_name}' ,[0.05, 0.9] , xycoords='axes fraction', fontsize=fontsize_base+1)

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

    ax_krad = fig_final.add_subplot(gs_final[0, 0:2])
    ax_krad.annotate('(b) $k_{\mathrm{rad}}$ [cm$^{3}$s$^{-1}$]' ,[0.05, 0.88] , xycoords='axes fraction', fontsize=fontsize_base+1)

    ax_td = fig_final.add_subplot(gs_final[0, 2:4])
    ax_td.annotate('(c) $k_{\mathrm{eff,bulk}}$ [s$^{-1}$]' ,[0.05, 0.88] , xycoords='axes fraction', fontsize=fontsize_base+1)

    ax_Ssub = fig_final.add_subplot(gs_final[0, 4:6])
    ax_Ssub.annotate('(d) SRV [cm s$^{-1}$]' ,[0.05, 0.88] , xycoords='axes fraction', fontsize=fontsize_base+1)

    ax_mob = fig_final.add_subplot(gs_final[1, 0:2])
    ax_mob.annotate('(e) $\mu_{\mathrm{vert}}$ [cm$^2$ (Vs)$^{-1}$]' ,[0.05, 0.88] , xycoords='axes fraction', fontsize=fontsize_base+1)

    ax_tau = fig_final.add_subplot(gs_final[1, 2:4])
    ax_tau.annotate('(f) $k_{\mathrm{nr}}$ [s$^{-1}$]' ,[0.05, 0.88] , xycoords='axes fraction', fontsize=fontsize_base+1)

    ax_PL_err_neq = fig_final.add_subplot(gs_final[1, 4:6])
    ax_PL_err_neq.annotate('(g) $p_{\mathrm{eq, min}}$ [cm$^{-3}$]' ,[0.05, 0.88] , xycoords='axes fraction', fontsize=fontsize_base+1)
 
    ax_diffl = fig_final.add_subplot(gs_final[2, 0:2])
    ax_diffl.annotate('(i) $L_{\mathrm{D}}$ [$\mu m$]' ,[0.05, 0.88] , xycoords='axes fraction', fontsize=fontsize_base+1)

    ax_LL = fig_final.add_subplot(gs_final[2, 2:4])
    ax_LL.annotate('(j) -log($LL$)$_{\mathrm{av.}}$ [-]' ,[0.05, 0.88] , xycoords='axes fraction', fontsize=fontsize_base+1) 

    ax_LLerr = fig_final.add_subplot(gs_final[2, 4:6])
    ax_LLerr.annotate('(k) $f_{\mathrm{\sigma LL}}$ [-]' ,[0.05, 0.88] , xycoords='axes fraction', fontsize=fontsize_base+1)

  
    Fluence_min_marker = np.where(Fluence == np.min(Fluence))[0]


    # Bulk Recombination
    k_rad_model_values = df_save['k_rad(cm3s-1)'] = trace.posterior.k_rad.values.ravel()[filter]       
    
    if PN_on_off == 0:
        ax_PL_err_neq.annotate('0.0 (fixed)',[0.1, 0.76] , xycoords='axes fraction', fontsize=fontsize_base-2, c=color_prism[0])
        p0_model_value = df_save[f'peq_min(cm-3)'] =  np.mean(trace.posterior.P_0_list.values[:,:,Fluence_min_marker],axis=2).ravel()[filter]*0
 

    else:
        p0_model_value = df_save[f'peq_min(cm-3)'] =  np.mean(trace.posterior.P_0_list.values[:,:,Fluence_min_marker],axis=2).ravel()[filter]*1e12
        plot_loghist(p0_model_value, ax_PL_err_neq, 'log', color_prism[0])
        ax_PL_err_neq.annotate(print_function(p0_model_value, 'log'),[0.1, 0.76] , xycoords='axes fraction', fontsize=fontsize_base-2, c=color_prism[0])

    if sh_defect == 0:
        k_capture = 0
        k_emission = 0
        k_deep = df_save['k_deep(s-1)'] = trace.posterior.k_deep.values.ravel()[filter]
    else:
    
        k_capture = df_save['k_capture(s-1)'] = trace.posterior.k_capture.values.ravel()[filter]
        k_emission = df_save['k_emission(s-1)'] = trace.posterior.k_emission.values.ravel()[filter]
        k_deep = df_save['k_deep(s-1)'] = trace.posterior.k_deep.values.ravel()[filter]
    
    df_save['k_nr_eff(s-1)'] = k_eff = k_deep*k_deep/(k_deep+k_capture) + k_capture*k_deep/(k_capture+k_deep)*k_emission/(k_emission+k_deep)  
    df_save['k_bulk_eff(s-1)'] = k_bulk = k_eff+ k_rad_model_values*(one_sun_carrier_density+p0_model_value)
 

    ## Surface Recombination    
    df_save['S_1(cm s-1)'] = trace.posterior.S_1.values.ravel()[filter]
    df_save['S_2(cm s-1)'] = trace.posterior.S_2.values.ravel()[filter]

    df_save['S_lower(cm s-1)']  = trace.posterior.S_lower.values.ravel()[filter]
    df_save['S_upper(cm s-1)'] = trace.posterior.S_upper.values.ravel()[filter]

    if SRV_display == 'upperlower':
        S_report_1 = df_save['S_lower(cm s-1)']
        S_plot_1_label = "S$_{lower}$"
        S_report_2 = df_save['S_upper(cm s-1)']
        S_plot_2_label = "S$_{upper}$"

    elif SRV_display == 'topbot':
        S_report_1 = df_save['S_1(cm s-1)']
        S_plot_1_label = "$S_{\mathrm{" + side_1 + "}}$"
        S_report_2 = df_save['S_2(cm s-1)']
        S_plot_2_label = "$S_{\mathrm{" + side_2 + "}}$"

    k_bulk_plus_surf = k_bulk + (S_report_1 + S_report_2)/(Thickness*1e-7)    
    

    ## Diffusion
    limit_mobility = (Thickness*1e-7)**2/(1e-9 * np.abs(np.array(df['Time'])[1]-np.array(df['Time'])[0]))/(1.380649e-23*292/1.6021766e-19)/4 #cm2 (Vs)-1
    min_mobility = (Thickness*1e-7)**2/(1e-9 * np.abs(np.array(df['Time'])[-1]))/(1.380649e-23*292/1.6021766e-19) * 5 #cm2 (Vs)-1


    Mob_vals = trace.posterior.mu_vert.values.ravel()[filter]
    mob_filter = [(Mob_vals >= min_mobility) & (Mob_vals <= limit_mobility)][0]
    
    if len(Mob_vals[mob_filter])/len(Mob_vals) >= 0.3:
        
        diff_infer = True
        
        Mob_vals[~mob_filter] = np.nan
        Mobility_values = df_save['Mobility_values(cm2V-1s-1)'] = Mob_vals
        
        Diffusion_coeff_values = trace.posterior.Diffusion_coefficient.values.ravel()[filter]
        Diffusion_coeff_values[~mob_filter] = np.nan

        Diffusion_length = df_save['Diffusion_length(um)'] = np.sqrt(Diffusion_coeff_values/k_eff)*1e4
        Diffusion_length_bulk  = np.sqrt(Diffusion_coeff_values/k_bulk)*1e4
        Diffusion_length_surface = np.sqrt(Diffusion_coeff_values/k_bulk_plus_surf)*1e4
    else:
        diff_infer = False

    ## Logp vals
    LL = np.abs(trace.posterior.Logp.values.ravel()[filter])
    df_save['LL'] = LL.ravel()/(len(time_plot)*len(Surface))
    df_save['sigma_LL_factor'] = 2+9*trace.posterior.sigma_fact.values.ravel()[filter]  


    ### Plotting the Probability Densities
    plot_loghist(k_rad_model_values, ax_krad, 'log', color_prism[0])
    ax_krad.annotate(print_function(k_rad_model_values, 'log'),[0.1, 0.76] , xycoords='axes fraction', fontsize=fontsize_base-2, c=color_prism[0])
    

    
    if sh_defect == 0:
        plot_loghist(k_deep, ax_tau, 'log', color_prism[0])
        ax_tau.annotate(str('$k_{\mathrm{d}}$: ' + print_function(k_deep, 'log')),[0.1, 0.76] , xycoords='axes fraction', fontsize=fontsize_base-2, c=color_prism[0])
    else:
        k_nr_combined = np.stack([k_capture, k_emission, k_deep]).ravel()
        bins = np.logspace(np.log10(k_nr_combined.min()/1.5), np.log10(k_nr_combined.max()*1.5),30)
    
        x1,_,_ = ax_tau.hist(k_capture, bins=bins, color=color_prism[0], alpha=0.6)
        x2,_,_ = ax_tau.hist(k_emission, bins=bins, color=color_prism[1], alpha=0.6)
        x3,_,_ = ax_tau.hist(k_deep, bins=bins, color=color_prism[2], alpha=0.6)

        x_array = np.array([x1.max(),x2.max(),x3.max()])
        ax_tau.set_ylim(bottom=0, top=2*x_array.max())
        ax_tau.set_yticks([])

        ax_tau.annotate(str('$k_{\mathrm{c}}$: ' + print_function(k_capture, 'log')),[0.1, 0.76] , xycoords='axes fraction', fontsize=fontsize_base-2, c=color_prism[0])
        ax_tau.annotate(str('$k_{\mathrm{e}}$: ' + print_function(k_emission, 'log')),[0.1, 0.69] , xycoords='axes fraction', fontsize=fontsize_base-2, c=color_prism[1])
        ax_tau.annotate(str('$k_{\mathrm{d}}$: ' + print_function(k_deep, 'log')),[0.1, 0.62] , xycoords='axes fraction', fontsize=fontsize_base-2, c=color_prism[2])



    k_bulk_combined = np.stack([k_eff, k_bulk]).ravel()
    bins = np.logspace(np.log10(k_bulk_combined.min()/1.5), np.log10(k_bulk_combined.max()*1.5),30)
    
    x1,_,_ = ax_td.hist(k_eff, bins=bins, color=color_prism[0], alpha=0.6)
    x2,_,_ = ax_td.hist(k_bulk, bins=bins, color='dimgrey', alpha=0.6)

    x_array = np.array([x1.max(),x2.max()])
    ax_td.set_ylim(bottom=0, top=2*x_array.max())
    ax_td.set_yticks([])

    ax_td.annotate(str('nr: ' +print_function(k_eff, 'log')),[0.1, 0.76] , xycoords='axes fraction', fontsize=fontsize_base-2, c=color_prism[0])
    ax_td.annotate(str('nr+rad: ' +print_function(k_bulk, 'log')),[0.1, 0.69] , xycoords='axes fraction', fontsize=fontsize_base-2, c='dimgrey')

    if diff_infer:
        plot_loghist(Mobility_values, ax_mob, 'log', color_prism[0])
        ax_mob.annotate(print_function(Mobility_values, 'log'),[0.1, 0.76] , xycoords='axes fraction', fontsize=fontsize_base-2, c=color_prism[0])

        plot_loghist(Diffusion_length, ax_diffl, 'lin', color_prism[0])
        ax_diffl.annotate(str('nr: ' +print_function(Diffusion_length, 'lin')),[0.1, 0.76] , xycoords='axes fraction', fontsize=fontsize_base-2, c=color_prism[0])
        ax_diffl.annotate(str('nr+r: ' + print_function(Diffusion_length_bulk, 'lin')),[0.1, 0.69] , xycoords='axes fraction', fontsize=fontsize_base-2, c='dimgrey')
        ax_diffl.annotate(str('nr+r+S: ' + print_function(Diffusion_length_surface, 'lin')),[0.1, 0.62] , xycoords='axes fraction', fontsize=fontsize_base-2, c='dimgrey')
    else:
        ax_diffl.annotate('cannot be inferred',[0.1, 0.76] , xycoords='axes fraction', fontsize=fontsize_base-2, c=color_prism[0])
        ax_mob.annotate('cannot be inferred',[0.1, 0.76] , xycoords='axes fraction', fontsize=fontsize_base-2, c=color_prism[0])


    plot_loghist(df_save['LL'], ax_LL, 'log', color_prism[0])
    ax_LL.annotate(print_function(df_save['LL'], 'log'),[0.1, 0.76] , xycoords='axes fraction', fontsize=fontsize_base-2, c=color_prism[0])
    
    plot_loghist(df_save['sigma_LL_factor'], ax_LLerr, 'log', color_prism[0])
    ax_LLerr.annotate(print_function(df_save['sigma_LL_factor'], 'log'),[0.1, 0.76] , xycoords='axes fraction', fontsize=fontsize_base-2, c=color_prism[0])


    S_report_1_kernel = S_report_1[S_report_1 != 0]
    S_report_2_kernel = S_report_2[S_report_2 != 0]
    

    S_combined = np.stack([S_report_1_kernel, S_report_2_kernel]).ravel()
    
    bins = np.logspace(np.log10(S_combined.min()/1.5), np.log10(S_combined.max()*1.5),30)
    
    x1,_,_ = ax_Ssub.hist(S_report_1_kernel, bins=bins, color=color_prism[0], alpha=0.6)
    x2,_,_ = ax_Ssub.hist(S_report_2_kernel, bins=bins, color=color_prism[1], alpha=0.6)
    
    x_array = np.array([x1.max(),x2.max()])
    ax_Ssub.set_ylim(bottom=0, top=2*x_array.max())
    ax_Ssub.set_yticks([])

    ax_Ssub.annotate(str(S_plot_1_label + ': ' + print_function(S_report_1_kernel, 'log')),[0.1, 0.76] , xycoords='axes fraction', fontsize=fontsize_base-2, c=color_prism[0])
    ax_Ssub.annotate(str(S_plot_2_label + ': ' + print_function(S_report_2_kernel, 'log')),[0.1, 0.69] , xycoords='axes fraction', fontsize=fontsize_base-2, c=color_prism[1])
    

    ### Plotting Simulated Curves alongside Data
    N_calc_vals = trace.posterior.N_calc_collect.values

    data_no_bckg = pd.DataFrame()
    data_no_bckg['Time (ns)'] = df['Time']
    median_of_simulations = pd.DataFrame()
    median_of_simulations['Time (ns)'] = time_plot


    N_calc_median = np.nanmedian(N_calc_vals,axis=[0, 1])


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


    for plot_name in [ax_main, ax_mob, ax_Ssub, ax_tau, ax_krad, ax_td, ax_PL_err_neq, ax_LL, ax_diffl, ax_LLerr]:
        for axis in ['top', 'bottom', 'left', 'right']:
            plot_name.spines[axis].set_linewidth(2)
        plot_name.tick_params(bottom=True,top=True,left=True,right=True,
                       direction='in',width=2, length=3.5, which='major', labelsize=fontsize_base-2, zorder=2000)
        plot_name.tick_params(bottom=True,top=True,left=True,right=True,
                       direction='in',width=0, length=0, which='minor')

        plot_name.set_xscale('log')
        

    Data_array = np.array(df.loc[:,df.columns != 'Time']).flatten()
    zero_marker = np.where(Data_array <= 0)
    Data_array[zero_marker] = np.nan

    ax_main.set_ylim(bottom=np.nanmin(Data_array), top=3.5)
    ax_main.set_xlim(np.array(df['Time'])[0], np.array(df['Time'])[-1])
    ax_main.set_yscale('log')
    ax_main.set_xscale(scaling)

    ax_PL_err_neq.set_yticks([])
    ax_diffl.set_xscale('linear')

    if scaling == 'log':
        legend_loc = 'lower left'
    else:
        legend_loc = 'upper right'
    ax_main.legend(title=str(str('pile-up $\leq$' + '{:.1f}'.format(np.max(pile_up)) + "%") +'\nFluences [cm$^{-2}$]'), loc=legend_loc, frameon=False)
    plt.tight_layout(w_pad=1)


    # Saving stuff
    data_no_bckg.to_csv(f'{data_folder_trpl}/{sample_name}_data_norm.dat', sep='\t', index= True, mode='w')
    median_of_simulations.to_csv(f'{data_folder_trpl}/{sample_name}_median_of_simulations.dat', sep='\t', index= True, mode='w')
    df_save.to_csv(f'{data_folder_trpl}/{sample_name}_drawn_samples.dat', sep='\t', index= True, mode='w')

    plt.savefig(f'{data_folder_trpl}/{sample_name}_plot.png', format='png', dpi=300, transparent=False)  
    plt.show()

    
    return df_save, trace   


def make_BayesFigure(trace_name, data_folder_trpl, df,  Fluence, Surface, spacing, max_arg, Thickness, scaling, one_sun_carrier_density, pile_up, side_1, side_2, SRV_display, bckg_list, PN_on_off, diffusion_on_off, sh_defect, filter):

    trace = az.from_netcdf(f'{data_folder_trpl}/{trace_name}')
    
    trace_file_split = trace_name.split('_trace_')
    sample_name = str(trace_file_split[0])

    a = spacing_choose(spacing, max_arg)

    df_save, trace = plot_and_save(trace, a, df, Fluence, Surface, Thickness, scaling, sample_name, data_folder_trpl, one_sun_carrier_density, max_arg, pile_up, side_1, side_2, SRV_display, bckg_list, PN_on_off, diffusion_on_off, sh_defect, filter)  
    

    return df_save, trace



def corner_plot_single(a, b, a_label, b_label, cornerlabel):
    centimeters = 1/2.54
    fontsize_base = 11

    fig_corner_1 = plt.figure(figsize=(10.5*centimeters, 10.5*centimeters))
    gs_corner_1 = gridspec.GridSpec(1, 1)
    ax_plot = fig_corner_1.add_subplot(gs_corner_1[0,0])

    
    ## Kernel Density Estimation
    def kernel_2D(a, b):
        
        a = np.log10(a)
        b = np.log10(b)

        Matrix_ab = np.vstack([a,b])
    
        x_a1, x_b1 = np.linspace(np.min(a)-0.5,np.max(a)+0.5,50) , np.linspace(np.min(b)-0.5,np.max(b)+0.5,50) 
        x_a, x_b = np.meshgrid(x_a1, x_b1)
        
        kernel_ab = stats.gaussian_kde(Matrix_ab)(np.vstack([x_a.ravel()[filter], x_b.ravel()[filter]]))
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
 














