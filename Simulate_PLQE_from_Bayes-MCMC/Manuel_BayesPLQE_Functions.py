import sys
import os
from pathlib import Path, PurePath
from os import chdir
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from scipy.integrate import simpson
from scipy.optimize import root, least_squares, minimize
#from lmfit import Parameters, minimize
from scipy import interpolate
import sys
import random

### Oxford color scheme
base_color = "#002147"
color_scheme = ["#44687d", "#69913b", '#cf7a30', '#be0f34']
highlight_color = "#ac48bf"

line_type_list = ['-', '-.', '--', ':']

centimeters = 1/2.54
fontsize_base = 11
figure_width = 21


def plqe_simulation(Generation_rate, Thickness, S_sum, krad, kc_n, kc_p, trap_depth, k_aug, P_esc, N_t, n0, p0):
    
    ne_1 = 1e18 * np.exp(-trap_depth/(8.62e-5 * 292))
    def p_estimate(p, n, kc_n, kc_p, ne1, Nt):
        p = n *(1+10**p)
        #ft = n/(n + kc_p/kc_n*p + ne1)
        ft = (p-n)/Nt

        return  -kc_n*n*(1-ft) + kc_n*ft*ne1 + kc_p*p*ft
    

    def rate_equations(vars, Gen, kc_n, kc_p, ne_1, krad, S_sum, k_aug, P_esc, N_t, n0, p0):            
            ### Pseudo-Equilibrium at each time-point
            n_dens = vars[0]#vars['n_eq'].value
            p_dens = vars[0]*(1+10**vars[1])#n_dens*vars['p_eq'].value
            #p_dens = n_dens *(1+10**root(p_estimate, 3, args=(n_dens, kc_n, kc_p, ne_1, N_t)).x)
            #nt = 10**vars[1]

            #print(n_dens)
            #if nt > N_t:
            #    nt = N_t
            #elif nt < 0:
            #    nt = 0
            #p_dens_list = [1, 1e20]
            #f_t = 0
            

             #    
            #    #nt = N_t*f_t
            #    p_dens = n_dens + f_t*N_t
            #    f_t = n_dens/(n_dens + kc_p/kc_n*p_dens + ne_1)
            #    #nt_dens_list.append(f_t*N_t)
            #    p_dens_list.append(p_dens)
    
            #f_t = n_dens/(n_dens + kc_p/kc_n*p_dens + ne_1)
            nt = p_dens - n_dens
            if nt >= N_t:
                nt = N_t
            elif nt <= 0:
                nt = 0
            f_t = nt/N_t


            dn_dt = Gen - krad*((n_dens)*(p_dens)) - kc_p*p_dens*f_t - (S_sum)/(Thickness*1e-7)*n_dens - k_aug*(n_dens**2*p_dens + p_dens**2*n_dens)
            #dnt_dt = kc_n*n_dens*(1-f_t) - kc_n*f_t*ne_1 - kc_p*p_dens*f_t
            #dp_dt = dn_dt+dnt_dt

            #dn_dt = Gen - P_esc*krad*n_dens*p_dens  - kc_p*p_dens*f_t - (S_sum)/(Thickness*1e-7)*n_dens - k_aug*(n_dens**2*p_dens + p_dens**2*n_dens)

            #dnt_dt = kc_n*n_dens*(1-f_t) - kc_n*f_t*ne_1 - kc_p*p_dens*f_t

            #dp_dt = dn_dt+dnt_dt
            #print(np.abs(dn_dt))
            return np.abs(dn_dt)
    
    test = False

    n_dens = p_dens = 1
    t_test = 1000
    dt = 10e-9
    #while t_test > 1:
    #    
    #    p_dens0 = p_dens#
    #
    #    RuKu1_n, RuKu1_p = rate_equations([n_dens, p_dens], Generation_rate, kc_n, kc_p, ne_1, krad, S_sum, k_aug, P_esc, N_t, n0, p0)
    #    RuKu2_n, RuKu2_p = rate_equations([n_dens + RuKu1_n*dt/2, p_dens + RuKu1_p*dt/2], Generation_rate, kc_n, kc_p, ne_1, krad, S_sum, k_aug, P_esc, N_t, n0, p0)
    #    RuKu3_n, RuKu3_p = rate_equations([n_dens + RuKu2_n*dt/2, p_dens + RuKu2_p*dt/2], Generation_rate, kc_n, kc_p, ne_1, krad, S_sum, k_aug, P_esc, N_t, n0, p0)
    #    RuKu4_n, RuKu4_p = rate_equations([n_dens + RuKu3_n*dt, p_dens + RuKu3_p*dt], Generation_rate, kc_n, kc_p, ne_1, krad, S_sum, k_aug, P_esc, N_t, n0, p0)
    #    
    #    #n_dens += dt/6*(RuKu1_n + 2*RuKu2_n + 2*RuKu3_n + RuKu4_n)
    #    #p_dens += dt/6*(RuKu1_p + 2*RuKu2_p + 2*RuKu3_p + RuKu4_p)
    #    n_dens += dt*RuKu1_n
    #    p_dens += dt*RuKu1_p
    #    #t_test = np.abs(p_dens-p_dens0)/p_dens0
    #    t_test -= 1
    
    #params = Parameters()
    #params.add('n_eq', value=1e14, min=1e10, max=1e20)
    #params.add('p_eq', value=5, min=1)


    #print(p_dens)
    while test == False:

        #vars_ss = root(rate_equations, [n_dens,p_dens], method='hybr', args=(Generation_rate, kc_n, kc_p, ne_1, krad, S_sum, k_aug, P_esc, N_t, n0, p0), tol=1e-8)
        #vars_ss = least_squares(rate_equations, [1e14, 1], bounds = (1e8,1e20), args=(Generation_rate, kc_n, kc_p, ne_1, krad, S_sum, k_aug, P_esc, N_t, n0, p0))
        #vars_ss = minimize(rate_equations, params, method='shgo', args=(Generation_rate, kc_n, kc_p, ne_1, krad, S_sum, k_aug, P_esc, N_t, n0, p0), nan_policy='omit')
        vars_ss = minimize(rate_equations, [1e14,0], method ='Nelder-Mead', args=(Generation_rate, kc_n, kc_p, ne_1, krad, S_sum, k_aug, P_esc, N_t, n0, p0))
        #vars_ss = root(rate_equations, 1e12, method = 'hybr', args=(Generation_rate, kc_n, kc_p, ne_1, krad, S_sum, k_aug, P_esc, N_t, n0, p0))
        #if vars_ss.x[1] >= 0:
        test = True
    #print(vars_ss.x)    
    #print(vars_ss.x[1]/vars_ss.x[0])  
            
    #print(vars_ss)
    #print(vars_ss.x[0], vars_ss.x[1])
    #print(p_dens/vars_ss.x[1])
    def PLQE_calc(vars_ss, Generation_rate, kc_n, kc_p, ne_1, krad, S_sum, k_aug, P_esc, N_t, n0, p0):
        n_dens = vars_ss[0]
        p_dens = vars_ss[0]*(1+10**vars_ss[1])
        #nt = 10**vars_ss[1]
        #p_dens = n_dens * (1+10**root(p_estimate, 3, args=(n_dens, kc_n, kc_p, ne_1, N_t)).x)
        
        print(p_dens/n_dens)
        
        #if nt > N_t:
        #    nt = N_t
        #elif nt < 0:
        #    nt = 0

        #p_dens_list = [1, 1e20]
        #f_t = 0

        #while np.abs(p_dens_list[-1] - p_dens_list[-2]) > 10:
        #    
        #    #nt = N_t*f_t
        #    p_dens = n_dens + f_t*N_t
        #    f_t = n_dens/(n_dens + kc_p/kc_n*p_dens + ne_1)
        #    #nt_dens_list.append(f_t*N_t)
        #    p_dens_list.append(p_dens)

        #nt = nt_dens_list[-1]

       

        #f_t = n_dens/(n_dens + kc_p/kc_n*p_dens + ne_1)

        PLQE_int = (krad*((n_dens)*(p_dens)))/(Generation_rate)
        #PLQE_int = (krad*n_dens)/(krad*n_dens + kc_n*n_dens/(n_dens+ne_1+n_dens) + S_sum/(Thickness*1e-7))

        PLQE_ext = PLQE_int*P_esc#/(1-(1-P_esc)*PLQE_int)

        return PLQE_ext
        
    PLQE_ext = PLQE_calc(vars_ss.x, Generation_rate, kc_n, kc_p, ne_1, krad, S_sum, k_aug, P_esc, N_t, n0, p0)
    return PLQE_ext


def one_sun_flux_estimate(BG):
    
    wavelength, _, NREL_15AM = np.loadtxt(r'C:\Users\kober-czerny\Desktop\Python Scripts\TRPL_Fitting\TRPL_Bayes\TRPL_Files\Solar_Spectrum.txt', unpack=True)
    
    h_norm = 4.135667696e-15  # eV s
    c_0 = 299792458  # m s-1

    
    y_line = np.flip(NREL_15AM*1e-9*wavelength**2/(h_norm*c_0*1.602e-19)/1239.847*wavelength*1e-4)

    x_line = np.flip(1239.847/wavelength)


    interpolated_am15 = interpolate.CubicSpline(x_line, y_line)

    def one_sun_photonflux(BG):
        x_bg = np.linspace(BG,4,1000)
        integrated = simpson(interpolated_am15(x_bg), x_bg, dx=(x_bg[1]-x_bg[0]))
        return integrated
    
    one_sun_flux = one_sun_photonflux(BG)
    
    return one_sun_flux


def simulate_PLQE(Thickness, laser, normalize, k_aug_guess, trpl_bayes_folder, sample_name, PLQE_file_name, PLQE_sample_name, save):
    
    # Make a new folder
    PLQE_simulation_folder = f'{trpl_bayes_folder}/PLQE_simulations'
    if not os.path.exists(PLQE_simulation_folder):
        os.makedirs(PLQE_simulation_folder)

    
        ## Find laser settings
    lasers = ['405 nm', '450 nm', '532 nm', '660 nm']
    wavelength_factors = [0.0604, 0.0691, 0.0615, 0.0425]
    wavelengths = [405, 445, 532, 660]

    laser_index = lasers.index(laser)
    wavelength_factor = wavelength_factors[laser_index]
    wavelength = wavelengths[laser_index]
    spotsize = 0.15     #cm2

    # Import Data
    Raw_File = pd.read_csv(PLQE_file_name)
    Power = Raw_File[f'Power_{PLQE_sample_name}(W)'].values/wavelength_factor/spotsize
    Data = Raw_File[f'PLQE_{PLQE_sample_name}(%)'].values
    Absorption = Raw_File[f'Abs_{PLQE_sample_name}(-)'].values

    Generation_Rates = Power*wavelength*1e-9/(6.63e-34*299792458)*Absorption

    G_rates_calculation = Generation_Rates#np.logspace(np.log10(np.nanmin(Generation_Rates)*0.8), np.log10(np.nanmax(Generation_Rates)/0.8), 100, base=10)


    #if bandgap > 0:
    #    one_sun_val = one_sun_flux_estimate(bandgap)
    #else:
    #    one_sun_val = 1
    #
    #if one_sun_val > Generation_Rates[-1]/0.8:
    #    one_sun_marker = -1
    #else:
    #    one_sun_marker = np.where(Generation_Rates/one_sun_val >= 0.8)[0][0]

    # Open and unpack trace
    Raw_File = pd.read_csv(f'{trpl_bayes_folder}/{sample_name}', sep="\t")


    ## Recombination
    k_rad_model_values = Raw_File['k_rad(cm3s-1)'].values
    k_aug_model_values = Raw_File['k_aug(cm6s-1)'].values
    N_t_values = Raw_File['N_t(cm-3)'].values
    kc_n_model_values = Raw_File['kn_trapping(s-1)'].values
    kc_p_model_values = Raw_File['kp_trapping(s-1)'].values
    trap_depth_model_values = Raw_File['td(eV)'].values
    #n0_model_values = Raw_File['n_0_min(cm-3)'].values
    #pD_model_values = Raw_File['p_D_min(cm-3)'].values

    #P_esc_values = (Raw_File['p_esc1(-)'].values + Raw_File['p_esc2(-)'].values)/2
    P_esc_values = 2/(1/Raw_File['p_esc1(-)'].values +1/Raw_File['p_esc2(-)'].values)
    #P_esc_values = Raw_File['p_esc2(-)'].values

    ## Reabsorption
    if 'S_1(cm s-1)' in Raw_File.columns.values.tolist():
        S_sum_model_values = Raw_File['S_1(cm s-1)'].values + Raw_File['S_2(cm s-1)'].values
    
    else:
        S_sum_model_values = Raw_File['S_low(cm s-1)'].values + Raw_File['S_high(cm s-1)'].values


    ## Simulate the PLQE
    PLQE_all = []
    P_esc_list = []
    S_sum_list = ()
    k_rad_list = ()
    k_aug_list = ()
    kc_n_list = ()
    kc_p_list = ()
    trap_depth_list = ()
    P_esc_list = ()

    random_samples = 500

    for rand_pick in np.arange(500):
        rand_pick = random.randint(0,len(k_rad_model_values)-1)
        #rand_pick = int(len(k_rad_model_values)/(random_samples))*number

        S_sum_list = np.append(S_sum_list, S_sum_model_values[rand_pick])
        k_rad_list = np.append(k_rad_list, k_rad_model_values[rand_pick])
        k_aug_list = np.append(k_aug_list, k_aug_model_values[rand_pick])
        n0 = 0#np.min(np.array([n0_model_values[rand_pick],p0_model_values[rand_pick]]))
        pD = 0#pD_model_values[rand_pick]

        kc_n_list = np.append(kc_n_list, kc_n_model_values[rand_pick])
        kc_p_list = np.append(kc_p_list, kc_p_model_values[rand_pick])
        trap_depth_list = np.append(trap_depth_list, trap_depth_model_values[rand_pick])
        N_t = N_t_values[rand_pick]
        P_esc_list = np.append(P_esc_list, P_esc_values[rand_pick])#np.median(P_esc_values)
        P_corr_list = []
        
        PLQE = ()
        PLQE_medvals = ()
        for Generation in G_rates_calculation:
            
            PLQE_i = plqe_simulation(Generation/(Thickness*1e-7), Thickness, S_sum_list[-1], k_rad_list[-1], kc_n_list[-1], kc_p_list[-1], trap_depth_list[-1], k_aug_list[-1], 1, N_t, n0, pD)           
            PLQE = np.append(PLQE, PLQE_i)

        PLQE_all.append(PLQE)

    for Generation in G_rates_calculation:
            
        PLQE_medvals = np.append(PLQE_medvals,  plqe_simulation(Generation/(Thickness*1e-7), Thickness, np.median(S_sum_list), np.median(k_rad_list), np.median(kc_n_list), np.median(kc_p_list) ,np.median(trap_depth_list), np.median(k_aug_list), np.median(P_esc_list), np.median(N_t_values), 0, 0))


    PLQE_median = np.median(P_esc_list)*np.median(PLQE_all,axis=0)
    print(PLQE_median)
    PLQE_q1 =PLQE_median-np.median(P_esc_list)*np.quantile(PLQE_all,0.25, axis=0)
    PLQE_q3 = (np.median(P_esc_list)*np.quantile(PLQE_all,0.75, axis=0)-PLQE_median)

    #PLQE_q1 = PLQE_medvals - PLQE_quart1
    #PLQE_q3 = PLQE_quart3 = PLQE_medvals

    mid_marker = int(len(Data)/2)

    mid_G_rate_marker = np.where(G_rates_calculation >= Generation_Rates[mid_marker])[0][0]

    PLQE_corr = Data[mid_marker]/100/PLQE_median[mid_G_rate_marker]

    if normalize:

        print(str(r'$P_{\mathrm{para.}}$: ' + '{:.2f}'.format(100*PLQE_corr) + "%"))
        PLQE_median_norm = PLQE_median*PLQE_corr
        PLQE_q1_norm = PLQE_q1*PLQE_corr
        PLQE_q3_norm = PLQE_q3*PLQE_corr

    PLQE_error = np.vstack((PLQE_q1,PLQE_q3))


    df_save = pd.DataFrame()

    df_save['Generation_rate (/cm2/s)'] = Generation_Rates
    df_save['PLQE_Data (%)'] = Data
    
    df_save_calc = pd.DataFrame()
    df_save_calc['Generation_rate (/cm2/s)'] = G_rates_calculation
    df_save_calc['PLQE_median (%)'] = PLQE_median*100
    df_save_calc['PLQE_q1'] = PLQE_q1*100
    df_save_calc['PLQE_q3'] = PLQE_q3*100
    df_save_calc['PLQE_median_norm (%)'] = PLQE_median_norm*100
    df_save_calc['PLQE_q1_norm'] = PLQE_q1_norm*100
    df_save_calc['PLQE_q3_norm'] = PLQE_q3_norm*100
    df_save_calc['PLQE_medvals (%)'] = PLQE_medvals*100

    if normalize:
        df_save_calc['PLQE_median_nonnorm (%)'] = PLQE_median*100/PLQE_corr


    #plt.errorbar(G_rates_calculation, PLQE_median*100, yerr=PLQE_error*100, c=color_scheme[1])
    plt.plot(G_rates_calculation, df_save_calc['PLQE_median (%)'],zorder=1000,  c=color_scheme[1], label='Median Calcuated PLQE')
    plt.plot(G_rates_calculation, df_save_calc['PLQE_median_norm (%)'],zorder=1000,  c=color_scheme[3], label='Median Calcuated PLQE (norm.)')
    #plt.plot(G_rates_calculation, df_save_calc['PLQE_medvals (%)'],zorder=1000,  c=color_scheme[2], linestyle='--', label='Calcuated PLQE of Medians')
    plt.plot(Generation_Rates, Data, marker='o', c=base_color, label='Measured PLQE')
    plt.fill_between(G_rates_calculation, df_save_calc['PLQE_median (%)']-df_save_calc['PLQE_q1'].values, df_save_calc['PLQE_median (%)']+df_save_calc['PLQE_q3'].values, color=color_scheme[1], alpha=0.2, zorder=-1000)


    plt.xscale('log')
    plt.yscale('log')
    #plt.ylim(5e-2,1e1)
    plt.xlabel('Photon Flux (cm$^{-2}$ s$^{-1}$)')
    plt.ylabel('PLQE (%)')
    plt.annotate(PLQE_sample_name, (0.05, 0.95), xycoords="axes fraction", fontsize=fontsize_base+1, c=base_color)
    plt.legend(frameon=False, loc='lower right')
    # Save the data

    
    
    if save == 'yes':
    
        plt.savefig(f'{PLQE_simulation_folder}/{sample_name}_plot.png', format='png', dpi=300, transparent=True)
        df_save.to_csv(f'{PLQE_simulation_folder}/{sample_name}_data.dat', sep='\t', index= True, mode='w')
        df_save_calc.to_csv(f'{PLQE_simulation_folder}/{sample_name}_simulation.dat', sep='\t', index= True, mode='w')
    

    plt.show()

    return df_save

