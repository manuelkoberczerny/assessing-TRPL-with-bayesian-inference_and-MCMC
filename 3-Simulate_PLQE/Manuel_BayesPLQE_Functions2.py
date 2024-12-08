import sys
import os
from pathlib import Path, PurePath
from os import chdir
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from scipy.integrate import simpson
from scipy.optimize import least_squares, root, minimize
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


def plqe_simulation(G, params, data):
    
    def rate_equations(n_pow, G, params):            
            ### Pseudo-Equilibrium at each time-point
            n_dens = 10**n_pow
            kS, krad, k_nr, k_aug, p_eq = params
                        
            p_dens =  n_dens + p_eq
                        
            R_nr =  - (k_nr+kS)*n_dens
            
            R_rad = - krad*n_dens*p_dens
            R_aug = - k_aug*(p_dens*n_dens**2 + p_dens**2*n_dens)

            dn_dt = R_nr + R_rad + R_aug
            
            return (G + dn_dt)**2
    
    
    def PLQE_calc(n_pow, G, params):
        
        n_dens = 10**n_pow
        kS, krad, k_nr, k_aug, p_eq = params

        p_dens =  n_dens + p_eq
            
        PLQE_int = (krad*(n_dens*(p_dens)))/(G)

        return PLQE_int

    vars_ss = minimize(rate_equations, -2, method='Powell', args=(G, params))    
    PLQE_int = PLQE_calc(vars_ss.x, G, params)

    return PLQE_int


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


def simulate_PLQE(Thickness, laser, normalize, k_aug_guess, trpl_bayes_folder, sample_name, PLQE_file_name, PLQE_sample_name, save, random_samples):
    
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

    G_rates_calculation = Generation_Rates

    # Open and unpack trace
    Raw_File = pd.read_csv(f'{trpl_bayes_folder}/{sample_name}', sep="\t")


    ## Recombination

    k_rad = Raw_File['k_rad(cm3s-1)'].values * 1e12    

    #k_deep = Raw_File['k_deep(s-1)'].values
    #k_capture = Raw_File['k_capture(s-1)'].values
    #k_emission = Raw_File['n_em(cm-3)'].values * 1e-12


    p_eq = 0.5*Raw_File['peq_min(cm-3)'].values * 1e-12
    k_nr_eff = Raw_File['k_nr_eff(s-1)'].values

    ## Reabsorption
    if 'S_1(cm s-1)' in Raw_File.columns.values.tolist():
        S1 = Raw_File['S_1(cm s-1)'].values 
        S2 = Raw_File['S_2(cm s-1)'].values
        S_sum_model_values = Raw_File['S_1(cm s-1)'].values + Raw_File['S_2(cm s-1)'].values
        D = 0.4*0.025
        kS = (S1 + S2) /(Thickness*1e-7) #+ (D/(Thickness*1e-7)**2 * np.pi**2)
    
    
    else:
        S_sum_model_values = Raw_File['S_low(cm s-1)'].values + Raw_File['S_high(cm s-1)'].values
        S1 = Raw_File['S_low(cm s-1)'].values
        S2 = Raw_File['S_high(cm s-1)'].values
        D = 0.4*0.025
        kS = (S1 + S2) /(Thickness*1e-7) #+ (D/(Thickness*1e-7)**2 * np.pi**2)

    ## Simulate the PLQE
    PLQE_all = []
    

    for rand_pick in np.arange(random_samples):
        rand_pick = random.randint(0,len(k_rad)-1)
        
                
        PLQE = ()
        for Generation in G_rates_calculation:
            
            params = kS[rand_pick], k_rad[rand_pick], k_nr_eff[rand_pick], k_aug_guess*1e24, p_eq[rand_pick]

            #params = np.median(S_sum_model_values)/(Thickness*1e-7), np.median(k_rad), np.median(k_nr_eff), 1e24*k_aug_guess, np.median(Nt), np.median(k_deep), np.median(k_capture), np.median(k_emission)


            PLQE_i = plqe_simulation(Generation/(Thickness*1e-7)*1e-12,  params, Data)           
            PLQE = np.append(PLQE, PLQE_i)

        PLQE_all.append(PLQE)
        
    
    PLQE_median = np.nanmedian(PLQE_all, axis=0)
    print(PLQE_median)
    PLQE_q1 = PLQE_median-np.quantile(PLQE_all,0.25, axis=0)
    PLQE_q3 = (np.quantile(PLQE_all,0.75, axis=0)-PLQE_median)

    #PLQE_q1 = PLQE_medvals - PLQE_quart1
    #PLQE_q3 = PLQE_quart3 = PLQE_medvals

    mid_marker = int(len(Data)/2)

    mid_G_rate_marker = np.where(G_rates_calculation >= Generation_Rates[mid_marker])[0][0]

    PLQE_corr = Data[mid_marker]/100/PLQE_median[mid_G_rate_marker]

    if normalize:

        print(str(r'$P_{\mathrm{esc.}}$: ' + '{:.2f}'.format(100*PLQE_corr) + "%"))
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

    if normalize:
        df_save_calc['PLQE_median_nonnorm (%)'] = PLQE_median*100/PLQE_corr


    #plt.errorbar(G_rates_calculation, PLQE_median*100, yerr=PLQE_error*100, c=color_scheme[1])
    plt.plot(G_rates_calculation, df_save_calc['PLQE_median_norm (%)'],zorder=1000,  c=color_scheme[1], label='Median Calcuated PLQE (norm.)')
    plt.plot(G_rates_calculation, df_save_calc['PLQE_median (%)'],zorder=1000, linestyle = '--',  c=color_scheme[3], label='Median Calcuated PLQE')
    #plt.plot(G_rates_calculation, df_save_calc['PLQE_medvals (%)'],zorder=1000,  c=color_scheme[2], linestyle='--', label='Calcuated PLQE of Medians')
    plt.plot(Generation_Rates, Data, marker='o', c=base_color, label='Measured PLQE')
    plt.fill_between(G_rates_calculation, df_save_calc['PLQE_median_norm (%)']-df_save_calc['PLQE_q1_norm'].values, df_save_calc['PLQE_median_norm (%)']+df_save_calc['PLQE_q3_norm'].values, color=color_scheme[1], alpha=0.2, zorder=-1000)


    plt.xscale('log')
    plt.yscale('log')
    plt.ylim(5e-2,1e2)
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

