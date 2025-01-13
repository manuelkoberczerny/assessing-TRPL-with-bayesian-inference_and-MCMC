import os
folder = os.getcwd()

import numpy as np
import pandas as pd
import pymc as pm
from scipy.integrate import odeint, simpson
from scipy import interpolate



def cut_data(Data, time):
    max_locator = np.argmax(Data)
    dtime = time[2] - time[1]
    pre_zero_time = np.linspace(max_locator, 1+dtime, max_locator)
    pre_zero_time = pre_zero_time * dtime * -1

    if time[0] < 0:
        time = time[1:-max_locator]
        Data = Data[max_locator+1:]
    else:
        time = time[:-max_locator] 
        Data = Data[max_locator:]

    return Data, time, max_locator



def find_bckg(max_locator, Data):
    # Find the highest value before the pulse as an upper limit for PL_err   
    Data_mask = np.array(Data[1:max_locator - 3])
    Bckg = np.max(Data_mask)
    
    return Bckg



def unpack_Data(FileName):
    """ Unpack Data File"""

    ## First the data is imported, the background removed, the maximum shifted to t=0 and everything is normalized to Exc_Density

    rows_to_skip = 73
    time1 = None
    while time1 is None:
        try:
            time1, Data1 = np.loadtxt(FileName, unpack=True, skiprows= rows_to_skip, encoding='unicode_escape')
        except:
            rows_to_skip += 1
            pass
        else:
            time1, Data1 = np.loadtxt(FileName, unpack=True, skiprows= rows_to_skip, encoding='unicode_escape')
            
    Data1 = np.array(Data1)
    len_Data = len(Data1)

    """ Cut Data into Shape"""
    Data2, time2, max_locator = cut_data(Data1, time1)
    
    """ Find Background from before Pulse"""
    Bckg = find_bckg(max_locator, Data1)
       
   
    """ Normalize Data"""
    Data3 = Data2 / Data2[0]

    Data3[Data3 <= 0] = np.min(Data3[Data3 > 0])


    return time2, Data3, len_Data



def make_Dataframe(time, data_raw, len_Data):

    Data = pd.DataFrame()


    ## The data is cut to the correct lengths and stored in a pd.Dataframe
    ### each dataset starts at t = 0
    
    Data['Time'] = time[0]
    Data['0'] = data_raw[0]

    i = 1
    while i < len(len_Data):

        Data2 = data_raw[i]

        if len(Data2) != len(np.array(Data['Time'])):
            if len(Data2) > len(np.array(Data['Time'])):
                a = np.abs(len(Data2)-len(np.array(Data['Time'])))
                Data2 = Data2[0:-a]
            else:
                a = np.abs(len(Data2) - len(np.array(Data['Time'])))
                Data2 = np.append(Data2, np.zeros(a))

        Data[str(i)] = Data2

        i += 1


    return Data



def unpack_Info(args):
    sample_names, info = np.loadtxt(args, unpack=True, skiprows=1, max_rows=2, delimiter=':', dtype=str, encoding='unicode_escape')
    sample_name = str(info[np.where(sample_names == '  Sample ')])[2:-2]

    with open(args, 'rb') as file:
    # read all lines using readline()
        lines = file.readlines()
        for line in lines:
            # check if string present on a current line
            word = b'  Exc_Wavelength :'
            if line.find(word) != -1:
                rows_to_skip = lines.index(line)

    names, info = np.loadtxt(args, unpack=True, skiprows=rows_to_skip, max_rows=21, delimiter=':', dtype=str, encoding='unicode_escape')
    wavelength = float(str(info[np.where(names == '  Exc_Wavelength ')])[3:-4])  # in nm
    sync_frequency = float(str(info[np.where(names == '  Sync_Frequency ')])[3:-4])  # in Hz
    signal_rate = float(str(info[np.where(names == '  Signal_Rate ')])[3:-5])  # in cps
    pile_up = signal_rate / sync_frequency * 100  # in %

    attenuation = str(info[np.where(names == '  Exc_Attenuation ')])[3:-6]
    if attenuation == 'open':
        attenuation = 1
    else:
        attenuation = float(attenuation[0:-1]) / 100

    return pile_up, attenuation, wavelength, sync_frequency, sample_name


def Fluence_Calc(wavelength, intensity, laserpower_file):
    """ Unpack Ref Data File"""
    path = os.getcwd()
    parent_directory = os.path.abspath(os.path.join(path, os.pardir))
    file_path = f'{parent_directory}\Ref_Files\{laserpower_file}'
    wl400, wl505, wl630 = np.loadtxt(file_path, unpack=True, skiprows=1)



    if wavelength == 397.7:

        laser_fluence = wl400[0] * intensity + wl400[1]

    elif wavelength == 505.5:

        laser_fluence = wl505[0] * intensity + wl505[1]

    elif wavelength == 633.8:

        laser_fluence = wl630[0] * intensity + wl630[1]

    return laser_fluence



def unpack_filenames(FileNames, intensity, Reflectance, laserpower_file):
    
    pile_up = []
    sample_names = []
    data = []
    time = []
    len_Data = []
    Fluence = []
    y_list = []
    
    for FileName in FileNames:
        pile_up_1, attenuation, wavelength, frequency, sample_name_1 = unpack_Info(FileName)
        pile_up.append(pile_up_1)
        sample_names.append(sample_name_1)
    
        ## Steps to calculate Laser Fluence and Excitation Density
        laser_fluence = Fluence_Calc(wavelength, intensity, laserpower_file)  # in cm-2
        laser_fluence = laser_fluence * attenuation   # in cm-2
        Fluence.append(laser_fluence)
        #laser_fluence_old.append(laser_fluence * (299792458 * 6.6261e-34) / (wavelength * 1e-9) * 1e9)  # in nJ cm-2
        
        
        time2, Data4, len_Data1 = unpack_Data(FileName)
        data.append(Data4)
        time.append(time2)
        len_Data.append(len_Data1)
    
    Fluence = np.array(Fluence) # in cm-2
        
    #### Length of Data is changed so it fits into a single Dataframe
    for y in range(1,len(FileNames)):
        len_Data[y] = len_Data[y] + len_Data[y - 1]
    
    #### The Dataframe is created    
    df = make_Dataframe(time, data, len_Data) 

    return df, pile_up, sample_names, Fluence, wavelength, frequency


def one_sun_condition_estimate(BG, sync_frequency, thickness):

    path = os.getcwd()
    parent_directory = os.path.abspath(os.path.join(path, os.pardir))
    file_path = f'{parent_directory}/Ref_Files/Solar_Spectrum.txt'
    wavelength, NREL_Sun, NREL_15AM = np.loadtxt(file_path, unpack=True)
    
    h_bar = 6.582119569e-16  # eV s
    h_norm = 4.135667696e-15  # eV s
    k_const = 8.617e-5  # eV K-1
    mu_gamma = 0
    mass = 9.10938356e-31 # kg
    c_0 = 299792458  # m s-1

    
    energy_per_photon_NREL = h_norm*1.602e-19*c_0/(wavelength)**2/1e-9

    y_line = np.flip(NREL_15AM*1e-9*wavelength**2/(h_norm*c_0*1.602e-19)/1239.847*wavelength*1e-4)

    x_line = np.flip(1239.847/wavelength)


    interpolated_am15 = interpolate.CubicSpline(x_line, y_line)

    def one_sun_photonflux(BG):
        x_bg = np.linspace(BG,4,1000)
        integrated = simpson(interpolated_am15(x_bg), x_bg, dx=(x_bg[1]-x_bg[0]))
        return integrated
    
    one_sun_condition_density = one_sun_photonflux(BG)/sync_frequency/thickness
    
    return one_sun_condition_density


def open_config_file(data_folder_trpl, trace_file):


    trace_file_split = trace_file.split('_trace_')
    config_file_name = str(trace_file_split[0] + ".csv")

    config = pd.read_csv(f"{data_folder_trpl}/{config_file_name}",header=None)
        
    ### These need to be entered via CONFIG file
    Files = config[0]
    Thickness = int(config[1][0]) #nm
    Surface = np.array(config[2])
    Absorption_coeff = np.array(config[3])
    Reflectance = np.array(config[4])
    intensity = np.array(config[5][0])
    max_time = np.array(config[6][0])
    
    FileNames = []
    for file in Files:
        FileNames.append(f"{data_folder_trpl}/{file}.dat")

    return FileNames, Thickness, Surface, Absorption_coeff, Reflectance, intensity, max_time


def open_log_file(data_folder_trpl, trace_file):

    log_file_name = trace_file.replace(".nc", "_logfile.txt")
    log_info = pd.read_csv(f"{data_folder_trpl}/{log_file_name}",header=None, sep='\t')
    
    spacing = log_info[0][0]
    laserpower_file = log_info[1][0]
    sh_defect = log_info[2][0]
    PN_on_off = log_info[3][0]
    diffusion_on_off = log_info[4][0]

    try:
        bckg_list = np.array(log_info[5][0][1:-2].split(' '))
    except:
        bckg_list = np.zeros(3)
    else:
        bckg_list = np.array(log_info[5][0][1:-2].split(' '))

    return spacing, laserpower_file, bckg_list, PN_on_off, diffusion_on_off, sh_defect


def Bayes_TRPL_Utils(data_folder_trpl, trace_file):

    spacing, laserpower_file, bckg_list, PN_on_off, diffusion_on_off, sh_defect = open_log_file(data_folder_trpl, trace_file)
    FileNames, Thickness, Surface, Absorption_coeff, Reflectance, intensity, max_time = open_config_file(data_folder_trpl, trace_file)    
    df, pile_up, sample_name, Fluence, wavelength, frequency = unpack_filenames(FileNames, intensity, Reflectance, laserpower_file)

    amax = np.where(np.array(df['Time']) <= max_time )[0][-1]
    
    one_sun_carrier_density = one_sun_condition_estimate(1240/wavelength, frequency, Thickness*1e-7)
    
    print('Files ready...')
    return df, pile_up, sample_name, Fluence, Thickness, Surface, Absorption_coeff, Reflectance, intensity, amax, one_sun_carrier_density, spacing, bckg_list, PN_on_off, diffusion_on_off, sh_defect









































