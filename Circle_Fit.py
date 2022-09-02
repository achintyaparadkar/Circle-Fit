# -*- coding: utf-8 -*-
"""
Created on Wed Nov 3 15:06 2021

Last edited Fri Feb 4 15:41 2022

@author: Avan Mirkhan

!v1.2 - Added stepchannels code structure to further automization of the script. The powers and resonator indexing is automatically taken from the VNA data itself.

!v1.1 - Added the pathlib module and adjusted the pathing of the script for further automization.

!v1.0 - First "release" of the code

This code is designed to be as general as possible. No edits in this specific file, as this is simply a "draft" to copy and paste locally to your computer
Local files can naturally be edited.

Credits goes to Sebastian Probst (2015) for having developed the Circle_Fit script.

Link to the paper: https://aip.scitation.org/doi/pdf/10.1063/1.4907935
Link to Circle Fit on GitHub: https://github.com/sebastianprobst/resonator_tools

This script file is written on Spyder

"""
#%%
# =============================================================================
# PACKAGES
# =============================================================================
import numpy as np
import sys
import csv
import pandas as pd
import matplotlib.pyplot as plt
import Labber
import pathlib
path_name = 'C:/Users/achintya/OneDrive - Chalmers/Documents/1. Project/5. Coupling Project/Experiments/'
file_name = '220509_Aluminum_SQUID_Chip_LCR_High_Power_Sweep_3.hdf5'
in_path = path_name+file_name #Directory path of the .hdf5 file that is going to be fitted
logFile = Labber.LogFile(in_path)
import DataModule as dm

# =============================================================================
# LOADING OF DATA AND PARTITIONING OF DATA
# =============================================================================

#Data
data = logFile.getData('VNA - S21') #Measurement data, usually magnitude. Should have the shape of (total # measurements, e.g. #power*#resonators x # data points)
# - each measurement should have 5000 + 10000 + 5000 pts
stepchannels = logFile.getStepChannels()

powers = [chan['values'] for chan in stepchannels if chan['name'] ==  'VNA - Output power']
powers = np.array(powers).flatten()
resonators = [chan['values'] for chan in stepchannels if chan['name'] == 'Central_frequency']
resonators = np.array(resonators).flatten()
# n_center = [chan['values'] for chan in stepchannels if chan['name'] == 'n_center']
# n_center = np.array(n_center).flatten()
# flux = [chan['values'] for chan in stepchannels if chan['name'] == 'Source Meter - Source current']
# #Partiotioning  and removal of NaN points of the data below
data = np.reshape(data, (len(resonators),len(powers),3,-1))
data_c = data[:, :, 1, :]
data_l = data[:, :, 0, :]
data_r = data[:, :, 2, :]

# data_c = data_c[~np.isnan(data_c)]
# data_c = np.reshape(data_c, (len(resonators),len(powers), -1))
# # print(data_l.shape) 
# data_l = data_l[~np.isnan(data_l)]
# data_l = np.reshape(data_l, (len(resonators),len(powers), -1))
# # print(data_l.shape)
# # print(data_r.shape) 
# data_r = data_r[~np.isnan(data_r)]
# data_r = np.reshape(data_r, (len(resonators),len(powers), -1))

data = np.concatenate((data_l, data_c, data_r), axis=2)
# # print(data_r.shape)
# data = data[~np.isnan(data)]
# data = np.reshape(data,(len(resonators),len(powers),-1))

#Frequency
fcenter = [chan['values'] for chan in stepchannels if chan['name'] == 'Central_frequency'] #Measurement data, here center frequency
fcenter = np.array(fcenter).flatten()
# freqs = np.reshape(fcenter, (len(resonators),len(powers),3,-1)) #Reshaping of data into (Powers, Resonators, [left, center, right])

# f_center = fcenter[1:209:3] #Center frequency
# f_left = fcenter[0:210:3] #Left tail of the frequency
# f_right = fcenter[2:210:3] #Right tail of the frequency

#Points
points = int([chan['values'] for chan in stepchannels if chan['name']=='VNA - # of points'][0])

#Span
span = [chan['values'] for chan in stepchannels if chan['name'] == 'Span_step'] #Span of the data, should be a constant
span = np.array(span).flatten()

f1 = np.linspace(span[0],span[1], points)
f2 = np.linspace(span[1], span[2], points)
f3 = np.linspace(span[2], -span[0], points)
f4 = np.concatenate((f1,f2,f3))

freq_span = [(f4+fcenter[i]) for i in range (0,len(fcenter))]
freqs = np.vstack((freq_span[0]))
for i in range(1,len(fcenter)):
    freqs = np.vstack((freq_span,freq_span[i]))
# span = np.linspace(-span/2, span/2, data.shape[-1])
# span_center = np.linspace(-span/2, span/2, data_c.shape[-1]) #Span of the center frequency
# span_lr = np.linspace(-span/2, span/2, data_r.shape[-1]) #Span of the frequencies to the left an right




# pwr = np.linspace(20,-20,powers) #Power applied from the VNA - Should normally be from 20 dBm to -25 dBm, with the interval taken inbetween through Labber
# res = np.linspace(1,1,resonators) #Number of resonators in your device


#Parameters which are going to be extracted from the circle fit
#Remains as empty list


QL = []             #Loaded Quality Factor
QC = []             #Coupling Quality Factor
Qint =[]            #Internal Quality Factor
R_sqr = []          #R^2
Error_QL = []       #Error of QL
Error_QC = []       #Error of QC
Error_Qint = []     #Error of Qint
res_freq = []       #Resonance Frequency
Error_res_freq = [] #Error of Resonance Frequency
phase = []          #Phase
Error_phase = []    #Error of Phase

pwr_idx_str = []    #Power index as string
res_idx_str = []    #Resonance index as string

out_path = path_name                      #Desired output path of files e.g. excel, csv or data files.
img_folder = pathlib.Path(path_name).joinpath('Data Fit images')
if not img_folder.exists():
    img_folder.mkdir()

# =============================================================================
# THE CIRCLE FIT
# =============================================================================

for res_idx in range(len(resonators)):               #Loop over the number of powers
    res_idx_str = str(res_idx+1)
    for pwr_idx in range(len(powers)):            #Loop over the number of resonators
        pwr_idx_str = str(powers[pwr_idx])
        # for flux_idx in range(len(flux)):
        # for flux_idx in range(np.where(flux == flux[41])[0][0],np.where(flux == flux[41])[0][0]):
            # flux_idx_str = str(flux[flux_idx])
        # fsweep = np.concatenate((freqs[res_idx, pwr_idx, 0] + span_lr,
        #                     freqs[res_idx, pwr_idx, 1] + span_center,
        #                     freqs[res_idx, pwr_idx, 2] + span_lr))                                                  #Concatenation of the frequencies into one vector
        # dat  = np.concatenate((data_l[res_idx,pwr_idx,:], data_c[res_idx,pwr_idx,:], data_r[res_idx,pwr_idx,:]))  #Concatenation of the measurement data into one vector
        # fsweep = (freqs[res_idx,pwr_idx, 1] + span_center)
        # dat  = data_c[res_idx,pwr_idx,:]
        # fsweep = ((freqs[res_idx,pwr_idx,:] + span))
        # dat = data[res_idx,pwr_idx,:]
        img_folder = img_folder.joinpath('Resonator '+res_idx_str)
        if not img_folder.exists():
            img_folder.mkdir()
        data_fitted = dm.data_complex(freqs.flatten(), data[res_idx,pwr_idx,:])                                      #Input variable to Cricle Fit with frequencies and data as in-data
        data_fitted.correct_delay()                                                     #Calculations of the electronic delay time (tau in the Probst paper)
        data_extracted = data_fitted.circle_fit_notch(comb_slopes=True)                                 #Circle Fit calculations - This is where the parameters are extracted from
        data_fitted.plot(engine='pyplot')                                               #Plots of the data
        # plt.savefig(str(img_folder)+'/'+'Resonator ' + res_idx_str + ' at ' + pwr_idx_str +' [dBm].png')  #Saving plots
        plt.show()                                                                      #Visualizing plot | Lines 90-92 can be hidden if plots are not interested.
        QL_temp = data_extracted.iat[0,0]
        QC_temp = data_extracted.iat[1,0]
        Qint_temp = data_extracted.iat[2,0]
        R_sqr_temp = data_extracted.iat[5,0]
        res_freq_temp = data_extracted.iat[3,0]
        Error_QL_temp = data_extracted.iat[0,1]
        Error_QC_temp = data_extracted.iat[1,1]
        Error_Qint_temp = data_extracted.iat[2,1]
        Error_res_freq_temp = data_extracted.iat[3,1]
        phase_temp = data_extracted.iat[4,0]
        Error_phase_temp = data_extracted.iat[4,1]                                     #Lines 93-103: Extraction of each individual parameter into their own variable
        img_folder = img_folder.parents[0]
        QL.append(QL_temp)
        QC.append(QC_temp)
        Qint.append(Qint_temp)
        R_sqr.append(R_sqr_temp)
        res_freq.append(res_freq_temp)
        Error_QL.append(Error_QL_temp)
        Error_QC.append(Error_QC_temp)
        Error_Qint.append(Error_Qint_temp)
        Error_res_freq.append(Error_res_freq_temp)
        phase.append(phase_temp)
        Error_phase.append(Error_phase_temp)
        print(powers[pwr_idx])
        # print(resonators[res_idx])
        
# =============================================================================
# GATHERING DATA IN A SINGLE VARIABLE
# =============================================================================
QL = np.reshape(QL,(len(powers),len(resonators)))
QC = np.reshape(QC,(len(powers),len(resonators)))
Qint = np.reshape(Qint,(len(powers),len(resonators)))
Error_QL = np.reshape(Error_QL,(len(powers),len(resonators)))
Error_QC = np.reshape(Error_QC,(len(powers),len(resonators)))
Error_Qint = np.reshape(Error_Qint,(len(powers),len(resonators)))
res_freq = np.reshape(res_freq,(len(powers),len(resonators)))
Error_res_freq = np.reshape(Error_res_freq,(len(powers),len(resonators)))
phase = np.reshape(phase,(len(powers),len(resonators)))
Error_phase = np.reshape(Error_phase,(len(powers),len(resonators)))
R_sqr = np.reshape(R_sqr,(len(powers),len(resonators)))
input_variable = [
     (res_freq),
     (Error_res_freq),
     (QL),
     (Error_QL),
     (QC),
     (Error_QC),
     (Qint), 
     (Error_Qint),
     (phase),
     (Error_phase),
     (R_sqr)
     ]                                                                                  #Lines 104-116: Inserting each parameter into a single variable

input_variable = np.array(input_variable)                                               #Converting from list to array      

# res_idx_str = ['Resonator {0}'.format(i) for i in range(1,int(res[-1])+1)]
# pwr_idx_str = str( pwr[pwr_idx] )
# input_variable = np.array(input_variable)
# input_variables_df = pd.DataFrame(data=input_variable, index = ['Resonance_Frequency','Error_Resonance_Frequency',
#                                                                   'QL', 'Error_QL', 'QC', 'Error_QC', 'Qint', 'Error_Qint','\phi_0','Error \phi_0','R^2'],
#                                       columns = res_idx_str*len(pwr))                            #Converting from array to DataFrame
# start_str = str(pwr[0])
# stop_str = str(pwr[-1])
# input_variables_df.to_excel(out_path+'VNA Data '+start_str+' to '+stop_str+' dBm.xlsx')                            #Export to Excel
    
#If needed, single plots are done here:

# plt.plot(fsweep, 20*np.log10(np.abs(dat)))
# data_fitted = dm.data_complex(fsweep, dat)
# data_fitted.correct_delay()
# data_fitted.circle_fit_notch()
# data_fitted.plot(engine='pyplot')

# =============================================================================
# CONVERSION AND WRITE TO .CSV FORMAT
# =============================================================================

# res_idx_str = ['Resonator {0}'.format(i) for i in range(1,int(res[-1])+1)]
# pwr_idx_str = str( pwr[pwr_idx] )
# input_variable = np.array(input_variable)
# input_variables_df = pd.DataFrame(data=input_variable, index = ['Resonance_Frequency','Error_Resonance_Frequency',
#                                                                   'QL', 'Error_QL', 'QC', 'Error_QC', 'Qint', 'Error_Qint','\phi_0','Error \phi_0','R^2'],
#                                       columns = res_idx_str*len(pwr))                            #Converting from array to DataFrame
# start_str = str(pwr[0])
# stop_str = str(pwr[-1])
# input_variables_df.to_excel(out_path+'VNA Data '+start_str+' to '+stop_str+' dBm.xlsx')                           #Export to Excel
    
#If needed, single plots are done here:

# plt.plot(fsweep, 20*np.log10(np.abs(dat)))
# data_fitted = dm.data_complex(fsweep, dat)
# data_fitted.correct_delay()
# data_fitted.circle_fit_notch()
# dada_fitted.plot(engine='pyplot')
#%%
# =============================================================================
# CALCULATION OF PHOTON NUMBERS AND PLOTTING
# =============================================================================
import math

# =============================================================================
# INTERNAL FRIDGE ATTENUATION
# =============================================================================

fridge_frequency = np.array([0.5, 1.0, 5.0, 10.0, 20.0])*1e9   #Frequency Data points for Coaxial cables
SN086attlen = np.array([1, 1.5, 3.2, 4.6, 6.5])   #Attenuation [dB]/meter [m] for SN086 Coaxial Cables
SN219attlen = np.array([0.4, 0.6, 1.3, 1.8, 2.6]) #Attenuation [dB]/meter [m] for SN219 Coaxial Cables
CryostatLength = 0.945 #Cryostat Length
SN086_Attenuation_Frequency = SN086attlen*CryostatLength
SN219_Attenuation_Frequency = SN219attlen*CryostatLength
SN086_Attenuation = np.polyfit(fridge_frequency,SN086_Attenuation_Frequency,1)
SN219_Attenuation = np.polyfit(fridge_frequency,SN219_Attenuation_Frequency,1)
RF_Att = [SN086_Attenuation, SN219_Attenuation]
Fridge_Attenuation = 66

# =============================================================================
# CABLE ATTENUATION EXPERIMENTAL DATA
# =============================================================================

# #ULC-6FT-SMSM+
# CableLoss1 = np.array([-3.30, -3.50, -3.99, -4.20, -4.88]) # 1 cable
# CableLoss2 = np.array([-5.44, -5.85, -6.62, -7.06, -7.88]) # 2 cables

#FLC-2M-SMSM+
CableLoss1 = np.array([-3.82, -4.12, -4.68, -4.94, -5.96]) # 1 cable
CableLoss2 = np.array([-5.50, -5.98, -6.78, -7.20, -8.02]) # 2 cables
f          = np.array([4,5,6,7,8])*1e9
p1 = np.polyfit(f,CableLoss1,1)
p2 = np.polyfit(f,CableLoss2,1)

#Cable Loss Downline
CableLoss  = np.array([-2.70, -3.37, -3.85, -4.33, -5.22])
p7         = np.polyfit(f,CableLoss,1)

#VNA Cable Losses Upline
Upline1    = np.array([-2.90, -3.07, -3.52, -3.68, -4.23]) # 1 cable
Upline2    = np.array([-4.16, -4.41, -5.00, -5.35, -5.98]) # 2 cables
p3 = np.polyfit(f,Upline1,1)
p4 = np.polyfit(f,Upline2,1)

#Fridge Downline Input
FridgeDownlineInput = np.array([-79.35, -80.9, -82.75, -84, -85.8])
FridgeDownlineInput = FridgeDownlineInput - CableLoss2 - Upline2 + Fridge_Attenuation
p5 = np.polyfit(f,FridgeDownlineInput,1)

#Fridge Upline Input
FridgeUplineInput = np.array([-4.84 -5.92 -5.90 -6.32 -7.13])
FridgeUplineInput = FridgeUplineInput - CableLoss1 - Upline1
p6 = np.polyfit(f,FridgeDownlineInput,1)

polyfits = [p1, p3, p5, p6, p7]


# =============================================================================
# CONSTANTS
# =============================================================================
Z0 = 50
# Zr = 49.5031 #From Mathematica, Niobium
Zr = 56.4019 #From Mathematica, Aluminum
Zmm = Z0/Zr
hbar = 1.0545718e-34
External_Attenuation = 10 #RT attenuation added

# =============================================================================
# CALCULATIONS
# =============================================================================

fr = res_freq
Qtot = QL
Qtot_error = Error_QL
Qc = QC
Qc_error = Error_QC
Qi = Qint
Qi_error = Error_Qint
# for i in range(len(resonators)):
#     fr.append(res_freq[i])
#     Qtot.append(QL[i])
#     Qtot_error.append(Error_QL[i])
#     Qc.append(QC[i])
#     Qc_error.append(Error_QC[i])
#     Qi.append(Qint[i])
#     Qi_error.append(Error_Qint[i])

P_app=[]
for i in range(len(powers)):
    for j in range(len(resonators)):
    # P_app = powers[0:4] - External_Attenuation - Fridge_Attenuation - (np.polyval(polyfits[0],res_freq[0]) + np.polyval(polyfits[1],res_freq[0]) + np.polyval(polyfits[2],res_freq[0])) - (np.polyval(RF_Att[0],res_freq[0]) + np.polyval(RF_Att[1],res_freq[0]))
        Powers =  powers[i] + np.polyval(polyfits[4],fr[i][j]) #+ (np.polyval(polyfits[0],fr[i][j]) + np.polyval(polyfits[1],fr[i][j]) + np.polyval(polyfits[2],fr[i][j])) - (np.polyval(RF_Att[0],fr[i][j]) + np.polyval(RF_Att[1],fr[i][j]))
        P_app.append(Powers)
P_app = np.array(P_app) - External_Attenuation - Fridge_Attenuation
# P_app = np.flip(P_app)
P_applied = 10**((P_app-30)/10)
P_applied = np.reshape(P_applied,(len(powers),len(resonators)))

n = (2/(hbar*(2*np.pi*np.array(fr))**2))*Zmm*(np.array(Qtot)**2/np.array(Qc))*P_applied

#n = Zmm*(2/np.pi)*P_applied*np.array(Qtot)**2/(hbar*np.array(Qc)*(2*np.pi*np.array(fr))**2)


for m in range(len(resonators)):
    fig, ax = plt.subplots(constrained_layout=True)
    # plt.errorbar(n, Qtot.flatten(),yerr = Qtot_error.flatten(), fmt='o', label = 'QL')
    # ax.errorbar(n, Qi.flatten(), yerr = Qi_error.flatten(), fmt='o', label = 'Qi')
    # ax.errorbar(P_app, Qi.flatten(), yerr = Qi_error.flatten(), fmt='o', label = 'Qi')
    # plt.errorbar(n, Qc.flatten(), yerr = Qc_error.flatten(), fmt='o', label = 'Qc')
    ax.set_xlabel('<n>')
    ax.set_ylabel('Quality Factors')
    ax.set_xscale('log')
    ax.set_yscale('log')
    #ax2 = ax.twinx()
    #ax2.errorbar(n, Qc.flatten(), yerr = Qc_error.flatten(), fmt='o', label = 'Qc', color='r')
    plt.title('ω = 2π *' + str(fr[0,m]))# + ' GHz')
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.show()
