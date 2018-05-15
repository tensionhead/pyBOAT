from __future__ import division,print_function
import os,sys
import pylab as ppl
import numpy as np
from os import path,walk
import pandas as pd

from wavelets_lib import *

ppl.ion()

# read in the excel file
#data = pd.
data = pd.read_excel('synth_signals2c.xlsx')

data = data.dropna()

#======================================
dt = 5                           # sampling interval in minutes
periods = np.linspace(2*dt,300,200) # the periods to scan for (Nyquist limit is 2dt!)
T_cut_off = 300               # the cut off period for the sinc filter
vmax = 15                     # max scale of the colorbar of the wavelet power spectrum
An = TFAnalyser(periods,dt,T_cut_off,vmax)   # construct the analyser
#======================================

raw_signal = data['signal4'].dropna()  # choose time series from data set, convert to array

# put a new signal into the Wavelet-Analyser
An.new_signal(raw_signal, name = 'Synth Data 1')

# detrend the raw signal with the sinc filter
An.sinc_detrend()                            

# plot the raw signal including the sinc-trend
An.plot_signal(with_trend = True,fig_num = 1)     
#savefig('signal_trend.pdf')

# plot the detrended signal
An.plot_detrended()
#savefig('detrended.pdf')

# compute the wavelet power spectrum
An.compute_spectrum(fig_num = 3)

# find max ridge where power above Threshold
An.draw_maxRidge(Thresh = 10, color = '0.3',smoothing = True)

# confidence regions for AR1 zero-hypothesis
An.draw_AR1_confidence(alpha = 0.4) # alpha = 0 corresponds to white noise
#savefig('spectrum.jpg') #

# #saves the identified periods + ridge power into results Data_Frame
An.save_ridge()                   
An.export_results('demo_output')  # exports results dataframe to excel sheet








#====================================================================
# for power-users: you can access all computed quantities directly
# from the analyser instance:
#====================================================================

signal = An.signal # the signal put into the analyser
det_signal = An.dsignal # the detrended signal

wavelet = An.wlet # the complex wavelet transform
modulus = An.modulus # ..and its real valued absolute value normalized to variance - the power 

ridge_data = An.ridge_data # dictionary holding the 'ridge', the 'time', and the 'power'

ax_spec = An.ax_spec # the axis object holding the spectrum, you can modify the plot e.g.:
ax_spec.set_title('Wavelet power spectrum',fontsize = 18)


