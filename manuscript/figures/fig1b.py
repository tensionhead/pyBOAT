import sys, os
import matplotlib.pyplot as ppl
from numpy.random import randn
import numpy as np
from numpy import pi

sys.path.insert(0, '../../tfapy/')

from tfa_lib.wanalyzer import WAnalyzer

ppl.ion()

# def synth_gen( freqs, eps

periods = np.linspace(2,120,250)
dt = 2
T_c = 65
time_unit = 's'

#+++++
save = False
# save = True
#+++++

# synthetic signal
Nt = 250
tvec = np.arange(Nt) * dt
tmax = tvec[-1]
# Noise intensity
eps = 1.25

# two harmonic components
T1 = 40
T2 = 90
# linearly sweeping frequency (chirp): dphi/dt |_tmax = T1 !!
signal2 = eps * randn(Nt) + np.sin( (0.5 * (2*pi/T1 - 2*pi/T2)/(tmax) * tvec + 2*pi/T2) * tvec )

# set up analyzing instance
wAn = WAnalyzer(periods, dt, T_c, unit_label = time_unit, p_max = 45)

# plot signal and trend
# wAn.plot_signal(signal2)
# wAn.plot_trend(signal)
# wAn.plot_detrended(signal2)

# compute and plot the spectrum
wAn.compute_spectrum(signal2, detrend = False)
fig1 = ppl.gcf()

ms = wAn.get_mean_spectrum()
# Fourier
wAn.plot_FFT(signal2, show_periods = False)
fig2 = ppl.gcf()
ax = ppl.gca()
ax.set_xlim( (-0.005, 0.081) )


# wAn.plot_FFT(signal2, show_periods = False)

wAn.get_maxRidge(Thresh = 6)
wAn.draw_Ridge()

# rd = wAn.ridge_data

if save: 
    fig2.savefig('chirp_Fspec.pdf')
    fig1.savefig('chirp_Wspec.pdf')
