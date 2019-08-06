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
# save = False
save = True
#+++++

# synthetic signal
Nt = 250
tvec = np.arange(Nt) * dt
# Noise intensity
eps = 0

# two harmonic components
T1 = 40
T2 = 90
signal1 = eps * randn(Nt) + np.sin(2*pi/T1 * tvec + pi)  +  np.sin(2*pi/T2 * tvec)


# set up analyzing instance
wAn = WAnalyzer(periods, dt, T_c, unit_label = time_unit, p_max = 40)

# plot signal and trend
# wAn.plot_signal(signal1)
# wAn.plot_trend(signal)
# wAn.plot_detrended(signal1)

# compute the spectrum
wAn.compute_spectrum(signal1, detrend = False)
fig1 = ppl.gcf()
ms = wAn.get_mean_spectrum()
# Fourier
wAn.plot_FFT(signal1, show_periods = False)
fig2 = ppl.gcf()
ax = ppl.gca()
ax.set_xlim( (-0.005, 0.081) )

# wAn.plot_FFT(signal2, show_periods = False)

# wAn.get_maxRidge()
#wAn.draw_Ridge()

# rd = wAn.ridge_data

if save:
    fig2.savefig('harmon_comp_Fspec.pdf')
    fig1.savefig('harmon_comp_Wspec.pdf')
