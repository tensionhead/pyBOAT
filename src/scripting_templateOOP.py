import matplotlib.pyplot as ppl
from numpy.random import randn
import numpy as np
from numpy import pi

from tfa_lib.wanalyzer import WAnalyzer

ppl.ion()

periods = np.linspace(2,80,150)
dt = 2
T_c = 65
time_unit = 's'

# synthetic signal
Nt = 250
tvec = np.arange(250) * dt
# Noise intensity
eps = 0

# two harmonic components
T1 = 50
T2 = 120
signal1 = eps * randn(250) + np.sin(2*pi/T1 * tvec) +  np.sin(2*pi/T2 * tvec)


# set up analyzing instance
wAn = WAnalyzer(periods, dt, T_c, time_unit_label = time_unit)

# plot signal and trend
wAn.plot_signal(signal1)
# wAn.plot_trend(signal)
# wAn.plot_detrended(signal1)

# compute the spectrum
wAn.compute_spectrum(signal1)

wAn.get_maxRidge(power_thresh = 5)
wAn.draw_Ridge()
wAn.plot_readout(draw_coi = True)
rd = wAn.ridge_data

