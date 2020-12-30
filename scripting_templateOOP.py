import matplotlib.pyplot as ppl
import numpy as np

from pyboat import WAnalyzer, ssg

ppl.ion()

# --- set basic parameters ---

dt = 3 # the sampling interval
periods = np.linspace(6,90,150)
time_unit = 's'

# --- create a synthetic signal ---

eps = 0.5 # noise intensity
alpha = 0.4 # AR1 parameter
Nt = 200 # number of samples

signal1 = ssg.create_noisy_chirp(T1 = 30 / dt, T2 = 50 / dt, Nt = Nt, eps = eps, alpha = alpha)

# add slower oscillatory trend
signal2 = ssg.create_chirp(T1 = 70 / dt, T2 = 70 / dt, Nt = Nt)

# linear superposition
signal = signal1 + 1.5 * signal2

# --- set up analyzing instance ---
wAn = WAnalyzer(periods, dt, time_unit_label=time_unit)

# calculate the trend with a 65min cutoff
trend = wAn.sinc_smooth(signal, T_c=65) 

# detrending is just a subtraction
detrended_signal = signal - trend

# plot signal and trend
wAn.plot_signal(signal, num=1)
wAn.plot_trend(trend)
# ppl.legend(ncol = 2)

# plot the detrended_signal in an extra figure
wAn.plot_signal(detrended_signal, num=2)

# compute and plot the spectrum without detrending
wAn.compute_spectrum(signal)

# compute and plot the spectrum of the detrended signal
wAn.compute_spectrum(detrended_signal)

wAn.get_maxRidge(power_thresh = 5)
wAn.draw_Ridge()
#ppl.savefig('detr_signal_spec.png')

#wAn.plot_readout(draw_coi = True)
#ppl.savefig('detr_signal_readout.png')

rd = wAn.ridge_data # this is a pandas DataFrame holding the ridge results

