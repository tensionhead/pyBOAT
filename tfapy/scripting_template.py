import matplotlib.pyplot as ppl
from numpy.random import randn
import numpy as np
from numpy import pi

import tfa_lib.core as wl # core wavelet library
import tfa_lib.plotting as pl

# monkey patch tick and label sizes
# defaults are good for the UI
pl.label_size = 16.5
pl.tick_label_size = 14


ppl.ion()

# the periods to scan for
periods = np.linspace(2,80,150)
# sampling interval
dt = 2
# cut off period for sinc filter
T_c = 30
time_unit = 's'

# create a synthetic signal

T = 47
tvec = np.arange(500) * dt
signal = 0.2*randn(500) + np.sin(2*pi/T * tvec) * np.exp(-tvec * 0.1/T)

# sinc detrending
trend = wl.sinc_smooth(signal, T_c, dt)
dsignal = signal - trend

# plot the signal/trend

ax = pl.mk_signal_ax(time_unit = 's')
pl.draw_signal(ax, tvec, signal)
# pl.draw_detrended(ax, tvec, signal)
# pl.draw_trend(ax, tvec, trend)
# ax.legend(fontsize = 16) # draw the legend

# compute spectrum
modulus, wlet = wl.compute_spectrum(signal, dt, periods)

# get maximum ridge
ridge = wl.get_maxRidge(modulus)

# evaluate along the ridge
ridge_results = wl.eval_ridge(ridge, wlet, signal, periods, tvec)

# plot spectrum and ridge
ax_sig, ax_spec = pl.mk_signal_modulus_ax(time_unit)

pl.plot_signal_modulus((ax_sig, ax_spec), tvec, signal, modulus, periods)
pl.draw_Wavelet_ridge(ax_spec, ridge_results)
# cone of influence
pl.draw_COI(ax_spec, tvec, wl.Morlet_COI())

# plot readout
pl.plot_readout(ridge_results)
