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

periods = np.linspace(2,80,150)
dt = 2
T_c = 30
time_unit = 's'

tvec = np.arange(500) * dt
signal = 0.2*randn(500) + np.sin(2*pi/47 * tvec)
trend = wl.sinc_smooth(signal, T_c, dt)
dsignal = signal - trend

# plot the signal/trend

fig = ppl.figure()
ax = pl.mk_signal_ax(fig, 's')
pl.draw_signal(ax, tvec, signal)
# pl.draw_detrended(ax, tvec, signal)
# pl.draw_trend(ax, tvec, trend)
ax.legend(fontsize = 16) # draw the legend

# compute spectrum
modulus, wlet = wl.compute_spectrum(signal, dt, periods)

# get maximum ridge
ridge = wl.get_maxRidge(modulus)

# evaluate along the ridge
ridge_results = wl.eval_ridge(ridge, wlet, signal, periods, tvec)

# plot spectrum and ridge
fig = ppl.figure(figsize = (6.5,7) )
ax_sig, ax_spec = pl.mk_signal_modulus_ax(fig, time_unit)

pl.plot_signal_modulus((ax_sig, ax_spec), tvec, signal, modulus, periods)
pl.draw_Wavelet_ridge(ax_spec, ridge_results)
fig.tight_layout()

# plot readout
fig = ppl.figure(figsize = (8.5,7) )
pl.plot_readout(fig, ridge_results)
