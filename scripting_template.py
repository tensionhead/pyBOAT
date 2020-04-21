import matplotlib.pyplot as ppl
from numpy.random import randn
import numpy as np
from numpy import pi

import pyboat
import pyboat.plotting as pl


ppl.ion()

periods = np.linspace(4,80,150)
dt = 2
T_c = 30
time_unit = 's'

tvec = np.arange(500) * dt
signal = 0.2*randn(500) + np.sin(2*pi/47 * tvec)
trend = pyboat.sinc_smooth(signal, T_c, dt)
dsignal = signal - trend

# plot the signal/trend

#ax = pl.mk_signal_ax(time_unit = 's')
#pl.draw_signal(ax, tvec, signal)
# pl.draw_detrended(ax, tvec, signal)
# pl.draw_trend(ax, tvec, trend)

# compute spectrum
modulus, wlet = pyboat.compute_spectrum(signal, dt, periods)

# get maximum ridge
ridge = pyboat.get_maxRidge(modulus)

# evaluate along the ridge
ridge_results = pyboat.eval_ridge(ridge, wlet, signal, periods, tvec)


# plot spectrum and ridge
ax_sig, ax_spec = pl.mk_signal_modulus_ax(time_unit)

pl.plot_signal_modulus((ax_sig, ax_spec), tvec, signal, modulus, periods)
pl.draw_Wavelet_ridge(ax_spec, ridge_results)
ppl.tight_layout()

# plot readout
pl.plot_readout(ridge_results)
