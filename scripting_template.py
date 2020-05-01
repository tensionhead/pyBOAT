import matplotlib.pyplot as ppl
from numpy.random import randn
import numpy as np
from numpy import pi

import pyboat
from pyboat import ssg
import pyboat.plotting as pl


ppl.ion()

periods = np.linspace(4,90,150)
dt = 2
T_cut_off = 65 # cut off period
time_unit = 's'

# --- create a synthetic signal ---

eps = 0.5 # noise intensity
alpha = 0.4 # AR1 parameter
Nt = 500 # number of samples

signal1 = ssg.create_noisy_chirp(T1 = 30 / dt, T2 = 50 / dt, Nt = Nt, eps = eps, alpha = alpha)

# add slower oscillatory trend
signal2 = ssg.create_chirp(T1 = 70 / dt, T2 = 70 / dt, Nt = Nt)

# linear superposition
signal = signal1 + 1.5 * signal2

# --- calculate trend ---

trend = pyboat.sinc_smooth(signal, T_cut_off, dt)
detr_signal = signal - trend

# plot the signal/trend
tvec = np.arange(len(signal)) * dt
ax = pl.mk_signal_ax(time_unit = 's')
pl.draw_signal(ax, tvec, signal)
# pl.draw_detrended(ax, tvec, signal)
pl.draw_trend(ax, tvec, trend)
ppl.legend(ncol = 2)
ppl.tight_layout()

# --- compute spectrum on the original signal ---
modulus, wlet = pyboat.compute_spectrum(signal, dt, periods)

# plot spectrum and ridge
ax_sig, ax_spec = pl.mk_signal_modulus_ax(time_unit)
pl.plot_signal_modulus((ax_sig, ax_spec), tvec, signal, modulus, periods)


# --- compute spectrum on the detrended signal ---
modulus, wlet = pyboat.compute_spectrum(detr_signal, dt, periods)

# get maximum ridge
ridge = pyboat.get_maxRidge(modulus)

# evaluate along the ridge
ridge_results = pyboat.eval_ridge(ridge, wlet, signal, periods, tvec)


# plot spectrum and ridge
ax_sig2, ax_spec2 = pl.mk_signal_modulus_ax(time_unit)

pl.plot_signal_modulus((ax_sig2, ax_spec2), tvec, signal, modulus, periods)
pl.draw_Wavelet_ridge(ax_spec2, ridge_results)
ppl.tight_layout()

# plot readout
pl.plot_readout(ridge_results)
