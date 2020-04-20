import matplotlib.pyplot as ppl
from numpy.random import randn
import numpy as np
from numpy import pi

from pyboat import core 
import pyboat.plotting as pl

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
trend = core.sinc_smooth(signal, T_c, dt)
dsignal = signal - trend

# plot the signal/trend

#ax = pl.mk_signal_ax(time_unit = 's')
#pl.draw_signal(ax, tvec, signal)
# pl.draw_detrended(ax, tvec, signal)
# pl.draw_trend(ax, tvec, trend)

# compute spectrum
modulus, wlet = core.compute_spectrum(signal, dt, periods)

# get maximum ridge
ridge = core.get_maxRidge(modulus)

# evaluate along the ridge
ridge_results = core.eval_ridge(ridge, wlet, signal, periods, tvec)

def find_COI_crossing(rd):

    '''
    checks for last/first time point
    which is outside the COI on the
    left/right boundary of the spectrum.

    Parameters
    ----------

    rd : pandas.DataFrame
        the ridge data from eval_ridge()
    '''

    coi_left = core.Morlet_COI() * rd.time
    # last time point outside left COI
    t_left = rd.index[~(coi_left > rd.periods)][-1]

    # use array to avoid inversed indexing
    coi_right = core.Morlet_COI() * rd.time.array[::-1]
    # first time point outside left COI
    t_right = rd.index[(coi_right < rd.periods)][0]
    
    return t_left, t_right


# ------------------------------

# plot spectrum and ridge
ax_sig, ax_spec = pl.mk_signal_modulus_ax(time_unit)

pl.plot_signal_modulus((ax_sig, ax_spec), tvec, signal, modulus, periods)
pl.draw_Wavelet_ridge(ax_spec, ridge_results)
ppl.tight_layout()

# plot readout
pl.plot_readout(ridge_results)
