'''
A short demonstration of the provided
ensemble measures
'''
import numpy as np
import pandas as pd
from pyboat import ensemble_measures as em
from pyboat import WAnalyzer, ssg
from pyboat import plotting as pl 

# set up analyzing instance
periods = np.linspace(5, 60, 100)
dt = 1
wAn = WAnalyzer(periods, dt)

# create a bunch of chirp signals
# with diverging period over time
Nsignals = 50 # times 2
Tstart = 30 # initial period
Tmax = 50   # slowest signal
Nt = 500 # number of samples per signal
signals = [
    ssg.create_noisy_chirp(T1 = Tstart, T2 = Tend, Nt = Nt, eps = 1)
    for Tend in np.linspace(Tstart, Tmax, Nsignals)
]

# add the same number of pure noise signals
noisy_ones = [
    ssg.ar1_sim(alpha = 0.5, Nt = Nt)
    for i in range(Nsignals)
]

# signals ids are just column numbers here
signals = pd.DataFrame(signals + noisy_ones).T

# get the the individual ridge readouts
ridge_results = {}

# store the individual time averaged Wavelet spectra
df_fouriers = pd.DataFrame(index = wAn.periods)

for ID in signals:

    wAn.compute_spectrum(signals[ID], do_plot = False)
    rd = wAn.get_maxRidge(smoothing_wsize = 11)
    ridge_results[ID] = rd

    df_fouriers[ID] = wAn.get_averaged_spectrum()

# the time-averaged power distribution,
# index holds the signal ids if given, otherwise just a column index
powers_series = em.average_power_distribution(
    ridge_results.values(),
    signal_ids = ridge_results.keys())

# is bi-modal!
pl.power_distribution(powers_series)

# filter out the pure noise signals with a power threshold
high_power_ids = powers_series[powers_series > 10].index
high_power_ridge_results = [ridge_results[ID] for ID in high_power_ids]

# creates a tuple of 4 DataFrames, one summary statistic over time              
# for period, amplitude, power and phase each
res = em.get_ensemble_dynamics(high_power_ridge_results)

pl.ensemble_dynamics(*res)

# the Fourier power distribution with pure noise signals
ax = pl.Fourier_distribution(df_fouriers)
ax.set_title('Whole ensemble', pad = -40)

# the Fourier power distribution without pure noise signals
ax = pl.Fourier_distribution(df_fouriers[high_power_ids])
ax.set_title('Noise filtered out', pad = -40)
