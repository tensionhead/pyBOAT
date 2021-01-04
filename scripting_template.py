import matplotlib.pyplot as ppl
import numpy as np

from pyboat import WAnalyzer, ssg

ppl.ion()

# --- set basic parameters and initialize the Analyzer---

dt = 2 # the sampling interval, 2s
periods = np.linspace(6, 90, 150) # period range, 6s to 90s
wAn = WAnalyzer(periods, dt, time_unit_label='s')

# --- create a synthetic signal ---

eps = 0.5 # noise intensity
alpha = 0.4 # AR1 parameter, set to 0 for white noise
Nt = 220 # number of samples

# oscillatory signal which sweeps from 30s to 50s
signal1 = ssg.create_noisy_chirp(T1 = 30 / dt, T2 = 50 / dt, Nt = Nt, eps = eps, alpha = alpha)

# add slower oscillatory trend with a period of 70s
signal2 = ssg.create_chirp(T1 = 70 / dt, T2 = 70 / dt, Nt = Nt)

# add exponential decay
syn_env = ssg.create_exp_envelope(tau = 0.65 * Nt, Nt = Nt)

# linear superposition
signal = syn_env * (signal1 + 2. * signal2)

# --- Filtering ---

# calculate the trend with a 60s cutoff
trend = wAn.sinc_smooth(signal, T_c=60) 

# detrending here is then just a subtraction
detrended_signal = signal - trend

# normalize the amplitude with a sliding window of 70s
norm_signal = wAn.normalize_amplitude(detrended_signal, window_size=70)

# plot signal and trend, arguments can be any which ppl.plot(...) accepts
wAn.plot_signal(signal, label='Raw signal', color='red', alpha=0.5)
wAn.plot_trend(trend, label='Trend with $T_c$=60s')

# make a new figure to show original signal and detrended + normalized
wAn.plot_signal(signal, num=2, label='Raw signal', color='red', alpha=0.5)
wAn.plot_signal(norm_signal, label='Detrended + normalized', alpha=0.8, marker='.')

# --- Wavelet Transforms ---

# compute and plot the spectrum without detrending
wAn.compute_spectrum(signal)
wAn.ax_spec_signal.set_title('Raw input')
# compute and plot the spectrum of the detrended signal
# the low frequencies are gone!
wAn.compute_spectrum(detrended_signal)
wAn.ax_spec_signal.set_title('Sinc detrended')

# compute and plot the spectrum of the detrended
# and normalized signal, this function also returns
# the Wavelet transformation results directly
modulus, transform = wAn.compute_spectrum(norm_signal)
wAn.ax_spec_signal.set_title('Sinc detrended + ampl. normalized')

# zoom into the spec 
# wAn.ax_spec.set_ylim((20,60))

# change x-ticks
wAn.ax_spec.set_xticks([0, 50, 100, 150, 200, 250, 300, 350])
# also vertical grid
wAn.ax_spec.grid(axis='both', color='white', alpha=0.4)

# --- Ridge Evaluation and Readout ---

# get the ridge of the last analysis
wAn.get_maxRidge(power_thresh = 10, smoothing_wsize=20)
wAn.draw_Ridge()
# ppl.savefig('detr_signal_spec.png')

wAn.plot_readout(draw_coi=True)
# ppl.savefig('detr_signal_readout.png')

rd = wAn.ridge_data # this is a pandas DataFrame holding the ridge results
print(rd)
