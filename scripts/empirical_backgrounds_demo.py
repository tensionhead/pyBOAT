'''
This script shows how to estimate an empirical background from
an ensemble of 'non-oscillatory' signals, and how to apply this
estimated background to the analysis.
'''
import numpy as np
import pandas as pd
from pyboat import WAnalyzer, ssg
from pyboat import plotting as pl
from pyboat.core import ar1_powerspec
import matplotlib.pyplot as ppl

# set up analyzing instance
periods = np.linspace(5, 80, 100)
dt = 1
wAn = WAnalyzer(periods, dt, p_max=20)

# create an ensemble of short AR(1) realizations
# in a real life scenario this would be the 'non-oscillatory'
# test signals
alpha = 0.7
signals = [ssg.ar1_sim(alpha, Nt=120) for i in range(100)]

# store the individual time averaged Wavelet spectra
df_fouriers = pd.DataFrame(index = wAn.periods)

# the individual Fourier estimates
for i,sig in enumerate(signals):

    wAn.compute_spectrum(sig, do_plot=False)
    df_fouriers[i] = wAn.get_averaged_spectrum()

# plot the Fourier spectra distribution
pl.Fourier_distribution(df_fouriers)

# compare to the theoretical spectrum,
# note the finite size effects suppressing the power for higher periods!
theo_bg = ar1_powerspec(alpha, wAn.periods, wAn.dt)
ppl.plot(wAn.periods, theo_bg, 'k--', label='theoretical')
ppl.legend()

# now we take the median of the Fourier power
# distribution as expected empirical background
emp_bg = df_fouriers.median(axis=1)

# note that we have have persistent and consistent periods
print(len(emp_bg.index) == len(wAn.periods))

# let's create a very noisy signal with the same
# background process as noise mixed in 
test_signal = ssg.create_noisy_chirp(T1=70,
                                     T2=40,
                                     Nt=500,
                                     alpha=alpha,
                                     eps=1.) # SNR is 1!

# plain wavelet analysis
wAn.compute_spectrum(test_signal)

# let's draw the confidence intervals from the theoretical spectrum
wAn.draw_confidence_from_bg(theo_bg, colors='black')

# now let's use the 'empirical' one, taking into
# account the finite size effect
wAn.draw_confidence_from_bg(emp_bg, colors='orange')

# finally let's get the ridge within the CI regions
# from the empirical background
ridge_data = wAn.get_sign_maxRidge(emp_bg)
wAn.draw_Ridge()
