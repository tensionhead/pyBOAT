import matplotlib.pyplot as ppl
import pandas
import numpy as np
import pytest

from pyboat import WAnalyzer, ssg


dt = 2  # the sampling interval, 2s
Nt = 220  # number of samples
T1 = 30  # in s
T2 = 50  # in s
dphi=np.pi / 4

@pytest.fixture
def signal():
    """Create a synthetic signal"""

    eps = 0.5  # noise intensity
    alpha = 0.4  # AR1 parameter

    # oscillatory signal which sweeps from T1 to T2
    signal1 = ssg.create_noisy_chirp(T1 = T1 / dt, T2 = T2 / dt,
                                     Nt = Nt, dphi=dphi, eps = eps, alpha = alpha)
    # add slower oscillatory trend with a period of 70s
    signal2 = ssg.create_chirp(T1 = 70 / dt, T2 = 70 / dt, Nt = Nt)
    # add exponential decay
    syn_env = ssg.create_exp_envelope(tau = 0.65 * Nt, Nt = Nt)

    # linear superposition
    signal = syn_env * (signal1 + 2. * signal2)
    return signal


def test_ssg(signal: np.ndarray):

    assert isinstance(signal, np.ndarray)
    assert signal.shape == (Nt, )


def test_wavelet_analysis(signal: np.ndarray):

    periods = np.linspace(6, 90, 150)  # period range, 6s to 90s
    wAn = WAnalyzer(periods, dt, time_unit_label='s')

    # calculate the trend with a 60s cutoff
    trend = wAn.sinc_smooth(signal, T_c=60)

    # detrending here is then just a subtraction
    detrended_signal = signal - trend

    # normalize the amplitude with a sliding window of 70s
    norm_signal = wAn.normalize_amplitude(detrended_signal, window_size=70)

    assert len(trend) == len(detrended_signal) == len(norm_signal)

    # plot signal and trend, arguments can be any which ppl.plot(...) accepts
    wAn.plot_signal(signal, label='Raw signal', color='red', alpha=0.5)
    wAn.plot_trend(trend, label='Trend with $T_c$=60s')
    # make a new figure to show original signal and detrended + normalized
    wAn.plot_signal(signal, num=2, label='Raw signal', color='red', alpha=0.5)
    wAn.plot_signal(norm_signal, label='Detrended + normalized', alpha=0.8, marker='.')
    # TODO: check for saved figures

    modulus, transform = wAn.compute_spectrum(norm_signal)
    assert np.iscomplexobj(modulus) is False
    assert np.iscomplexobj(transform) is True

    wAn.ax_spec_signal.set_title('Sinc detrended + ampl. normalized')

    # get the complete ridge of the last analysis as pandas DataFrame
    rd = wAn.get_maxRidge(power_thresh = 0, smoothing_wsize=20)
    assert isinstance(rd, pandas.DataFrame)
    assert rd.shape == (Nt, 6)
    assert 'periods' in rd
    assert 'phase' in rd
    assert 'amplitude' in rd
    assert 'power' in rd
    assert 'frequencies' in rd

    # mean period is between T1 and T2 (the sweep)
    assert T1 < rd.mean()['periods'] < T2

    # get a thresholded ridge
    rd2 = wAn.get_maxRidge(power_thresh = 10, smoothing_wsize=20)
    # low power samples are excluded
    assert len(rd2) < len(rd)
    assert rd2.mean()['power'] > rd.mean()['power']

    wAn.draw_Ridge()
    # ppl.savefig('signal_spec.png')

    wAn.plot_readout(draw_coi=True)
    # ppl.savefig('signal_readout.png')

    # raw signal is dominated by low freq trend
    _ = wAn.compute_spectrum(signal)
    rd_raw = wAn.get_maxRidge()
    # ridge traces the trend in the spectrum
    assert rd_raw.mean()['periods'] > T2
