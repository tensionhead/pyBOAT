'''OOP API for Wavelet Analysis of pyBOAT'''

import matplotlib.pyplot as ppl
import numpy as np

import pyboat.core as core
import pyboat.plotting as pl
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# for publications monkey patch font sizes
# pl.label_size = 24
# pl.tick_label_size = 20
# -----------------------------------------------------------


class WAnalyzer:

    '''
    Convenience class to access all of pyBOATs capabilities
    through an object oriented API
    '''
    
    def __init__(
            self,
            periods,
            dt,
            p_max=None,
            time_unit_label="a.u.",            
            M=None
    ):

        """
        Sets up an analyzer instance with the following parameters:

        periods   : sequence of periods to compute the Wavelet spectrum for, 
                    must have same units as dt!

        dt        : the sampling interval scaled to desired time units


        p_max      : Maximum power for spectrum z-axis colormap display, 
                    if *None* scales automatically

        time_unit_label: the string label for the time unit 

        M         : Length of the sinc filter window, defaults to length
                    of input signal. Set to a lower value to 
                    speed up sinc-detrending.

        """
        

        # sanitize periods
        if periods[0] < 2 * dt:
            logger.warning(f"Nyquist limit is {2*dt:.2f} {time_unit_label}!!")
            logger.info(f"Setting lower period limit to {2*dt:.2f}")
            periods[0] = 2 * dt

        self.periods = np.linspace(periods[0], periods[-1], len(periods))
        self.dt = dt
        self.p_max = p_max
        self.M = M

        self.time_unit_label = time_unit_label

        self.ana_signal = None

        self._has_spec = False
        self._has_ridge = False

        self.ax_signal = None
        self.ax_spec = None
        self.transform = None

    def compute_spectrum(self,
                         raw_signal,
                         T_c=None,
                         window_size=None,
                         do_plot=True,
                         draw_coi=False):

        """
        Computes the Wavelet spectrum for a given *raw_signal*.

        After a successful analysis, the analyser instance updates 

        self.transform 
        self.modulus

        with the results.

        Parameters
        ----------
        
        signal  : a sequence, the time-series to be analyzed

        T_c : float, optional
              Cut off period for the sinc-filter detrending, all periods
              larger than that one are removed from the signal. If not given,
              no sinc-detending will be done.

        window_size : float, optional
                      Length of the sliding window for amplitude
                      envelope estimation in real time units, e.g. 17 minutes.
                      If not given no amplitude normalization will be done.
        
        do_plot      : boolean, set to False if no plot is desired, 
                       good for batch processing

        draw_coi: boolean, set to True if cone of influence 
                           shall be drawn on the wavelet power spectrum
                           
        
        """

        if T_c:
            detrended = self.sinc_detrend(raw_signal, T_c)
            ana_signal = detrended
        else:
            ana_signal = raw_signal

        # only after potential detrending!
        if window_size:
            ana_signal = self.normalize_amplitude(ana_signal, window_size)

        self.ana_signal = ana_signal

        modulus, transform = core.compute_spectrum(ana_signal, self.dt, self.periods)

        if do_plot:

            tvec = np.arange(len(ana_signal)) * self.dt

            axs = pl.mk_signal_modulus_ax(self.time_unit_label)
            pl.plot_signal_modulus(
                axs,
                time_vector=tvec,
                signal=ana_signal,
                modulus=modulus,
                periods=self.periods,
                p_max=self.p_max,
            )

            fig = ppl.gcf()
            fig.tight_layout()
            self.ax_spec = axs[1]

            if draw_coi:
                pl.draw_COI(axs[1], time_vector=tvec)

        self.transform = transform
        self.modulus = modulus
        self._has_spec = True

    def get_maxRidge(self, power_thresh=0):

        """
        Computes and evaluates the ridge as consecutive 
        maxima of the modulus.

        Returns the ridge_data dictionary, see also `core.eval_ridge`!

        Additionally the analyser instance updates 

        self.ridge_data 

        with the results.
        

        Parameters
        ----------
        
        power_thresh : float, threshold for the ridge. 

        Returns
        -------
        
        A DataFrame with the following columns:

        time      : the t-values of the ridge, can have gaps if thresholding!
        periods   : the instantaneous periods 
        frequencies : the instantaneous frequencies 
        phase    : the arg(z) values
        power     : the Wavelet Power normalized to white noise (<P(WN)> = 1)
        amplitude : the estimated amplitudes of the signal

        """

        if not self._has_spec:
            logger.warning("Need to compute a wavelet spectrum first!")
            return

        # for easy integration
        modulus = self.modulus

        Nt = modulus.shape[1]  # number of time points
        tvec = np.arange(Nt) * self.dt

        # ================ridge detection=====================================
                
        ridge_y = core.get_maxRidge_ys(modulus)

        rd = core.eval_ridge(
            ridge_y,
            self.transform,
            self.ana_signal,
            self.periods,
            tvec=tvec,
            power_thresh=power_thresh
        )

        self.ridge_data = rd
        self._has_ridge = True

        # return also directly
        return rd

    def plot_readout(self, draw_coi = False, num=None):

        """
        Wraps the readout from pyboat.plotting.
        """

        if not self._has_ridge:
            logger.warning("Need to extract a ridge first!")
            return

        pl.plot_readout(self.ridge_data, time_unit=self.time_unit_label, draw_coi = draw_coi)
        
    def draw_Ridge(self):

        if not self._has_ridge:
            logger.warning("Can't draw ridge, need to do a ridge detection first!")
            return

        if not self.ax_spec:
            logger.warning("Can't draw ridge, plot the spectrum first!")
            return

        pl.draw_Wavelet_ridge(self.ax_spec, self.ridge_data)

    def plot_signal(self, signal, legend=False, num=None):

        '''
        Creates the signal-figure and plots the signal.
        '''

        if self.ax_signal is None:
            fig = ppl.figure(num, figsize=(6, 3.5))
            self.ax_signal = pl.mk_signal_ax(self.time_unit_label, fig=fig)

        tvec = np.arange(len(signal)) * self.dt
        pl.draw_signal(self.ax_signal, tvec, signal)

        if legend:
            self.ax_signal.legend(fontsize=pl.label_size, ncol=3)
            ymin, ymax = self.ax_signal.get_ylim()
            self.ax_signal.set_ylim((ymin, 1.3 * ymax))

        fig = ppl.gcf()
        fig.subplots_adjust(bottom=0.18)
        fig.tight_layout()

    def plot_trend(self, trend, legend=False):

        if self.ax_signal is None:
            return

        tvec = np.arange(len(trend)) * self.dt
        pl.draw_trend(self.ax_signal, tvec, trend)

        if legend:
            self.ax_signal.legend(fontsize=pl.label_size, ncol=3)
            ymin, ymax = self.ax_signal.get_ylim()
            self.ax_signal.set_ylim((ymin, 1.3 * ymax))

        fig = ppl.gcf()
        fig.subplots_adjust(bottom=0.18)
        fig.tight_layout()

    def plot_detrended(self, signal, num=None):

        if not num or self.ax_signal is None:
            fig = ppl.figure(num, figsize=(6, 3.5))
            self.ax_signal = pl.mk_signal_ax(self.time_unit_label, fig = fig)

        tvec = np.arange(len(signal)) * self.dt
        pl.draw_detrended(self.ax_signal, tvec, signal)

        fig = ppl.gcf()
        fig.subplots_adjust(bottom=0.18)
        fig.tight_layout()

    def plot_envelope(self, envelope, legend=False, num=None):

        '''
        Plot the sliding window amplitude envelope onto the signal.
        '''
        
        if self.ax_signal is None:
            fig = ppl.figure(num, figsize=(6, 3.5))
            self.ax_signal = pl.mk_signal_ax(self.time_unit_label, fig = fig)

        tvec = np.arange(len(signal)) * self.dt
        envelope = self.get_envelope(signal, self.L)
        pl.draw_envelope(self.ax_signal, tvec, envelope)

        if legend:
            self.ax_signal.legend(fontsize=pl.label_size, ncol=3)
            ymin, ymax = self.ax_signal.get_ylim()
            self.ax_signal.set_ylim((ymin, 1.3 * ymax))

        fig = ppl.gcf()
        fig.subplots_adjust(bottom=0.18)
        fig.tight_layout()
        
    def get_averaged_spectrum(self):

        """ 
        Average Wavelet spectrum over time
        to give a Fourier estimate. A Wavelet spectrum
        has to be computed first.        

        Returns
        -------
        
        mfourier : Fourier spectrum estimate
        """

        if not self._has_spec:
            logger.warning("Need to compute a wavelet spectrum first!")
            return

        mfourier = np.sum(self.modulus, axis=1) / self.modulus.shape[1]

        return mfourier

    def draw_AR1_confidence(self, alpha):

        if not self._has_spec:
            logger.warning("Need to compute a wavelet spectrum first!")
            return

        tvec = np.arange(self.transform.shape[1]) * self.dt
        x, y = np.meshgrid(tvec, self.periods)  # for plotting the wavelet transform

        ar1power = core.ar1_powerspec(alpha, self.periods, self.dt)

        scaled_mod = np.zeros(self.modulus.shape)

        # maybe there is a more clever way
        for i, col in enumerate(self.modulus.T):
            scaled_mod[:, i] = col / ar1power

        CS = self.ax_spec.contour(
            x,
            y,
            scaled_mod,
            levels=[core.xi2_95 / 2.0],
            linewidths=1.7,
            colors="0.95",
            alpha=0.8,
        )
        CS = self.ax_spec.contour(
            x,
            y,
            scaled_mod,
            levels=[core.xi2_99 / 2.0],
            linewidths=1.7,
            colors="orange",
            alpha=0.8,
        )

        # check confidence levels on (long) ar1 realisations !
        # print (len(where(scaled_mod > conf95)[0])/prod(transform.shape))
        # should be ~0.05
        # print (len(where(scaled_mod > conf99)[0])/prod(transform.shape))
        # should be ~0.01

    def sinc_smooth(self, signal, T_c):
        '''

        Convolve the signal with a sinc filter
        of cut-off period *T_c*. Returns
        the smoothed signal representing
        the non-linear trend.

        Parameters
        ----------
        
        signal : a sequence
        
        T_c : float, Cut off period for the sinc-filter detrending, all periods
              larger than that one are removed from the signal

        Returns
        -------

        trend : numpy 1d-array
        '''

        trend = core.sinc_smooth(signal, T_c, self.dt, M=self.M)

        return trend

    def sinc_detrend(self, signal, T_c):

        '''
        Convenience function which right away subtracts the
        trend obtained by sinc filtering. See 'sinc_smooth'
        for details.
        '''

        trend = core.sinc_smooth(signal, T_c, self.dt, self.M)

        detrended = signal - trend

        return detrended

    def normalize_amplitude(self, signal, window_size):

        '''
        Estimates the amplitude envelope with a sliding window
        and normalizes the signal with 1/envelope. 

        Best to do a detrending first!        
        The signal mean gets subtracted in any case.

        Note that the *window_size* should be at least
        of the length of the lowest period to be expected,
        otherwise oscillatory components get damped.

        Parameters
        ----------
        
        signal : a sequence        
        window_size : Length of the sliding window for amplitude
                      envelope estimation in real time units, e.g. 17 minutes

        Returns
        -------
        
        ANsignal : ndarray, the amplitude normalized signal

        See Also
        --------

        get_envelope : returns the envelope itself
        
        '''
        
        ANsignal = core.normalize_with_envelope(
            signal,
            window_size,
            dt = self.dt)

        return ANsignal
    
    def get_envelope(self, signal, window_size, SGsmooth = True):

        '''
        Max - Min sliding window operation
        to estimate amplitude envelope.

        Parameters
        ----------
        
        signal : a sequence        
        window_size : Length of the sliding window for amplitude
                      envelope estimation in real time units, e.g. 17 minutes
        SGsmooth : bool, optional Savitzky-Golay smoothing of the
                         envelope with the same window size


        Returns
        -------
        
        ANsignal : ndarray, the amplitude normalized signal

        optional:
        '''
        
        envelope = core.sliding_window_amplitude(
            signal,
            window_size,
            dt = self.dt,
            SGsmooth = SGsmooth
        )

        return envelope
        
    def plot_FFT(self, signal, show_periods=True):
        fig = ppl.figure(figsize=(5, 2.5))

        ax = pl.mk_Fourier_ax(
            fig, time_unit=self.time_unit_label, show_periods=show_periods
        )

        fft_freqs, fft_power = core.compute_fourier(signal, self.dt)
        logger.info(f"mean fourier power: {np.mean(fft_power):.2f}")
        pl.Fourier_spec(ax, fft_freqs, fft_power, show_periods)
        fig.tight_layout()
