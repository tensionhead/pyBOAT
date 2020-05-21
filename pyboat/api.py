# ========================================
#  OOP API for Wavelet Analysis of pyBOAT
# =======================================

import matplotlib.pyplot as ppl
import numpy as np
from numpy import pi, e, cos, sin, sqrt
import pandas as pd

import pyboat.core as core
import pyboat.plotting as pl

# globals
# -----------------------------------------------------------
# default dictionary for ridge detection by annealing
ridge_def_dic = {
    "Temp_ini": 0.2,
    "Nsteps": 25000,
    "max_jump": 3,
    "curve_pen": 0.2,
    "sub_s": 2,
    "sub_t": 2,
}

# Significance levels from Torrence Compo 1998
xi2_95 = 5.99
xi2_99 = 9.21

# for publications
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
            T_cut_off,
            L = None,
            p_max=None,
            time_unit_label="a.u.",            
            M=None
    ):

        """
        Sets up an analyzer instance with the following parameters:

        periods   : sequence of periods to compute the Wavelet spectrum for, 
                    must have same units as dt!

        dt        : the sampling interval scaled to desired time units

        T_cut_off : Cut off period for the sinc-filter detrending, all periods
                    larger than that one are removed from the signal

        p_max      : Maximum power for z-axis colormap display, 
                    if *None* scales automatically

        L         : Length of the sliding window for amplitude
                    envelope estimation. Set to None per default.

        time_unit_label: the string label for the time unit 

        M         : Length of the sinc filter window, defaults to length
                     of input signal. Set to a lower value to speed up sinc-detrending.

        """
        

        # sanitize periods
        if periods[0] < 2 * dt:
            print(f"Warning, Nyquist limit is {2*dt:.2f} {time_unit_label}!!")
            print(f"Setting lower period limit to {2*dt:.2f}")
            periods[0] = 2 * dt

        self.periods = np.linspace(periods[0], periods[-1], len(periods))
        self.dt = dt
        self.T_c = T_cut_off
        self.L = L
        self.p_max = p_max
        self.M = M

        self.time_unit_label = time_unit_label

        self.ana_signal = None

        self._has_spec = False
        self._has_ridge = False

        self.ax_signal = None
        self.ax_spec = None
        self.wlet = None

    def compute_spectrum(self,
                         raw_signal,
                         sinc_detrend=True,
                         norm_amplitude = False,
                         do_plot=True,
                         draw_coi=False):

        """
        Computes the Wavelet spectrum for a given *signal* for the given *periods*
        
        signal  : a sequence, the time-series to be analyzed

        sinc_detrend : boolean, if True sinc-filter detrending 
                       will be done with the set T_cut_off parameter

        norm_amplitude : boolean, if True sliding window amplitude envelope
                          estimation is performed, and normalisation with
                          1/envelope
        
        do_plot      : boolean, set to False if no plot is desired, 
                       good for for batch processing

        draw_coi: boolean, set to True if cone of influence 
                           shall be drawn on the wavelet power spectrum
                   
        
        After a successful analysis, the analyser instance updates 

        self.wlet 
        self.modulus

        with the results.
        
        """

        if sinc_detrend:
            detrended = self.sinc_detrend(raw_signal)
            ana_signal = detrended
        else:
            ana_signal = raw_signal

        # only after potential detrending!
        if norm_amplitude:
            ana_signal = self.normalize_amplitude(ana_signal)

        self.ana_signal = ana_signal

        modulus, wlet = core.compute_spectrum(ana_signal, self.dt, self.periods)

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
                coi_m = core.Morlet_COI()
                pl.draw_COI(axs[1], time_vector=tvec, coi_slope=coi_m)

        self.wlet = wlet
        self.modulus = modulus
        self._has_spec = True

    def get_maxRidge(self, power_thresh=0, smoothing_wsize=None):

        """
        Computes the ridge as consecutive maxima of the modulus.

        Returns the ridge_data dictionary (see core.eval_ridge)!

        """

        if not self._has_spec:
            print("Need to compute a wavelet spectrum first!")
            return

        # for easy integration
        modulus = self.modulus

        Nt = modulus.shape[1]  # number of time points
        tvec = np.arange(Nt) * self.dt

        # ================ridge detection============================================

        # just pick the consecutive modulus
        # (squared complex wavelet transform) maxima as the ridge

        ridge_y = core.get_maxRidge_ys(modulus)

        rd = core.eval_ridge(
            ridge_y,
            self.wlet,
            self.ana_signal,
            self.periods,
            tvec=tvec,
            power_thresh=power_thresh,
            smoothing_wsize=smoothing_wsize,
        )

        self.ridge_data = rd
        self._has_ridge = True

        # return also directly
        return rd

    def plot_readout(self, draw_coi = False, num=None):

        """
        wraps the readout plotting
        """

        if not self._has_ridge:
            print("Need to extract a ridge first!")
            return

        pl.plot_readout(self.ridge_data, time_unit=self.time_unit_label, draw_coi = draw_coi)

    def get_annealRidge(self):

        """ not implemented yet """

        if not self._has_spec:
            print("Need to compute a wavelet spectrum first!")
            return

        ridge_y, cost = core.find_ridge_anneal(
            self.modulus, y0, ini_T, Nsteps, mx_jump=max_jump, curve_pen=curve_pen
        )

    def draw_Ridge(self):

        if not self._has_ridge:
            print("Can't draw ridge, need to do a ridge detection first!")
            return

        if not self.ax_spec:
            print("Can't draw ridge, plot the spectrum first!")
            return

        pl.draw_Wavelet_ridge(self.ax_spec, self.ridge_data)

    def plot_signal(self, signal, legend=False, num=None):

        if self.ax_signal is None:
            self.ax_signal = pl.mk_signal_ax(self.time_unit_label)

        tvec = np.arange(len(signal)) * self.dt
        pl.draw_signal(self.ax_signal, tvec, signal)

        if legend:
            self.ax_signal.legend(fontsize=pl.label_size, ncol=3)
            ymin, ymax = self.ax_signal.get_ylim()
            self.ax_signal.set_ylim((ymin, 1.3 * ymax))

        fig = ppl.gcf()
        fig.subplots_adjust(bottom=0.18)
        fig.tight_layout()

    def plot_trend(self, signal, legend=False, num=None):

        if self.ax_signal is None:
            fig = ppl.figure(num, figsize=(6, 3.5))
            self.ax_signal = pl.mk_signal_ax(self.time_unit_label, fig = fig)

        tvec = np.arange(len(signal)) * self.dt
        trend = self.get_trend(signal)
        pl.draw_trend(self.ax_signal, tvec, trend)

        if legend:
            self.ax_signal.legend(fontsize=pl.label_size, ncol=3)
            ymin, ymax = self.ax_signal.get_ylim()
            self.ax_signal.set_ylim((ymin, 1.3 * ymax))

        fig = ppl.gcf()
        fig.subplots_adjust(bottom=0.18)
        fig.tight_layout()

    def plot_detrended(self, signal, legend=False, num=None):

        if self.ax_signal is None:
            fig = ppl.figure(num, figsize=(6, 3.5))
            self.ax_signal = pl.mk_signal_ax(self.time_unit_label, fig = fig)

        tvec = np.arange(len(signal)) * self.dt
        trend = self.get_trend(signal)
        pl.draw_detrended(self.ax_signal, tvec, signal - trend)

        if legend:
            # detrended lives on 2nd axis :/
            pass

        fig = ppl.gcf()
        fig.subplots_adjust(bottom=0.18)
        fig.tight_layout()

    def plot_envelope(self, signal, legend=False, num=None):

        '''
        Sliding window amplitude envelope
        '''

        if self.L is None:
            print('Set a window size for the sliding window first!')
            return 
        
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

        """ Average over time """

        if not self._has_spec:
            print("Need to compute a wavelet spectrum first!")
            return

        mfourier = np.sum(self.modulus, axis=1) / self.modulus.shape[1]

        return mfourier

    def draw_AR1_confidence(self, alpha):

        if not self._has_spec:
            print("Need to compute a wavelet spectrum first!")
            return

        tvec = np.arange(self.wlet.shape[1]) * self.dt
        x, y = np.meshgrid(tvec, self.periods)  # for plotting the wavelet transform

        ar1power = core.ar1_powerspec(alpha, self.periods, self.dt)
        conf95 = xi2_95 / 2.0
        conf99 = xi2_99 / 2.0

        scaled_mod = np.zeros(self.modulus.shape)

        # maybe there is a more clever way
        for i, col in enumerate(self.modulus.T):
            scaled_mod[:, i] = col / ar1power

        CS = self.ax_spec.contour(
            x,
            y,
            scaled_mod,
            levels=[xi2_95 / 2.0],
            linewidths=1.7,
            colors="0.95",
            alpha=0.8,
        )
        CS = self.ax_spec.contour(
            x,
            y,
            scaled_mod,
            levels=[xi2_99 / 2.0],
            linewidths=1.7,
            colors="orange",
            alpha=0.8,
        )

        # check confidence levels on (long) ar1 realisations !
        # print (len(where(scaled_mod > conf95)[0])/prod(wlet.shape))
        # should be ~0.05
        # print (len(where(scaled_mod > conf99)[0])/prod(wlet.shape))
        # should be ~0.01

    def get_trend(self, signal):

        trend = core.sinc_smooth(signal, self.T_c, self.dt, M=self.M)

        return trend

    def sinc_detrend(self, signal):

        trend = core.sinc_smooth(signal, self.T_c, self.dt, self.M)

        detrended = signal - trend

        # for easier interface return directly
        return detrended

    def normalize_amplitude(self, signal):

        '''
        Best to do a detrending first..
        
        Mean gets subtracted in any case.
        '''

        if self.L is None:
            print('Set a window size for the sliding window first!')
            print("Can't perform amplitude envelope normalization..")
            return signal
        
        ANsignal = core.normalize_with_envelope(signal, window_size = self.L)

        return ANsignal
    
    def get_envelope(self, signal, SGsmooth = True):

        '''
        Sliding window amplitude envelope

        optional:
        Savitzky-Golay smoothing of the
        envelope with the same window size
        '''

        if self.L is None:
            print('Set a window size for the sliding window first!')
            return None
        
        envelope = core.sliding_window_amplitude(
            signal,
            window_size = self.L,
            SGsmooth = SGsmooth
        )

        return envelope
    
    
    def plot_FFT(self, signal, show_periods=True):
        fig = ppl.figure(figsize=(5, 2.5))

        ax = pl.mk_Fourier_ax(
            fig, time_unit=self.time_unit_label, show_periods=show_periods
        )

        fft_freqs, fft_power = core.compute_fourier(signal, self.dt)
        print(f"mean fourier power: {np.mean(fft_power):.2f}")
        pl.Fourier_spec(ax, fft_freqs, fft_power, show_periods)
        fig.tight_layout()
