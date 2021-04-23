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
    through an object oriented API.

    Parameters
    ----------
    periods   : ordered sequence of periods to compute the Wavelet spectrum for, 
                must have same units as dt!

    dt        : float
                the sampling interval scaled to desired time units, e.g. use 
                dt=120 to get results in seconds or dt=2 to get results in minutes 


    p_max      : Maximum power for Wavelet spectrum z-axis colormap range, 
                 if *None* scales automatically and individually for each 
                 signal. Set to a fixed value to allow for visual comparisons 
                 between signals.

    time_unit_label: string
                     the label for the time unit, e.g. 's' or 'min'

    M         : Length of the sinc filter window, defaults to length
                of input signal. Set to a lower value to 
                speed up sinc-detrending, should be at least around 50
                for a somehwat sharp roll-off.


    Attributes (other than the above given during initialization)
    ----------

    modulus : 2d ndarray
              the real Wavelet power spectrum normalized by signal
              variance, has shape len(periods) x len(signal). Is set to None
              before any analysis is done.
    
    transform : 2d ndarray 
                the complex results of the Wavelet transform with 
                shape len(periods) x len(signal). Is set to None
                before any analysis is done.

    ridge_data : Pandas DataFrame
                 Analysis results obtained by ridge extraction and evaluation. See
                 `get_maxRidge` method documentation for details. Is set to None
                 before any analysis is done.

    ax_signal : matplotlib.axes.Axes
                Reference to the signal axis if created with `plot_signal`, 
                otherwise is None.

    ax_spec_signal : matplotlib.axes.Axes
                Reference to the signal axis of the spectrum figure
                if created with `compute_spectrum`, otherwise is None.

    ax_spec : matplotlib.axes.Axes
              Reference to the power spectrum axis of the spectrum figure
              if created with `compute_spectrum`, otherwise is None.

    Methods
    -------

    sinc_smooth(signal, T_c)
        Convolves the signal with the sinc filter with 
        cut-off period *T_c*. Returns the trend.

    sinc_detrend(signal, T_c)
        Convenience function which returns the detrended
        signal after sinc filtering.

    get_envelope(signal, window_size, SGsmooth = True)
        Uses a sliding window Min-Max operation to estimate
        and return the amplitude envelope.

    normalize_amplitude(signal, window_size)
        Convenience function to estimate
        the amplitude envelope and right away normalizes the signal with
        1/envelope.
        
    compute_spectrum(signal, T_c=None, window_size=None, do_plot=True, draw_coi=False)
        Performs the Wavelet transform, optional ad-hoc filtering if *T_c* and/or 
        *window_size* are set. See also `sinc_detrend` and `normalize_amplitude`.
        If *do_plot* is True, creates the *ax_spec* axis.

    get_averaged_spectrum()
        Returns the time averaged Wavelet spectrum, a (very) good
        Fourier estimate.

    get_maxRidge(power_thresh=0, smoothing_wsize=None)
        Computes and evaluates the ridge as consecutive 
        maxima of the modulus.

    get_sign_maxRidge(empirical_background, confidence=5.99, smoothing_wsize=None)
        Computes and evaluates the ridge as consecutive 
        maxima of the modulus, thresholded by a significance test 
        given a background spectrum. Confidence defaults to 95% interval.

    calc_rhythmicity()
        Returns a rhythmicity score: the time averaged ridge power.

    draw_Ridge()
        Draws the ridge on the Wavelet spectrum.
    
    plot_readout(draw_coi=False, num=None)
        Creates a summary plot after the complete analysis (spectrum+ridge)
        was performed. Shows instantaneous period, phase, amplitude and power.
    
    plot_signal(signal, legend=False, num=None)
        Creates the signal axis *ax_signal* and plots the signal.

    plot_trend(trend, legend=False)
        Plots the trend on the signal axis.

    plot_envelope(envelope, legend=False, num=None)
        Plots the amplitude envelope on the signal axis.

    plot_FFT(signal, show_periods=True)
        Shows the Fourier power spectrum.
    '''
    
    def __init__(
            self,
            periods,
            dt,
            p_max=None,
            time_unit_label="a.u.",            
            M=None
    ):

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

        # to allow for detrending + Wavelet analysis
        self.ana_signal = None
        
        # analysis results
        self.transform = None
        self.modulus = None
        self.ridge_data = None

        # the figure axes
        self.ax_signal = None
        self.ax_spec_signal = None
        self.ax_spec = None

    def compute_spectrum(self,
                         raw_signal,
                         T_c=None,
                         window_size=None,
                         do_plot=True,
                         draw_coi=False):

        """
        Computes the Wavelet spectrum for a given *raw_signal*.

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
                           
        Returns
        -------
        modulus : 2d ndarray
              the real Wavelet power spectrum normalized by signal
              variance, has shape len(periods) x len(signal). Is set to None
              before any analysis is done.
        transform : 2d ndarray 
                the complex results of the Wavelet transform with 
                shape len(periods) x len(signal). Is set to None
                before any analysis is done.

        After a successful analysis, the analyzer instance updates 

        self.transform 
        self.modulus

        with the results. If do_plot was set to True, the attributes

        self.ax_spec_signal
        self.ax_spec

        allow to access the created subplots directly.
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
            # some space for a title
            fig.subplots_adjust(top=0.95)            
            self.ax_spec_signal = axs[0]
            self.ax_spec = axs[1]

            if draw_coi:
                pl.draw_COI(axs[1], time_vector=tvec)

        self.transform = transform
        self.modulus = modulus

        # return also directly
        return modulus, transform

    def get_averaged_spectrum(self):

        """ 
        Average Wavelet spectrum over time
        to give a Fourier estimate. A Wavelet spectrum
        has to be computed first.        

        Returns
        -------
        
        mfourier : Fourier spectrum estimate
        """

        if self.transform is None:
            logger.warning("Need to compute a wavelet spectrum first!")
            return

        mfourier = np.sum(self.modulus, axis=1) / self.modulus.shape[1]

        return mfourier
    
    def get_maxRidge(self, power_thresh=0, smoothing_wsize=None):

        """
        Computes and evaluates the ridge as consecutive 
        maxima of the modulus.

        Returns the ridge_data DataFrame, see also `core.eval_ridge`!

        Additionally the analyser instance updates 

        self.ridge_data 

        with the results.
        

        Parameters
        ----------        
        power_thresh : float, threshold for the ridge. 
        smoothing_wsize : int, optional 
                          Savitkzy-Golay smoothing window size 
                          for ridge smoothing

        Returns
        -------        
        A DataFrame with the following columns:

        time      : the t-values of the ridge, can have gaps if thresholding!
        periods   : the instantaneous periods 
        frequencies : the instantaneous frequencies 
        phase    : the instantaneous phases
        power     : the Wavelet Power normalized to white noise (<P(WN)> = 1)
        amplitude : the estimated amplitudes of the signal

        """

        if self.transform is None:
            logger.warning("Need to compute a wavelet spectrum first!")
            return

        # for easy integration
        modulus = self.modulus

        Nt = modulus.shape[1]  # number of time points
        tvec = np.arange(Nt) * self.dt

        # has to be odd
        if smoothing_wsize and smoothing_wsize % 2 == 0:
            smoothing_wsize = smoothing_wsize + 1
            
        # ================ridge detection=====================================
                
        ridge_ys = core.get_maxRidge_ys(modulus)

        rd = core.eval_ridge(
            ridge_ys,
            self.transform,
            self.ana_signal,
            self.periods,
            tvec=tvec,
            power_thresh=power_thresh,
            smoothing_wsize=smoothing_wsize
        )

        self.ridge_data = rd

        # return also directly
        return rd

    def get_sign_maxRidge(self,
                          empirical_background,
                          confidence=core.chi2_95,
                          smoothing_wsize=None):

        """
        Computes and evaluates the ridge as consecutive 
        maxima of the modulus, thresholded by a significance test for 
        the given background spectrum. Confidence defaults to 95% interval.

        Returns the ridge_data DataFrame, see also `core.eval_ridge`!

        Additionally the analyser instance updates 

        self.ridge_data 

        with the results.
        

        Parameters
        ----------        
        empirical_background : 1d sequence, Fourier estimate of the background.
                               Must hold the powers 
                               at exactly the periods used for
                               the wavelet analysis!

        confidence : float, the Chi-squared value at the desired 
                    confidence level. Defaults to the 95% confidence interval.

        smoothing_wsize : int, optional 
                          Savitkzy-Golay smoothing window size 
                          for ridge smoothing

        Returns
        -------        
        A DataFrame with the following columns:

        time      : the t-values of the ridge, can have gaps if thresholding!
        periods   : the instantaneous periods 
        frequencies : the instantaneous frequencies 
        phase    :  the instantaneous phases
        power     : the Wavelet Power normalized to white noise (<P(WN)> = 1)
        amplitude : the estimated amplitudes of the signal

        """

        if self.transform is None:
            logger.warning("Need to compute a wavelet spectrum first!")
            return

        # for easy integration
        modulus = self.modulus

        Nt = modulus.shape[1]  # number of time points
        tvec = np.arange(Nt) * self.dt

        # has to be odd
        if smoothing_wsize and smoothing_wsize % 2 == 0:
            smoothing_wsize = smoothing_wsize + 1
            
        # ================ridge detection=====================================
                
        ridge_ys = core.get_maxRidge_ys(modulus)

        rd = core.eval_ridge(
            ridge_ys,
            self.transform,
            self.ana_signal,
            self.periods,
            tvec=tvec,
            power_thresh=0, # we need the complete ridge here!
            smoothing_wsize=smoothing_wsize
        )

        spectrum_bool = core.get_significant_regions(self.modulus,
                                                     empirical_background)

        # boolean mask for the ridge
        ridge_bool = spectrum_bool[ridge_ys, rd.index]

        # the significant parts of the ridge
        sign_ridge = rd[ridge_bool]
        # attach additional data, total length and sampling interval
        sign_ridge.Nt = rd.Nt
        sign_ridge.dt = rd.dt

        self.ridge_data = sign_ridge

        # return also directly
        return sign_ridge

    def calc_rhythmicity(self):

        '''
        Returns a rhythmicity score: the time averaged ridge power.
        The averaging is done with respect to total signal length, 
        not ridge length! Therefore, thresholded and short but high 
        power ridge segments get penalized.
        '''

        if self.ridge_data is None:
            logger.warning("Can't calculate rhythmicity, extract a ridge first!")
            return

        R = self.ridge_data.power.sum() / len(self.ana_signal)

        return R
    
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
    
    def get_envelope(self, signal, window_size, SGsmooth=True):

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
        '''
        
        envelope = core.sliding_window_amplitude(
            signal,
            window_size,
            dt = self.dt,
            SGsmooth = SGsmooth
        )

        return envelope
    
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
                
    def draw_Ridge(self, marker_size=1.5):

        if self.ridge_data is None:
            logger.warning("Can't draw ridge, need to do a ridge detection first!")
            return

        if self.transform is None:
            logger.warning("Can't draw ridge, plot the spectrum first!")
            return

        if self.ax_spec is None:
            logger.warning("No spectrum plotted, can't draw ridge!")
            
        pl.draw_Wavelet_ridge(self.ax_spec, self.ridge_data, marker_size)

    def plot_signal(self, signal, num=None, figsize=(6.5, 4.5), **pkwargs):

        '''
        Creates the signal-figure and plots the signal.
        
        Parameters
        ----------        
        signal : a sequence
        num : int
              The number of the figure to be created
        figsize : tuple
                  the size of the signal axis in inches (x,y)
        **pkwargs : keyword arguments for the matlotlib `plot`
                    call, e.g. marker='o', lw=3, color='red'
        '''

        # create plotting style dictionary from defaults
        # and potentially overwrite with the arguments of **pkwargs        
        style_dic = dict(pl.SIGNAL_STYLE, **pkwargs)
        
        if num or self.ax_signal is None:
            fig = ppl.figure(num, figsize=figsize)
            self.ax_signal = pl.mk_signal_ax(self.time_unit_label, fig=fig)

        tvec = np.arange(len(signal)) * self.dt
        self.ax_signal.plot(tvec, signal, **style_dic)

        if 'label' in style_dic:
            self.ax_signal.legend(fontsize=pl.tick_label_size, ncol=2)
            ymin, ymax = self.ax_signal.get_ylim()
            self.ax_signal.set_ylim((ymin, 1.2 * ymax))

        fig = ppl.gcf()
        # fig.subplots_adjust(bottom=0.18)
        fig.tight_layout()

    def plot_trend(self, trend, **pkwargs):
        
        '''
        Plots the trend into the signal axis.
        
        Parameters
        ----------
        trend : a sequence
        **pkwargs : keyword arguments for the matlotlib `plot`
                    call, e.g. marker='o', lw=3, color='red'
        '''

        if self.ax_signal is None:
            logger.warning("No axis to plot trend, plot a signal first!")
            return
        
        # create plotting style dictionary from defaults
        # and potentially overwrite with the arguments of **pkwargs
        style_dic = dict(pl.TREND_STYLE, **pkwargs)

        tvec = np.arange(len(trend)) * self.dt
        self.ax_signal.plot(tvec, trend, **style_dic)

        if 'label' in style_dic:
            self.ax_signal.legend(fontsize=pl.tick_label_size, ncol=2)
            ymin, ymax = self.ax_signal.get_ylim()
            self.ax_signal.set_ylim((ymin, 1.2 * ymax))

        fig = ppl.gcf()
        fig.subplots_adjust(bottom=0.18)
        fig.tight_layout()

    def plot_envelope(self, envelope, **pkwargs):

        '''
        Plot the sliding window amplitude envelope onto the signal.

        Parameters
        ----------
        envelope : a sequence
        **pkwargs : keyword arguments for the matlotlib `plot`
                    call, e.g. marker='o', lw=3, color='red'
        '''
        
        if self.ax_signal is None:
            logger.warning("Can't plot envelope, plot a signal first!")

        # create plotting style dictionary from defaults
        # and potentially overwrite with the arguments of **pkwargs
        style_dic = dict(pl.TREND_STYLE, **pkwargs)
                        
        tvec = np.arange(len(envelope)) * self.dt
        self.ax_signal.plot(tvec, envelope, **style_dic)
        
        if 'label' in style_dic:
            self.ax_signal.legend(fontsize=pl.tick_label_size, ncol=2)
            ymin, ymax = self.ax_signal.get_ylim()
            self.ax_signal.set_ylim((ymin, 1.2 * ymax))

        fig = ppl.gcf()
        fig.subplots_adjust(bottom=0.18)
        fig.tight_layout()

    def plot_readout(self, draw_coi=False, num=None):

        """
        Wraps the readout plot from `pyboat.plotting.plot_readout`.
        Set *num* explicit to control the figure instance created.
        Uses default plotting styles.
        """

        if self.ridge_data is None:
            logger.warning("Need to extract a ridge first!")
            return

        pl.plot_readout(
            self.ridge_data,
            time_unit=self.time_unit_label,
            draw_coi = draw_coi
        )

    def draw_confidence_from_bg(self, empirical_background,
                                confidence=core.chi2_95,
                                **pkwargs):

        '''
        Given an (empirical) background Fourier spectrum,
        draws the contours of the 95% confidence interval
        on the Wavelet power spectrum. 

        empirical_background: a sequence, must hold the powers 
                              at exactly the periods used for
                              the wavelet analysis (self.periods)!

        confidence : float, the Chi-squared value at the desired 
                    confidence level. Defaults to the 95% confidence interval.

        **pkwargs : additional plotting options passed to 
                    matplotlib's contour()
        '''
        if self.transform is None:
            print("Need to compute a wavelet spectrum first!")
            return

        # every period needs a background power value
        # (constant 1 for white noise for examle)
        if not len(empirical_background) == self.modulus.shape[0]:
            raise ValueError("Empirical background doesn't fit"
                             " to wavelet spectrum!")
        
        tvec = np.arange(self.transform.shape[1]) * self.dt
        x, y = np.meshgrid(tvec, self.periods)  

        # remove DOF factor
        confidence = confidence / 2.0

        scaled_mod = np.zeros(self.modulus.shape)

        # rescale every column along the period axis
        # with the assumed 0-model spectrum
        for i, col in enumerate(self.modulus.T):
            scaled_mod[:, i] = col / empirical_background

        default_style = {'linewidths' : 1.4,
                         'colors' : "orange",
                         'alpha' : 0.8}

        # update with user arguments
        style_dic = dict(default_style, **pkwargs)

        CS = self.ax_spec.contour(
            x,
            y,
            scaled_mod,
            levels=[confidence],
            **style_dic
        )
        
    def draw_AR1_confidence(self, alpha):

        '''
        This is a special function which only makes
        sense if AR(1) is the 0-model, and alpha is known! 
        Uses the theoretical AR(1) power spectrum to mark 
        significant regions in the spectrum. Plots both the
        95% and 99% confidence intervals.
        '''

        if self.transform is None:
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
            levels=[core.chi2_95 / 2.0],
            linewidths=1.7,
            colors="0.95",
            alpha=0.8,
        )
        
        CS = self.ax_spec.contour(
            x,
            y,
            scaled_mod,
            levels=[core.chi2_99 / 2.0],
            linewidths=1.7,
            colors="orange",
            alpha=0.8,
        )

        # check confidence levels on (long) ar1 realisations !
        # print (len(where(scaled_mod > conf95)[0])/prod(transform.shape))
        # should be ~0.05
        # print (len(where(scaled_mod > conf99)[0])/prod(transform.shape))
        # should be ~0.01
        
    def plot_FFT(self, signal, show_periods=True):
        fig = ppl.figure(figsize=(5, 2.5))

        ax = pl.mk_Fourier_ax(
            fig, time_unit=self.time_unit_label, show_periods=show_periods
        )

        fft_freqs, fft_power = core.compute_fourier(signal, self.dt)
        logger.info(f"mean fourier power: {np.mean(fft_power):.2f}")
        pl.Fourier_spec(ax, fft_freqs, fft_power, show_periods)
        fig.tight_layout()
