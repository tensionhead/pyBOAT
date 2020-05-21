import matplotlib.pyplot as ppl
import numpy as np
from numpy import pi
from scipy.stats import gaussian_kde, iqr

from pyboat.core import Morlet_COI, find_COI_crossing

# --- define colors ---
rgb_2mpl = lambda R, G, B: np.array((R, G, B)) / 255
rgba_2mpl = lambda R, G, B, A: np.array((R, G, B, A)) / 255

SIG_COLOR = "darkslategray"
TREND_COLOR = rgb_2mpl(165, 105, 189)  # orchidy
ENVELOPE_COLOR = 'orange'

DETREND_COLOR = 'black'
FOURIER_COLOR = 'slategray'
RIDGE_COLOR = "crimson"

COI_COLOR = '0.6' # light gray

# average power histogram
HIST_COLOR = 'lightslategray'

# the readouts
PERIOD_COLOR = 'cornflowerblue'
PHASE_COLOR = 'crimson'
AMPLITUDE_COLOR = 'k'

POWERKDE_COLOR = rgba_2mpl(10,10,10,180)

# the colormap for the wavelet spectra
CMAP = "YlGnBu_r"
# CMAP = 'cividis'
# CMAP = 'magma'

# --- define line widths ---
TREND_LW = 1.5
SIGNAL_LW = 1.5

# --- standard sizes ---
label_size = 18
tick_label_size = 16


# size of x-axis in inches to
# match dimensions of spectrum and signal plots
x_size = 6.5

# --- Signal and Trend -----------------------------------------------


def mk_signal_ax(time_unit="a.u.", fig=None):

    if fig is None:
        fig = ppl.figure(figsize=(x_size, 3.2))
        fig.subplots_adjust(bottom=0.18)

    ax = fig.add_subplot()

    ax.set_xlabel("time (" + time_unit + ")", fontsize=label_size)
    ax.set_ylabel(r"signal", fontsize=label_size)
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    ax.yaxis.offsetText.set_fontsize(tick_label_size)
    ax.tick_params(axis="both", labelsize=tick_label_size)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    return ax


def draw_signal(ax, time_vector, signal):
        
    ax.plot(time_vector, signal, lw=SIGNAL_LW,
            color=SIG_COLOR, alpha=0.8, label="signal")


def draw_trend(ax, time_vector, trend):
    ax.plot(time_vector, trend, color=TREND_COLOR, alpha=0.8, lw=TREND_LW, label="trend")

def draw_envelope(ax, time_vector, envelope):
    ax.plot(time_vector, envelope, color=ENVELOPE_COLOR,
            alpha=0.8, lw=TREND_LW, label="envelope")
    

def draw_detrended(ax, time_vector, detrended):

    ax2 = ax.twinx()
    ax2.plot(time_vector, detrended, "-",
             color=DETREND_COLOR, lw=1.5, alpha=0.6, label = 'detrended')

    ax2.set_ylabel("detrended", fontsize=label_size)
    ax2.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    ax2.yaxis.offsetText.set_fontsize(tick_label_size)

    return ax2


# --- Fourier Spectrum ------------------------------------------------


def mk_Fourier_ax(fig, time_unit="a.u.", show_periods=False):

    fig.clf()
    ax = fig.subplots()

    if show_periods:
        ax.set_xlabel("period (" + time_unit + ")", fontsize=label_size)
        # ax.set_xscale("log")

    else:
        ax.set_xlabel("frequency (" + time_unit + r"$^{-1}$)", fontsize=label_size)

    ax.set_ylabel("Fourier power", fontsize=label_size)
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    ax.tick_params(axis="both", labelsize=tick_label_size)
    ax.yaxis.offsetText.set_fontsize(tick_label_size)

    return ax


def Fourier_spec(ax, fft_freqs, fft_power, show_periods=False):

    freq_bin_width = np.diff(fft_freqs)[0]

    pow_max_ind = np.argmax(fft_power)

    # heuristically determine bin width by looking
    # at the most prominent period
    if pow_max_ind < len(fft_freqs):
        per_bin_width = 1/fft_freqs[pow_max_ind] - 1/fft_freqs[pow_max_ind + 1]

    else:
        per_bin_width = 1/fft_freqs[pow_max_ind] - 1/fft_freqs[pow_max_ind - 1]

    
    if show_periods:

        # period view, omit the last bin 2/(N*dt)

        # skip 0-frequency

        if len(fft_freqs) < 300:

            ax.bar(
                1 / fft_freqs[1:],
                fft_power[1:],
                alpha=0.4,
                edgecolor="k",
                color=FOURIER_COLOR,
                width= 0.5 * per_bin_width,
            )

            # ax.vlines(
            #     1 / fft_freqs[1:],
            #     0,
            #     fft_power[1:],
            #     lw=2,
            #     alpha=0.8,
            #     color=FOURIER_COLOR,
            # )

        # plot differently for very long data
        else:
            ax.plot(
                1 / fft_freqs[1:],
                fft_power[1:],
                "--",
                lw=1.5,
                alpha=0.8,
                color=FOURIER_COLOR,
            )

    else:

        if len(fft_freqs) < 300:

            # frequency view
            ax.bar(
                fft_freqs,
                fft_power,
                alpha=0.4,
                edgecolor="k",
                color=FOURIER_COLOR,
                width=0.8 * freq_bin_width,
            )
        else:
            ax.plot(fft_freqs, fft_power, ".", ms=1, alpha=0.8, color=FOURIER_COLOR)


# --- Wavelet spectrum  ------


def mk_signal_modulus_ax(time_unit="a.u.", height_ratios=[1, 2.5], fig=None):

    if fig is None:
        fig = ppl.figure(figsize=(x_size, 6.5))
        fig.subplots_adjust(bottom=0.07, top=0.97)
    # 1st axis is for signal, 2nd axis is the spectrum
    axs = fig.subplots(2, 1, gridspec_kw={"height_ratios": height_ratios}, sharex=True)

    sig_ax = axs[0]
    mod_ax = axs[1]

    sig_ax.set_ylabel("signal (a.u.)", fontsize=label_size)

    mod_ax.set_xlabel("time (" + time_unit + ")", fontsize=label_size)
    mod_ax.set_ylabel("period (" + time_unit + ")", fontsize=label_size)
    mod_ax.tick_params(axis="both", labelsize=tick_label_size)

    return axs


def plot_signal_modulus(axs, time_vector, signal, modulus, periods, p_max=None):

    """
    Plot the signal and the wavelet power spectrum.
    axs[0] is signal axis, axs[1] spectrum axis
    """

    sig_ax = axs[0]
    mod_ax = axs[1]

    # Plot Signal above spectrum

    sig_ax.plot(time_vector, signal, color="black", lw=1.5, alpha=0.7)
    # sig_ax.plot(time_vector, signal, ".", color="black", ms=2.0, alpha=0.5)
    sig_ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    sig_ax.tick_params(axis="both", labelsize=tick_label_size)
    sig_ax.yaxis.offsetText.set_fontsize(tick_label_size)

    # Plot Wavelet Power Spectrum

    extent = (time_vector[0], time_vector[-1], periods[0], periods[-1])
    im = mod_ax.imshow(
        modulus[::-1], cmap=CMAP, vmin = 0, vmax=p_max, extent=extent, aspect="auto"
    )

    mod_ax.set_ylim((periods[0], periods[-1]))
    mod_ax.set_xlim((time_vector[0], time_vector[-1]))
    mod_ax.grid(axis="y", color="0.6", lw=1.0, alpha=0.5)  # vertical grid lines

    min_power = modulus.min()
    if p_max is None:
        cb_ticks = [0, int(np.floor(modulus.max()))]
    else:
        cb_ticks = [0, p_max]

    cb = ppl.colorbar(
        im, ax=mod_ax, orientation="horizontal", fraction=0.08, shrink=0.6, pad=0.22
    )
    cb.set_ticks(cb_ticks)
    cb.ax.set_xticklabels(cb_ticks, fontsize=tick_label_size)
    # cb.set_label('$|\mathcal{W}_{\Psi}(t,T)|^2$',rotation = '0',labelpad = 5,fontsize = 15)
    cb.set_label("wavelet power", rotation="0", labelpad=-17, fontsize=0.9 * label_size)


def draw_Wavelet_ridge(ax, ridge_data, marker_size=1.5):

    """
    *ridge_data* comes from core.eval_ridge !
    """

    ax.plot(
        ridge_data.time,
        ridge_data["periods"],
        "o",
        color=RIDGE_COLOR,
        alpha=0.6,
        ms=marker_size,
    )


def draw_COI(ax, time_vector, coi_slope):

    """
    Draw Cone of influence on spectrum, period version
    """

    # Plot the COI
    N_2 = int(len(time_vector) / 2.0)

    # ascending left side
    ax.plot(
        time_vector[: N_2 + 1], coi_slope * time_vector[: N_2 + 1],
        "-.", alpha= 0.7, color = COI_COLOR
    )

    # descending right side
    ax.plot(
        time_vector[N_2:],
        coi_slope * (time_vector[-1] - time_vector[N_2:]),
        "-.",
        alpha=0.7, color = COI_COLOR
    )


# --- Wavelet readout ----------------------------


def plot_readout(ridge_data, time_unit="a.u.", draw_coi = False, fig=None):

    """
    ridge_data from core.eval_ridge(...)
    creates four axes: period, phase, amplitude and power
    """
    
    i_left, i_right = find_COI_crossing(ridge_data)
    
    if fig is None:
        fig = ppl.figure(figsize=(7, 4.8))

    periods = ridge_data["periods"]
    phases = ridge_data["phase"]
    powers = ridge_data["power"]
    amplitudes = ridge_data["amplitude"]
    tvec = ridge_data["time"] # indexable 

    # check for discontinuous ridge
    if np.all(np.diff(tvec, n = 2) < 1e-12):
        # draw continuous lines
        lstyle = '-'
        mstyle = ''
    else:
        lstyle = ''
        mstyle = 'o'
    
    fig.subplots_adjust(wspace=0.3, left=0.11, top=0.98, right=0.97, bottom=0.14)

    axs = fig.subplots(2, 2, sharex=True)

    ax1 = axs[0, 0]
    ax2 = axs[0, 1]
    ax3 = axs[1, 0]
    ax4 = axs[1, 1]

    # periods
    ax1.plot(tvec[i_left:i_right], periods[i_left:i_right], color=PERIOD_COLOR, alpha=0.8, lw=2.5, ls = lstyle, marker = mstyle, ms = 1.5)
    # inside COI
    ax1.plot(tvec[:i_left], periods[:i_left], '--', color=PERIOD_COLOR, alpha=0.8, ms = 2.5)
    ax1.plot(tvec[i_right:], periods[i_right:], '--', color=PERIOD_COLOR, alpha=0.8, ms = 2.5)
    
    ax1.set_ylabel(f"Period ({time_unit})", fontsize=label_size)
    ax1.grid(True, axis="y")
    yl = ax1.get_ylim()
    ax1.set_ylim((max([0, 0.75 * yl[0]]), 1.25 * yl[1]))
    
    # only now draw the COI?
    if draw_coi:
        draw_COI(ax1, np.array(ridge_data.time), Morlet_COI())
    
    ax1.tick_params(axis="both", labelsize=tick_label_size)

    # ax1.set_ylim( (120,160) )

    # phases
    ax2.plot(tvec[i_left:i_right], phases[i_left:i_right], "-", c=PHASE_COLOR, alpha=0.8,
             ls = lstyle, marker = mstyle, ms = 1.5)

    # inside COI
    ax2.plot(tvec[:i_left], phases[:i_left], '-.',color=PHASE_COLOR, alpha=0.5, ms = 2.5)
    ax2.plot(tvec[i_right:], phases[i_right:], '-.', color=PHASE_COLOR, alpha=0.5, ms = 2.5)
    
    ax2.set_ylabel("Phase (rad)", fontsize=label_size, labelpad=0.5)
    ax2.set_yticks((0, pi, 2 * pi))
    ax2.set_yticklabels(("$0$", "$\pi$", "$2\pi$"))
    ax2.tick_params(axis="both", labelsize=tick_label_size)

    # amplitudes
    ax3.plot(tvec[i_left:i_right], amplitudes[i_left:i_right], c = AMPLITUDE_COLOR, lw=2.5, alpha=0.9,
             ls = lstyle, marker = mstyle, ms = 1.5)

    # inside COI
    ax3.plot(tvec[:i_left], amplitudes[:i_left], '--',color = AMPLITUDE_COLOR, alpha=0.6, ms = 2.5)
    ax3.plot(tvec[i_right:], amplitudes[i_right:], '--', color = AMPLITUDE_COLOR, alpha=0.6, ms = 2.5)
    
    ax3.set_ylim((0, 1.1 * amplitudes.max()))
    ax3.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    ax3.yaxis.offsetText.set_fontsize(tick_label_size)
    ax3.set_ylabel("Amplitude (a.u.)", fontsize=label_size)
    ax3.set_xlabel("Time (" + time_unit + ")", fontsize=label_size)
    ax3.tick_params(axis="both", labelsize=tick_label_size)

    # powers
    ax4.plot(tvec[i_left:i_right], powers[i_left:i_right], "k-", lw=2.5, alpha=0.5,
             ls = lstyle, marker = mstyle, ms = 1.5)

    # inside COI
    ax4.plot(tvec[:i_left], powers[:i_left], '--',color='gray', alpha=0.6, ms = 2.5)
    ax4.plot(tvec[i_right:], powers[i_right:], '--', color='gray', alpha=0.6, ms = 2.5)
    
    ax4.set_ylim((0, 1.1 * powers.max()))
    ax4.set_ylabel("Power (wnp)", fontsize=label_size)
    ax4.set_xlabel("Time (" + time_unit + ")", fontsize=label_size)
    ax4.tick_params(axis="both", labelsize=tick_label_size)

# -------- Ensemble Measures Plots -------------------------------------

def Freedman_Diaconis_rule(samples):

    '''
    Get optimal number of bins from samples,
    a bit too small in most cases for the power distribution it seems..
    '''

    h = 2 * iqr(samples) / pow(samples.size, 1/3)
    Nbins = int( ( samples.max() - samples.min() ) / h)
    return Nbins

def Rice_rule(samples):

    '''
    Get optimal number of bins from samples, looks about
    right for 'typical' bi-modal power distributions.
    '''

    Nbins = int(2 * pow(len(samples), 1/3))
    return Nbins

def plot_power_distribution(powers, kde = True, fig = None):

    '''
    Histogram (bin-counts) of Wavelet powers, intended
    for the time-averaged powers as computed
    by ensemble_measures.average_power_distribution(...).

    Parameters
    ----------
    
    powers : a sequence of floats
    fig :    matplotlib figure instance
    '''

    powers = np.array(powers)

    if powers.ndim != 1:
        raise ValueError('Powers must be a sequence!')
    
    if fig is None:
        fig = ppl.figure(figsize=(5, 3.2))

    ax = fig.subplots()
    ax.set_ylabel("Counts", fontsize=label_size)
    ax.set_xlabel("Average Wavelet Power", fontsize=label_size)
    ax.tick_params(axis="both", labelsize=tick_label_size)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # use automated number of bins
    Nbins = Rice_rule(powers)
    
    # the raw histogram
    counts, bins,_ = ax.hist(
        powers,
        bins = Nbins,
        rwidth = 0.9,
        color = HIST_COLOR,
        alpha = 0.7,
        density = False
    )

    delta_bin = bins[1] - bins[0] # evenly spaced

    if kde:
        
        # get the KDE support
        support = np.linspace(0.2*powers.min(), 1.2*powers.max(), 200)
        # standard KDE, bandwith from Silverman
        dens = gaussian_kde(powers, bw_method = 'silverman')
        
        ax2 = ax.twinx()
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)

        # normalize relative to count axis
        ax2.set_ylim( (0, ax.get_ylim()[1]/( len(powers) * delta_bin) ))
        ax2.set_yticks( () )
        ax2.plot(
            support,
            dens(support),
            color = POWERKDE_COLOR,
            lw = 2.,
            alpha = 0.8,
            label = 'KDE'
        )

        # not needed?!
        # ax2.legend(fontsize = tick_label_size)

    fig.tight_layout()
    #fig.subplots_adjust(bottom = 0.22, left = 0.15)
    
def plot_ensemble_dynamics(
        periods,
        amplitudes,
        phases,
        dt = 1,
        time_unit = 'a.u',        
        fig = None):

    '''
    Taking the summary statistics from

    core.get_ensemble_dynamics()
    
    plots the median and quartiles for the periods
    and amplitudes over time, as well as the 
    phase coherence measure R.

    Parameters
    ----------

    periods : DataFrame containing the 'median' and the
              two quartiles 'Q1' and 'Q3' over time
    amplitudes : DataFrame containing the 'median' and the
                 two quartiles 'Q1' and 'Q3' over time

    phases : DataFrame containing 'R'

    dt : float, the sampling interval to get a proper time axis

    time_unit : str, the time unit label

    '''

    # all DataFrames come from the same ensemble, so their shape should match
    tvec = np.arange(periods.shape[0]) * dt
    
    if fig is None:
        fig = ppl.figure( figsize = (4.5,6.8) )

    # create the 2 axes
    axs = fig.subplots(3, 1, sharex = True)

    for ax in axs:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis = 'y')
    ax1, ax2, ax3 = axs
    
    # the periods
    ax1.plot(tvec, periods['median'], c = PERIOD_COLOR, lw = 2, alpha = 0.7)
    ax1.fill_between(tvec, periods['Q1'], periods['Q3'], color = PERIOD_COLOR, alpha = 0.3)
    ax1.set_ylabel(f"Period ({time_unit})", fontsize = label_size)
    ax1.tick_params(axis="both", labelsize=tick_label_size)

    # amplitudes
    ax2.plot(tvec, amplitudes['median'], c = AMPLITUDE_COLOR, lw = 2, alpha = 0.6)
    ax2.fill_between(tvec, amplitudes['Q1'], amplitudes['Q3'], color = AMPLITUDE_COLOR, alpha = 0.2)
    ax2.set_ylabel(f"Amplitude (a.u.)", fontsize=label_size)
    ax2.tick_params(axis="both", labelsize=tick_label_size)

    # phase coherence
    ax3.plot(tvec, phases['R'], c = PHASE_COLOR, lw = 3, alpha = 0.8)

    ax3.set_ylim( (0, 1.1) )
    ax3.set_ylabel(f"Phase Coherence", fontsize=label_size)    
    ax3.set_xlabel(f'Time ({time_unit})', fontsize = label_size)
    ax3.tick_params(axis="both", labelsize=tick_label_size)

    fig.subplots_adjust(bottom = 0.1, left = 0.2, top = 0.95, hspace = 0.1)
