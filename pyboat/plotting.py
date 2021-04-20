""" This module provides all visualizations, both for the ui and the API """

import matplotlib.pyplot as ppl
import numpy as np
from numpy import pi
from scipy.stats import gaussian_kde, iqr

from pyboat.core import get_COI_branches, find_COI_crossing

# --- define colors ---
rgb_2mpl = lambda R, G, B: np.array((R, G, B)) / 255
rgba_2mpl = lambda R, G, B, A: np.array((R, G, B, A)) / 255

SIG_COLOR = "darkslategray"
TREND_COLOR = rgb_2mpl(165, 105, 189)  # orchidy
ENVELOPE_COLOR = "orange"

DETREND_COLOR = "black"
FOURIER_COLOR = "cadetblue"
RIDGE_COLOR = "crimson"

COI_COLOR = "0.6"  # light gray

# average power histogram
HIST_COLOR = "cadetblue"

# the readouts
PERIOD_COLOR = "cornflowerblue"
PHASE_COLOR = "crimson"
AMPLITUDE_COLOR = "k"
POWER_COLOR = "cadetblue"
POWERKDE_COLOR = rgba_2mpl(10, 10, 10, 160)

# the colormap for the wavelet spectra
CMAP = "YlGnBu_r"
# CMAP = 'cividis'
# CMAP = 'magma'

# --- max size of signal to plot also the sample points with explicit marker o's
Nmax = 250

# --- define line widths ---
TREND_LW = 2.0
SIGNAL_LW = 1.5
MARKER_SIZE = 4

# --- standard sizes ---
label_size = 18
tick_label_size = 16

# size of x-axis in inches to
# match dimensions of spectrum and signal plots
x_size = 6.5

# --- default styles as dictionaries ---
# can be used with ppl.plot(..., **STYLE)

SIGNAL_STYLE = {
    "lw": SIGNAL_LW,
    "marker": None,
    "ms": MARKER_SIZE,
    "color": SIG_COLOR,
    "alpha": 0.8,
}
TREND_STYLE = {
    "lw": TREND_LW,
    "marker": None,
    "ms": MARKER_SIZE,
    "color": TREND_COLOR,
    "alpha": 0.8,
}
ENVELOPE_STYLE = {
    "lw": TREND_LW,
    "marker": None,
    "ms": MARKER_SIZE,
    "color": ENVELOPE_COLOR,
    "alpha": 0.8,
}


# signal plot style, show markers only for short signals
def get_marker_lw(signal):

    if len(signal) <= Nmax:
        m = "o"
        lw = 1.2
    else:
        m = ""
        lw = SIGNAL_LW
    return m, lw


# --- Signal and Trend -----------------------------------------------


def mk_signal_ax(time_unit="a.u.", fig=None):

    if fig is None:
        fig = ppl.figure(figsize=(x_size, 3.2))
        fig.subplots_adjust(bottom=0.18)

    ax = fig.subplots()

    ax.set_xlabel("Time (" + time_unit + ")", fontsize=label_size)
    ax.set_ylabel(r"Signal", fontsize=label_size)
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    ax.yaxis.offsetText.set_fontsize(tick_label_size)
    ax.tick_params(axis="both", labelsize=tick_label_size)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return ax


def draw_signal(ax, time_vector, signal):

    m, lw = get_marker_lw(signal)

    ax.plot(
        time_vector, signal, lw=lw, marker=m, ms=MARKER_SIZE, color=SIG_COLOR, alpha=0.8
    )


def draw_trend(ax, time_vector, trend, label="Trend"):
    ax.plot(time_vector, trend, color=TREND_COLOR, alpha=0.8, lw=TREND_LW, label=label)


def draw_envelope(ax, time_vector, envelope):
    ax.plot(
        time_vector,
        envelope,
        color=ENVELOPE_COLOR,
        alpha=0.8,
        lw=TREND_LW,
        label="Envelope",
    )


def draw_detrended(ax, time_vector, detrended):

    m, lw = get_marker_lw(detrended)

    ax2 = ax.twinx()
    ax2.plot(
        time_vector,
        detrended,
        "-",
        marker=m,
        ms=MARKER_SIZE,
        color=DETREND_COLOR,
        lw=lw,
        alpha=0.6,
        label="Detrended",
    )

    ax2.set_ylabel("detrended", fontsize=label_size)
    ax2.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    ax2.yaxis.offsetText.set_fontsize(tick_label_size)

    return ax2


# --- Fourier Spectrum --------------------------


def mk_Fourier_ax(fig, time_unit="a.u.", show_periods=False):

    fig.clf()
    ax = fig.subplots()

    if show_periods:
        ax.set_xlabel("Period (" + time_unit + ")", fontsize=label_size)
        # ax.set_xscale("log")

    else:
        ax.set_xlabel("Frequency (" + time_unit + r"$^{-1}$)", fontsize=label_size)

    ax.set_ylabel("Fourier power", fontsize=label_size)
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    ax.tick_params(axis="both", labelsize=tick_label_size)
    ax.yaxis.offsetText.set_fontsize(tick_label_size)

    return ax


def Fourier_spec(ax, fft_freqs, fft_power, show_periods=False):

    if show_periods:
        # period view, omit the last bin 2/(N*dt)
        # skip 0-frequency
        if len(fft_freqs) < 1000:
            ax.vlines(
                1 / fft_freqs[1:],
                0,
                fft_power[1:],
                lw=1.5,
                alpha=0.8,
                color=FOURIER_COLOR,
            )
        # plot differently for very long signals
        else:
            ax.plot(
                1 / fft_freqs[1:],
                fft_power[1:],
                "-",
                lw=1.5,
                alpha=0.8,
                color=FOURIER_COLOR,
            )
    else:
        if len(fft_freqs) < 1000:
            # frequency view
            ax.vlines(fft_freqs, 0, fft_power, alpha=0.8, color=FOURIER_COLOR, lw=1.5)
        else:
            ax.plot(fft_freqs, fft_power, "-", lw=1.5, alpha=0.8, color=FOURIER_COLOR)


# --- time averaged Wavelet spectrum -> Fourier estimate


def averaged_Wspec(averaged_Wspec, periods, time_unit="a.u", fig=None):

    """
    Plots the time averaged Wavelet spectrum, which is a good Fourier estimate.

    Parameters
    ----------

    averaged_Wspec : sequence, holding the average power for each period

    periods :   sequence, the periods of the original Wavelet transform

    time_unit : str, the time unit label

    fig : matplotlib figure instance, a new figure is created per default

    """

    if fig is None:
        fig = ppl.figure(figsize=(5, 3.2))

    ax = fig.subplots()
    ax.set_ylabel("Power (wnp)", fontsize=label_size)
    ax.set_xlabel(f"Period ({time_unit})", fontsize=label_size)

    ax.plot(periods, averaged_Wspec, lw=SIGNAL_LW, color=FOURIER_COLOR)
    ax.fill_between(periods, 0, averaged_Wspec, color=FOURIER_COLOR, alpha=0.3)

    return ax
# --- Wavelet spectrum  ------


def mk_signal_modulus_ax(time_unit="a.u.", height_ratios=[1, 2.5], fig=None):

    if fig is None:
        fig = ppl.figure(figsize=(x_size, 6.5))
        fig.subplots_adjust(bottom=0.07, top=0.97)
    # 1st axis is for signal, 2nd axis is the spectrum
    axs = fig.subplots(2, 1, gridspec_kw={"height_ratios": height_ratios}, sharex=True)

    sig_ax = axs[0]
    mod_ax = axs[1]

    sig_ax.set_ylabel("Signal (a.u.)", fontsize=label_size)

    mod_ax.set_xlabel("Time (" + time_unit + ")", fontsize=label_size)
    mod_ax.set_ylabel("Period (" + time_unit + ")", fontsize=label_size)
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

    m, lw = get_marker_lw(signal)

    sig_ax.plot(
        time_vector, signal, color="black", lw=lw, alpha=0.65, marker=m, ms=MARKER_SIZE
    )

    # sig_ax.plot(time_vector, signal, ".", color="black", ms=2.0, alpha=0.5)
    sig_ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    sig_ax.tick_params(axis="both", labelsize=tick_label_size)
    sig_ax.yaxis.offsetText.set_fontsize(tick_label_size)

    # Plot Wavelet Power Spectrum

    extent = (time_vector[0], time_vector[-1], periods[0], periods[-1])
    im = mod_ax.imshow(
        modulus[::-1], cmap=CMAP, vmin=0, vmax=p_max, extent=extent, aspect="auto"
    )

    mod_ax.set_ylim((periods[0], periods[-1]))
    mod_ax.set_xlim((time_vector[0], time_vector[-1]))
    mod_ax.grid(axis="y", color="0.6", lw=1.0, alpha=0.5)  # vertical grid lines

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
    cb.set_label("Wavelet Power", rotation="0", labelpad=-12, fontsize=0.9 * label_size)


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


def draw_COI(ax, time_vector):

    """
    Draw Cone of influence on spectrum, period version
    
    """

    coi, coi_t = get_COI_branches(time_vector)

    ax.plot(coi_t, coi, "-.", alpha=0.7, color=COI_COLOR)


# --- Wavelet readout ----------------------------


def plot_readout(ridge_data, time_unit="a.u.", draw_coi=False, fig=None):

    """
    ridge_data from core.eval_ridge(...)
    creates four axes: period, phase, amplitude and power
    """

    # restore the original complete time vector
    tvec = np.arange(ridge_data.Nt) * ridge_data.dt

    i_left, i_right = find_COI_crossing(ridge_data)

    if fig is None:
        fig = ppl.figure(figsize=(7, 4.8))

    periods = ridge_data["periods"]
    phases = ridge_data["phase"]
    powers = ridge_data["power"]
    amplitudes = ridge_data["amplitude"]
    ridge_t = ridge_data["time"]  # indexable

    # check for discontinuous ridge
    if np.all(np.diff(ridge_t, n=2) < 1e-12):
        # draw continuous lines
        lstyle = "-"
        mstyle = ""
    else:
        lstyle = ""
        mstyle = "o"

    fig.subplots_adjust(wspace=0.3, left=0.11, top=0.98, right=0.97, bottom=0.14)

    axs = fig.subplots(2, 2, sharex=True)

    for ax in axs.flatten():
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    ax1 = axs[0, 0]
    ax2 = axs[0, 1]
    ax3 = axs[1, 0]
    ax4 = axs[1, 1]

    # periods
    ax1.plot(
        ridge_t.loc[i_left:i_right],
        periods.loc[i_left:i_right],
        color=PERIOD_COLOR,
        alpha=0.8,
        lw=2.5,
        ls=lstyle,
        marker=mstyle,
        ms=1.5,
    )
    # inside COI
    ax1.plot(
        ridge_t.loc[:i_left],
        periods.loc[:i_left],
        "--",
        color=PERIOD_COLOR,
        alpha=0.8,
        ms=2.5,
    )
    ax1.plot(
        ridge_t.loc[i_right:],
        periods.loc[i_right:],
        "--",
        color=PERIOD_COLOR,
        alpha=0.8,
        ms=2.5,
    )

    ax1.set_ylabel(f"Period ({time_unit})", fontsize=label_size)
    ax1.grid(True, axis="y")
    yl = ax1.get_ylim()
    ax1.set_ylim((max([0, 0.75 * yl[0]]), 1.25 * yl[1]))

    # only now draw the COI?
    if draw_coi:
        draw_COI(ax1, tvec)

    ax1.tick_params(axis="both", labelsize=tick_label_size)

    # ax1.set_ylim( (120,160) )

    # phases
    ax2.plot(
        ridge_t.loc[i_left:i_right],
        phases.loc[i_left:i_right],
        "-",
        c=PHASE_COLOR,
        alpha=0.8,
        ls=lstyle,
        marker=mstyle,
        ms=1.5,
    )

    # inside COI
    ax2.plot(
        ridge_t.loc[:i_left],
        phases.loc[:i_left],
        "-.",
        color=PHASE_COLOR,
        alpha=0.5,
        ms=2.5,
    )
    ax2.plot(
        ridge_t.loc[i_right:],
        phases.loc[i_right:],
        "-.",
        color=PHASE_COLOR,
        alpha=0.5,
        ms=2.5,
    )

    ax2.set_ylabel("Phase (rad)", fontsize=label_size, labelpad=0.5)
    ax2.set_yticks((0, pi, 2 * pi))
    ax2.set_yticklabels(("$0$", "$\pi$", "$2\pi$"))
    ax2.tick_params(axis="both", labelsize=tick_label_size)

    # amplitudes
    ax3.plot(
        ridge_t.loc[i_left:i_right],
        amplitudes.loc[i_left:i_right],
        c=AMPLITUDE_COLOR,
        lw=2.5,
        alpha=0.9,
        ls=lstyle,
        marker=mstyle,
        ms=1.5,
    )

    # inside COI
    ax3.plot(
        ridge_t.loc[:i_left],
        amplitudes.loc[:i_left],
        "--",
        color=AMPLITUDE_COLOR,
        alpha=0.6,
        ms=2.5,
    )
    ax3.plot(
        ridge_t.loc[i_right:],
        amplitudes.loc[i_right:],
        "--",
        color=AMPLITUDE_COLOR,
        alpha=0.6,
        ms=2.5,
    )

    ax3.set_ylim((0, 1.1 * amplitudes.max()))
    ax3.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    ax3.yaxis.offsetText.set_fontsize(tick_label_size)
    ax3.set_ylabel("Amplitude (a.u.)", fontsize=label_size)
    ax3.set_xlabel("Time (" + time_unit + ")", fontsize=label_size)
    ax3.tick_params(axis="both", labelsize=tick_label_size)

    # powers
    ax4.plot(
        ridge_t.loc[i_left:i_right],
        powers.loc[i_left:i_right],
        lw=2.5,
        alpha=0.8,
        color=POWER_COLOR,
        ls=lstyle,
        marker=mstyle,
        ms=1.5,
    )

    # inside COI
    ax4.plot(
        ridge_t.loc[:i_left],
        powers.loc[:i_left],
        "--",
        color=POWER_COLOR,
        alpha=0.6,
        ms=2.5,
    )
    ax4.plot(
        ridge_t.loc[i_right:],
        powers.loc[i_right:],
        "--",
        color=POWER_COLOR,
        alpha=0.6,
        ms=2.5,
    )

    ax4.set_ylim((0, 1.1 * powers.max()))
    ax4.set_ylabel("Power (wnp)", fontsize=label_size)
    ax4.set_xlabel("Time (" + time_unit + ")", fontsize=label_size)
    ax4.tick_params(axis="both", labelsize=tick_label_size)

    fig.tight_layout()

    return axs
    
# -------- Ensemble Measures Plots -------------------------------------


def Freedman_Diaconis_rule(samples):

    """
    Get optimal number of bins from samples,
    a bit too small in most cases for the power distribution it seems..
    """

    h = 2 * iqr(samples) / pow(samples.size, 1 / 3)
    Nbins = int((samples.max() - samples.min()) / h)
    return Nbins


def Rice_rule(samples):

    """
    Get optimal number of bins from samples, looks about
    right for 'typical' bi-modal power distributions.
    """

    Nbins = int(2 * pow(len(samples), 1 / 3))
    return Nbins


def power_distribution(powers, kde=True, fig=None):

    """
    Histogram (bin-counts) of Wavelet powers, intended
    for the time-averaged powers as computed
    by ensemble_measures.average_power_distribution(...).

    Parameters
    ----------
    
    powers : a sequence of floats
    kde  : bool, Gaussian kde 
    fig :    matplotlib figure instance
    """

    powers = np.array(powers)

    if powers.ndim != 1:
        raise ValueError("Powers must be a sequence!")

    if fig is None:
        fig = ppl.figure(figsize=(5, 3.2))

    ax = fig.subplots()
    ax.set_ylabel("Counts", fontsize=label_size)
    ax.set_xlabel("Average Signal Power", fontsize=label_size)
    ax.tick_params(axis="both", labelsize=tick_label_size)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # use automated number of bins
    Nbins = Rice_rule(powers)

    # the raw histogram
    counts, bins, _ = ax.hist(
        powers, bins=Nbins, rwidth=0.9, color=HIST_COLOR, alpha=0.7, density=False
    )

    delta_bin = bins[1] - bins[0]  # evenly spaced

    if kde:

        # get the KDE support
        support = np.linspace(0.2 * powers.min(), 1.2 * powers.max(), 200)
        # standard KDE, bandwith from Silverman
        dens = gaussian_kde(powers, bw_method="silverman")

        ax2 = ax.twinx()
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)

        # normalize relative to count axis
        ax2.set_ylim((0, ax.get_ylim()[1] / (len(powers) * delta_bin)))
        ax2.set_yticks(())
        ax2.plot(support, dens(support), color=POWERKDE_COLOR, lw=2.0, label="KDE")

        # not needed?!
        # ax2.legend(fontsize = tick_label_size)

    fig.tight_layout()
    # fig.subplots_adjust(bottom = 0.22, left = 0.15)


def ensemble_dynamics(
    periods, amplitudes, powers, phases, dt=1, time_unit="a.u", fig=None
):

    """
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

    powers : DataFrame containing the 'median' and the
                 two quartiles 'Q1' and 'Q3' over time

    phases : DataFrame containing 1st order parameter 'R'

    dt : float, the sampling interval to get a proper time axis

    time_unit : str, the time unit label

    """

    # all DataFrames come from the same ensemble, so their shape should match
    tvec = np.arange(periods.shape[0]) * dt

    if fig is None:
        fig = ppl.figure(figsize=(7, 4.8))

    # create the 2 axes
    axs = fig.subplots(2, 2, sharex=True)

    for ax in axs.flatten():
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y")

    ax1 = axs[0, 0]
    ax2 = axs[0, 1]
    ax3 = axs[1, 0]
    ax4 = axs[1, 1]

    # the periods
    ax1.plot(tvec, periods["median"], c=PERIOD_COLOR, lw=2, alpha=0.7)
    ax1.fill_between(tvec, periods["Q1"], periods["Q3"], color=PERIOD_COLOR, alpha=0.3)
    ax1.set_ylabel(f"Period ({time_unit})", fontsize=label_size)
    ax1.tick_params(axis="both", labelsize=tick_label_size)

    # phase coherence
    ax2.plot(tvec, phases["R"], c=PHASE_COLOR, lw=3, alpha=0.8)

    ax2.set_ylim((0, 1.1))
    ax2.set_ylabel(f"Phase Coherence", fontsize=label_size)
    ax2.tick_params(axis="both", labelsize=tick_label_size)

    # amplitudes
    ax3.plot(tvec, amplitudes["median"], c=AMPLITUDE_COLOR, lw=2, alpha=0.6)
    ax3.fill_between(
        tvec, amplitudes["Q1"], amplitudes["Q3"], color=AMPLITUDE_COLOR, alpha=0.2
    )
    ax3.set_ylabel(f"Amplitude (a.u.)", fontsize=label_size)
    ax3.tick_params(axis="both", labelsize=tick_label_size)
    ax3.set_xlabel(f"Time ({time_unit})", fontsize=label_size)

    # powers
    ax4.plot(tvec, powers["median"], c=POWER_COLOR, lw=2, alpha=0.6)
    ax4.fill_between(tvec, powers["Q1"], powers["Q3"], color=POWER_COLOR, alpha=0.2)
    ax4.set_ylabel(f"Power (a.u.)", fontsize=label_size)
    ax4.tick_params(axis="both", labelsize=tick_label_size)
    ax4.set_xlabel(f"Time ({time_unit})", fontsize=label_size)

    fig.subplots_adjust(wspace=0.3, left=0.11, top=0.98, right=0.97, bottom=0.14)

    # fig.subplots_adjust(bottom = 0.1, left = 0.2, top = 0.95, hspace = 0.1)
    return axs


def Fourier_distribution(
    df_fouriers, time_unit="a.u.", label=None, color=FOURIER_COLOR, fig=None
):

    """
    Plots the median and quartiles of a distribution of 
    Fourier power spectra. Typical use case are the
    time averaged Wavelet spectra of a population.

    Parameters
    ----------

    df_fouriers : DataFrame, columns hold the individual Fourier power spectra,
                            index holds the periods

    time_unit : str, the time unit label

    fig : matplotlib figure instance

    """

    periods = df_fouriers.index

    med = df_fouriers.median(axis=1)
    q1 = df_fouriers.quantile(q=0.25, axis=1)
    q3 = df_fouriers.quantile(q=0.75, axis=1)

    if fig is None:
        fig = ppl.figure(figsize=(5.6, 4.5))

    ax = fig.subplots()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y")

    ax.tick_params(axis="both", labelsize=tick_label_size)
    ax.set_xlabel(f"Period ({time_unit})", fontsize=label_size)
    ax.set_ylabel("Power (wnp)", fontsize=label_size)

    ax.plot(periods, med, lw=2.5, color=color, label=label)
    ax.fill_between(periods, q1, q3, color=color, alpha=0.3)

    if label:
        ax.legend()

    fig.tight_layout()
    return ax
