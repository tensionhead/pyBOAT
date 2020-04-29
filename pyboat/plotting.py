import matplotlib.pyplot as ppl
import numpy as np
from numpy import pi

from pyboat.core import Morlet_COI, find_COI_crossing

# --- define colors ---
rgb_2mpl = lambda R, G, B: np.array((R, G, B)) / 255

SIG_COLOR = "darkslategray"
TREND_COLOR = rgb_2mpl(165, 105, 189)  # orchidy
ENVELOPE_COLOR = 'orange'

DETREND_COLOR = "black"
FOURIER_COLOR = "slategray"
RIDGE_COLOR = "crimson"

COI_COLOR = '0.6' # light gray

# the colormap for the wavelet spectra
CMAP = "YlGnBu_r"
# CMAP = 'cividis'
# CMAP = 'magma'

# --- define line widths ---
TREND_LW = 1.5
SIGNAL_LW = 1.5

# --- standard sizes ---
label_size = 20
tick_label_size = 18


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
        modulus[::-1], cmap=CMAP, vmax=p_max, extent=extent, aspect="auto"
    )

    mod_ax.set_ylim((periods[0], periods[-1]))
    mod_ax.set_xlim((time_vector[0], time_vector[-1]))
    mod_ax.grid(axis="y", color="0.6", lw=1.0, alpha=0.5)  # vertical grid lines

    min_power = modulus.min()
    if p_max is None:
        cb_ticks = [np.ceil(min_power), int(np.floor(modulus.max()))]
    else:
        cb_ticks = [np.ceil(min_power), p_max]

    cb = ppl.colorbar(
        im, ax=mod_ax, orientation="horizontal", fraction=0.08, shrink=0.6, pad=0.22
    )
    cb.set_ticks(cb_ticks)
    cb.ax.set_xticklabels(cb_ticks, fontsize=tick_label_size)
    # cb.set_label('$|\mathcal{W}_{\Psi}(t,T)|^2$',rotation = '0',labelpad = 5,fontsize = 15)
    cb.set_label("wavelet power", rotation="0", labelpad=-17, fontsize=0.9 * label_size)


def draw_Wavelet_ridge(ax, ridge_data, marker_size=1.5):

    """
    *ridge_data* comes from wavelets.eval_ridge !
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
    
    t_left, t_right = find_COI_crossing(ridge_data)
    
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
    ax1.plot(tvec[t_left:t_right], periods[t_left:t_right], color="cornflowerblue", alpha=0.8, lw=2.5, ls = lstyle, marker = mstyle, ms = 1.5)
    # inside COI
    ax1.plot(tvec[:t_left], periods[:t_left], '--',color="cornflowerblue", alpha=0.8, ms = 2.5)
    ax1.plot(tvec[t_right:], periods[t_right:], '--', color="cornflowerblue", alpha=0.8, ms = 2.5)
    
    ax1.set_ylabel(f"period ({time_unit})", fontsize=label_size)
    ax1.grid(True, axis="y")
    yl = ax1.get_ylim()
    ax1.set_ylim((max([0, 0.75 * yl[0]]), 1.25 * yl[1]))
    
    # only now draw the COI?
    if draw_coi:
        draw_COI(ax1, np.array(ridge_data.time), Morlet_COI())
    
    ax1.tick_params(axis="both", labelsize=tick_label_size)

    # ax1.set_ylim( (120,160) )

    # phases
    ax2.plot(tvec[t_left:t_right], phases[t_left:t_right], "-", c="crimson", alpha=0.8,
             ls = lstyle, marker = mstyle, ms = 1.5)

    # inside COI
    ax2.plot(tvec[:t_left], phases[:t_left], '-.',color='crimson', alpha=0.5, ms = 2.5)
    ax2.plot(tvec[t_right:], phases[t_right:], '-.', color='crimson', alpha=0.5, ms = 2.5)
    
    ax2.set_ylabel("phase (rad)", fontsize=label_size, labelpad=0.5)
    ax2.set_yticks((0, pi, 2 * pi))
    ax2.set_yticklabels(("$0$", "$\pi$", "$2\pi$"))
    ax2.tick_params(axis="both", labelsize=tick_label_size)

    # amplitudes
    ax3.plot(tvec[t_left:t_right], amplitudes[t_left:t_right], "k-", lw=2.5, alpha=0.9,
             ls = lstyle, marker = mstyle, ms = 1.5)

    # inside COI
    ax3.plot(tvec[:t_left], amplitudes[:t_left], '--',color='k', alpha=0.6, ms = 2.5)
    ax3.plot(tvec[t_right:], amplitudes[t_right:], '--', color='k', alpha=0.6, ms = 2.5)
    
    ax3.set_ylim((0, 1.1 * amplitudes.max()))
    ax3.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    ax3.yaxis.offsetText.set_fontsize(tick_label_size)
    ax3.set_ylabel("amplitude (a.u.)", fontsize=label_size)
    ax3.set_xlabel("time (" + time_unit + ")", fontsize=label_size)
    ax3.tick_params(axis="both", labelsize=tick_label_size)

    # powers
    ax4.plot(tvec[t_left:t_right], powers[t_left:t_right], "k-", lw=2.5, alpha=0.5,
             ls = lstyle, marker = mstyle, ms = 1.5)

    # inside COI
    ax4.plot(tvec[:t_left], powers[:t_left], '--',color='gray', alpha=0.6, ms = 2.5)
    ax4.plot(tvec[t_right:], powers[t_right:], '--', color='gray', alpha=0.6, ms = 2.5)
    
    ax4.set_ylim((0, 1.1 * powers.max()))
    ax4.set_ylabel("power (wnp)", fontsize=label_size)
    ax4.set_xlabel("time (" + time_unit + ")", fontsize=label_size)
    ax4.tick_params(axis="both", labelsize=tick_label_size)
