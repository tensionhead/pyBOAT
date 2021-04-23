###########################################################################
# Tools for time-frequency analysis with Morlet Wavelets
# Inspired by 'A Practical Guide to Wavelet Analysis' from Torrence
# and Compo 1998
# and 'Identification of Chirps with Continuous Wavelet Transform'
# from Carmona,Hwang and Torresani 1995
###########################################################################

import numpy as np
from numpy.fft import rfft, rfftfreq
from numpy.random import uniform, randn, randint, choice
from numpy import pi
from scipy.signal import savgol_filter
import pandas as pd

# global variables
# -----------------------------------------------------------
omega0 = 2 * pi  # central frequency of the mother wavelet
chi2_95 = 5.99    # 0.05 confidence interval 
chi2_99 = 9.21    # 0.01 confidence interval 

# clip Wavelets at 1/peak_fraction envelope
peak_fraction = 1e8
clip_support = True

# max sinc length, increases performance for long signals
M_max = 2000
# -----------------------------------------------------------


def compute_spectrum(signal, dt, periods):

    """

    Computes the Wavelet spectrum for a given *signal* 
    for the given *periods*.

    Parameters
    ----------

    signal  : a sequence
              the time-series to be analyzed, 
              you might want to detrend beforehand!

    dt      : the sampling interval scaled to desired time units

    periods : the list of periods to compute the Wavelet spectrum for, 
              must have same units as dt!


    Returns
    -------

    modulus : 2d ndarray of reals, 
              the Wavelet power spectrum normalized by signal variance

    transform : 2d complex ndarray, 
           the Wavelet transform with dimensions len(periods) x len(signal) 

    """

    if periods[0] < 2 * dt:
        print()
        print(f"Warning, Nyquist limit is {2*dt:.2f}!!")
        print()

    # -- subtract the mean --
    signal = np.array(signal) - np.mean(signal)

    periods = np.array(periods)
    dt = float(dt)
    sfreq = 1 / dt  # the sampling frequency

    # --------------------------------------------
    scales = scales_from_periods(periods, sfreq, omega0)
    # --------------------------------------------

    # mx_per = 4*len(signal)/((omega0+sqrt(2+omega0**2))*sfreq)
    mx_per = dt * len(signal)
    if max(periods) > mx_per:

        print()
        print("Warning: Very large periods chosen!")
        print("Max. period should be <", np.rint(mx_per))
        print("proceeding anyways...")

    Morlet = mk_Morlet(omega0)
    transform = CWT(signal, Morlet, scales)  # complex wavelet transform
    sig2 = np.var(signal)  # white noise has then mean power of one
    modulus = np.abs(transform) ** 2 / sig2  # normalize with variance of signal

    return modulus, transform


def get_maxRidge_ys(modulus):

    """
    Just pick the time-consecutive modulus 
    (squared complex wavelet transform)
    maxima as the ridge.
    
    Parameters
    ----------

    modulus : 2d ndarray of reals, 
              the Wavelet power spectrum (periods x time) 
              normalized by signal variance

    Returns
    -------

    ridge_y  : the y-coordinates of the ridge

    """

    ridge_y = np.argmax(modulus, axis=0)

    return ridge_y


def eval_ridge(
        ridge_y,
        transform,
        signal,
        periods,
        tvec,
        power_thresh=0,
        smoothing_wsize=None
):

    """
    
    Given the ridge y-coordinates, evaluates the spectrum along  
    the time axis return the readout along the ridge.

    Parameters
    ----------

    ridge_y :  sequence of indices, has the length of the time axis
               of the spectrum, e.i. the y-coordinates of a ridge

    transform :     2d complex ndarray, holds the complex Wavelet transform 
               with dimensions len(periods) x len(signal)

    signal :   1d ndarray, the original signal to be analyzed, 
               only needed for variance calculation

    periods :  1d ndarray, the (central) periods of the Morlet wavelets
               used for the Wavelet transform
               
    tvec :     1d sequency holding the time axis of the signal

    power_thresh : float, minimal power a point (t, y) of a ridge must have
                   to be included in the result, this thresholds the ridge

    smoothing_wsize : int, optional, Savitkzy-Golay smoothing of the ridge


    Returns
    -------

    A DataFrame with the following columns:

    time      : the t-values of the ridge, can have gaps if thresholding!
    periods   : the instantaneous periods 
    frequencies : the instantaneous frequencies 
    phase    : the arg(z) values
    power     : the Wavelet Power normalized to white noise (<P(WN)> = 1)
    amplitude : the estimated amplitudes of the signal
    (z        : the (complex) z-values of the Wavelet along the ridge) 
                not attached as redundant

    Additional attributes:

    dt : the sampling interval
    Nt : the length of the original signal
    """

    sigma2 = np.var(signal)
    modulus = np.abs(transform) ** 2 / sigma2  # normalize with variance of signal

    # calculate here to minimize arguments needed
    dt = tvec[1] - tvec[0]

    Nt = modulus.shape[1]  # number of time points

    # for the DataFrame, to keep absolute time references    
    index = np.arange(Nt) 
    
    ridge_per = periods[ridge_y]
    ridge_z = transform[ridge_y, np.arange(Nt)]  # picking the right t-y values !

    ridge_power = modulus[ridge_y, np.arange(Nt)]

    inds = (
        ridge_power > power_thresh
    )  # boolean array of positions exceeding power threshold
    sign_per = ridge_per[inds]  # periods which cross the power threshold
    ridge_t = tvec[inds]
    ridge_phi = np.angle(ridge_z)[inds] % (2 * pi)  # map to [0,2pi]
    sign_power = ridge_power[inds]
    sign_amplitudes = power_to_amplitude(sign_per, sign_power, np.sqrt(sigma2), dt)

    # sign_z = ridge_z[inds] # not needed

    if smoothing_wsize is not None:
        Ntt = len(ridge_per)

        # sanitize ridge smoothing window size
        if Ntt < smoothing_wsize:
            # need an odd window size
            smoothing_wsize = Ntt if Ntt % 2 == 1 else Ntt - 1

        # print('inds:', inds)
        # smoothed maximum estimate of the whole ridge..
        sign_per = savgol_filter(ridge_per, smoothing_wsize, polyorder=3)[inds]

    # set truncated index to allow for
    # proper concatenation along time axis
    # of multiple ridge readouts (summary stats..)
    
    ridge_data = pd.DataFrame(
        columns=["time", "periods", "phase", "amplitude", "power", "frequencies"],
        index = index[inds] 
    )

    ridge_data["time"] = ridge_t
    ridge_data["phase"] = ridge_phi
    ridge_data["power"] = sign_power
    ridge_data["amplitude"] = sign_amplitudes
    ridge_data["periods"] = sign_per
    ridge_data["frequencies"] = 1 / sign_per

    # attach additional data for
    # internal use only
    ridge_data.dt = dt
    ridge_data.Nt = Nt
    
    MaxPowerPer = ridge_per[
        np.nanargmax(ridge_power)
    ]  # period of highest power on ridge

    return ridge_data


# --- confidence intervals ---

def get_significant_regions(modulus,
                            empirical_background,
                            confidence = chi2_95):

    '''
    Given an (empirical) background Fourier power spectrum,
    returns a boolean mask indicating areas of significant 
    oscillations within the wavelet power spectrum for 
    the desired confidence interval.

    Parameters
    ----------

    modulus : 2d ndarray of reals, 
              the wavelet power spectrum normalized by signal variance

    empirical_background: 1d sequence, Fourier estimate of the background. 
                          Must hold the powers at exactly the periods used for
                          the wavelet analysis!

    confidence : float, the Chi-squared value at the desired confidence level.
                 Defaults to the 95% confidence interval.

    Returns
    -------

    sign_regions : 2d boolean ndarray with the shape of *modulus*,
                   is True for every significant point of the
                   power spectrum

    '''
        
    # every period needs a background power value
    # (constant 1 for white noise for examle)
    if not len(empirical_background) == modulus.shape[0]:
        raise ValueError("Empirical background doesn't fit"
                         " to wavelet spectrum!")

    # remove DOF factor from chi2
    conf = confidence / 2.0

    # rescale every column along the period axis
    # with the assumed 0-model spectrum    
    scaled_mod = np.zeros(modulus.shape)
    for i, col in enumerate(modulus.T):
        scaled_mod[:, i] = col / empirical_background

    # return boolean mask of the power spectrum
    # indicating significant oscillations
    sign_regions = scaled_mod > conf
    
    return sign_regions


# --- COI business ---


def get_COI_branches(time_vector):

    """
    Get the left and right branches of the COI

    Parameters
    ----------

    time_vector : ndarray, the complete time vector along the signal
    """

    coi_slope = Morlet_COI()
    
    N_2 = int(len(time_vector) / 2.0)
    
    # ascending left side
    left = coi_slope * time_vector[: N_2]
    left_t = time_vector[: N_2]    

    # descending right side
    right = - coi_slope * (time_vector[N_2:] -  time_vector[-1])
    right_t = time_vector[N_2:]

    return np.r_[left, right], np.r_[left_t, right_t]


def find_COI_crossing(rd):

    """
    checks for first/last time point
    which is outside the COI on the
    left/right boundary of the spectrum.

    Returns indices!

    Parameters
    ----------

    rd : pandas.DataFrame
        the ridge data from eval_ridge()
    """

    # restore the original complete time vector
    tvec = np.arange(rd.Nt) * rd.dt
    coi, coi_t = get_COI_branches(tvec)
    
    N2 = len(tvec) // 2
        
    left_inds = np.intersect1d(np.arange(N2), rd.index)

    # first time point outside left COI
    coi_inds = left_inds[coi[left_inds] > rd.periods[left_inds]]
    # left ridge might be entirely outside COI
    i_left = coi_inds[0] if coi_inds.size > 0 else 0

    right_inds = np.intersect1d(N2 + np.arange(N2), rd.index)
    
    # last time point outside right COI
    coi_inds = right_inds[coi[right_inds] > rd.periods[right_inds]]
    # right ridge might be entirely outside COI
    i_right = coi_inds[-1] if coi_inds.size > 0 else -1

    return i_left, i_right


# ============ Snake Annealing =====================================


def find_ridge_anneal(landscape, y0, T_ini, Nsteps, mx_jump=2, curve_pen=0):

    """ 
    Taking an initial straight line guess at *y0* finds a ridge in *landscape* which 
    minimizes the cost_func_anneal by the simulated annealing method.

    landscape - time x scales signal representation (modulus of Wavelet transform)
    y0        - initial ridge guess is straight line at scale landscape[y0] 
                -> best to set it close to a peak in the Wavelet modulus (*landscape*)
    T_ini     - initial value of the temperature for the annealing method
    Nsteps    - Max. number of steps for the algorithm
    mx_jump   - Max. distance in scale direction covered by the random steps
    curve_pen - Penalty weight for the 2nd derivative of the ridge to estimate -> 
                high values lead to  less curvy ridges

    """

    print()
    print("started annealing..")

    incr = np.arange(-mx_jump, mx_jump + 1)  # possible jumps in scale direction
    incr = incr[incr != 0]  # remove middle zero

    Nt = landscape.shape[-1]  # number of time points
    Ns = landscape.shape[0]  # number of scales
    t_inds = np.arange(Nt)
    ys = y0 * np.ones(
        Nt, dtype=int
    )  # initial ridge guess is straight line at scale landscape[y0]

    Nrej = 0

    tfac = 0.01  # still arbitrary :/
    T_ini = T_ini * tfac
    curve_pen = curve_pen * tfac

    T_k = T_ini  # for more natural units ->  0 < T_ini < 100 should be ok

    for k in range(Nsteps):

        F = cost_func_anneal(ys, t_inds, landscape, 0, curve_pen)

        pos = randint(
            0, len(ys), size=1
        )  # choose time position to make random scale jump

        # dealing with the scale domain boundaries
        if ys[pos] >= Ns - mx_jump - 1:
            eps = -1

        elif ys[pos] < mx_jump:
            eps = +1

        # jump!
        else:
            eps = choice(incr, size=1)

        ys[pos] = ys[pos] + eps  # the candidate

        F_c = cost_func_anneal(ys, t_inds, landscape, 0, curve_pen)

        accept = True

        # a locally non-optimal move occured
        if F_c > F:
            u = uniform()

            # reject bad move? exp(-(F_c - F)/T_k) is (Boltzmann) probability for bad move to be accepted
            if u > np.exp(-(F_c - F) / T_k):
                accept = False

        if not accept:
            ys[pos] = ys[pos] - eps  # revert the wiggle
            Nrej += 1

        if accept:
            Nrej = 0

        print(T_k)
        T_k = T_ini / np.log(2 + k)  # update temperature

    print()
    print("annealing done!")
    print("final cost:", F_c)
    print("number of final still steps:", Nrej)
    print("final temperature:", T_k * tfac)
    return ys, F_c


def cost_func_anneal(ys, t_inds, landscape, l, m):

    """
    Evaluates ridge candidate *ys* on *landscape* plus penalizing terms
    for 1st (*l*) and 2nd (*m*) derivative of the ridge curve.
    """

    N = len(ys)
    D = -sum(landscape[ys, t_inds])
    S1 = l * sum(abs(np.diff(ys, 1)))
    S2 = m * sum(abs(np.diff(ys, 2)))

    # print D,S1,S2,D + S1 + S2

    return (D + S1 + S2) / N


# =============== Filters +  Detrending =================================


def smooth(x, window_len=11, window="flat", data=None):
    """
    smooth the data using a window with requested size.

    input:
    x: the input signal
    window_len: the dimension of the smoothing window; should be an odd integer
    window: the type of window from 'flat' or 'extern'

    flat window will produce a moving average smoothing.
    data: if not None, will be used as evaluated window!

    """

    x = np.array(x)

    # use externally derieved window evaluation
    if data is not None:
        window_len = len(data)
        window = "extern"

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        raise ValueError("window must not be shorter than 3")

    if window_len % 2 == 0:
        raise ValueError("window_len should be odd")

    if window not in ["flat", "extern"]:
        raise ValueError("Window is none of 'flat' or 'extern'")

    s = np.r_[x[window_len - 1 : 0 : -1], x, x[-1:-window_len:-1]]
    # print(len(s))
    if window == "flat":  # moving average
        w = np.ones(window_len, "d")

    elif window == "extern":
        w = data

    else:
        w = eval(window + "(window_len)")

    y = np.convolve(w / w.sum(), s, mode="valid")

    return y[int((window_len - 1) / 2) : len(y) - int((window_len - 1) / 2)]


def sinc_filter(M, f_c=0.2):

    """ 
    Cutoff frequency f_c in sampling frequency unit, max 0.5!
    M is blackman window length and must be even, output length will be M+1.

    """

    # not very effective, but should be gets called only once per convolution

    # limit the filter's maximal size
    if M > M_max:
        M = M_max
    
    assert M % 2 == 0, "M must be even!"
    res = []

    for x in np.arange(0, M + 1):

        # this is the non-vectorizable part
        if x == M / 2:
            res.append(2 * pi * f_c)
            continue

        r = np.sin(2 * pi * f_c * (x - M / 2)) / (
            x - M / 2
        )  # the sinc filter unwindowed
        r = r * (
            0.42 - 0.5 * np.cos(2 * pi * x / M) + 0.08 * np.cos(4 * pi * x / M)
        )  # blackman window
        res.append(r)

    res = np.array(res)
    res = res / np.sum(res)

    return res


def sinc_smooth(raw_signal, T_c, dt, M=None):

    """
    Convolve the signal with a sinc filter
    of cut-off period *T_c*.

    Length of the filter controlled by
    M, defaults to length of the raw_signal
    """

    signal = np.array(raw_signal)
    dt = float(dt)

    # relative cut_off frequency
    f_c = dt / T_c

    if M is None:

        M = len(signal) - 1  # max for sharp roll-off

        # M needs to be even
        if M % 2 != 0:
            M = M - 1

    w = sinc_filter(M, f_c)  # the evaluated windowed sinc filter
    sinc_smoothed = smooth(signal, data=w)

    return sinc_smoothed


def sliding_window_amplitude(signal, window_size, dt, SGsmooth=True):

    """
    Max - Min sliding window operation
    to estimate amplitude envelope.

    free boundaries -> half the window_size + 1
    at 1st and last entry

    optional:
    Savitzky-Golay smoothing of the
    envelope with the same window size

    Parameters
    ----------

    signal : ndarray, the (detrended) signal
    window_size : int, the window size in time units
    dt : float, the sampling interval 
    """

    # get the underlying array
    vector = np.array(signal)

    if window_size > (len(signal) - 1) * dt:
        window_size = (len(signal) - 1) * dt
        print(f'Warning, setting window_size to {window_size}!')
    
    # window size in sampling interval units
    window_size = int(window_size / dt)
        
    # has to be odd
    if window_size % 2 != 1:
        window_size = window_size + 1

    # rolling matrix
    # Stack Overflow move
    shape = vector.shape[:-1] + (vector.shape[-1] - window_size + 1, window_size)
    strides = vector.strides + (vector.strides[-1],)
    aa = np.lib.stride_tricks.as_strided(vector, shape=shape, strides=strides)

    # treat the boundaries, fill with NaNs
    b_size = window_size // 2
    b_left = np.zeros((b_size, aa.shape[1]), dtype=float) * np.nan
    b_right = np.zeros((b_size, aa.shape[1]), dtype=float) * np.nan

    for i in range(b_size):
        b_left[i, : b_size + i + 1] = vector[: b_size + 1 + i]
        b_right[-(i + 1), -(b_size + 1 + i) :] = vector[-(b_size + 1 + i) :]

    rm = np.vstack([b_left, aa, b_right])

    # max-min/2 in sliding window
    amplitudes = (np.nanmax(rm, axis=1) - np.nanmin(rm, axis=1)) / 2

    if SGsmooth:
        amplitudes = savgol_filter(amplitudes, window_length=window_size, polyorder=3)

    return amplitudes


def normalize_with_envelope(dsignal, window_size, dt):

    """
    Best to use detrending beforehand!

    Mean subtraction is still always performed.

    Does NOT check for zero-divide errors,
    arrising in cases where the signal is constant.

    Parameters
    ----------

    dsignal : ndarray, the (detrended) signal
    window_size : int, the window size in time units, e.g. 17 minutes
    dt : float, the sampling interval 
    """

    # mean subtraction
    signal = dsignal - dsignal.mean()

    if window_size > (len(signal) - 1) * dt:
        window_size = (len(signal) - 1) * dt
        print(f'Warning, setting window_size to {window_size}!')
              
    # ampl. normalization
    env = sliding_window_amplitude(signal, window_size, dt)
    ANsignal = 1 / env * dsignal

    return ANsignal


# =============WAVELETS=====================================================


def scales_from_periods(periods, sfreq, omega0=2 * pi):
    # conversion from periods to morlet scales
    # strictly admissable version from Torrence-Compo
    scales = (omega0 + np.sqrt(2 + omega0 ** 2)) * periods * sfreq / (4 * pi)
    return scales


def scales_from_periodsNA(periods, sfreq, omega0=2 * pi):
    # conversion from periods to morlet scales
    # fits the non-admissable Morlet definition!
    scales = omega0 * periods * sfreq / (2 * pi)
    return scales


# is normed to have unit energy on all scales! ..to be used with CWT underneath
def mk_Morlet(omega0):
    def Morlet(t, scale):
        res = (
            pi ** (-0.25)
            * np.exp(omega0 * 1j * t / scale)
            * np.exp(-0.5 * (t / scale) ** 2)
        )
        return 1 / np.sqrt(scale) * res

    return Morlet


def gauss_envelope(t, scale):

    """
    The gaussian envelope of the Morlet Wavelet
    """

    return 1 / np.sqrt(scale) * pi ** (-0.25) * np.exp(-0.5 * (t / scale) ** 2)


def inverse_gauss(y, scale):

    """
    Invert the right Gaussian branch to
    determine Morlet support cut off 
    """

    a = pi ** (-0.25) * 1 / np.sqrt(scale)
    return scale * np.sqrt(-2 * np.log(y / a))


# allows for complex wavelets, needs scales scaled with sampling freq!
def CWT(signal, wavelet, scales, clip_support=clip_support):

    # test for complexity
    if np.iscomplexobj(wavelet(10, 1)):
        output = np.zeros([len(scales), len(signal)], dtype=complex)
    else:
        output = np.zeros([len(scales), len(signal)])

    # we want to take always the maximum support available
    # .. no we don't -> performance, otherwise convolutions scale with N * N!!
    # vec = np.arange(-len(signal)/2, len(signal)/2) # old default

    for ind, scale in enumerate(scales):

        # Morlet main peak value:
        y0 = gauss_envelope(0, scale)

        if clip_support:
            # support cut off at 1/peak_fraction of Morlet peak
            x_max = int(inverse_gauss(y0 / peak_fraction, scale))
        else:
            x_max = len(signal) / 2

        # print(ind, scale, 2*x_max)

        # max support is length of signal
        if 2 * x_max > len(signal):
            vec = np.arange(-len(signal) / 2, len(signal) / 2)

        else:
            vec = np.arange(-x_max, x_max)

        wavelet_data = wavelet(vec, scale)
        output[ind, :] = np.convolve(signal, wavelet_data, mode="same")
    return output


def Morlet_COI(omega0=omega0):
    # slope of Morlet e-folding time in tau-periods (spectral) view
    m = 4 * pi / (np.sqrt(2) * (omega0 + np.sqrt(2 + omega0 ** 2)))
    return m


def power_to_amplitude(periods, powers, signal_std, dt):

    """
    Rescale Morlet wavelet powers according to the
    definition of 

    "On the Analytic Wavelet Transform",
    Jonathan M. Lilly 2010

    to get an amplitude estimate.
    """

    scales = scales_from_periods(periods, 1 / dt)

    kappa = 1 / np.sqrt(scales) * pi ** -0.25
    kappa = kappa * signal_std * np.sqrt(2)

    return np.sqrt(powers) * kappa


# ===== Fourier FFT Spectrum ================================================


def compute_fourier(signal, dt):

    N = len(signal)

    df = 1.0 / (N * dt)  # frequency bins

    rf = rfft(signal, norm="ortho")  # positive frequencies
    # use numpy routine for sampling frequencies
    fft_freqs = rfftfreq(len(signal), d=dt)

    # print(N,dt,df)
    # print(len(fft_freqs),len(rf))

    fpower = np.abs(rf) ** 2 / np.var(signal)

    return fft_freqs, fpower


# ============== AR1 spectrum and simulation =============


def ar1_powerspec(alpha, periods, dt):
    res = (1 - alpha ** 2) / (
        1 + alpha ** 2 - 2 * alpha * np.cos(2 * pi * dt / periods)
    )

    return res


def ar1_sim(alpha, N, sigma=1, x0=None):

    N = int(N)
    sol = np.zeros(N)

    if x0 is None:
        x0 = randn()

    sol[0] = x0

    for i in range(1, N):
        sol[i] = alpha * sol[i - 1] + sigma * randn()

    return sol


# ========= Utility functions ==============


def complex_average(phis, axis=0):

    """
    Vectorial/directional average on complex plane
    
    Parameters
    ----------
    phis: ndarray, holding phase values in rad
    axis: int, axis along which to perform the averaging

    """

    Z = np.sum(np.e ** (phis * 1j), axis=axis)

    # normalization
    Z = Z / phis.shape[axis]

    # 1st Order parameter - circular std
    R = np.abs(Z)

    # mean phase
    Psi = np.angle(Z)

    return R, Psi


# =============== NaN - Missing Values interpolation ====================


def interpolate_NaNs(y):

    """
    linearly interpolates through NaNs
    in a sequence of scalars
    """

    bool_ar = np.isnan(y)
    indexer = lambda z: np.nonzero(z)[0]

    # the interpolated values
    interp_yp = np.interp(indexer(bool_ar), indexer(~bool_ar), y[~bool_ar])

    yy = y.copy()

    # replace NaNs with interpolated values
    yy[bool_ar] = interp_yp

    return yy
