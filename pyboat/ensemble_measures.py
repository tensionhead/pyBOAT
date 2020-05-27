''' Methods to assess oscillatory readouts of an ensemble of signals '''

import numpy as np
import pandas as pd

from pyboat.core import find_COI_crossing, complex_average

def average_power_distribution(ridge_results, signal_ids = None, exclude_coi = False):

    '''
    Compute the power distribution of an ensemble
    of (ridge-)analyzed signals.

    Central measure is the time-averaged ridge-power of individual
    signals.

    Parameters
    ----------

    ridge_results : sequence of DataFrames,
                   holds the ridge data for the individual signals.
                   Check pyboat.core.eval_ridge for details
                   about the DataFrame structure
    
    signal_ids : sequence, optional 
                 labels of the analyzed signals, of not given
                 a numeric sequence of len(ridge_results)
                 will be used as labels

    exclude_coi :  bool, if True only average the ridge outside of the COI,
                  for short noisy trajectories there might be no such points!
    '''

    powers = []

    if signal_ids is None:
        signal_ids = np.arange( len(ridge_results) )

    assert len(signal_ids) == len(ridge_results)

    # collect the time-averaged powers
    for rd in ridge_results:

        if exclude_coi:
            # take only ridge power outside of COI
            i_left, i_right = find_COI_crossing(rd)
            mpower= (rd.power[i_left : i_right]).mean()
        else:
            mpower= rd.power.mean()
        powers.append( mpower )
        
    powers_series = pd.Series(index = signal_ids, data = powers)
        
    # sort by power, descending
    powers_series.sort_values(
        ascending = False,
        inplace = True)

    return powers_series

def get_ensemble_dynamics(ridge_results):

    '''
    Aggregate all the ridge-readouts (period, amplitude and phase)
    of a ridge-analyzed signal ensemble and return time-continuous median and
    quartiles (as for a time-continuous box plot).
    In the case of phase return the 1st order parameter 
    as a phase coherence measure over time.

    Signals/ridge-readouts of unequal length in time are Ok! 

    Parameters
    ----------

    ridge_results : sequence (or iterable) of DataFrames
                    holding the ridge data for the individual signals
                    check pyboat.core.eval_ridge for details
                    about the DataFrame structure

    Returns
    --------

    A tuple holding 5 data frames, one summary statistic over time
    for period, amplitude, power and phase each.
    
    '''

    # aggregate the observables, missing values because of unequal length
    # get a NaN entry
    periods = pd.concat([r['periods'] for r in ridge_results], axis = 1)
    phases = pd.concat([r['phase'] for r in ridge_results], axis = 1)
    amplitudes = pd.concat([r['amplitude'] for r in ridge_results], axis = 1)
    powers = pd.concat([r['power'] for r in ridge_results], axis = 1)

    # median and the quantiles, NaNs get skipped over
    periods_mq1q3 = pd.DataFrame()
    periods_mq1q3['median'] = periods.median(axis = 1, skipna = True)
    periods_mq1q3['Q1'] = periods.quantile(q = 0.25, axis = 1)
    periods_mq1q3['Q3'] = periods.quantile(q = 0.75, axis = 1)
    
    amplitudes_mq1q3 = pd.DataFrame()
    amplitudes_mq1q3['median'] = amplitudes.median(axis = 1, skipna = True)
    amplitudes_mq1q3['Q1'] = amplitudes.quantile(q = 0.25, axis = 1)
    amplitudes_mq1q3['Q3'] = amplitudes.quantile(q = 0.75, axis = 1)

    powers_mq1q3 = pd.DataFrame()
    powers_mq1q3['median'] = powers.median(axis = 1, skipna = True)
    powers_mq1q3['Q1'] = powers.quantile(q = 0.25, axis = 1)
    powers_mq1q3['Q3'] = powers.quantile(q = 0.75, axis = 1)

    
    # 1st order parameter, NaNs of DataFrames get masked for numpy functions!
    R, Psi = complex_average(phases, axis = 1)
    phases_R = pd.DataFrame()
    phases_R['R'] = R
    
    return periods_mq1q3, amplitudes_mq1q3, powers_mq1q3, phases_R

if __name__ == '__main__':

    '''
    A short demonstration of the provided
    ensemble measures
    '''

    from pyboat import WAnalyzer, ssg
    from pyboat.plotting import plot_ensemble_dynamics, plot_power_distribution
    
    # set up analyzing instance
    periods = np.linspace(5, 60, 100)
    dt = 1
    wAn = WAnalyzer(periods, dt, T_cut_off = None)
        
    # create a bunch of chirp signals
    Nsignals = 50 # times 2
    T1 = 30 # initial period
    Nt = 500 # number of samples per signal
    signals = [
        ssg.create_noisy_chirp(T1 = T1, T2 = T, Nt = Nt, eps = 1)
        for T in np.linspace(T1, 50, Nsignals) ]

    # add the same amount of pure noise
    noisy_ones = [
        ssg.ar1_sim(alpha = 0.5, N = Nt)
        for i in range(Nsignals)]

    signals = signals + noisy_ones

    # get the the individual ridge readouts
    ridge_results = []
    for signal in signals:
        
        wAn.compute_spectrum(signal, sinc_detrend = False, do_plot = False)
        rd = wAn.get_maxRidge(smoothing_wsize = 11)
        ridge_results.append(rd)

    # the time-averaged power distribution
    powers_series = average_power_distribution( ridge_results, signal_ids = None )
    plot_power_distribution(powers_series)
    
    # keeping the pure noise signal out
    res = get_ensemble_dynamics(ridge_results[:Nsignals])
    plot_ensemble_dynamics(*res)
