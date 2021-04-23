''' Methods to assess oscillatory readouts of an ensemble of signals '''

import numpy as np
import pandas as pd

from pyboat.core import find_COI_crossing, complex_average


def average_power_distribution(ridge_results, signal_ids = None, exclude_coi = False):

    '''
    Compute the power distribution of an ensemble
    of (ridge-)analyzed signals.

    Central measure is the signal-length (not ridge length!) averaged 
    ridge-power of individual signals.

    Parameters
    ----------

    ridge_results : sequence of DataFrames,
                   holds the ridge data for the individual signals.
                   Check pyboat.core.eval_ridge for details
                   about the DataFrame structure
    
    signal_ids : sequence, optional 
                 labels of the analyzed signals, if not given
                 a numeric sequence of len(ridge_results)
                 will be used as labels

    exclude_coi :  bool, if True only average the ridge outside of the COI,
                  for short noisy trajectories there might be no such points!

    Returns
    -------
    power_series : pandas Series with the signal_ids as index 
                   and averaged powers as values
    
    '''

    powers = []
    ids = []

    if signal_ids is None:
        signal_ids = np.arange( len(ridge_results) )

    assert len(signal_ids) == len(ridge_results)

    # collect the time-averaged powers
    # rd.Nt is the total signal length!
    for rd,_id in zip(ridge_results, signal_ids):

        if exclude_coi:
            # take only ridge power outside of COI
            i_left, i_right = find_COI_crossing(rd)
            mpower= (rd.power[i_left : i_right]).sum() / rd.Nt            
        else:
            mpower= rd.power.sum() / rd.Nt

        # can happen if ridge exclusively inside COI
        if not np.isnan(mpower):
            powers.append( mpower )
            ids.append(_id)
        
    powers_series = pd.Series(index = ids, data = powers)
        
    # sort by power, descending
    powers_series.sort_values(
        ascending = False,
        inplace = True)

    return powers_series


def get_ensemble_dynamics(ridge_results):

    '''
    Aggregate all the ridge-readouts (period, amplitude and phase)
    of a ridge-analyzed signal ensemble and return time-continuous median and
    quartiles (as in a time-continuous box plot).
    In the case of phase return the 1st order parameter 
    (length of resultant vector) 
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

    A tuple holding 4 data frames, one summary statistic over time
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
