import matplotlib.pyplot as ppl
import numpy as np
from numpy import pi, e, cos, sin, sqrt
import pandas as pd


import tfa_lib.wavelets as wl
import tfa_lib.plotting as pl

# globals 
#-----------------------------------------------------------
# default dictionary for ridge detection by annealing
ridge_def_dic = {'Temp_ini' : 0.2, 'Nsteps' : 25000,
                 'max_jump' : 3, 'curve_pen' : 0.2,
                 'sub_s' : 2, 'sub_t' : 2} 

#Significance levels from Torrence Compo 1998
xi2_95 = 5.99
xi2_99 = 9.21

# monkey patch tick and label sizes 
pl.label_size = 16.5
pl.tick_label_size = 14
#-----------------------------------------------------------


class TFAnalyzer:

    def __init__(self, periods, dt, T_cut_off,
                 p_max = None, time_label = 'a.u.'):


        '''
        Sets up an analyzer instance with the following parameters:

        periods   : sequence of periods to compute the Wavelet spectrum for, 
                    must have same units as dt!

        dt        : the sampling interval scaled to desired time units

        T_cut_off : Cut off period for the sinc-filter detrending, all periods
                    larger than that one are removed from the signal

        p_max      : Maximum power for z-axis colormap display, 
                    if *None* scales automatically

        time_label: the string label for the time unit 
        '''

        # sanitize periods
        if periods[0] < 2*dt:
            print('Warning, Nyquist limit is',2*dt,time_label,'!!')
            print(f'Setting lower period limit to {2*dt}')
            periods[0] = 2*dt
                  

        self.periods = np.linspace(periods[0], periods[-1], len(periods))
        self.dt = dt
        self.T_c = T_cut_off
        self.p_max = p_max

        self.time_unit = time_label

        self._has_spec = False
        self._has_ridge = False
        
        self.ax_spec = None
        self.wlet = None


    def compute_spectrum(self, raw_signal, detrend = True, Plot = True,  draw_coi = True):


        '''
        Computes the Wavelet spectrum for a given *signal* for the given *periods*
        
        signal  : a sequence, the time-series to be analyzed

        detrend : boolean, if True sinc-filter detrending will be done with the
                  set T_cut_off parameter
        
        Plot    : boolean, set to False if no plot is desired, 
                  good for for batch processing

        draw_coi: boolean, set to True if cone of influence shall be drawn on the
                  wavelet power spectrum
 
        returns:

        wlet : the Wavelet transform with dimensions len(periods) x len(signal) 
        
        '''
        

        
        if detrend:
            detrended = self.sinc_detrend(raw_signal)
            ana_signal = detrended
        else:
            ana_signal = raw_signal
            
        modulus, wlet = wl.compute_spectrum(ana_signal, self.dt, self.periods)

        if Plot:

            # 1st axis is for normalized signal, 2nd axis is the spectrum
            fig, axs = ppl.subplots(2, 1, figsize = (6.5,7),
                                    gridspec_kw = {'height_ratios':[1, 2.5]},
                                    sharex = True)

            tvec = np.arange(len(ana_signal)) * self.dt
                        
            pl.signal_modulus(axs, time_vector = tvec, signal = ana_signal,
                              modulus = modulus, periods = self.periods,
                              v_max = self.p_max)

            pl.format_signal_modulus(axs, self.time_unit)

            fig.tight_layout()
            self.ax_spec = axs[1]

            if draw_coi:
                coi_m = wl.Morlet_COI(self.periods)
                pl.draw_COI(axs[1], time_vector = tvec, coi_slope = coi_m)
            
        self.wlet = wlet
        self.modulus = modulus
        self._has_spec = True

    def get_maxRidge(self, Thresh = 0, smoothing = True, smooth_win_len = 17):

        '''
        Computes the ridge as consecutive maxima of the modulus.

        Returns the ridge_data dictionary (see wl.mk_ridge_data)!

        '''

        if not self._has_spec:
            print('Need to compute a wavelet spectrum first!')
            return

        # for easy integration
        modulus = self.modulus        

        Nt = modulus.shape[1] # number of time points
        tvec = np.arange(Nt) * self.dt

        #================ridge detection============================================

        # just pick the consecutive modulus
        # (squared complex wavelet transform) maxima as the ridge

        ridge_y = np.array( [np.argmax(modulus[:,t]) for t in np.arange(Nt)],
                            dtype = int)
        
        self._has_ridge = True
        rd = wl.make_ridge_data(ridge_y, modulus, self.wlet, self.periods,
                                tvec = tvec, Thresh = Thresh, smoothing = smoothing,
                                win_len = smooth_win_len)
        self.ridge_data = rd

        # return also directly
        return rd


    def get_annealRidge(self):

        ''' not working yet '''
        
        if not self._has_spec:
            print('Need to compute a wavelet spectrum first!')
            return

        ridge_y, cost = wl.find_ridge_anneal(self.modulus, y0, ini_T, Nsteps,
                                             mx_jump = max_jump, curve_pen = curve_pen)

        

    def draw_Ridge(self):

        if not self._has_ridge:
            print("Can't draw ridge, Need to a ridge detection first!")
            return

        pl.draw_Wavelet_ridge(self.ax_spec, self.ridge_data)
            
    
    def get_mean_spectrum(self):

        ''' Average over time '''

        if not self._has_spec:
            print('Need to compute a wavelet spectrum first!')
            return

        mfourier = np.sum(self.modulus,axis = 1)/len(self.signal)

        return mfourier

        
    def draw_AR1_confidence(self,alpha):

        if not self._has_spec:
            print('Need to compute a wavelet spectrum first!')
            return

        tvec = np.arange(self.wlet.shape[1]) * self.dt
        x,y = np.meshgrid(tvec, self.periods) # for plotting the wavelet transform
        
        ar1power = wl.ar1_powerspec(alpha,self.periods,self.dt)
        conf95 = xi2_95/2.
        conf99 = xi2_99/2.
            
        scaled_mod = np.zeros(self.modulus.shape)

        # maybe there is a more clever way
        for i,col in enumerate(self.modulus.T):
            scaled_mod[:,i] = col/ar1power
            
        CS = self.ax_spec.contour(x,y,scaled_mod,levels = [xi2_95/2.],linewidths = 1.,colors = '0.95', alpha = 0.7)
        CS = self.ax_spec.contour(x,y,scaled_mod,levels = [xi2_99/2.],linewidths = 1.,colors = 'orange', alpha = 0.7)


        # check confidence levels on (long) ar1 realisations !
        # print (len(where(scaled_mod > conf95)[0])/prod(wlet.shape)) # should be ~0.05
        # print (len(where(scaled_mod > conf99)[0])/prod(wlet.shape)) # should be ~0.01
        
    def save_ridge(self):

        if not self._has_ridge:
            print()
            print('No ridge analysis found, can not write any new results ..')
            print()
            return

        r = self.ridge_data['periods']
        t = self.ridge_data['time']
        power = self.ridge_data['power']
        phases = self.ridge_data['phase']
        inds = self.ridge_data['inds']
        index = np.arange(len(self.signal))
    
        s1 = pd.Series(data = r, index = index[inds])
        self.results['Periods ' + self.name] = s1 # add a column with the smoothed_maxpers to the dataframe
        s2 = pd.Series(data = power, index = index)
        
        self.results['RidgePower ' + self.name]= s2 # add a column with the ridge powers to the dataframe
        s3 = pd.Series(data = phases, index = index)
        self.results['Phases ' + self.name] = s3 # add a column with the smoothed_maxpers to the 

    def export_results(self,outname):

        if not self._has_results:
            print()
            print('No results to export yet..')
            print()
            return

        self.results.to_excel(outname+'.xlsx',index=False,header=True)
        print('Wrote {}.xlsx'.format(outname))

        
    def get_trend(self, signal):

        trend = wl.sinc_smooth(signal,self.T_c,self.dt)
        
        return trend

    def sinc_detrend(self, signal):
        
        trend = wl.sinc_smooth(signal, self.T_c, self.dt)
        
        detrended = signal - trend         

        # for easier interface return directly
        return detrended
