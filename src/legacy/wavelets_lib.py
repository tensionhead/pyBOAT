##################################################################################
# Tools for time-frequency analysis with Morlet Wavelets
# Inspired by 'A Practical Guide to Wavelet Analysis' from Torrence and Compo 1998
# and 'Identification of Chirps with Continuous Wavelet Transform' from Carmona,Hwang and Torresani 1995
#
# Version 0.3 May 2017, Gregor Moenke (gregor.moenke@embl.de)
##################################################################################


from __future__ import division,print_function
import os,sys
import matplotlib.pyplot as ppl
import numpy as np
from numpy.random import uniform,randn,randint,choice
from numpy import linspace, ones, zeros, arange, array, pi, sin, cos, argmax,var,nanargmax,nanmax,exp,log
from math import atan2
from os import path,walk
from scipy.optimize import leastsq
from scipy.signal import hilbert,cwt,ricker,lombscargle,welch,morlet, bartlett
import pandas as pd

from matplotlib import rc
rc('font', family='sans-serif', size = 14)
rc('lines', markeredgewidth = 0)
rc('text', usetex=False) # better for the UI

# global variables
#-----------------------------------------------------------
# thecmap = 'plasma' # the colormap for the wavelet spectra
thecmap = 'viridis' # the colormap for the wavelet spectra
omega0 = 2*pi # central frequency of the mother wavelet
ridge_def_dic = {'y0' : 0.5,'T_ini' : 0.1, 'Nsteps' : 15000, 'max_jump' : 3, 'curve_pen' : 0.2, 'sub_s' : 2, 'sub_t' : 2} # default dictionary for ridge detection by annealing
xi2_95 = 5.99
xi2_99 = 9.21
#-----------------------------------------------------------

def plot_signal_trend(signal,trend, dt, fig_num = 1, ptitle = None,time_label = 'min'):

    tvec = arange(0, len(signal) * dt, step = dt)

    fsize = (8,6)
    fig1 = ppl.figure(fig_num,figsize = fsize)
    ppl.clf()
    ax1 = ppl.gca()

    if ptitle:
        ax1.set_title(ptitle)
        
    ax1.plot(tvec,signal,lw = 1.5, color = 'royalblue',alpha = 0.8)

    # plot the trend
    ax1.plot(tvec,trend,color = 'orange',lw = 1.5) 

    # plot detrended signal
    ax2 = ax1.twinx()
    ax2.plot(tvec, signal - trend,'--', color = 'k',lw = 1.5) 
    ax2.set_ylabel('trend')
    
    ax1.set_xlabel('Time [' + time_label + ']')
    ax1.set_ylabel(r'signal') 
    ppl.ticklabel_format(style='sci',axis='y',scilimits=(0,0)) 
    fig1.subplots_adjust(bottom = 0.15,left = 0.15, right = 0.85)


def sinc_smooth(raw_signal,T_c,dt,M = None):

    ''' 
    Smoothing of a signal with the sinc-filter.

    T_c: cutoff period
    dt:  sampling interval
    M:   optional filter window length, defaults to signal lengths

    '''

    signal = array(raw_signal)
    dt = float(dt)

    # relative cut_off frequency
    f_c = dt/T_c

    if M is None:
        
        M = len(signal) - 1 # max for sharp roll-off

        # M needs to be even
        if M%2 != 0:
            M = M - 1

    w = sinc_filter(M, f_c)  # the evaluated windowed sinc filter
    sinc_smoothed = smooth(signal, data = w)

    return sinc_smoothed




#def compute_spectrum(signal, dt, periods, vmax = 20, Plot = True, time_label = 'min',fig_num = None, ptitle = None):
def compute_spectrum(signal, dt, periods):

        '''

        Computes the Wavelet spectrum for a given *signal* for the given *periods*
        
        signal  : a sequence
        the time-series to be analyzed, detrend beforehand!
        dt      : the sampling interval scaled to desired time units
        periods : the list of periods to compute the Wavelet spectrum for, 
              must have same units as dt!

        vmax       : Maximum power for z-axis colormap display, if *None* scales automatically
        
        Plot       : set to False if no plot is desired
        time_label : the label for the time unit when plotting
        fig_num    : the figure number when plotting, if *None* a new figure will be created

        returns:

        modulus : the Wavelet power spectrum normalized by signal variance
        wlet : the Wavelet transform with dimensions len(periods) x len(signal) 
        
        '''

        #if periods[0] < 2*dt:
        #    print()
        #    print('Warning, Nyquist limit is',2*dt,time_label,'!!')
        #    print()

        signal = array(signal)
        periods = array(periods)
        dt = float(dt)
        sfreq = 1/dt # the sampling frequency

        Nt = len(signal) # number of time points

        #--------------------------------------------
        scales = scales_from_periods(periods,sfreq,omega0)
        #--------------------------------------------

        #mx_per = 4*len(signal)/((omega0+sqrt(2+omega0**2))*sfreq)
        mx_per = dt*len(signal)
        if max(periods) > mx_per:

            print()
            print ('Warning: Very large periods chosen!')
            print ('Max. period should be <',rint(mx_per),time_label)
            print ('proceeding anyways...')

        Morlet = mk_Morlet(omega0)
        wlet = CWT(signal, Morlet, scales) #complex wavelet transform
        sig2 = np.var(signal) # white noise has then mean power of one
        modulus = np.abs(wlet)**2/sig2 # normalize with variance of signal

        return modulus, wlet
    
# TODO move to UI lib
def _plot_modulus(modulus,periods,dt,offset = 0,vmax = None, fig_num = None, ptitle = None,time_label = 'min'):

    tvec = arange(0,modulus.shape[1]*dt,dt) + offset
    
    x,y = np.meshgrid(tvec,periods) # for plotting the wavelet transform
    
    fsize = (8,7)
    fig1 = ppl.figure(fig_num,figsize = fsize)
    ppl.clf()
    ax1 = ppl.gca()

    #im = ax1.pcolor(x,y,modulus,cmap = thecmap,vmax = vmax)
    aspect = len(tvec)/len(periods)
    im = ax1.imshow(modulus[::-1],cmap = thecmap,vmax = vmax,extent = (tvec[0],tvec[-1],periods[0],periods[-1]),aspect = 'auto')
    ax1.set_ylim( (periods[0],periods[-1]) )
    ax1.set_xlim( (tvec[0],tvec[-1]) )
    if ptitle:
        ax1.set_title(ptitle)

 
    cb = ppl.colorbar(im,ax = ax1,orientation='horizontal',fraction = 0.08,shrink = 1.)
    cb.set_label('$|\mathcal{W}_{\Psi}(t,T)|^2$',rotation = '0',labelpad = 5,fontsize = 23)

    ax1.set_xlabel('Time [' + time_label + ']')
    ax1.set_ylabel('Period [' + time_label + ']')
    ppl.subplots_adjust(bottom = 0.11, right=0.95,left = 0.13,top = 0.95)

    return ax1

def get_maxRidge(modulus):

        '''

        returns: 

        ridge_y  : the y-coordinates of the ridge

        '''
        Nt = modulus.shape[1] # number of time points

        #================ridge detection============================================

        # just pick the consecutive modulus (squared complex wavelet transform) maxima as the ridge

        ridge_y = array( [argmax(modulus[:,t]) for t in arange(Nt)] ,dtype = int)

        return ridge_y
        


def mk_rdata(ridge_y, modulus, wlet, periods, tvec,Thresh = 0, smoothing = True, win_len = 17):
    
    '''
    
    Given the ridge coordinates, returns a dictionary containing:

    periods  : the instantaneous periods from the ridge detection    
    time     : the t-values of the ridge
    z        : the (complex) z-values of the Wavelet along the ridge
    phases   : the arg(z) values
    amplitudes : ...

    Moving average smoothing of the ridge supported.

    '''
    Nt = modulus.shape[1] # number of time points
    
    ridge_maxper = periods[ridge_y]
    ridge_z = wlet[ ridge_y, arange(Nt) ] # picking the right t-y values !

    ridge_power = modulus[ridge_y, arange(Nt)]

    inds = ridge_power > Thresh # boolean array of positions of significant oscillations
    sign_maxper = ridge_maxper[inds] # periods which cross the power threshold
    ridge_t = tvec[inds]
    ridge_phi = np.angle(ridge_z)[inds]

    if (sum(inds)) < 1: 
        print( 'Can not identify ridge, no significant oscillations found, check spectrum/threshold!')
        return None

    if smoothing is True:

        if (sum(inds)) < win_len: # ridge smoothing window len
            print( 'Can not identify ridge, no significant oscillations found, check spectrum/threshold!')
            return None

        sign_maxper = smooth(ridge_maxper,win_len)[inds] # smoothed maximum estimate of the whole ridge..

        
    ridge_data = {'periods' : sign_maxper, 'time' : ridge_t, 'z' : ridge_z, 'amplitudes' : ridge_power, 'inds' : inds, 'phase' : ridge_phi}

    MaxPowerPer=ridge_maxper[nanargmax(ridge_power)]  # period of highest power on ridge
        
    print('Period with max power of {:.2f} is {:.2f}'.format(nanmax(ridge_power),MaxPowerPer)) 
        
    return ridge_data

    
def find_ridge_anneal(landscape,y0,T_ini,Nsteps,mx_jump = 2,curve_pen = 0):

    ''' 
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

    '''

    print()
    print('started annealing..')
    
    incr = arange(-mx_jump,mx_jump+1) #possible jumps in scale direction
    incr = incr[incr!=0] #remove middle zero
    
    Nt = landscape.shape[-1] # number of time points
    Ns = landscape.shape[0] # number of scales
    t_inds = arange(Nt)
    ys = y0*ones(Nt,dtype = int) #initial ridge guess is straight line at scale landscape[y0]

    Nrej = 0

    T_k = T_ini/10. # for more natural units
    for k in range(Nsteps):
        
        F = cost_func_anneal(ys,t_inds,landscape,0,curve_pen)
        
        pos = randint(0,len(ys),size = 1) # choose time position to make random scale jump

        # dealing with the scale domain boundaries
        if ys[pos] >= Ns-mx_jump-1:
            eps = -1

        elif ys[pos] < mx_jump :
            eps = +1

        # jump!
        else:
            eps = choice(incr,size = 1)
            
        ys[pos] = ys[pos] + eps # the candidate
            
        F_c = cost_func_anneal(ys,t_inds,landscape,0,curve_pen)
        
        accept = True
        
        # a locally non-optimal move occured
        if F_c > F:
            u = uniform()
            
            # reject bad move? exp(-(F_c - F)/T_k) is (Boltzmann) probability for bad move to be accepted
            if u > exp(-(F_c - F)/T_k):
                accept = False

        if not accept:
            ys[pos] = ys[pos] - eps #revert the wiggle
            Nrej += 1

        if accept:
            Nrej = 0

                
        T_k = T_ini/log(2+k)/10. # update temperature

    print()
    print('annealing done!')
    print('final cost:',F_c)
    print('number of final still steps:',Nrej)
    return ys,F_c

def cost_func_anneal(ys,t_inds,landscape,l,m):

    '''
    Evaluates ridge candidate *ys* on *landscape* plus penalizing terms
    for 1st (*l*) and 2nd (*m*) derivative of the ridge curve.
    '''

    N = len(ys)
    D = -sum(landscape[ys,t_inds])
    S1 = l*sum(abs(np.diff(ys,1)))
    S2 = m*sum(abs(np.diff(ys,2)))

    #print D,S1,S2,D + S1 + S2
    
    return (D + S1 + S2)/N
    
#-------------------UI imported functions END-------------------

class TFAnalyser:

    def __init__(self,periods,dt, T_cut_off,vmax = 20, max_sig_len = None, offset = 0):

        self.periods = periods
        self.dt = dt
        self.vmax = vmax
        self.max_sig_len = max_sig_len
        self.offset = offset
        self.T_cut_off = T_cut_off

        self._has_spec = False
        self._has_dsignal = False
        self._has_signal = False        
        self._has_ridge = False
        self._has_results = False
        
        self.ax_spec = None
        self.signal = None
        self.dsignal = None
        self.name = ''

    def new_signal(self,raw_signal, name = ''):

        self.signal = raw_signal
        self.name = str(name)
        self.tvec = arange(0,len(raw_signal)*self.dt,self.dt) + self.offset

        # if self._has_results:

            # if len(self.results) != len(raw_signal):
            #     print()
            #     print( 'Warning, different input signal lengths encountered, saving results might not properly work!')
            #     print()
                       
        if not self._has_results:

            if self.max_sig_len:
                results=pd.DataFrame(index = range(self.max_sig_len)) # initialize DataFrame index
                tvec = arange(0,self.max_sig_len*self.dt,self.dt) + self.offset
                results.insert(0,'Time (min)',tvec)
            else:
                results=pd.DataFrame(index = range(len(raw_signal))) # initialize DataFrame index             
                results.insert(0,'Time (min)',self.tvec) # assign an extra column in the front

            self.results = results
            self._has_results = True
            
        self._has_spec = False
        self._has_dsignal = False
        self._has_ridge = False
        self.ridge_data = None
        self.ax_spec = None
        
        self._has_signal = True

    def compute_spectrum(self, raw_signal = None, Plot = True, time_label = 'min',fig_num = None, ptitle = None, detrend = True):

        if raw_signal is not None:
            self.new_signal(raw_signal)
        if detrend:
            self.sinc_detrend()
            signal = self.dsignal
        else:
            signal = self.signal
        
        # easy 
        dt = self.dt
        periods = self.periods
        vmax = self.vmax
        signal = self.dsignal
        '''

        Computes the Wavelet spectrum for a given *signal* for the given *periods*
        
        signal  : a sequence
        the time-series to be analyzed, detrend beforehand!
        dt      : the sampling interval scaled to desired time units
        periods : the list of periods to compute the Wavelet spectrum for, 
              must have same units as dt!

        vmax       : Maximum power for z-axis colormap display, if *None* scales automatically
        
        Plot       : set to False if no plot is desired
        time_label : the label for the time unit when plotting
        fig_num    : the figure number when plotting, if *None* a new figure will be created

        returns:

        wlet : the Wavelet transform with dimensions len(periods) x len(signal) 
        
        '''

        if periods[0] < 2*dt:
            print()
            print('Warning, Nyquist limit is',2*dt,time_label,'!!')
            print()

        signal = array(signal)
        periods = array(periods)
        dt = float(dt)
        sfreq = 1/dt # the sampling frequency

        Nt = len(signal) # number of time points

        #--------------------------------------------
        scales = scales_from_periods(periods,sfreq,omega0)
        #--------------------------------------------

        #mx_per = 4*len(signal)/((omega0+sqrt(2+omega0**2))*sfreq)
        mx_per = dt*len(signal)
        if max(periods) > mx_per:

            print()
            print ('Warning: Very large periods chosen!')
            print ('Max. period should be <',rint(mx_per),time_label)
            print ('proceeding anyways...')

        Morlet = mk_Morlet(omega0)
        wlet = CWT(signal, Morlet, scales) #complex wavelet transform
        sig2 = np.var(signal)
        modulus = np.abs(wlet)**2/sig2 # normalize with variance of signal

        if Plot:

            _plot_modulus(modulus,periods,dt,offset = self.offset,vmax = self.vmax,fig_num=fig_num,ptitle = ptitle, time_label = time_label)

        self.wlet = wlet
        self.modulus = modulus
        self.ax_spec = ppl.gca()
        self._has_spec = True

    def get_maxRidge(self, Thresh = 0, smoothing = True):

        if not self._has_spec:
            print('Need to compute a wavelet spectrum first!')
            return

        # for easy integration
        modulus = self.modulus
        wlet = self.wlet
        dt = self.dt
        periods = self.periods
        dsignal = self.dsignal
        tvec = self.tvec

        '''

        returns: 

        ridge_per  : the instantaneous periods from the ridge detection    
        ridge_t    : the t-values of the ridge
        ridge_z    : the (complex) z-values of the Wavelet along the ridge

        '''
        Nt = modulus.shape[1] # number of time points

        #================ridge detection============================================

        # just pick the consecutive modulus (squared complex wavelet transform) maxima as the ridge

        ridge_y = array( [argmax(modulus[:,t]) for t in arange(Nt)] ,dtype = int)
        ridge_maxper = periods[ridge_y]
        ridge_z = wlet[ ridge_y, arange(Nt) ] # picking the right t-y values !

        ridge_power = abs(ridge_z)**2/var(dsignal)

        inds = ridge_power > Thresh # boolean array of positions of significant oscillations
        sign_maxper = ridge_maxper[inds] # periods which cross the power threshold
        ridge_t = tvec[inds]

        if (sum(inds)) < 17: # ridge smoothing window len
            print( 'Can not identify ridge, no significant oscillations found, check spectrum/threshold!')
            return None
            
        if smoothing is True:

            sign_maxper = smooth(ridge_maxper,17)[inds] # smoothed maximum estimate of the whole ridge..

        
        ridge_data = {'ridge' : sign_maxper, 'time' : ridge_t, 'z' : ridge_z, 'power' : ridge_power, 'inds' : inds}

        self._has_ridge = True
        self.ridge_data = ridge_data

        MaxPowerPer=ridge_maxper[nanargmax(ridge_power)]  # period of highest power on ridge
        
        print('Period with max power of {:.2f} in sample {} is {:.2f}'.format(nanmax(ridge_power),self.name,MaxPowerPer)) 
        # self.maxPeriods.append(MaxPowerPer) # not implemented yet
        
        return ridge_data


    def draw_maxRidge(self, Thresh = 0, smoothing = True, color = 'orangered'):

        if not self._has_dsignal:
            print('Need to detrend an input signal first!')
            return

        if not self._has_spec:
            print('Need to compute a wavelet spectrum first!')
            return

        rdata = self.get_maxRidge(Thresh,smoothing)

        if rdata is None:
            return

        self.ax_spec.plot(rdata['time'],rdata['ridge'],'o',color = color,alpha = 0.5,ms = 5)


    def draw_AR1_confidence(self,alpha):

        if not self._has_spec:
            print('Need to compute a wavelet spectrum first!')
            return

        x,y = np.meshgrid(self.tvec,self.periods) # for plotting the wavelet transform
        
        ar1power = ar1_powerspec(alpha,self.periods,self.dt)
        conf95 = xi2_95/2.
        conf99 = xi2_99/2.
            
        scaled_mod = zeros(self.modulus.shape)

        # maybe there is a more clever way
        for i,col in enumerate(self.modulus.T):
            scaled_mod[:,i] = col/ar1power
            
        CS = self.ax_spec.contour(x,y,scaled_mod,levels = [xi2_95/2.],linewidths = 1.5,colors = '0.95')
        CS = self.ax_spec.contour(x,y,scaled_mod,levels = [xi2_99/2.],linewidths = 1.5,colors = 'orange')


        # check confidence levels on (long) ar1 realisations !
        # print (len(where(scaled_mod > conf95)[0])/prod(wlet.shape)) # should be ~0.05
        # print (len(where(scaled_mod > conf99)[0])/prod(wlet.shape)) # should be ~0.01
        
    def save_ridge(self):

        if not self._has_ridge:
            print()
            print('No ridge analysis found, can not write any new results ..')
            print()
            return

        r = self.ridge_data['ridge']
        t = self.ridge_data['time']
        power = self.ridge_data['power']
        inds = self.ridge_data['inds']
        index = arange(len(self.dsignal))
    
        s1 = pd.Series(data = r, index = index[inds])
        self.results['Periods ' + self.name] = s1 # add a column with the smoothed_maxpers to the dataframe
        s2 = pd.Series(data = power, index = index)
        
        self.results['RidgePower ' + self.name]= s2 # add a column with the ridge powers to the dataframe

    def export_results(self,outname):

        if not self._has_results:
            print()
            print('No results to export yet..')
            print()
            return

        self.results.to_excel(outname+'.xlsx',index=False,header=True)
        print('Wrote {}.xlsx'.format(outname))

        
    def get_trend(self):

        if not self._has_dsignal:
            print()
            print('No detrended signal..aborting!')
            print()
            return

        
        return self.trend

    def sinc_detrend(self, T_c = None):

        if not T_c:
            T_c = self.T_cut_off

        if not self._has_signal:
            print()
            print('No input signal..exiting!')
            print()
            return

        
        trend = sinc_smooth(self.signal,T_c,self.dt)
        detrended = self.signal - trend

        self._has_dsignal = True
        self.dsignal = detrended
        self.trend = trend

        return self.tvec, detrended

    def plot_signal(self,with_trend = True, fig_num = 1, ptitle = None,time_label = 'min'):
        
        if not self._has_signal:
            print()
            print('No input signal found..exiting!')
            print()
            return
        
        if with_trend:
            trend = self.get_trend()

        dt = self.dt
        
        tvec = self.tvec
        
        fsize = (8,6)
        fig1 = ppl.figure(fig_num,figsize = fsize)
        ppl.clf()
        ax1 = ppl.gca()

        if ptitle:
            ax1.set_title(ptitle)
        ax1.plot(tvec,self.signal,lw = 1.5, color = 'royalblue',alpha = 0.8)
        print(tvec.shape)
        print(trend.shape)
        ax1.plot(tvec,self.trend,color = 'orange',lw = 1.5) # plot the trend
        ax1.set_xlabel('Time [' + time_label + ']')
        ax1.set_ylabel(r'Intensity $\frac{I}{I_0}$') # some latex moves :)
        ppl.ticklabel_format(style='sci',axis='y',scilimits=(0,0)) 
        fig1.subplots_adjust(bottom = 0.11,left = 0.17)


    def plot_detrended(self, fig_num = 2, ptitle = None,time_label = 'min'):
            
        if not self._has_dsignal:
            print()
            print('No detrended signal found..exiting!')
            print()
            return

        dt = self.dt
        
        tvec = self.tvec
        
        fsize = (8,6)

        fig1 = ppl.figure(fig_num,figsize = fsize)
        ppl.clf()
        ax1 = ppl.gca()

        if ptitle:
            ax1.set_title(ptitle)

        if ptitle:
            ax1.set_title(ptitle)
            
        ax1.plot(tvec,self.dsignal,lw = 1.5, color = 'royalblue',alpha = 0.8)
        ax1.set_xlabel('Time [' + time_label + ']')
        ax1.set_ylabel(r'Intensity $\frac{I}{I_0}$') # some latex moves :)
        ppl.ticklabel_format(style='sci',axis='y',scilimits=(0,0))             
        fig1.subplots_adjust(bottom = 0.11,left = 0.17)



#--------------------the general plotting routines--------------------------------------

def Plot_signal(signal,dt,fig_num = None, time_label = 'min', fsize = (8,4)):
    
    tvec = arange(0,len(signal)*dt,dt)
    

    fig1 = ppl.figure(fig_num,figsize = fsize)
    ppl.clf()
    ax1 = ppl.gca()

    ax1.plot(tvec,signal,lw = 2., color = 'royalblue',alpha = 0.8)
    ax1.set_xlabel('Time [' + time_label + ']')
    ax1.set_ylabel('Signal')
    fig1.subplots_adjust(bottom = 0.2)

    return ax1

#------------------------------------------------------------------------------------------


def ar1_powerspec(alpha,periods,dt):
    res = (1-alpha**2)/(1+alpha**2 - 2*alpha*np.cos(2*pi*dt/periods))

    return res


# vectorial mean -> 2nd Order parameter
def mean_phase(phis):

    my = sum( [np.sin(phi) for phi in phis] )
    mx = sum( [np.cos(phi) for phi in phis] )

    return atan2(my,mx) # phi = tan(y/x)

# 1st order par
def order_par(thetas):
    N = len(thetas)
    x_tot = sum(np.cos(thetas))/N
    y_tot = sum(np.sin(thetas))/N
    
    return np.abs( array( (x_tot,y_tot) ) )


# difference of phases on the unit circle
def phase_diff(phi1,phi2):
    delta1 = 2*pi - phi1 # rotate reference frame
    sp2 = phi2 + delta1 

    return atan2(sin(sp2),cos(sp2))

#===============Filter===Detrending==================================

def smooth(x,window_len=11,window='bartlett',data = None):
    """smooth the data using a window with requested size.

    input:
    x: the input signal
    window_len: the dimension of the smoothing window; should be an odd integer
    window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
    flat window will produce a moving average smoothing.
    data: if not None, will be used as evaluated window!

    """

    x = array(x)

    # use externally derieved window evaluation
    if data is not None:
        window_len = len(data)
        window = 'extern'

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")


    if window_len<3:
        raise ValueError("window must not be shorter than 3")

    if window_len%2 is 0:
        raise ValueError("window_len should be odd")

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman','triang','extern']:
       raise ValueError("Window is none of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman','triang','extern'")

   
    s=np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
                                        #print(len(s))
    if window == 'flat': #moving average
        w=ones(window_len,'d')

    elif window == 'triang':
        w = triang(window_len)

    elif window == 'extern':
        w = data
        
    else:
        w=eval(window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    
    return y[int((window_len-1)/2):len(y)-int((window_len-1)/2)]


def sinc_filter(M, f_c = 0.2):

    ''' 
    Cutoff frequency f_c in sampling frequency unit, max 0.5!
    M is blackman window length and must be even, output length will be M+1.

    '''

    # not very effective, but should be get called only once per convolution

    assert M%2 == 0,'M must be even!'
    res = []

    for x in arange(0,M+1):
            
        if x == M/2:
            res.append(2*pi*f_c)
            continue
    
        r = np.sin(2*pi*f_c*(x - M/2))/( x - M/2 ) # the sinc filter unwindowed
        r = r * (0.42 - 0.5*np.cos(2*pi*x/M) + 0.08*np.cos(4*pi*x/M)) # blackman window
        res.append(r)

    res = array(res)
    res = res/sum(res)
            
    return res


def sinc_detrend(raw_signal,T_c,dt):

    signal = array(raw_signal)
    dt = float(dt)

    # relative cut_off frequency
    f_c = dt/T_c
    M = len(signal) - 1 # max for sharp roll-off

    # M needs to be even
    if M%2 != 0:
        M = M - 1

    w = sinc_filter(M, f_c)  # the evaluated windowed sinc filter
    sinc_smoothed = smooth(signal, data = w)
    sinc_detrended = signal - sinc_smoothed

    return sinc_detrended

def sinc_smooth(raw_signal,T_c,dt,M = None):

    signal = array(raw_signal)
    dt = float(dt)

    # relative cut_off frequency
    f_c = dt/T_c

    if M is None:
        
        M = len(signal) - 1 # max for sharp roll-off

        # M needs to be even
        if M%2 != 0:
            M = M - 1

    w = sinc_filter(M, f_c)  # the evaluated windowed sinc filter
    sinc_smoothed = smooth(signal, data = w)

    return sinc_smoothed


def detrend(raw_signal,winsize = 7,window = 'flat', data = None):

    avsignal = smooth(raw_signal,winsize,window = window, data = data) 
    dsignal = raw_signal - avsignal             # detrend by subtracting filter convolution

    return dsignal

#=============WAVELETS===============================================================

def scales_from_periods(periods,sfreq,omega0):
    scales = (omega0+np.sqrt(2+omega0**2))*periods*sfreq/(4*pi) #conversion from periods to morlet scales
    return scales

# is normed to have unit energy on all scales! ..to be used with CWT underneath
def mk_Morlet(omega0):

    def Morlet(t,scale):
        res = pi**(-0.25)*np.exp(omega0*1j*t/scale)*np.exp(-0.5*(t/scale)**2)
        return 1/np.sqrt(scale)*res
    
    return Morlet

# allows for complex wavelets, needs scales scaled with sampling freq!
def CWT(data,wavelet,scales):

    # test for complexity
    if np.iscomplexobj( wavelet(10,1) ):
        output = np.zeros([len(scales), len(data)],dtype = complex)
    else:
        output = np.zeros([len(scales), len(data)])

    vec = arange(-len(data)/2, len(data)/2) # we want to take always the maximum support available
    for ind, scale in enumerate(scales):
        wavelet_data = wavelet( vec, scale)
        output[ind, :] = np.convolve(data, wavelet_data,
                                  mode='same')
    return output

