import matplotlib.pyplot as ppl
import numpy as np
from numpy import pi

# --- define colors ---
SIG_COLOR = 'royalblue'
TREND_COLOR = 'orange'
DETREND_COLOR = 'black'
FOURIER_COLOR = 'slategray'
RIDGE_COLOR = 'red'

# --- label sizes ---
tick_label_size = 10
label_size = 12


# --- Signal and Trend -----------------------------------------------

def format_signal_ax(ax, time_unit):
        
    ax.set_xlabel('time (' + time_unit + ')', fontsize = label_size) 
    ax.set_ylabel(r'signal', fontsize = label_size) 
    ax.ticklabel_format(style='sci',axis='y',scilimits=(0,0))
    ax.tick_params(axis = 'both',labelsize = tick_label_size)

def draw_signal(ax, time_vector, signal):    
    ax.plot(time_vector, signal, lw = 1.5,
            color = SIG_COLOR,alpha = 0.8)

def draw_trend(ax,  time_vector, trend):
    ax.plot(time_vector, trend,
            color = TREND_COLOR, lw = 1.5)

def draw_detrended(ax, time_vector, detrended):

    ax2 = ax.twinx()
    ax2.plot(time_vector, detrended,'-',
             color = DETREND_COLOR,lw = 1.5, alpha = 0.6) 

    ax2.set_ylabel('detrended', fontsize = label_size)
    ax2.ticklabel_format(style='sci',axis='y',scilimits=(0,0))
        
    return ax2

# --- Fourier Spectrum ------------------------------------------------

def format_Fourier_ax(ax, time_unit, show_periods = False):

    if show_periods:
        ax.set_xlabel('Periods (' + time_unit + ')',
                      fontsize = label_size)

    else:
        ax.set_xlabel('Frequency (' + time_unit + r'$^{-1}$)',
                      fontsize = label_size)

    ax.set_ylabel('Fourier power', fontsize = label_size)
    ax.ticklabel_format(style='sci',axis='y',scilimits=(0,0))
    ax.tick_params(axis = 'both',labelsize = tick_label_size)
        
           
def Fourier_spec(ax, fft_freqs, fft_power, show_periods = False):
    
    if show_periods:
        
        # period view, omit the last bin 2/(N*dt)

        # skip 0-frequency 
        ax.vlines(1/fft_freqs[1:],0,
                        fft_power[1:],lw = 1.8,
                        alpha = 0.7, color = FOURIER_COLOR)
        ax.set_xscale('log')

    else:
        # frequency view
        ax.vlines(fft_freqs,0,
                        fft_power,lw = 1.8,
                        alpha = 0.7, color = FOURIER_COLOR)
        

# --- Wavelet spectrum  ---


def format_signal_modulus(axs, time_unit):
    
    sig_ax = axs[0]
    mod_ax = axs[1]
    
    mod_ax.set_xlabel('time (' + time_unit + ')', fontsize = label_size) 
    mod_ax.set_ylabel('period (' + time_unit + ')', fontsize = label_size)
    mod_ax.tick_params(axis = 'both',labelsize = tick_label_size)


def signal_modulus(axs, time_vector, signal, modulus, periods, v_max):

    '''
    axs[0] is signal axis, axs[1] spectrum axis
    '''

    sig_ax = axs[0]
    mod_ax = axs[1]

    # Plot Signal above spectrum

    sig_ax.plot(time_vector, signal, color = 'black', lw = .75, alpha = 0.7)
    sig_ax.plot(time_vector, signal, '.', color = 'black', ms = 3., alpha = 0.7)
    sig_ax.set_ylabel('signal [a.u.]', fontsize = label_size) 
    sig_ax.ticklabel_format(style='sci',axis='y',scilimits=(0,0))
    sig_ax.tick_params(axis = 'both',labelsize = tick_label_size)

    # Plot Wavelet Power Spectrum

    im = mod_ax.imshow(modulus[::-1], cmap = 'viridis', vmax = v_max,
                       extent = (time_vector[0], time_vector[-1],periods[0],periods[-1]),
                       aspect = 'auto')
    
    mod_ax.set_ylim( (periods[0],periods[-1]) )
    mod_ax.set_xlim( (time_vector[0], time_vector[-1]) )
    mod_ax.grid(axis = 'y',color = '0.6', lw = 1., alpha = 0.5) # vertical grid lines

    min_power = modulus.min()
    cb_ticks = [np.ceil(min_power),v_max]
    cb = ppl.colorbar(im,ax = mod_ax,orientation='horizontal',fraction = 0.08,shrink = .6, pad = 0.25)
    cb.set_ticks(cb_ticks)
    cb.ax.set_xticklabels(cb_ticks, fontsize=tick_label_size)
    #cb.set_label('$|\mathcal{W}_{\Psi}(t,T)|^2$',rotation = '0',labelpad = 5,fontsize = 15)
    cb.set_label('Wavelet power',rotation = '0',labelpad = -5,fontsize = 10)

def draw_Wavelet_ridge(ax, ridge_data, marker_size = 2):

    '''
    *ridge_data* comes from wavelets.make_ridge_data !
    '''
    
    ax.plot(ridge_data['time'], ridge_data['periods'],'o',
            color = RIDGE_COLOR,alpha = 0.6,ms = marker_size)


def draw_COI(ax, time_vector, coi_slope, alpha = 0.3):

    '''
    Cone of influence, period version
    '''
    
    # Plot the COI
    N_2 = int(len(time_vector) / 2.)

    # ascending left side
    ax.plot(time_vector[:N_2+1], coi_slope * time_vector[:N_2+1],
            'k-.', alpha = alpha)
    
    # descending right side
    ax.plot(time_vector[N_2:], coi_slope * (time_vector[-1]- time_vector[N_2:]),
            'k-.',alpha = alpha)

# --- Wavelet readout ----------------------------


def plot_readout(axs, ridge_data, time_unit):

    '''
    *axs* has three axes: period, phase and power
    '''

    ps = ridge_data['periods']
    phases = ridge_data['phase']
    powers = ridge_data['power']
    tvec = ridge_data['time']

    ax1 = axs[0]
    ax2 = axs[1]
    ax3 = axs[2]

    ax1.plot(tvec,ps, alpha = 0.8)
    ax1.set_ylabel(f'period ({time_unit})', fontsize = label_size)
    ax1.grid(True,axis = 'y')
    yl = ax1.get_ylim()
    ax1.set_ylim( ( max([0,0.75*yl[0]]), 1.25*yl[1] ) )
    # ax1.set_ylim( (120,160) )

    ax2.plot(tvec,phases,'-', c = 'crimson', alpha = 0.8)
    ax2.set_ylabel('phase (rad)', fontsize = label_size)
    ax2.set_yticks( (-pi,0,pi) )
    ax2.set_yticklabels( ('$-\pi$','$0$','$\pi$') )
    ax2.tick_params(axis = 'both',labelsize = tick_label_size)

    ax3.plot(tvec,powers,'k-',lw = 2.5, alpha = 0.5)
    ax3.set_ylim( (0,1.1*powers.max()) )
    ax3.set_ylabel('power', fontsize = label_size)
    ax3.set_xlabel('time (' + time_unit + ')', fontsize = label_size)
    ax3.tick_params(axis = 'both',labelsize = tick_label_size)

    ppl.subplots_adjust(top = 0.98, left = 0.18)
