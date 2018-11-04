#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
from PyQt5.QtWidgets import QCheckBox, QTableView, QComboBox, QFileDialog, QAction, QMainWindow, QApplication, QLabel, QLineEdit, QPushButton, QMessageBox, QSizePolicy, QWidget, QVBoxLayout, QHBoxLayout, QDialog, QGroupBox, QFormLayout, QGridLayout


from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

#from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtCore import pyqtSlot, pyqtSignal

import wavelets_lib as wl
from helper.pandasTable import PandasModel
import random
import numpy as np
import pandas as pd

def synth_signal1(T, amp, per, sigma, slope):  
    
    tvec = np.arange(T)
    trend = slope*tvec**2/tvec[-1]**2*amp
    noise = np.random.normal(0,sigma, len(tvec))
    sin = amp*np.sin(2*np.pi/per*tvec)+noise+trend

    return tvec, sin
    
def plot_signal_trend(signal,trend, dt, fig_num = 1, ptitle = None,time_label = 'min'):

    tvec = np.arange(0, len(signal) * dt, step = dt)

    fsize = (8,6)
    fig1 = plt.figure(fig_num,figsize = fsize)
    plt.clf()
    ax1 = plt.gca()

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
    plt.ticklabel_format(style='sci',axis='y',scilimits=(0,0)) 
    fig1.subplots_adjust(bottom = 0.15,left = 0.15, right = 0.85)
    return fig1
        


pdic = {'T' : 900, 'amp' : 6, 'per' : 70, 'sigma' : 2, 'slope' : -10.}
tvec, raw_signal = synth_signal1(**pdic)
trend = wl.sinc_smooth(raw_signal = raw_signal,T_c = 100, dt = 1)

figure=plot_signal_trend(raw_signal,trend, 1, fig_num = 1, ptitle = None,time_label = 'min')
figure.show()

