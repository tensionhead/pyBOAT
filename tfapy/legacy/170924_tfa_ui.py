#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
from PyQt5.QtWidgets import QMainWindow, QApplication, QLabel, QLineEdit, QPushButton, QMessageBox, QSizePolicy, QWidget, QVBoxLayout, QHBoxLayout


from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from PyQt5.QtCore import pyqtSlot

import random
import numpy as np

DEBUG = True

class MainWindow(QMainWindow):
    def __init__(self,tvec, signal,default_para_dic):
        super().__init__()
        plotTools = InterActiveTimeSeriesPlotter(self,tvec, signal,default_para_dic)


class InterActiveTimeSeriesPlotter(QWidget):
    ''' 
    tvec: array containing the time vector
    signal: array containing the signal or 'synthetic' if synthetic siganl shall be used
    default_para_dic: dictonary containing default parameters for synthetic signal creation


    '''
    
    def __init__(self,tvec,signal, default_para_dic): 
        super().__init__()
        self.default_para_dic = default_para_dic
        self.tvec = tvec 
        self.signal = signal
        
        self.initUI()

        #if DEBUG:
        #    print('Calling gen_func..')
        #    res = gen_func(**default_para_dic)
        #    print('Success')
        
    def initUI(self):

        plotWindow = TimeSeriesWindow(tvec = self.tvec, signal = self.signal) 
        detrendWindow = TimeSeriesWindow(tvec = self.tvec,signal =self.signal)

        self.setWindowTitle('Enter parameters')
        self.setGeometry(300,300,320,220) #???

        self.dialog = NumericParameterDialog(self.default_para_dic)
        if 0:
            print ('default para{}'.format(self.default_para_dic))
        main_layout = QVBoxLayout()
        button_layout = QHBoxLayout()
        main_layout.addWidget(self.dialog)
        self.button = PlotButton(plot_command = plotWindow.update, label = 'Plot signal', dialog = self.dialog)
        self.button_detrend = PlotButton(plot_command=detrendWindow.update_detrend, label = 'Detrend', dialog =self.dialog)
        
        self.button_reset = ResetButton(label = 'Reset', dialog=self.dialog, default_para_dic = self.default_para_dic)
        
        button_layout.addWidget(self.button_reset)
        button_layout.addWidget(self.button)
        button_layout.addWidget(self.button_detrend)
        main_layout.addLayout(button_layout)
        
        # TODO button to reset to default parameters

        
        #self.plotWindow = TimeSeriesWindow(genfunc = sin_plot)
        
        self.setLayout(main_layout)
        self.show()
        



class NumericParameterDialog(QWidget):    
  
    def __init__(self, default_para_dic):
        super().__init__()
        self.default_para_dic = default_para_dic
        self.input_fields = {} # holds the par_names:textbox
        self.initUI()

       
    def initUI(self):

        if DEBUG:
            print ('default para{}'.format(self.default_para_dic))
        
        #default_para_dic = default_para_dic
        self.para_dic = self.default_para_dic.copy()

        vbox = QVBoxLayout()
        #vbox.addStretch(1)
        for par_name, value in self.default_para_dic.items():
            hbox = QHBoxLayout()
        #label for textbox                                                                                  
            label = QLabel(par_name, self)
           
        # Create textbox                                                                                    
            textbox = QLineEdit(self)
            textbox.clear()
            textbox.insert(str(value))
            #textbox.update()
            if DEBUG:
                print ('set values'+str(value))
            
            # add textbox to field dictionary
            self.input_fields[par_name] = textbox
            if DEBUG:
                print (textbox.text())

            hbox.addStretch(1)
            hbox.addWidget(label)
            hbox.addStretch(0)
            hbox.addWidget(textbox)

            # add hbox to vbox
            vbox.addLayout(hbox)
        
        self.setLayout(vbox)

    def read(self):
         
        for pname in self.para_dic.keys():   

            textbox = self.input_fields[pname]
            textboxString = textbox.text()
            
            self.para_dic[pname] = float(textboxString)        

        return self.para_dic


    def reset(self):
        self.__init__(self.default_para_dic)

class ResetButton(QWidget):
    def __init__(self, label, dialog, default_para_dic):
        super().__init__()
        self.dialog = dialog
        self.default_para_dic = default_para_dic
        resetButton = QPushButton(label, self)
        resetButton.clicked.connect(self.reset_on_click)
    
    @pyqtSlot()
    def reset_on_click(self):
        self.dialog.reset()
        if DEBUG:
            print ('Parameters were resetted')

class PlotButton(QWidget):
    
    #vbox = QVBoxLayout()
    
    def __init__(self, plot_command, label, dialog, checked=False):
        super().__init__()
        self.dialog = dialog
        self.plot_command = plot_command
        
         # Create a button in the window                                                                     
        okButton = QPushButton(label, self)
      
        # connect button to function save_on_click                                                          
        okButton.clicked.connect(self.do_on_click)
        

    @pyqtSlot()
    def do_on_click(self):

        para_dic = self.dialog.read()

        if DEBUG:    
            print('Button clicked', para_dic)
            
        self.plot_command(para_dic)


class TimeSeriesWindow(QWidget):
    def __init__(self, tvec, signal):
        super().__init__()
        self.tvec = tvec
        self.signal = signal
        self.initUI()
    
    def initUI(self):
        self.setWindowTitle('Plot')
        self.setGeometry(10,10,640,400)
        self.mplot = TimeSeriesPlot(parent = self, tvec = self.tvec, signal = self.signal)
        self.mplot.move(0,0)

    def update(self, gf_kwargs):
        self.mplot.mpl_update(gf_kwargs)
        self.show()
    def update_detrend(self,gf_kwargs):
        self.mplot.mpl_update_detrend(gf_kwargs)
        self.show()
        


class TimeSeriesPlot(FigureCanvas):
    def __init__(self, parent=None, width=6, height=4, dpi=100,tvec = None, signal = None):
        fig = Figure(figsize=(width,height), dpi=dpi)
        self.axes = fig.add_subplot(111)


        #if not signal:
        #    raise ValueError('No time or signal supplied') ###gen_func
        self.tvec = tvec
        self.signal = signal
        self.signal_type = signal

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)
 
        FigureCanvas.setSizePolicy(self,
                QSizePolicy.Expanding,
                QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        
    def mpl_update(self,gf_kwargs):

        if DEBUG:
            print('mpl update called with {}'.format(gf_kwargs))
        if self.signal_type == 'synthetic':
            self.signal = sin_func( t =self.tvec, **gf_kwargs )

        self.axes.cla()
        self.axes.plot(self.tvec,self.signal)
        self.axes.set_xlabel('time')
        self.draw()

    def mpl_update_detrend(self,gf_kwargs):

        if DEBUG:
            print('mpl update detrend called with {}'.format(gf_kwargs))
        if self.signal_type == 'synthetic':
            self.signal = sin_func( t =self.tvec, **gf_kwargs )
            trend = wl.sinc_smooth(self.signal, 100, 1)
            detrend_signal = self.signal -trend



        self.axes.cla()
        self.axes.plot(self.tvec,trend)
        self.axes.plot(self.tvec, detrend_signal)
        self.axes.set_xlabel('time')
        self.draw()

# test case for data generating function, standard synthetic signal
def sin_func(t, amp, per, sigma, slope):  
        
    trend = slope*t**2/t[-1]**2*amp
    noise = np.random.normal(0,sigma, len(t))
    sin = amp*np.sin(2*np.pi/per*t)+noise+trend

    return sin
        
    

if __name__ == '__main__':

    import wavelets_lib as wl


    app = QApplication(sys.argv)
    pdic = {'amp' : 6, 'per' : 70, 'sigma' : 2, 'slope' : -10.}
    tvec = np.array(range(0,256))

    signal = 'synthetic'
    dt = 1
    T_c = 100
    raw_signal =sin_func(tvec,**pdic)

    trend = wl.sinc_smooth(raw_signal, T_c, dt)
    detrend = (raw_signal - trend)

    pDialog = InterActiveTimeSeriesPlotter(tvec,signal,pdic)
    #detrDialog = InterActiveTimeSeriesPlotter(tvec, detrend, pdic)

    sys.exit(app.exec_())
        
