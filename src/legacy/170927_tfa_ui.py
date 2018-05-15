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


class Detrender(QWidget):
    pass

class SyntheticSignalGenerator(QWidget):
    ''' 
    tvec: array containing the time vector
    signal: array containing the signal or 'synthetic' if synthetic siganl shall be used
    default_para_dic: dictonary containing default parameters for synthetic signal creation


    '''
    
    def __init__(self,gen_func, default_para_dic): 
        super().__init__()
        self.default_para_dic = default_para_dic
        self.gen_func = gen_func

        if DEBUG:
            print ('default para{}'.format(self.default_para_dic))

        self.initUI()

        #if DEBUG:
        #    print('Calling gen_func..')
        #    res = gen_func(**default_para_dic)
        #    print('Success')
        

    def initUI(self):

        self.plotWindow = TimeSeriesWindow('Synthetic Signal')

        self.setWindowTitle('Enter parameters')
        self.setGeometry(300,300,620,820) #???

        main_layout = QVBoxLayout()
        button_layout = QHBoxLayout()


        # add/create dialog
        self.dialog = NumericParameterDialog(self.default_para_dic)

        main_layout.addWidget(self.plotWindow)
        main_layout.addWidget(self.dialog)




        # Create a plot button in the window                                                                     
        PlotButton = QPushButton('Plot signal', self)
        # connect button to function save_on_click                                                          
        PlotButton.clicked.connect(self.doPlot)
        
        #button_layout.addWidget(self.button_reset)
        button_layout.addWidget(PlotButton)

        main_layout.addLayout(button_layout)
        
        # TODO button to reset to default parameters

        
        #self.plotWindow = TimeSeriesWindow(genfunc = sin_plot)
        
        self.setLayout(main_layout)
        self.show()

    @pyqtSlot()
    def doPlot(self):
        pdic = self.dialog.read()
        print('Plotting with {}'.format(pdic))
        tvec, signal = self.gen_func( **pdic)

        self.plotWindow.update(tvec, signal)
        
        # plot 2nd signal
        #self.plotWindow.update(tvec, -0.5*signal, clear = False)




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
    def __init__(self, title = None):
        super().__init__()
        self.initUI(title)
    
    def initUI(self, title):
        self.setWindowTitle(title)
        self.setGeometry(10,10,440,300)
        self.mplot = TimeSeriesPlot(parent = self)
        self.mplot.move(0,0)

    # transfer function
    def update(self, tvec, signal, clear = True):
        self.mplot.mpl_update(tvec, signal, clear = clear)
        self.show()
        

class TimeSeriesPlot(FigureCanvas):
    def __init__(self, parent=None, width=4, height=3, dpi=100):
        fig = Figure(figsize=(width,height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        self.axes.set_xlabel('time')

        #if not signal:
        #    raise ValueError('No time or signal supplied') ###gen_func

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)
 
        FigureCanvas.setSizePolicy(self,
                QSizePolicy.Expanding,
                QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        
    def mpl_update(self, tvec, signal, clear = True):

        if DEBUG:
            print('mpl update called with {}, {}'.format(tvec[:10], signal[:10]))

        if clear:
            self.axes.cla()
        self.axes.plot(tvec, signal)
        self.draw()

# test case for data generating function, standard synthetic signal
def synth_signal1(T, amp, per, sigma, slope):  
    
    tvec = np.arange(T)
    trend = slope*tvec**2/tvec[-1]**2*amp
    noise = np.random.normal(0,sigma, len(tvec))
    sin = amp*np.sin(2*np.pi/per*tvec)+noise+trend

    return tvec, sin
        
    

if __name__ == '__main__':

    import wavelets_lib as wl


    app = QApplication(sys.argv)
    pdic = {'T' : 900, 'amp' : 6, 'per' : 70, 'sigma' : 2, 'slope' : -10.}
    tvec = np.array(range(0,256))

    signal = 'synthetic'
    dt = 1
    T_c = 100
    tvec, raw_signal = synth_signal1(**pdic)

    trend = wl.sinc_smooth(raw_signal, T_c, dt)
    detrend = (raw_signal - trend)

    pDialog = SyntheticSignalGenerator(synth_signal1, pdic)
    pDialog2 = SyntheticSignalGenerator(synth_signal1, pdic)
    #detrDialog = InterActiveTimeSeriesPlotter(tvec, detrend, pdic)

    sys.exit(app.exec_())
        
