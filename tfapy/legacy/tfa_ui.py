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

class InterActiveTimeSeriesPlotter(QWidget):
    ''' what is it, what shall it do.. '''
    
    def __init__(self, gen_func, default_para_dic):
        super().__init__()
        self.default_para_dic = default_para_dic
        self.gen_func = gen_func
        
        self.initUI()

        if DEBUG:
            print('Calling gen_func..')
            res = gen_func(**default_para_dic)
            print('Success')
        
    def initUI(self):

        plotWindow = TimeSeriesWindow(gen_func = self.gen_func)
        
        self.setWindowTitle('Enter parameters')
        self.setGeometry(300,300,320,220) #???

        self.dialog = NumericParameterDialog(self.default_para_dic)
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.dialog)
        self.button = PlotButton(plot_command = plotWindow.update, dialog = self.dialog)
        main_layout.addWidget(self.button)
        
        # TODO reset to default parameters

        
        #self.plotWindow = TimeSeriesWindow(genfunc = sin_plot)
        
        self.setLayout(main_layout)
        self.show()
        



class NumericParameterDialog(QWidget):    
  
    def __init__(self, default_para_dic):
        super().__init__()
        self.input_fields = {} # holds the par_names:textbox
        self.initUI(default_para_dic)

       
    def initUI(self, default_para_dic):
        
        
        self.para_dic = default_para_dic.copy()

        vbox = QVBoxLayout()
        #vbox.addStretch(1)
        for par_name, value in default_para_dic.items():
            hbox = QHBoxLayout()
        #label for textbox                                                                                  
            label = QLabel(par_name, self)
           
        # Create textbox                                                                                    
            textbox = QLineEdit(self)
            textbox.setText(str(value))
            
            # add textbox to field dictionary
            self.input_fields[par_name] = textbox
                  

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

class PlotButton(QWidget):
    
    #vbox = QVBoxLayout()
    
    def __init__(self, plot_command, dialog):
        super().__init__()
        self.dialog = dialog
        self.plot_command = plot_command
        
         # Create a button in the window                                                                     
        okButton = QPushButton('Save+Plot', self)
      
        # connect button to function save_on_click                                                          
        okButton.clicked.connect(self.do_on_click)
        

    @pyqtSlot()
    def do_on_click(self):

        para_dic = self.dialog.read()

        if DEBUG:    
            print('Button clicked', para_dic)
            
        self.plot_command(para_dic)


class TimeSeriesWindow(QWidget):
    def __init__(self, gen_func):
        super().__init__()
        self.gen_func = gen_func
        self.initUI()
    
    def initUI(self):
        self.setWindowTitle('Plot')
        self.setGeometry(10,10,640,400)
        self.mplot = TimeSeriesPlot(parent = self, gen_func = self.gen_func)
        self.mplot.move(0,0)

    def update(self, gf_kwargs):
        self.mplot.mpl_update(gf_kwargs)
        self.show()
        
        


class TimeSeriesPlot(FigureCanvas):
    def __init__(self, parent=None, width=6, height=4, dpi=100, gen_func = None):
        fig = Figure(figsize=(width,height), dpi=dpi)
        self.axes = fig.add_subplot(111)


        if not gen_func:
            raise ValueError('No generating function supplied')
        self.gen_func = gen_func

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)
 
        FigureCanvas.setSizePolicy(self,
                QSizePolicy.Expanding,
                QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        
    def mpl_update(self,gf_kwargs):

        if DEBUG:
            print('mpl update called with {}'.format(gf_kwargs))
        t, data = self.gen_func( **gf_kwargs )

        self.axes.cla()
        self.axes.plot(t,data)
        self.axes.set_xlabel('time')
        self.draw()

# test case for data generating function, standard synthetic signal
def sin_func(amp, per, sigma, slope):  
    
    t = np.array(range(0,256))
       
        
    trend = slope*t**2/t[-1]**2*amp
    noise = np.random.normal(0,sigma, len(t))
    sin = amp*np.sin(2*np.pi/per*t)+noise+trend

    return t, sin
        
    

if __name__ == '__main__':

    import wavelets_lib as wl


    app = QApplication(sys.argv)
    pdic = {'amp' : 6, 'per' : 70, 'sigma' : 2, 'slope' : -10.}

    tvec, signal = sin_func(**pdic)
    dt = 1

    trend = wl.sinc_smooth(signal, T_c = 100, dt = 1)
    detrended = signal - trend

    pDialog = InterActiveTimeSeriesPlotter(sin_func,pdic)
    detrDialog = InterActiveTimeSeriesPlotter(gf_sinc_smooth,{'T_c' : 100})

    sys.exit(app.exec_())
        
