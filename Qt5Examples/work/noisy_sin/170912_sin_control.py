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

class InterActivePlotter(QWidget):
    pass


class NumericParameterDialog(QWidget):    

    def __init__(self, default_para_dic):
        super().__init__()

        self.initUI(default_para_dic)

        self.plotWindow = TimeSeriesWindow(genfunc = sin_plot) ### a TimeSeriesWindow is always created when NumericParamerterDialog is called

    def initUI(self, default_para_dic):
        
        self.setWindowTitle('Enter parameter')
        self.setGeometry(300,300,320,220)

        self.input_fields = {}
        self.para_dic = default_para_dic.copy()

        vbox = QVBoxLayout()
        #vbox.addStretch(1)
        for par_name, value in default_para_dic.items():
            hbox = QHBoxLayout()
        #label for textbox                                                                                  
            label = QLabel(par_name, self)
           
        # Create textbox                                                                                    
            textbox = QLineEdit(self)
            textbox.insert(str(value))

            # add textbox to field dictionary
            self.input_fields[par_name] = textbox

            hbox.addStretch(1)
            hbox.addWidget(label)
            hbox.addStretch(0)
            hbox.addWidget(textbox)

            # add hbox to vbox
            vbox.addLayout(hbox)

        self.setLayout(vbox)



        
         # Create a button in the window                                                                     
        okButton = QPushButton('Save+Plot', self)
        hbox = QHBoxLayout()
        hbox.addStretch(1)
        hbox.addWidget(okButton)
        vbox.addLayout(hbox)### Connection

        # connect button to function save_on_click                                                          
        okButton.clicked.connect(self.save_on_click)
        self.show()

    @pyqtSlot()
    def save_on_click(self):
        #pass

        for pname in self.para_dic.keys():  #Connection

            textbox = self.input_fields[pname] #Connection
            textboxString = textbox.text()
            
            self.para_dic[pname] = float(textboxString)
        if DEBUG:    
            print(self.para_dic)
        self.plotWindow.update(self.para_dic)



#TODO combine parameter dialog with Button -> TimeSeriesWindow
#class InteractivePlotter(
        
        

class TimeSeriesWindow(QWidget):
    def __init__(self, genfunc):
        super().__init__()
        self.genfunc = genfunc
        self.initUI()
    
    def initUI(self):
        self.setWindowTitle('Plot')
        self.setGeometry(10,10,640,400)
        self.mplot = TimeSeriesPlot(parent = self, genfunc = self.genfunc)
        self.mplot.move(0,0)

    def update(self, gf_kwargs):
        self.mplot.mpl_update(gf_kwargs)
        self.show()
        
        


class TimeSeriesPlot(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100, genfunc = None):
        fig = Figure(figsize=(width,height), dpi=dpi)
        self.axes = fig.add_subplot(111)


        if not genfunc:
            raise ValueError('No generating function supplied')
        self.genfunc = genfunc

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)
 
        FigureCanvas.setSizePolicy(self,
                QSizePolicy.Expanding,
                QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        
    def mpl_update(self,gf_kwargs):

        if DEBUG:
            print('mpl update called with {}'.format(gf_kwargs))
        t, data = self.genfunc( **gf_kwargs )

        self.axes.cla()
        self.axes.plot(t,data)
        self.draw()

def sin_plot(amp, per, sigma):
    
    t = np.array(range(0,256))
       
        
    noise = np.random.normal(0,sigma, len(t))
    sin = amp*np.sin(2*np.pi/per*t)+noise

    return t, sin
        
    

if __name__ == '__main__':
    
    
    app = QApplication(sys.argv)
    pdic = {'amp' : 6, 'per' : 70, 'sigma' : 2}

    #plotWindow = TimeSeriesWindow(genfunc = sin_plot)
    #plotWindow.update(gf_kwargs = pdic)

    pDialog = NumericParameterDialog(pdic)
    sys.exit(app.exec_())
        
