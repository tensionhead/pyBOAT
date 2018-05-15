#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys
from PyQt5.QtWidgets import QMainWindow, QApplication, QLabel, QLineEdit, QPushButton, QMessageBox, QSizePolicy

import matplotlib
matplotlib.use('QT5Agg')

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from PyQt5.QtCore import pyqtSlot

import random
import numpy as np


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        self.setGeometry(300,300,250,150)
        self.setWindowTitle('Main Window')

class para_control(MainWindow):
    

    def initUI(self):
        
        self.setWindowTitle('Enter parameter')
        self.setGeometry(300,300,320,180)

        #label for textbox                                                                                  
        label = QLabel('Amplitude:', self)
        label.move(50,25)

        # Create textbox                                                                                    
        self.textbox = QLineEdit(self)
        self.textbox.move(220, 30)
        self.textbox.resize(60,20)
        
         # Create a button in the window                                                                     
        self.button = QPushButton('Save', self)
        self.button.resize(60,20)
        self.button.move(220,140)


        # connect button to function save_on_click                                                          
        self.button.clicked.connect(self.save_on_click)
        self.show()

    @pyqtSlot()
    def save_on_click(self):
        textboxValue = self.textbox.text()

        QMessageBox.question(self, 'Parameter', "You typed: " + textboxValue , QMessageBox.Ok, QMessageBox.Ok)

class PlotWindow(QMainWindow):
    def __init__(self, amp):
        super().__init__()
        self.amp=amp
        print (self.amp)
        self.initUI()
    
    
    def initUI(self):
        self.setWindowTitle('Plot')
        self.setGeometry(10,10,640,400)
        print (self.amp)
        p = MyPlot(self.amp)
        p.move(0,0)
        self.show()
        
        


class MyPlot(FigureCanvas):
    def __init__(self, amp=3, parent=None, width=5, height=4, dpi=100):
        sin_fig = Figure(figsize=(width,height), dpi=dpi)
        self.axes = sin_fig.add_subplot(111)
        self.amp = amp
        print (self.amp)

 
        FigureCanvas.__init__(self, sin_fig)
        self.setParent(parent)
 
        FigureCanvas.setSizePolicy(self,
                QSizePolicy.Expanding,
                QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self.sin_plot()

    def sin_plot(self):
        t = np.array(range(0,256))
        ax = self
        sigma = 1
        #amp= 3
        per = 40
        
        noise = np.random.normal(0,sigma, len(t))
        sin = self.amp*np.sin(2*np.pi/per*t)+noise
        self.axes.plot(t,sin)
    
    



        
        
    

if __name__ == '__main__':
    
    
    app = QApplication(sys.argv)
    x = PlotWindow(amp=6)
   
    sys.exit(app.exec_())
        
