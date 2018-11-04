#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
from PyQt5.QtWidgets import QComboBox, QFileDialog, QAction, QMainWindow, QApplication, QLabel, QLineEdit, QPushButton, QMessageBox, QSizePolicy, QWidget, QVBoxLayout, QHBoxLayout, QDialog, QGroupBox, QFormLayout


from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

#from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtCore import pyqtSlot, pyqtSignal

import wavelets_lib as wl
import random
import numpy as np
import pandas as pd

DEBUG = True

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setGeometry(100,100,400,100)
        self.setWindowTitle('TFAnalyzer')

        self.quitAction = QAction("&Quit", self)
        self.quitAction.setShortcut("Ctrl+Q")
        self.quitAction.setStatusTip('Leave The App')
        self.quitAction.triggered.connect(self.close_application)

        openFile = QAction("&Load data", self)
        openFile.setShortcut("Ctrl+O")
        openFile.setStatusTip('Laod data')
        openFile.triggered.connect(self.load)

        detrending = QAction('&Detrend signal', self)
        detrending.setShortcut('Ctrl+D')
        detrending.setStatusTip('Detrends siganl')
        detrending.triggered.connect(self.detrending)

        plotSynSig = QAction('&Plot synthetic signal',self)
        plotSynSig.setShortcut('Ctrl+Y')
        plotSynSig.setStatusTip('Plot synthetic signal')
        plotSynSig.triggered.connect(self.plotSynSig)

        self.statusBar()

        mainMenu = self.menuBar()
        
        fileMenu = mainMenu.addMenu('&File')
        fileMenu.addAction(self.quitAction)
        fileMenu.addAction(openFile)
        
        analyzerMenu = mainMenu.addMenu('&Analyzer')
        analyzerMenu.addAction(plotSynSig)
        analyzerMenu.addAction(detrending)
        
        self.home()



    def home(self):
        quitButton = QPushButton("Quit", self)
        quitButton.clicked.connect(self.close_application)
        quitButton.resize(quitButton.minimumSizeHint())
        quitButton.move(0,100)

        self.quitAction.triggered.connect(self.close_application)
        self.toolBar = self.addToolBar("Extraction")
        self.toolBar.addAction(self.quitAction)


        self.openFileButton = QPushButton("Load data",self)
        self.openFileButton.move(200,120)
        self.openFileButton.clicked.connect(self.load)

        self.show()
        
    def close_application(self):
        choice = QMessageBox.question(self, 'Quitting',
                                            'Do you want to exit this application?',
                                            QMessageBox.Yes | QMessageBox.No)
        if choice == QMessageBox.Yes:
            print("Quitting ...")
            sys.exit()
        else:
            pass
        
    def load (self):
        name = QFileDialog.getOpenFileName(self, 'Open File')
        file = open(name,'r')
        if DEBUG:
            print (file)
    def detrending (self):
        print ('function detrending called')
        self.detr = Detrender()
        
        self.detr.make_connection(self.plotSynSig)
        print ('detr and plotSynSig connected')
        

    def plotSynSig (self):
        pdic = {'T' : 900, 'amp' : 6, 'per' : 70, 'sigma' : 2, 'slope' : -10.}
        gen_func = synth_signal1 #TODO dropdown which synthetic signal
        default_para_dic = pdic
        self.plotSynSig=SyntheticSignalGenerator(gen_func, default_para_dic)
        

class DataHandler(QWidget):
    def __init__(self):
        super().__init__()
        file_name = QFileDialog.getOpenFileName(self, 'Open File')
        
        
        if DEBUG:
            print (file_name[0])

        raw_data = pd.read_excel(str(file_name[0]), header=0)
        #print (raw_data.head())


class Detrender(QWidget):
    def __init__(self):
        super().__init__()
        #self.raw_signal = signal
        #self.tvec = tvec
        self.initUI()
    def initUI(self):
        self.plotWindow = TimeSeriesWindow()
        self.plotWindow_signal = TimeSeriesWindow()
        
        self.setWindowTitle('Detrender')
        self.setGeometry(310,310,450,900)
        
        main_layout_v =QVBoxLayout()
        button_layout_h = QHBoxLayout()
        self.dialog = NumericParameterDialog({'T_c': 100})
        
        
        self.dataChoice = QLabel('synth_signal1', self)

        comboBox = QComboBox(self)
        comboBox.addItem('data1')
        comboBox.addItem('synth_signal1')
        dataChoic_layout_h =QHBoxLayout()
        dataChoic_layout_h.addWidget(self.dataChoice)
        dataChoic_layout_h.addWidget(comboBox)

        main_layout_v.addWidget(self.plotWindow)
        main_layout_v.addWidget(self.plotWindow_signal)
        main_layout_v.addWidget(self.dialog)
        main_layout_v.addLayout(dataChoic_layout_h)

        plotButton = QPushButton('Detrend signal', self)
        plotButton.clicked.connect(self.doPlot)
        button_layout_h.addStretch(0)
        button_layout_h.addWidget(plotButton)
        main_layout_v.addLayout(button_layout_h)
        self.setLayout(main_layout_v)
        self.show()
        
        
        comboBox.activated[str].connect(self.data_choice)

    def data_choice(self, text):
        self.dataChoice.setText(text)


    def make_connection(self, signal_generator_object):        ###########################
        signal_generator_object.timeSignal.connect(self.get_time_signal) #######################

    @pyqtSlot('PyQt_PyObject')
    def get_time_signal(self, time_signal_list):

        self.tvec = time_signal_list[0]
        self.raw_signal = time_signal_list[1]

    #@pyqtSlot()
    def doPlot(self):

        pdic = self.dialog.read()
        print('Plotting with {}'.format(pdic))
        trend = wl.sinc_smooth(raw_signal = self.raw_signal,T_c = pdic['T_c'], dt = 1)
        detrended_signal= self.raw_signal - trend
        #plot trend and signal
        self.plotWindow.update(self.tvec, self.raw_signal)
        self.plotWindow.update(self.tvec, trend, clear = False)

        #plot dtrended signal
        self.plotWindow_signal.update(self.tvec, detrended_signal)


class SyntheticSignalGenerator(QWidget):
    ''' 
    tvec: array containing the time vector
    signal: array containing the signal or 'synthetic' if synthetic siganl shall be used
    default_para_dic: dictonary containing default parameters for synthetic signal creation


    '''
    # Added a signal
    timeSignal = pyqtSignal('PyQt_PyObject')  #########################

    def __init__(self,gen_func, default_para_dic): 
        super().__init__()
        self.default_para_dic = default_para_dic
        self.gen_func = gen_func

        if DEBUG:
            print ('default para{}'.format(self.default_para_dic))

        self.initUI()
           
        
        

    def initUI(self):

        self.plotWindow = TimeSeriesWindow('Synthetic Signal')

        self.setWindowTitle('Synthetic Signal Generator')
        self.setGeometry(300,300,450,720) #???

        main_layout_v = QVBoxLayout()
        button_layout_h = QHBoxLayout()

        # add/create dialog
        self.dialog = NumericParameterDialog(self.default_para_dic)

        main_layout_v.addWidget(self.plotWindow)
        main_layout_v.addWidget(self.dialog)

        # Create a plot button in the window                                                                     
        plotButton = QPushButton('Save / Plot signal', self)
        # connect button to function save_on_click                                                          
        plotButton.clicked.connect(self.doPlot)
        
        button_layout_h.addStretch(1)
        button_layout_h.addWidget(plotButton)
        
        main_layout_v.addLayout(button_layout_h)
        
        # TODO button to reset to default parameters        

        
        self.setLayout(main_layout_v)
        if DEBUG:
            print ('Showing Syn Plot')
        self.show()
        if DEBUG:
            print ('Closing Syn Plot')

    def doPlot(self):
        if DEBUG:
            if not self.gen_func:
                raise ValueError('No gen_func supplied')
        pdic = self.dialog.read()
        print('Plotting with {}'.format(pdic))
        tvec, signal = self.gen_func( **pdic)
        self.timeSignal.emit([tvec,signal])
        self.plotWindow.update(tvec, signal)
        
        # plot 2nd signal
        #self.plotWindow.update(tvec, -0.5*signal, clear = False)


class NumericParameterDialog(QDialog):

    def __init__(self,default_para_dic):
        super().__init__()
        self.default_para_dic = default_para_dic
        self.input_fields ={} #holds para_names:textbox
        self.para_dic = self.default_para_dic.copy()

        self.createFormGroupBox()

        mainLayout = QVBoxLayout()
        mainLayout.addWidget(self.formGroupBox)
        
        self.setLayout(mainLayout)
        self.show()
        

    def createFormGroupBox(self):

        self.formGroupBox = QGroupBox('Parameters')
        layout = QFormLayout()
        for par_name, value in self.default_para_dic.items():
            textbox = QLineEdit()
            textbox.insert(str(value))
            layout.addRow(QLabel(par_name),textbox)
            self.input_fields[par_name] = textbox
        
        self.formGroupBox.setLayout(layout)
    
    def read(self):

        for pname in self.para_dic.keys():

            textbox = self.input_fields[pname]
            textboxString = textbox.text()

            self.para_dic[pname] = float(textboxString)

        return self.para_dic


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




    app = QApplication(sys.argv)
    pdic = {'T' : 900, 'amp' : 6, 'per' : 70, 'sigma' : 2, 'slope' : -10.}



    #dt = 1
    #T_c = 100
    tvec, raw_signal = synth_signal1(**pdic)
    

    #pDialog = SyntheticSignalGenerator(synth_signal1, pdic)
    #pDialog2 = Detrender()
    #pDialog2.make_connection(pDialog)
    #detrDialog = InterActiveTimeSeriesPlotter(tvec, detrend, pdic)
    
    #open_file = DataHandler()
    window = MainWindow()
    sys.exit(app.exec_())
        
