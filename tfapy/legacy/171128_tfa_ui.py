#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
from PyQt5.QtWidgets import QCheckBox, QTableView, QComboBox, QFileDialog, QAction, QMainWindow, QApplication, QLabel, QLineEdit, QPushButton, QMessageBox, QSizePolicy, QWidget, QVBoxLayout, QHBoxLayout, QDialog, QGroupBox, QFormLayout, QGridLayout


from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from PyQt5.QtCore import pyqtSlot, pyqtSignal, Qt

import wavelets_lib as wl
from helper.pandasTable import PandasModel
import random
import numpy as np
import pandas as pd

DEBUG = True

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.dh = DataHandler()
        self.initUI()
    def initUI(self):
        self.setGeometry(100,100,400,100)
        self.setWindowTitle('TFAnalyzer')

        self.quitAction = QAction("&Quit", self)
        self.quitAction.setShortcut("Ctrl+Q")
        self.quitAction.setStatusTip('Leave The App')
        self.quitAction.triggered.connect(self.close_application)

        openFile = QAction("&Load data", self)
        openFile.setShortcut("Ctrl+L")
        openFile.setStatusTip('Laod data')
        openFile.triggered.connect(self.load)

        detrending = QAction('&Detrend signal', self)
        detrending.setShortcut('Ctrl+T')
        detrending.setStatusTip('Detrends signal')
        detrending.triggered.connect(self.detrending)

        plotSynSig = QAction('&Plot synthetic signal',self)
        plotSynSig.setShortcut('Ctrl+P')
        plotSynSig.setStatusTip('Plot synthetic signal')
        plotSynSig.triggered.connect(self.plotSynSig)
        
        viewer = QAction('&View Data',self)
        viewer.setShortcut('Ctrl+D')
        viewer.setStatusTip('View data')
        viewer.triggered.connect(self.viewing)

        self.statusBar()

        mainMenu = self.menuBar()
        
        fileMenu = mainMenu.addMenu('&File')
        fileMenu.addAction(self.quitAction)
        fileMenu.addAction(openFile)
        
        analyzerMenu = mainMenu.addMenu('&Analyzer')
        analyzerMenu.addAction(plotSynSig)
        analyzerMenu.addAction(detrending)
        analyzerMenu.addAction(viewer)
        
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
        
    def close_application(self, event):
        choice = QMessageBox.question(self, 'Quitting',
                                            'Do you want to exit this application?',
                                            QMessageBox.Yes | QMessageBox.No)
        if choice == QMessageBox.Yes:
            print("Quitting ...")
            #sys.exit()
            appc = QApplication.instance()
            appc.closeAllWindows()
        else:
            pass
        
    def load (self):

        self.new_data = DataLoader()
        
        self.dh.make_connection(self.new_data)
        self.new_data.initUI()
###################################
    def viewing (self):
        print ('function viewing called')
        #self.view = DataViewer()
        
        #self.view.make_connection(self.dh) 
        #self.dh.data_request_connection(self.view)
        
        #self.dh.ini_connect_analyzer_tools()
        self.dh.initUI()
        #print ('DataViewer and DH connected')
        
    def detrending (self):
        print ('function detrending called')
        self.detr = Detrender()
        
        self.detr.make_connection(self.dh) 
        self.dh.data_request_connection(self.detr)
        self.dh.ini_connect_analyzer_tools() #emit ids
        self.detr.initUI()
        print ('detr and plotSynSig connected')
        

    def plotSynSig (self):
        pdic = {'T' : 900, 'amp' : 6, 'per' : 70, 'sigma' : 2, 'slope' : -10.}
        gen_func = synth_signal1 #TODO dropdown which synthetic signal
        default_para_dic = pdic
        self.synSig=SyntheticSignalGenerator(gen_func, default_para_dic)
        self.dh.make_connection(self.synSig)
        self.synSig.initUI()
        
class DataLoader(QWidget):
    timeSignal = pyqtSignal('PyQt_PyObject')
    def __init__(self):
        super().__init__()
        self.raw_data = pd.DataFrame()
        #self.initUI()
    def initUI(self):
        file_name = QFileDialog.getOpenFileName(self, 'Open File')
        ###If correct file path/type is supplied data are read in and emitted
        #try: 
        if DEBUG:
            print (file_name[0])

        self.raw_data = pd.read_excel(file_name[0], header=0)
        ## TODO drop NaNs
        ## later TODO deal with 'holes'
        self.emit_values()

        
        #except:
           # self.noFile = Error('No valid path or file supplied!', 'No File')
        
    def emit_values(self):
        for name, values in self.raw_data.iteritems():
            signal = np.array(values)
            #signal =signal[~np.isnan(signal)]
            tvec = range(len(signal))
            self.timeSignal.emit([name,tvec,signal])
    


class DataHandler(QWidget):
    #Add signal
    signalIds = pyqtSignal('PyQt_PyObject')
    dataSignal= pyqtSignal('PyQt_PyObject')
    def __init__(self):
        super().__init__()
        self.raw_data= pd.DataFrame()
        self.signal_ids = []
        #self.df_return_data = pd.DataFrame()
        self.tvec_dic = {}
        self.signal_dic = {}
        #self.ini_connect_analyzer_tools()
        ##self.initUI()
    #def ini_connect_analyzer_tools(self):
    #    self.signalIds.emit(self.signal_ids)
    
    
    def initUI(self):
        self.plotWindow = TimeSeriesWindowView()
        
        print(self.plotWindow.sizeHint())
        print (type(self.plotWindow))
        
        self.setWindowTitle('DataViewer')
        self.setGeometry(20,30,900,650)
        
        main_layout_v =QVBoxLayout()
        #Data selction drop-down
        dataLable = QLabel('Select signal', self)
        comboBox = QComboBox(self)
        comboBox.addItem('None')
        for i in self.signal_ids:
            print (i)
            comboBox.addItem(i)
        
        self.table = QTableView()
        
        data_selection_layout_h =QHBoxLayout()
        data_selection_layout_h.addWidget(dataLable)
        data_selection_layout_h.addWidget(comboBox)
        data_selection_layout_h.addStretch(0)
        main_layout_v.addLayout(data_selection_layout_h)
        
        comboBox.activated[str].connect(self.data_request)
        
        # plot options box
        plot_options_box = QGroupBox('Show')
        cb_layout= QFormLayout()
        cb_raw = QCheckBox('Raw signal', self)
        cb_trend = QCheckBox('Trend', self)
        cb_detrend = QCheckBox('Detrended Signal', self)
        plotButton = QPushButton('Plot signal', self)
        button_layout_h = QHBoxLayout()
        plotButton.clicked.connect(self.doPlot)
        button_layout_h.addStretch(0)
        button_layout_h.addWidget(plotButton)
        
        
        dialog = NumericParameterDialog({'T_c': 100})
        
        para_box = QGroupBox('Detrend')
        para_layout = QFormLayout()
        para_layout.addRow(dialog)
        
        para_box.setLayout(para_layout)
        
        #checkbox layout
        cb_layout.addRow(cb_raw)
        cb_layout.addRow(para_layout) #Does not show
        cb_layout.addRow(cb_trend)
        cb_layout.addRow(cb_detrend)
        cb_layout.addRow(button_layout_h)
        
        plot_options_box.setLayout(cb_layout)
        
        # checkbox signal
        cb_raw.toggle()
        cb_trend.toggle()
        cb_detrend.toggle()
        cb_raw.stateChanged.connect(self.set_raw_true)
        cb_trend.stateChanged.connect(self.set_trend_true)
        cb_detrend.stateChanged.connect(self.set_detrended_true)
        self.plot_raw = True
        self.plot_trend = True
        self.plot_detrended = True
        
        
        #Analyzer options box
        ana_options_box = QGroupBox('Wavlet Analysis')
        ana_layout= QFormLayout()
        
        ana_options_box.setLayout(ana_layout)
        
        
        #Options box (big box)
        options_box = QGroupBox('Options')
        options_layout=QFormLayout()
        options_layout.addRow(plot_options_box)
        options_layout.addRow(ana_options_box)
        
        options_box.setLayout(options_layout)
        
        
        #plot_box = QGroupBox('Plot')
        #plot_layout = QFormLayout()
        #plot_layout.addRow(self.plotWindow)
        
        #plot_box.setLayout(plot_layout)

        ############################
        horizontalGroupBox = QGroupBox('Signal')
        layout = QGridLayout()
        layout.addWidget(self.table,0,0,2,6)
        layout.addWidget(self.plotWindow,2,0,3,3)
        layout.addWidget(options_box, 2,3,3,3)
        horizontalGroupBox.setLayout(layout)
        

        
        main_layout_v.addWidget(horizontalGroupBox)
        
        
        
        self.setLayout(main_layout_v)
        self.show()
        
        
        
    def set_raw_true (self, state):
        if state == Qt.Checked:
            self.plot_raw = True
        else:
            self.plot_raw = False
        print (self.plot_raw)
    def set_trend_true (self, state):
        print (self.plot_trend)
        if state == Qt.Checked:
            self.plot_trend = True
        else:
            self.plot_trend = False
        print (self.plot_trend)
    def set_detrended_true (self, state):
        if state == Qt.Checked:
            self.plot_detrended = True
        else:
            self.plot_detrended = False
        print (self.plot_detrended)
        
    def table_view(self):

        model= PandasModel(self.raw_data)
        self.table.setModel(model)

        
    def data_request(self, text):
        self.id = text
        #self.dataRequest.emit(text)
        
        
    def doPlot(self):
        #pdic = self.dialog.read()
        try:
            raw_signal= np.array(self.raw_data[self.id].values)
            tvec = np.array(self.raw_data[self.id+'time'].values)
            tvec =tvec[~np.isnan(raw_signal)]
            raw_signal =raw_signal[~np.isnan(raw_signal)]
            trend = wl.sinc_smooth(raw_signal = raw_signal,T_c = 100, dt = 1)
            self.plotWindow.update(tvec, raw_signal, trend, plot_raw= self.plot_raw, plot_trend=self.plot_trend, plot_detrended=self.plot_detrended)

        except:
            self.noDataSelected = Error('Please selcte one dataseries from the drop down menu or load data first!','Missing data')

    '''def make_connection(self, datahandler_object):
        datahandler_object.signalIds.connect(self.get_signal_ids)
        datahandler_object.dataSignal.connect(self.get_data)
        
        
    @pyqtSlot('PyQt_PyObject')
    def get_signal_ids(self, signal_ids):
        self.series_ids = signal_ids
        
    @pyqtSlot('PyQt_PyObject')
    def get_data(self, raw_data):
        self.raw_data = raw_data
        self.table_view()
    '''
    
    
    

    def make_connection(self, signal_object):        ###########################
        signal_object.timeSignal.connect(self.get_time_signal) #######################
    #def data_request_connection(self, signal_object):
    #    signal_object.dataRequest.connect(self.return_data)
        

    @pyqtSlot('PyQt_PyObject')
    def get_time_signal(self, id_signal_list): # list of length 3 with id, time, signal
        self.signal_ids.append(id_signal_list[0])
        self.tvec_dic[id_signal_list[0]] = id_signal_list[1]
        print (id_signal_list[1])
        self.signal_dic[id_signal_list[0]] =id_signal_list[2]
        
        for id in self.signal_ids:
            self.raw_data[id+'time'] = self.tvec_dic[id]
            self.raw_data[id] = self.signal_dic[id]
            print (self.signal_ids)
        
    #@pyqtSlot('PyQt_PyObject')
    #def return_data(self, id):
    #    print (self.tvec_dic.keys())
    #    print (id)
    #    self.df_return_data[id+'time'] = self.tvec_dic[id]
    #    self.df_return_data[id] = self.signal_dic[id]
    #    self.dataSignal.emit(self.df_return_data)


        
class WavletAnalyzer(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
    def initUI(self):
        self.setWindowTitle('WavletAnalyzer')
        self.setGeometry(510,310,600,700)
        signalPlot = TimeSeriesWindow()
        pdic = {'T' : 900, 'amp' : 6, 'per' : 70, 'sigma' : 2, 'slope' : -10.}
        tvec, signal= synth_signal1(**pdic)
        dt = 1.0
        periods=range(30, 100, 2)
        signalPlot.update(tvec, signal)
        
        waveletPlot = FrequencyWindow()
        waveletPlot.update(signal, dt, periods)
        print (waveletPlot)
        main_layout_v = QVBoxLayout()
        main_layout_v.addWidget(signalPlot)
        main_layout_v.addWidget(waveletPlot)
        self.setLayout(main_layout_v)
        
        self.show()
        
        
class FrequencyWindow(QWidget):
    def __init__(self, title = None):
        super().__init__()
        self.initUI()
    
    def initUI(self):
        #self.setWindowTitle(title)
        self.setGeometry(10,10,440,300)
        self.mplot = FrequencyPlot(self)
        self.mplot.move(0,0)

    # transfer function
    def update(self, signal, dt, periods, clear = True):
        self.mplot.mpl_update(tvec, signal,periods, clear = clear)
        self.show()
    
        

class FrequencyPlot(FigureCanvas):
    def __init__(self, parent=None, width=8, height=7, dpi=100):
        fig = Figure(figsize=(width,height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        #self.axes.set_xlabel('time')

        #if not signal:
        #    raise ValueError('No time or signal supplied') ###gen_func

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)
        
        print ('Time Series Size', FigureCanvas.sizeHint(self))
        FigureCanvas.setSizePolicy(self,
                QSizePolicy.Expanding,
                QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        
    def mpl_update(self,signal, dt, periods, clear = True):

        if DEBUG:
            print('mpl update called with {}, {}'.format(tvec[:10], signal[:10]))

        if clear:
            self.axes.cla()
        self.axes = wl.compute_spectrum(signal, dt, periods)
        self.draw()
        self.show()
        
        
        


class Detrender(QWidget):
    dataRequest =pyqtSignal('PyQt_PyObject')
    def __init__(self):
        super().__init__()
        self.raw_data= pd.DataFrame()
        self._connected=False
        
        #self.initUI()
    def initUI(self):
        self.plotWindow = TimeSeriesWindow()
        print(self.plotWindow.sizeHint())
        self.plotWindow_signal = TimeSeriesWindow()
        print (type(self.plotWindow))
        
        self.setWindowTitle('Detrender')
        self.setGeometry(310,310,450,900)
        
        main_layout_v =QVBoxLayout()
        button_layout_h = QHBoxLayout()
        self.dialog = NumericParameterDialog({'T_c': 100})
        
        
        dataLable = QLabel('Select signal', self)
        self.dataChoice = QLabel('',self)

        self.comboBox = QComboBox(self)
        self.comboBox.addItem('None')
        if self._connected:
            for i in self.series_names_list:
                print (i)
                self.comboBox.addItem(i)
        
        dataChoic_layout_h =QHBoxLayout()
        dataChoic_layout_h.addWidget(dataLable)
        dataChoic_layout_h.addWidget(self.comboBox)

        main_layout_v.addWidget(self.plotWindow)
        print (type(self.plotWindow))
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
        
        
        self.comboBox.activated[str].connect(self.data_request)
        
    def data_request(self, text):
        self.id = text
        self.dataRequest.emit(text)
        print ('data requested')

    def make_connection(self, datahandler_object):
        datahandler_object.signalIds.connect(self.get_signal_ids)
        datahandler_object.dataSignal.connect(self.get_data)
        self._connected= True
        
    @pyqtSlot('PyQt_PyObject')
    def get_signal_ids(self, signal_ids):
        self.series_names_list = signal_ids
        
    @pyqtSlot('PyQt_PyObject')
    def get_data(self, raw_data):
        self.raw_data = raw_data
        #self.table_view()
        print (self.raw_data.head())

    
    def doPlot(self):

        pdic = self.dialog.read()
        if 1:
            #print('Plotting {}'.format(self.dataChoice.text()))
            #dt =self.tvec_dic[self.dataChoice.text()][1]-self.tvec_dic[self.dataChoice.text()][0]
            trend = wl.sinc_smooth(raw_signal = self.raw_data[self.id].values,T_c = pdic['T_c'], dt = 1)
            print(len(self.raw_data[self.id].values), len(trend))
            print(self.raw_data[self.id].values[:-20])
            print(trend[:-20])

            detrended_signal= self.raw_data[self.id].values - trend
            #plot trend and signal
            self.plotWindow.update(self.raw_data[self.id+'time'].values, self.raw_data[self.id].values)
            self.plotWindow.update(self.raw_data[self.id+'time'].values, trend, clear = False)
            print (type(self.plotWindow))

            #plot dtrended signal
            self.plotWindow_signal.update(self.raw_data[self.id+'time'].values, detrended_signal)
        #except:
        #    self.noDataSelected = Error('Please selcte one dataseries from the drop down menu!','Missing data')

class SyntheticSignalGenerator(QWidget):
    ''' 
    tvec: array containing the time vector
    signal: array containing the signal or 'synthetic' if synthetic signal shall be used
    default_para_dic: dictonary containing default parameters for synthetic signal creation


    '''
    # Added a signal, that emits signal name, tvec and signal values
    timeSignal = pyqtSignal('PyQt_PyObject')  #########################

    def __init__(self,gen_func, default_para_dic): 
        super().__init__()
        self.default_para_dic = default_para_dic
        self.gen_func = gen_func

        if DEBUG:
            print ('default para{}'.format(self.default_para_dic))

        #self.initUI()
           
        
        

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
        
        self.timeSignal.emit(['synthetic siganl1_{}'.format(pdic),tvec,signal])
        self.plotWindow.update(tvec, signal)
        

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
        self.initUI()
    
    def initUI(self):
        #self.setWindowTitle(title)
        self.setGeometry(10,10,440,300)
        self.mplot = TimeSeriesPlot(self, width= 6)
        self.mplot.move(0,0)

    # transfer function
    def update(self, tvec, signal, clear = True):
        self.mplot.mpl_update(tvec, signal, clear = clear)
        self.show()
    
        

class TimeSeriesPlot(FigureCanvas):
    def __init__(self, parent=None, width=4, height=3, dpi=100):
        fig = Figure(figsize=(width,height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        #self.axes.set_xlabel('time')

        #if not signal:
        #    raise ValueError('No time or signal supplied') ###gen_func

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)
        
        print ('Time Series Size', FigureCanvas.sizeHint(self))
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
        self.show()




class TimeSeriesWindowView(QWidget):
    def __init__(self, title = None):
        super().__init__()
        self.initUI()
    
    def initUI(self):
        #self.setWindowTitle(title)
        self.setGeometry(10,10,460,330)
        self.mplot = TimeSeriesPlotView(self, width = 8)
        self.mplot.move(0,0)

    # transfer function
    def update(self, tvec, signal, trend, plot_raw, plot_trend, plot_detrended, clear = True, time_label = 'min'):
        self.mplot.mpl_update(tvec, signal, trend, plot_raw, plot_trend, plot_detrended, clear = True, time_label = 'min')
        self.show()
    
        

class TimeSeriesPlotView(FigureCanvas):
    def __init__(self, parent=None, width=4, height=3, dpi=100):
        self.fig1 = Figure(figsize=(width,height), dpi=dpi)
        self.fig1.clf()
        self.ax1 = self.fig1.add_subplot(111)
        #self.axes.set_xlabel('time')

        #if not signal:
        #    raise ValueError('No time or signal supplied') ###gen_func

        FigureCanvas.__init__(self, self.fig1)
        self.setParent(parent)
        
        print ('Time Series Size', FigureCanvas.sizeHint(self))
        FigureCanvas.setSizePolicy(self,
                QSizePolicy.Expanding,
                QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        
    def mpl_update(self, tvec, signal,trend, plot_raw, plot_trend, plot_detrended, clear = True, time_label = 'min'):
        self.fig1.clf()
        self.ax1 = self.fig1.add_subplot(111)
        print (plot_raw, plot_trend, plot_detrended)
        if DEBUG:
            print('mpl update called with {}, {}'.format(tvec[:10], signal[:10]))

        if clear:
            self.ax1.cla()
        if plot_raw:
            self.ax1.plot(tvec,signal,lw = 1.5, color = 'royalblue',alpha = 0.8)
        if plot_trend:
            self.ax1.plot(tvec,trend,color = 'orange',lw = 1.5) 
        if plot_detrended:
            ax2 = self.ax1.twinx()
            ax2.plot(tvec, signal - trend,'--', color = 'k',lw = 1.5) 
            ax2.set_ylabel('trend')
    
        self.ax1.set_xlabel('Time [' + time_label + ']')
        self.ax1.set_ylabel(r'signal') 
        plt.ticklabel_format(style='sci',axis='y',scilimits=(0,0))
        self.fig1.subplots_adjust(bottom = 0.15,left = 0.15, right = 0.85)
        self.draw()
        self.show()
    

        

# test case for data generating function, standard synthetic signal
def synth_signal1(T, amp, per, sigma, slope):  
    
    tvec = np.arange(T)
    trend = slope*tvec**2/tvec[-1]**2*amp
    noise = np.random.normal(0,sigma, len(tvec))
    sin = amp*np.sin(2*np.pi/per*tvec)+noise+trend

    return tvec, sin
        
    

class Error(QWidget):
    def __init__(self, message,title):
        super().__init__()
        self.message = message
        self.title = title
        self.initUI()
       
    def initUI(self):
        error = QLabel(self.message)
        self.setGeometry(300,300,220,100)
        self.setWindowTitle(self.title)
        okButton = QPushButton('OK', self)
        okButton.clicked.connect(self.close)
        main_layout_v = QVBoxLayout()
        main_layout_v.addWidget(error)
        main_layout_v.addWidget(okButton)
        self.setLayout(main_layout_v)
        self.show()



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
    
    #open_file = DataLoader()
    #dh = DataHandler()
    #dh.make_connection(open_file)
    
    #testWavlet = WavletAnalyzer()
    
    window = MainWindow()
    sys.exit(app.exec_())
        
