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
        openFile.setStatusTip('Load data')
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
        mainMenu.setNativeMenuBar(False)
        
        fileMenu = mainMenu.addMenu('&File')
        fileMenu.addAction(self.quitAction)
        fileMenu.addAction(openFile)
        
        analyzerMenu = mainMenu.addMenu('&Analyzer')
        analyzerMenu.addAction(plotSynSig)
        analyzerMenu.addAction(detrending)
        analyzerMenu.addAction(viewer)
        
        #self.home()

    #def home(self):

        #self.quitAction.triggered.connect(self.close_application)
        #self.toolBar = self.addToolBar("Extraction")
        #self.toolBar.addAction(self.quitAction)
        #self.toolBar.addAction(openFile)

        quitButton = QPushButton("Quit", self)
        quitButton.clicked.connect(self.close_application)
        quitButton.resize(quitButton.minimumSizeHint())
        quitButton.move(50,50)

        openFileButton = QPushButton("Load data",self)
        openFileButton.clicked.connect(self.load)
        quitButton.resize(quitButton.minimumSizeHint())
        openFileButton.move(120,50)


        self.show()
        
    def close_application(self):
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
        self.viewing() # initialize data viewer
###################################
    def viewing (self):
        print ('function viewing called')
        self.view = DataViewer()
        
        self.view.make_connection(self.dh) 
        #self.dh.data_request_connection(self.view)
        
        self.dh.ini_connect_analyzer_tools()
        self.view.initUI()
        print ('DataViewer and DH connected')
        
    def detrending (self):
        print ('function detrending called')
        self.detr = Detrender()
        
        self.detr.make_connection(self.dh) 
        #self.dh.data_request_connection(self.detr)
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

        try:
            self.raw_data = pd.read_excel(file_name[0], header=0)
        #except:
        #    try:
        #        self.raw_data = pd.read_csv(file_name[0])
        except:
            self.noFile = Error('No valid path or file supplied!', 'No File')
        
        
        ## TODO drop NaNs
        ## later TODO deal with 'holes'
        self.emit_values()

        

        
    def emit_values(self):
        for name, values in self.raw_data.iteritems():

            signal = values
            self.timeSignal.emit([name,signal])
            
    


class DataHandler(QWidget):
    #Add signal
    signalIds = pyqtSignal('PyQt_PyObject')
    
    def __init__(self):
        super().__init__()
        self.signal_ids = []
        self.signal_dic = {}

    def ini_connect_analyzer_tools(self):
        print ('ini_connect_analyzer_tolls called')
        
        self.signalIds.emit([self.signal_ids, self.signal_dic])
    

    def make_connection(self, signal_object):        ###########################
        signal_object.timeSignal.connect(self.get_signal) #######################

        

    @pyqtSlot('PyQt_PyObject')
    def get_signal(self, id_signal_list):
        self.signal_ids.append(id_signal_list[0])
        self.signal_dic[id_signal_list[0]] = id_signal_list[1]


    

class DataViewer(QWidget):

    @pyqtSlot('PyQt_PyObject')
    def get_signal_ids(self, signal_ids):
        print ('get_signal_ids called')
        self.series_ids = signal_ids[0]
        self.signal_dic= signal_ids[1]
        #self.tvec_dic = signal_ids[2]
        
        for id in self.series_ids:
            #self.raw_data[id+'time'] = self.tvec_dic[id]
            self.raw_data[id] = self.signal_dic[id]
        self.table_view()

    def __init__(self):
        super().__init__()
        print ('__init__ of DataViewer called')
        self.raw_data= pd.DataFrame()
        self.signal_dic = {}
        self.wletWindow = {}
        self.i = 0
        self.id='None'
    def initUI(self):
        self.plotWindow = TimeSeriesViewCanvas()
        
        print(self.plotWindow.sizeHint())
        print (type(self.plotWindow))
        
        self.setWindowTitle('DataViewer')
        self.setGeometry(20,30,900,650)
        
        main_layout_v =QVBoxLayout()
        #Data selction drop-down
        dataLable = QLabel('Select signal', self)
        signalBox = QComboBox(self)
        signalBox.addItem('None')
        for i in self.series_ids:
            print (i)
            signalBox.addItem(i)
        
        #self.table = QTableView()
        #dt = NumericParameterDialog({'Cut_off frequency': 100})
        dt_label= QLabel('Sampling intervall:')
        self.dt_edit = QLineEdit()
        self.dt_edit.insert(str(1))
        unit_label= QLabel('in:')
        unitBox = QComboBox(self)
        unitBox.addItem('min')
        unitBox.addItem('s')
        #layout.addRow(QLabel(par_name),textbox)
        
        data_selection_layout_h =QHBoxLayout()
        data_selection_layout_h.addWidget(dataLable)
        data_selection_layout_h.addWidget(signalBox)
        data_selection_layout_h.addStretch(0)
        data_selection_layout_h.addWidget(dt_label)
        data_selection_layout_h.addWidget(self.dt_edit)
        data_selection_layout_h.addStretch(0)
        data_selection_layout_h.addWidget(unit_label)
        data_selection_layout_h.addWidget(unitBox)
        data_selection_layout_h.addStretch(0)
        main_layout_v.addLayout(data_selection_layout_h)
        
        self.dt_edit.textChanged.connect(self.set_periods)
        signalBox.activated[str].connect(self.set_signal_id)
        unitBox.activated[str].connect(self.unit)
        
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
        
        #detrending parameters
        
        self.T_c = QLineEdit()
        self.T_c.insert(str(100)) 
        
        #checkbox layout
        cb_layout.addRow(cb_raw)
        cb_layout.addRow(QLabel('Cut-off period for detrending:'),self.T_c)
        cb_layout.addRow(cb_trend)
        cb_layout.addRow(cb_detrend)
        cb_layout.addRow(button_layout_h)
        
        plot_options_box.setLayout(cb_layout)
        
        # checkbox signal
        cb_raw.toggle()
        #cb_trend.toggle()
        #cb_detrend.toggle()
        
        cb_raw.stateChanged.connect(self.toggle_raw)
        cb_trend.stateChanged.connect(self.toggle_trend)
        cb_detrend.stateChanged.connect(self.toggle_detrended)
        
        self.plot_raw = bool(cb_raw.checkState() )
        self.plot_trend = bool(cb_trend.checkState() )
        self.plot_detrended = bool(cb_detrend.checkState() )
        
        # for wavlet params, button, etc.
        self.T_min = QLineEdit()
        self.step_num = QLineEdit()
        self.step_num.insert('100')
        
        self.T_max = QLineEdit()
        
        wletButton = QPushButton('Analyze signal', self)
        wletButton.clicked.connect(self.doPlot)
        # add  button to layout
        wlet_button_layout_h = QHBoxLayout()
        wletButton.clicked.connect(self.wlet_ana)
        wlet_button_layout_h.addStretch(0)
        wlet_button_layout_h.addWidget(wletButton)
        
        #Wavelet params
        
        self.cb_use_detrended = QCheckBox('Use detrended signal', self)
        self.cb_use_detrended.stateChanged.connect(self.toggle_detrended)
        
        
        
        #Analyzer options box
        ana_options_box = QGroupBox('Wavelet Analysis')
        ana_layout= QFormLayout()
        ana_layout.addRow(QLabel('Smallest sampling period:'),self.T_min)
        ana_layout.addRow(QLabel('In steps'), self.step_num)
        ana_layout.addRow(QLabel('Highest sampling period:'),self.T_max)
        ana_layout.addRow(self.cb_use_detrended)
        ana_layout.addRow(wlet_button_layout_h)
        
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
        layout.addWidget(self.table,0,0,3,6)
        layout.addWidget(self.plotWindow,3,0,3,4)
        layout.addWidget(options_box, 3,4,3,2)
        horizontalGroupBox.setLayout(layout)
        

        
        main_layout_v.addWidget(horizontalGroupBox)
        
        
        
        self.setLayout(main_layout_v)
        self.show()
        
    def toggle_raw (self, state):
        if state == Qt.Checked:
            self.plot_raw = True
        else:
            self.plot_raw = False
        print (self.plot_raw)
        
    def toggle_trend (self, state):
        print (self.plot_trend)
        if state == Qt.Checked:
            self.plot_trend = True
        else:
            self.plot_trend = False
        print (self.plot_trend)
        
    def toggle_detrended (self, state):
        if state == Qt.Checked:
            self.plot_detrended = True
            self.cb_use_detrended.setCheckState(Qt.Checked)
        else:
            self.plot_detrended = False
            self.cb_use_detrended.setCheckState(Qt.Unchecked)
        print (self.plot_detrended)
        
    def table_view(self):
        print ('table_view called')
        self.table = QTableView()
        model= PandasModel(self.raw_data)
        self.table.setModel(model)
    def unit(self,text):
        self.unit = text
        
    def set_signal_id(self, text):
        self.signal_id = text
        self.set_periods()
        self.doPlot()
        
    def set_periods(self):
        self.data_prep()
        self.T_min.clear()
        self.T_max.clear()
        self.T_min.insert(str(2*self.dt))
        self.T_max.insert(str(self.dt*len(self.raw_signal)))
        
    def data_prep(self):
        self.dt = int(self.dt_edit.text())
        try:
            self.raw_signal = np.array(self.raw_data[self.signal_id].values)
            self.raw_signal =self.raw_signal[~np.isnan(self.raw_signal)]
            self.tvec =np.arange(0,len(self.raw_signal)*self.dt,self.dt)
        except:
            pass

    def periods_changed (self):
        
        self.T_min_value = float(self.T_min.text())
        self.step_num_value = float(self.step_num.text())
        self.T_max_value =float(self.T_max.text())
       
        
    def doPlot(self):
        self.data_prep()
        T_c = float(self.T_c.text())
        try:
            self.trend = wl.sinc_smooth(raw_signal = self.raw_signal,T_c = T_c, dt = self.dt)
            self.plotWindow.mpl_update(self.tvec, self.raw_signal, self.trend, plot_raw= self.plot_raw, plot_trend=self.plot_trend, plot_detrended=self.plot_detrended)

        except:
            self.noDataSelected = Error('Please selcte one dataseries from the drop down menu or load data first!','Missing data')
            
    def wlet_ana(self):
        self.periods_changed()
        if self.T_min_value < 2*self.dt:
            self.outofBoundary = Error('Out of boundary: Please select value bigger than '+str(2*self.dt)+'(Nyqvist limit is 2*sampling interval)!','Out of boundary')
            return
        elif self.T_max_value > self.dt*len(self.raw_signal):
            self.outofBoundary = Error('Out of boundary: Please select value smaller than '+str(self.dt*len(self.raw_signal))+'(Length of data series)!','Out of boundary')
            return
        if self.step_num_value > 1000:
            
            choice = QMessageBox.question(self, 'High number: ',
                                            'High number: Do you want to continue?',
                                            QMessageBox.Yes | QMessageBox.No)
            if choice == QMessageBox.Yes:
                pass
            else:
                return
            
        if self.plot_detrended:
            signal= self.raw_signal-self.trend
        else:
            signal= self.raw_signal
        self.i = self.i+20
        
        self.wletWindow[self.i] = WaveletAnalyzer(signal=signal, dt=self.dt, T_min= self.T_min_value, T_max= self.T_max_value, position= self.i, signal_id =self.signal_id, step_num= self.step_num_value)

    def make_connection(self, datahandler_object):
        datahandler_object.signalIds.connect(self.get_signal_ids)
        #datahandler_object.dataSignal.connect(self.get_data)
        
        
        

class WaveletAnalyzer(QWidget):
    def __init__(self, signal, dt, T_min, T_max, position, signal_id, step_num):
        super().__init__()
        self.signal_id = signal_id


        periods=np.linspace(T_min, T_max, step_num)
        
        print (periods[-1])
        
        # Plot input signal
        tvec = np.arange(0,len(signal)*dt,dt)

        #=============Compute Spectrum============================
        modulus, wlet = wl.compute_spectrum(signal, dt, periods)
        #========================================================
        
        # Wavelet and signal plot
        self.waveletPlot = SpectrumCanvas()
        self.waveletPlot.plot_signal_modulus(tvec, signal,modulus,periods)
        
        self.initUI(position)
        
    def initUI(self, position):
        self.setWindowTitle('WaveletAnalyzer - '+str(self.signal_id))
        self.setGeometry(510+position,30+position,600,700)

        #Save output
        saveButton = QPushButton('Save', self)
        saveButton.clicked.connect(self.save_out)
        
        saveButton_layout_h = QHBoxLayout()
        saveButton_layout_h.addStretch(0)
        saveButton_layout_h.addWidget(saveButton)
        
        save_options_box = QGroupBox('Options')
        save_layout= QFormLayout()
        save_layout.addRow(saveButton_layout_h)
        
        save_options_box.setLayout(save_layout)
        
        main_layout = QGridLayout()
        main_layout.addWidget(self.waveletPlot, 0,0,5,5)
        main_layout.addWidget(save_options_box,5,0)
        self.setLayout(main_layout)
        
        self.show()
        
    def save_out (self):
        self.waveletPlot.save(self.signal_id)

class SpectrumCanvas(FigureCanvas):
    def __init__(self, parent=None): #, width=6, height=3, dpi=100):
        self.fig, self.axs = plt.subplots(2,1,gridspec_kw = {'height_ratios':[1, 2.5]}, sharex = True)
        
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)
        
        FigureCanvas.setSizePolicy(self,
                QSizePolicy.Expanding,
                QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

        
    def plot_signal_modulus(self,tvec, signal,modulus,periods, time_label = 'min'):
        
        # self.fig.clf() # not needed as only once initialised?!
        sig_ax = self.axs[0]
        mod_ax = self.axs[1]

        # Plot Signal

        sig_ax.plot(tvec, signal, color = 'black', lw = 1.5, alpha = 0.7)
        sig_ax.set_ylabel('signal [a.u.]')
        # Plot Wavelet Power Spectrum
        
        # aspect = len(tvec)/len(periods)
        im = mod_ax.imshow(modulus[::-1], cmap = 'viridis', vmax = 20,extent = (tvec[0],tvec[-1],periods[0],periods[-1]),aspect = 'auto')
        mod_ax.set_ylim( (periods[0],periods[-1]) )
        mod_ax.set_xlim( (tvec[0],tvec[-1]) )
        mod_ax.set_xlabel('time ' + time_label)
        
        cb = self.fig.colorbar(im,ax = mod_ax,orientation='horizontal',fraction = 0.08,shrink = .6, pad = 0.25)
        #cb.set_label('$|\mathcal{W}_{\Psi}(t,T)|^2$',rotation = '0',labelpad = 5,fontsize = 15)
        cb.set_label('Wavelet power',rotation = '0',labelpad = 5,fontsize = 10)

        mod_ax.set_xlabel('Time [' + time_label + ']')
        mod_ax.set_ylabel('Period [' + time_label + ']')
        plt.subplots_adjust(bottom = 0.11, right=0.95,left = 0.13,top = 0.95)
        self.fig.tight_layout()
        
    def save (self, signal_id):
        self.fig.savefig(str(signal_id)+'.png')

### end from wavelet_lib


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
            for i in self.series_ids:
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
        #datahandler_object.dataSignal.connect(self.get_data)
        self._connected= True
        
    @pyqtSlot('PyQt_PyObject')
    def get_signal_ids(self, signal_ids):

        self.series_ids = signal_ids[0]
        self.signal_dic= signal_ids[1]
        self.tvec_dic = signal_ids[2]
        
        for id in self.series_ids:
            self.raw_data[id+'time'] = self.tvec_dic[id]
            self.raw_data[id] = self.signal_dic[id]
        
        
    

    
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

        self.plotWindow = TimeSeriesCanvas('Synthetic Signal')

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
        self.plotWindow.mpl_update(tvec, signal)

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
        
        
        

class TimeSeriesCanvas(FigureCanvas):
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


class TimeSeriesViewCanvas(FigureCanvas):
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
            ax2.plot(tvec, signal - trend,'-', color = 'k',lw = 1.5, alpha = 0.6) 
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
    
    #testWavelet = WaveletAnalyzer()
    
    window = MainWindow()
    sys.exit(app.exec_())
        
