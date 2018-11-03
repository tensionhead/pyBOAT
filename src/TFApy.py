#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys,os
from PyQt5.QtWidgets import QCheckBox, QTableView, QComboBox, QFileDialog, QAction, QMainWindow, QApplication, QLabel, QLineEdit, QPushButton, QMessageBox, QSizePolicy, QWidget, QVBoxLayout, QHBoxLayout, QDialog, QGroupBox, QFormLayout, QGridLayout, QTabWidget, QTableWidget

from PyQt5.QtGui import QDoubleValidator, QIntValidator, QScreen

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from PyQt5.QtCore import pyqtSlot, pyqtSignal, Qt

from tfa_lib import wavelets as wl
from helper.pandasTable import PandasModel

import random
import numpy as np
from numpy import pi,e
import pandas as pd

# set matplotlib settings, no fontsize effect??!
from matplotlib import rc
# rc('font', family='sans-serif', size = 10)
# rc('lines', markeredgewidth = 0)
rc('text', usetex=False) # better for the UI

tick_label_size = 10
label_size = 12
DEBUG = False

# some Qt Validators, they accept floats with ','!         
posfloatV = QDoubleValidator(bottom = 1e-16, top = 1e16)
posintV = QIntValidator(bottom = 1,top = 9999999999)

class MainWindow(QMainWindow):
    
    def __init__(self):
        super().__init__()
        self.nViewers = 0
        self.DataViewers = {} # no Viewers yet
        self.detr = {}
        self.initUI()
        
    def initUI(self):
        
        self.setGeometry(1400,100,400,100)
        self.setWindowTitle('TFAnalyzer')

        # Actions for the menu - status bar
        main_widget = QWidget()
        
        self.quitAction = QAction("&Quit", self)
        self.quitAction.setShortcut("Ctrl+Q")
        self.quitAction.setStatusTip('Quit pyTFA')
        self.quitAction.triggered.connect(self.close_application)

        openFile = QAction("&Load data", self)
        openFile.setShortcut("Ctrl+L")
        openFile.setStatusTip('Load data')
        openFile.triggered.connect(self.Load_init_Viewer)


        # ?
        self.statusBar()

        # the menu bar
        
        mainMenu = self.menuBar()
        mainMenu.setNativeMenuBar(False)
        
        fileMenu = mainMenu.addMenu('&File')
        fileMenu.addAction(self.quitAction)
        fileMenu.addAction(openFile)
        
        # analyzerMenu = mainMenu.addMenu('&Analyzer')
        # analyzerMenu.addAction(plotSynSig)

        # main groups
        
        main_layout = QGridLayout()

        load_box = QGroupBox('Import Data')
        load_box_layout = QVBoxLayout()

        synsig_box = QGroupBox('Synthetic Signals')
        synsig_box_layout = QVBoxLayout()
        
        synsigButton = QPushButton("Start Generator",self)
        synsigButton.clicked.connect(self.init_synsig_generator)

        synsig_box_layout.addWidget(synsigButton)
        synsig_box_layout.addStretch(0)
        synsig_box.setLayout(synsig_box_layout)

        
        openFileButton = QPushButton("Open",self)
        openFileButton.clicked.connect(self.Load_init_Viewer)
        # quitButton.resize(quitButton.minimumSizeHint())
        load_box_layout.addWidget(openFileButton)
                
        self.cb_header = QCheckBox('With table header', self)
        self.cb_header.setChecked(True) # detrend by default
                
        load_box_layout.addWidget(self.cb_header)

        load_box.setLayout(load_box_layout)

        # not used right now
        quitButton = QPushButton("Quit", self)
        quitButton.clicked.connect(self.close_application)
 
        #quitButton.setMinimumSize(40,20)
        #quitButton.resize(quitButton.minimumSizeHint())

        # fill grid main layout
        main_layout.addWidget(load_box,0,0,3,2)
        main_layout.addWidget(synsig_box,0,2,2,2)
        main_layout.addWidget(quitButton,2,3,1,1)

        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        self.show()


    def init_synsig_generator(self):
        self.e = Error('Sorry, not implemented yet!',"Error")
        return
        self.test = TestWindow(self)
        
    def close_application(self):

        # no confirmation window
        if DEBUG:
            appc = QApplication.instance()
            appc.closeAllWindows()
            return

        choice = QMessageBox.question(self, 'Quitting',
                                            'Do you want to exit TFApy?',
                                            QMessageBox.Yes | QMessageBox.No)
        if choice == QMessageBox.Yes:
            print("Quitting ...")
            #sys.exit()
            appc = QApplication.instance()
            appc.closeAllWindows()
        else:
            pass
        
    def Load_init_Viewer(self, cb_status):

        cb_status = self.cb_header.isChecked()
        
        self.new_data = DataLoader()
        succ = self.new_data.load_data(cb_status)
        if not succ: # error handling done in data loader
            return

###################################

        print ('function Viewer_Ini called')
        self.nViewers += 1

        # make new DataViewer and get the data
        self.DataViewers[self.nViewers] = DataViewer()
        
        self.DataViewers[self.nViewers].make_connection(self.new_data)
        self.new_data.emit_values()
        if DEBUG:
            print ('DataLoader transferred data to DataViewer')

        # only now after transfer call the UI
        self.DataViewers[self.nViewers].initUI()



class TestWindow(QWidget):

    def __init__(self, parent = None):
        
        super().__init__()
        self.initUI()
        
    def initUI(self):

        self.setWindowTitle('Test Window')
        # self.setGeometry(510,30,550,600)

        
        main_layout = QGridLayout()

        load_box = QGroupBox('Import Data')
        load_box_layout = QVBoxLayout()

        synsig_box = QGroupBox('Synthetic Signals')
        synsig_box_layout = QVBoxLayout()
        synsigButton = QPushButton("Start Generator",self)
        synsig_box_layout.addWidget(synsigButton)
        synsig_box.setLayout(synsig_box_layout)

        # not connected
        openFileButton = QPushButton("Open",self)
        # quitButton.resize(quitButton.minimumSizeHint())
        load_box_layout.addWidget(openFileButton)
                
        cb_headers = QCheckBox('With table headers', self)
        load_box_layout.addWidget(cb_headers)
        load_box.setLayout(load_box_layout)

        # fill grid main layout
        main_layout.addWidget(load_box,0,0,3,1)
        main_layout.addWidget(synsig_box,0,2,3,1)

        self.setLayout(main_layout)
        self.show()

    
class DataLoader(QWidget):

    # the signal
    DataTransfer = pyqtSignal('PyQt_PyObject')
    
    def __init__(self):
        super().__init__()
        self.raw_df = pd.DataFrame()
        self.signal_dic = {}
        
    def load_data(self, with_header = False):

        ### If a valid file path/type is supplied data is read in and emitted

        if DEBUG:            
            file_names = ['../data_examples/PSMexamples_nohead.csv']
            print('with_header', with_header)
        else:
            # returns a list
            file_names = QFileDialog.getOpenFileName(self, 'Open File','../')
                        
        # pandas needs an integer number of header rows
        if with_header:
            header_rows = 0
        else:
            # pandas then puts sequence of integers
            header_rows = None
        
        if DEBUG:
            print (file_names)

        # defaults to white space if no extension
        file_name = file_names[0]        
        file_ext = file_name.split('.')[-1]
            
        # check for valid path
        if not os.path.isfile(file_name):
            
            self.noFile = Error('No valid path or file supplied!', 'No File')
            print ('DataLoader returned ..no values emitted')
            return False

        try:
            print('Loading',file_ext, file_name)
            # open file according to extension 
            if file_ext == 'csv':
                print("CSV")
                self.raw_df = pd.read_csv(file_name, header = header_rows)
                
            elif file_ext == 'tsv':
                print("TSV")
                self.raw_df = pd.read_csv(file_name, header=  header_rows, sep = '\t')

            elif file_ext in ['xls','xlsx']:
                print("EXCEL")
                self.raw_df = pd.read_excel(file_name, header= header_rows)

            # try white space separation as a fallback
            else:
                print('WHITESPACE')
                self.raw_df = pd.read_csv(file_name, delim_whitespace = True, header = header_rows)
            print('Raw Columns:',self.raw_df.columns)
            
        except pd.errors.ParserError:
            self.raw_df = pd.DataFrame()
            self.invalidInput = Error('Non-numeric values encountered in\n\n{}\n\nCheck input file/header?!'.format(file_name), 'Data Input Error')
            print ('DataLoader returned from error ..no values emitted')
            return False

        if DEBUG:
            print('Data Types:',self.raw_df.dtypes)
        
        # catch wrongly parsed input data
        for dtype in self.raw_df.dtypes:
            if dtype == object:
                self.raw_df = pd.DataFrame()
                self.invalidInput = Error('Non-numeric values encountered in\n\n{}\n\nCheck input file/header?!'.format(file_name), 'Data Input Error')
                print ('DataLoader returned from error ..no values emitted')
                return False
        
        # map all columns to strings for the PandasModel and SignalBox
        str_col = map(str,self.raw_df.columns)
        self.raw_df.columns = str_col
        
        #print(self.raw_df)
        
        ## TODO drop NaNs
        ## later TODO deal with 'holes'

        # gets called externally!
        # self.emit_values()
        return True
        
    def emit_values(self):
        if DEBUG:
            print('raw columns:',self.raw_df.columns)
        self.DataTransfer.emit(self.raw_df)
            
    


class DataViewer(QWidget):
        
    def __init__(self):
        super().__init__()
        self.df = None # initialize empty  
        #self.signal_dic = {}
        self.anaWindows = {}
        self.w_position = 0 # analysis window position offset
        
        self.signal_id= None # no signal id initially selected
        self.raw_signal = None # no signal initial array
        self.dt = None # gets initialized from the UI -> qset_dt
        self.T_c = None # gets initialized from the UI -> qset_T_c
        self.tvec = None # gets initialized by vector_prep
        self.time_unit = None # gets initialized by qset_time_unit

        
        # gets updated with dt in -> qset_dt
        self.periodV = QDoubleValidator(bottom = 1e-16, top = 1e16)


    
    # to get data from DataLoader instance,
    # get_df gets called when data_loader emits
    def make_connection(self, data_loader):
        data_loader.DataTransfer.connect(self.get_df)

    # should be called by DataLoader emission before initUI 
    @pyqtSlot('PyQt_PyObject')
    def get_df(self, df):
        if DEBUG:
            print ('get_df called')
        self.df = df
        
    #===============UI=======================================

    def initUI(self):
        self.plotWindow = TimeSeriesViewerCanvas()
        main_frame = QWidget()
        self.plotWindow.setParent(main_frame)
        ntb = NavigationToolbar(self.plotWindow, main_frame) # full toolbar

        # the table instance,
        # self.df created by get_df <-> DataLoader.DataTransfer signal
        DataTable = QTableView()
        model= PandasModel(self.df)
        DataTable.setModel(model)
        DataTable.setSelectionBehavior(2) # columns only
        DataTable.clicked.connect(self.Table_select) # magically transports QModelIndex
        # so that it also works for header selection
        header = DataTable.horizontalHeader() # returns QHeaderView
        header.sectionClicked.connect(self.Header_select) # magically transports QModelIndex

        # size policy for DataTable, not needed..?!
        # size_pol= QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # DataTable.setSizePolicy(size_pol)

        # the signal selection box
        SignalBox = QComboBox(self)
        
        # needs to be connected befor calling initUI
        self.setWindowTitle('DataViewer')
        self.setGeometry(2,30,900,650)
        

        main_layout_v =QVBoxLayout() # The whole Layout
        #Data selction drop-down
        dataLabel = QLabel('Select signal', self)
        
        dt_label= QLabel('Sampling intervall:')
        dt_edit = QLineEdit()
        dt_edit.setValidator(posfloatV)
                
        unit_label= QLabel('time unit:')
        unit_edit = QLineEdit(self)
        

        top_bar_box = QWidget()
        top_bar_layout = QHBoxLayout()

        top_bar_layout.addWidget(dataLabel)
        top_bar_layout.addWidget(SignalBox)
        top_bar_layout.addStretch(0)
        top_bar_layout.addWidget(dt_label)
        top_bar_layout.addWidget(dt_edit)
        top_bar_layout.addStretch(0)
        top_bar_layout.addWidget(unit_label)
        top_bar_layout.addWidget(unit_edit)
        top_bar_layout.addStretch(0)
        top_bar_box.setLayout(top_bar_layout)

        top_and_table = QGroupBox('Settings and Data')
        top_and_table_layout = QVBoxLayout()
        top_and_table_layout.addWidget(top_bar_box)
        top_and_table_layout.addWidget(DataTable)
        top_and_table.setLayout(top_and_table_layout)
        main_layout_v.addWidget(top_and_table)

        ##detrending parameters
        
        T_c_edit = QLineEdit()
        T_c_edit.setValidator(posfloatV)
        
        sinc_options_box = QGroupBox('Detrending')
        sinc_options_layout = QGridLayout()
        sinc_options_layout.addWidget(QLabel('Cut-off period for sinc:'),0,0)
        sinc_options_layout.addWidget(T_c_edit,0,1)
        sinc_options_box.setLayout(sinc_options_layout)


        # plot options box
        plot_options_box = QGroupBox('Plotting Options')
        plot_options_layout = QGridLayout()
        
        cb_raw = QCheckBox('Raw signal', self)
        cb_trend = QCheckBox('Trend', self)
        cb_detrend = QCheckBox('Detrended signal', self)
        plotButton = QPushButton('Refresh plot', self)
        plotButton.clicked.connect(self.doPlot)

        saveButton = QPushButton('Save Filter Results', self)
        saveButton.clicked.connect(self.save_out_trend)

        
        ## checkbox layout
        plot_options_layout.addWidget(cb_raw,0,0)
        plot_options_layout.addWidget(cb_trend,0,1)
        plot_options_layout.addWidget(cb_detrend,0,2)
        plot_options_layout.addWidget(plotButton,1,0)
        plot_options_layout.addWidget(saveButton,1,1,1,2)
        plot_options_box.setLayout(plot_options_layout)
                
        ## checkbox signal set and change
        cb_raw.toggle()
        
        cb_raw.stateChanged.connect(self.toggle_raw)
        cb_trend.stateChanged.connect(self.toggle_trend)
        cb_detrend.stateChanged.connect(self.toggle_detrended)
        
        self.plot_raw = bool(cb_raw.checkState() )
        self.plot_trend = bool(cb_trend.checkState() )
        self.plot_detrended = bool(cb_detrend.checkState() )
        
        #Ploting box/Canvas area
        plot_box = QGroupBox('Signal and Trend')
        plot_layout = QVBoxLayout()
        plot_layout.addWidget(self.plotWindow)
        plot_layout.addWidget(ntb)
        plot_box.setLayout(plot_layout)
        
        #Analyzer box with tabs
        ana_widget = QGroupBox("Analysis")
        ana_box = QVBoxLayout()
 
        ## Initialize tab scresen
        tabs = QTabWidget()
        tab1 = QWidget()
        tab2 = QWidget()

        ## Add tabs
        tabs.addTab(tab1,"Wavelet analysis")
        tabs.addTab(tab2,"Fourier transform")
 
        ## Create first tab
        tab1.parameter_box = QFormLayout()
        
        ## for wavlet params, button, etc.
        self.T_min = QLineEdit()
        self.step_num = QLineEdit()
        self.step_num.insert('150')
        self.T_max = QLineEdit()
        self.v_max = QLineEdit()
        self.v_max.insert(str(20))
        
        T_min_lab = QLabel('Smallest period')
        step_lab = QLabel('Number of periods')
        T_max_lab = QLabel('Highest  period')
        v_max_lab = QLabel('Expected maximal power')
        
        T_min_lab.setWordWrap(True)
        step_lab.setWordWrap(True)
        T_max_lab.setWordWrap(True)
        v_max_lab.setWordWrap(True)
        
        
        wletButton = QPushButton('Analyze Signal', self)
        wletButton.clicked.connect(self.run_wavelet_ana)
        
        ## add  button to layout
        wlet_button_layout_h = QHBoxLayout()

        wlet_button_layout_h.addStretch(0)
        wlet_button_layout_h.addWidget(wletButton)
        wlet_button_layout_h.addStretch(0)
        
        self.cb_use_detrended = QCheckBox('Use detrended signal', self)
        # self.cb_use_detrended.stateChanged.connect(self.toggle_use)
        self.cb_use_detrended.setChecked(True) # detrend by default
        
        ## Add Wavelet analyzer options to tab1.parameter_box layout
        
        tab1.parameter_box.addRow(T_min_lab,self.T_min)
        tab1.parameter_box.addRow(step_lab, self.step_num)
        tab1.parameter_box.addRow(T_max_lab,self.T_max)
        tab1.parameter_box.addRow(v_max_lab, self.v_max)
        tab1.parameter_box.addRow(self.cb_use_detrended)
        tab1.parameter_box.addRow(wlet_button_layout_h)
        
        tab1.setLayout(tab1.parameter_box)

        # fourier button
        fButton = QPushButton('Analyze signal', self)
        ## add  button to layout
        f_button_layout_h = QHBoxLayout()
        fButton.clicked.connect(self.run_fourier_ana)
        f_button_layout_h.addStretch(0)
        f_button_layout_h.addWidget(fButton)

        # fourier detrended switch
        self.cb_use_detrended2 = QCheckBox('Use detrended signal', self)
        # self.cb_use_detrended2.stateChanged.connect(self.toggle_use)
        self.cb_use_detrended2.setChecked(True) # detrend by default
        
        # fourier period or frequency view
        self.cb_FourierT = QCheckBox('Show frequencies', self)
        self.cb_FourierT.setChecked(False) # show periods per default 

        ## Create second tab
        tab2.parameter_box = QFormLayout()
        #tab2.parameter_box.addRow(T_min_lab,self.T_min)
        #tab2.parameter_box.addRow(T_max_lab,self.T_max)
        tab2.parameter_box.addRow(self.cb_use_detrended2)
        tab2.parameter_box.addRow(self.cb_FourierT)
        tab2.parameter_box.addRow(f_button_layout_h)
        tab2.setLayout(tab2.parameter_box)
        
        
        #Add tabs to Vbox
        ana_box.addWidget(tabs)
        #set layout of ana_widget (will be added to options layout)
        # as ana_box (containing actual layout)
        ana_widget.setLayout(ana_box)
        
        # Fix size of table_widget containing parameter boxes - it's all done via column stretches of
        # the GridLayout below
        # size_pol= QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        # ana_widget.setSizePolicy(size_pol)
        
        #==========Plot and Options Layout=======================================
        # Merge Plotting canvas and options
        plot_and_options = QWidget()
        layout = QGridLayout()
        plot_and_options.setLayout(layout)
        # layout.addWidget(top_bar_box,0,0,1,6)
        # layout.addWidget(DataTable,1,0,3,6)
        layout.addWidget(plot_box, 0,0,4,1)
        layout.addWidget(sinc_options_box, 0,5,1,1)
        layout.addWidget(plot_options_box, 1,5,1,1)
        layout.addWidget(ana_widget, 2,5,2,1)

        # plotting-canvas column stretch <-> 1st (0th) column
        layout.setColumnStretch(0,1) # plot should stretch
        layout.setColumnMinimumWidth(0,360) # plot should not get too small

        layout.setColumnStretch(1,0) # options shouldn't stretch

        #==========Main Layout=======================================
        main_layout_v.addWidget(plot_and_options) # is a VBox

        # populate signal selection box
        SignalBox.addItem('') # empty initial selector

        for col in self.df.columns:
            SignalBox.addItem(col)
            
        # connect to plotting machinery
        SignalBox.activated[str].connect(self.select_signal_and_Plot)
        self.SignalBox = SignalBox # to modify current index by table selections
        
        # initialize parameter fields
        dt_edit.textChanged[str].connect(self.qset_dt)
        dt_edit.insert(str(1)) # initial sampling interval is 1

        T_c_edit.textChanged[str].connect(self.qset_T_c)
        
        # set default for quick testing
        # if DEBUG:
        #    T_c_edit.insert(str(25))

        unit_edit.textChanged[str].connect(self.qset_time_unit)
        unit_edit.insert( 'min' ) # standard time unit is minutes

        self.setLayout(main_layout_v)
        self.show()

        # trigger initial plot?!
        # self.select_signal_and_Plot(self.df.columns[0])

    # when clicked into the table
    def Table_select(self,qm_index):
        # recieves QModelIndex
        col_nr = qm_index.column()
        self.SignalBox.setCurrentIndex(col_nr + 1)
        if DEBUG:
            print('table column number clicked:',col_nr)
        signal_id = self.df.columns[col_nr] # DataFrame column name
        self.select_signal_and_Plot(signal_id)


    # when clicked on the header
    def Header_select(self,index):
        # recieves index
        col_nr = index
        self.SignalBox.setCurrentIndex(col_nr + 1)

        if DEBUG:
            print('table column number clicked:',col_nr)
            
        signal_id = self.df.columns[col_nr] # DataFrame column name
        self.select_signal_and_Plot(signal_id)
        
    # the signal to work on, connected to selection box
    def select_signal_and_Plot(self, text):
        self.signal_id = text
        succ =  self.vector_prep() # fix a raw_signal + time vector
        if not succ: # error handling done in data_prep
            print('Could not load', self.signal_id)
            return
        self.set_initial_periods()
        self.doPlot()


    # probably all the toggle state variables are not needed -> read out checkboxes directly
    def toggle_raw (self, state):
        if state == Qt.Checked:
            self.plot_raw = True
        else:
            self.plot_raw = False

        # signal selected?
        if self.signal_id:
            self.doPlot()

        
    def toggle_trend (self, state):

        if DEBUG:
            print ('old state:',self.plot_trend)
        if state == Qt.Checked:

            # user warning - no effect without T_c set
            if not self.T_c:
                self.NoTrend = Error('Specify a cut-off period!','Missing value')
                
            self.plot_trend = True
        else:
            self.plot_trend = False

        # signal selected?
        if self.signal_id:
            self.doPlot()
        
    def toggle_detrended (self, state):
        if state == Qt.Checked:

            # user warning - no effect without T_c set
            if not self.T_c:
                self.NoTrend = Error('Specify a cut-off period!','Missing value')

            self.plot_detrended = True

        else:
            self.plot_detrended = False

        # signal selected?
        if self.signal_id:
            self.doPlot()

    #connected to unit_edit
    def qset_time_unit(self,text):
        self.time_unit = text #self.unit_edit.text()
        if DEBUG:
            print('time unit changed to:',text)


    # connected to dt_edit 
    def qset_dt(self, text):

        # checking the input is done automatically via .setValidator!
        # check,str_val,_ = posfloatV.validate(t,  0) # pos argument not used
        t = text.replace(',','.')
        try:
            self.dt = float(t)
            self.set_initial_periods()
            # update period Validator
            self.periodV = QDoubleValidator(bottom = 2*self.dt,top = 1e16)


        # empty input
        except ValueError:
            if DEBUG:
                print('dt ValueError',text)
            pass 
       
        if DEBUG:
            print('dt set to:',self.dt)


    # connected to T_c_edit
    def qset_T_c(self, text):

        # value checking done by validator, accepts also comma '1,1' !
        tc = text.replace(',','.')
        try:
            self.T_c = float(tc)

        # empty line edit
        except ValueError:
            if DEBUG:
                print('T_c ValueError',text)
            pass

        if DEBUG:
            print('T_c set to:',self.T_c)

        
    def set_initial_periods(self):

        if DEBUG:
            print('set_initial_periods called')
        
        self.T_min.clear()
        self.T_min.insert(str(2*self.dt)) # Nyquist
        
        if np.any(self.raw_signal): # check if raw_signal already selected
            if not bool(self.T_max.text()): # check if a T_max was already entered
                # default is # half the observation time
                self.T_max.clear()
                self.T_max.insert(str(self.dt*0.5*len(self.raw_signal))) 

    # retrieve and check set wavelet paramers
    def set_wlet_pars (self):

        # period validator
        vali = self.periodV

        # read all the LineEdits:
        
        text = self.T_min.text()
        T_min = text.replace(',','.')
        check,_,_ = vali.validate(T_min, 0)
        if DEBUG:
            print('Min periodValidator output:',check, 'value:',T_min)
        if check == 0:
            self.OutOfBounds = Error("Wavelet periods out of bounds!","Value Error")
            return False
        self.T_min_value = float(T_min)

        step_num = self.step_num.text()
        check,_,_ = posintV.validate(step_num, 0)
        if DEBUG:
            print('# Periods posintValidator:',check, 'value:', step_num)
        if check == 0:
            self.OutOfBounds = Error("Number of periods must be a positive integer!","Value Error")
            return False
        self.step_num_value = int(step_num)

        text = self.T_max.text()
        T_max = text.replace(',','.')
        check,_,_ = vali.validate(T_max, 0)
        if DEBUG:
            print('Max periodValidator output:',check)
        if check == 0:
            self.OutOfBounds = Error("Wavelet periods out of bounds!","Value Error")
            return False
        self.T_max_value = float(T_max)

        text = self.v_max.text()
        v_max = text.replace(',','.')
        check,_,_ = posfloatV.validate(v_max, 0) # checks for positive float
        if check == 0:
            self.OutOfBounds = Error("Powers are positive!", "Value Error")
            return False

        self.v_max_value = float(v_max)
        
        # success!
        return True
        
    def vector_prep(self):
        ''' 
        prepares signal vector (NaN removal) and
        corresponding time vector 
        '''
        if DEBUG:
            print('trying to prepare', self.signal_id)

        # checks for empty signal_id string
        if self.signal_id:
            print('preparing signal',self.signal_id)
            self.raw_signal = self.df[self.signal_id]

            # remove NaNs
            self.raw_signal =self.raw_signal[~np.isnan(self.raw_signal)]
            self.tvec =np.arange(0,len(self.raw_signal), step = 1) * self.dt
            return True # success
            
        else:
            self.NoSignalSelected = Error('Please select a signal!','No Signal')
            return False
        
    def calc_trend(self):
        
        trend = wl.sinc_smooth(raw_signal = self.raw_signal,T_c = self.T_c, dt = self.dt)
        return trend

        
    def doPlot(self):
        
        # update raw_signal and tvec
        succ = self.vector_prep() # error handling done here
        
        if not succ:
            return False

        if DEBUG:
            print("called Plotting [raw] [trend] [derended]",self.plot_raw,self.plot_trend,self.plot_detrended)
            
        # no trend plotting without T_cut_off value set by user
        if self.T_c and (self.plot_trend or self.plot_detrended):
            if DEBUG:
                print("Calculating trend with T_c = ", self.T_c)
            trend = self.calc_trend()
                
        else:
            trend = None

        # call the plotting routine
        self.plotWindow.mpl_update(self.tvec, self.raw_signal, trend, plot_raw= self.plot_raw, plot_trend=self.plot_trend, plot_detrended=self.plot_detrended, time_unit = self.time_unit)


    def run_wavelet_ana(self):
        ''' run the Wavelet Analysis '''

        if not np.any(self.raw_signal):
            self.NoSignalSelected = Error('Please select a signal first!','No Signal')
            return False
        
        succ = self.set_wlet_pars() # Error handling done there
        if not succ:
            if DEBUG:
                print('Wavelet parameters could not be set!')
            return False
        
        if self.step_num_value > 1000:
            
            choice = QMessageBox.question(self, 'Too much periods?: ',
                                            'High number of periods: Do you want to continue?',
                                            QMessageBox.Yes | QMessageBox.No)
            if choice == QMessageBox.Yes:
                pass
            else:
                return


        if self.cb_use_detrended.isChecked() and not self.T_c:
            self.NoTrend = Error('Detrending parameter not set,\n' +
                                 'specify a cut-off period!','No Trend')
            return

        elif self.cb_use_detrended.isChecked():
            trend = self.calc_trend()
            signal= self.raw_signal - trend
        else:
            signal= self.raw_signal
            
        self.w_position += 20
        
        self.anaWindows[self.w_position] = WaveletAnalyzerWindow(signal=signal, dt=self.dt, T_min= self.T_min_value, T_max= self.T_max_value, position= self.w_position, signal_id =self.signal_id, step_num= self.step_num_value, v_max = self.v_max_value, time_unit= self.time_unit)

    def run_fourier_ana(self):
        if not np.any(self.raw_signal):
            self.NoSignalSelected = Error('Please select a signal first!','No Signal')
            return False

        # shift new analyser windows 
        self.w_position += 20

        if self.cb_use_detrended2.isChecked() and not self.T_c:                
            self.NoTrend = Error('Detrending not set, can not use detrended signal!','No Trend')
            return
        
        elif self.cb_use_detrended2.isChecked():
            trend = self.calc_trend()
            signal= self.raw_signal- trend
        else:
            signal= self.raw_signal

        # periods or frequencies?
        if self.cb_FourierT.isChecked():
            show_T = False
        else:
            show_T = True
            
        self.anaWindows[self.w_position] = FourierAnalyzer(signal = signal,
                                                           dt = self.dt,
                                                           signal_id = self.signal_id,
                                                           position = self.w_position,
                                                           time_unit = self.time_unit,
                                                           show_T = show_T
        )

    def save_out_trend(self):

        if not self.T_c:
            self.NoTrend = Error('Detrending parameter not set,\n' + 
                                 'specify a cut-off period!','No Trend')
            return

        if not np.any(self.raw_signal):
            self.NoSignalSelected = Error('Please select a signal first!','No Signal')
            return 

        if DEBUG:
            print('saving trend out')

        #-------calculate trend and detrended signal------------
        trend = self.calc_trend()
        dsignal= self.raw_signal - trend

        # add everything to a pandas data frame
        data = np.array([self.raw_signal,trend,dsignal]).T # stupid pandas..
        columns = ['raw', 'trend', 'detrended']
        df_out = pd.DataFrame(data = data, columns = columns)
        #------------------------------------------------------

        if DEBUG:
            print('df_out', df_out[:10])
            print('trend', trend[:10])
        dialog = QFileDialog()
        options = QFileDialog.Options()

        #----------------------------------------------------------
        default_name = 'trend_' + str(self.signal_id)
        format_filter = "Text File (*.txt);; CSV ( *.csv);; Excel (*.xlsx)"
        #-----------------------------------------------------------
        file_name, sel_filter = dialog.getSaveFileName(self,"Save as",
                                              default_name,
                                              format_filter,
                                              None,
                                              options=options)

        # dialog cancelled
        if not file_name:
            return
        
        file_ext = file_name.split('.')[-1]

        if DEBUG:
            print('selected filter:',sel_filter)
            print('out-path:',file_name)
            print('extracted extension:', file_ext)
        
        if file_ext not in ['txt','csv','xlsx']:
            self.e = Error("Ouput format not supported..\n" +
                           "Please append .txt, .csv or .xlsx\n" +
                           "to the file name!",
                           "Unknown format")
            return        

        # ------the write out calls to pandas----------------
        
        float_format = '%.2f' # still old style :/
            
        if file_ext == 'txt':
            df_out.to_csv(file_name, index = False,
                          sep = '\t',
                          float_format = float_format
            )

        elif file_ext == 'csv':
            df_out.to_csv(file_name, index = False,
                          sep = ',',
                          float_format = float_format
            )

        elif file_ext == 'xlsx':
            df_out.to_excel(file_name, index = False,
                          float_format = float_format
            )

        else:
            if DEBUG:
                print("Something went wrong during save out..")
            return
        if DEBUG:
            print('Saved!')

class FourierAnalyzer(QWidget):
    def __init__(self, signal, dt, signal_id, position,time_unit, show_T, parent = None):
        super().__init__()

        self.fCanvas = FourierCanvas()
        self.fCanvas.plot_spectrum(signal,dt, time_unit, show_T)

        self.initUI(position, signal_id)

    def initUI(self, position, signal_id):

        self.setWindowTitle('Fourier spectrum ' + signal_id)
        self.setGeometry(510+position,80+position,550,600)

        main_frame = QWidget()
        self.fCanvas.setParent(main_frame)
        ntb = NavigationToolbar(self.fCanvas, main_frame)

        main_layout = QGridLayout()
        main_layout.addWidget(self.fCanvas,0,0,9,1)
        main_layout.addWidget(ntb,10,0,1,1)

        # test canvas resize
        # dummy_h = QHBoxLayout()
        # dummy_h.addStretch(0)
        # button = QPushButton('push me', self)
        # dummy_h.addWidget(button)
        # main_layout.addLayout(dummy_h,11,0)
        
        self.setLayout(main_layout)
        self.show()
        
class FourierCanvas(FigureCanvas):
    def __init__(self):
        self.fig, self.axs = plt.subplots(1,1)

        FigureCanvas.__init__(self, self.fig)

        FigureCanvas.setSizePolicy(self,
                QSizePolicy.Expanding,
                QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        
    def plot_spectrum(self,signal,dt, time_unit, show_T = True):

        #time_label = '[min]'

        fft_freqs, fpower = wl.compute_fourier(signal, dt)

        if show_T:
            # period view, omit the last bin 2/(N*dt)
            if DEBUG:
                print('Plotting Fourier Periods')
                
            # skip 0-frequency 
            self.axs.vlines(1/fft_freqs[1:],0,
                            fpower[1:],lw = 1.8,
                            alpha = 0.7, color = 'slategray')
            self.axs.set_xlabel('Periods ' + time_unit,
                                fontsize = label_size)
            self.axs.set_xscale('log')

        else:
            # frequency view
            self.axs.vlines(fft_freqs,0,
                            fpower,lw = 1.8,
                            alpha = 0.7, color = 'slategray')
            self.axs.set_xlabel('Frequency ' + time_unit + r'$^{-1}$',
                                fontsize = label_size)

        
        self.axs.set_ylabel('Fourier power', fontsize = label_size)
        self.axs.ticklabel_format(style='sci',axis='y',scilimits=(0,0))
        self.axs.tick_params(axis = 'both',labelsize = tick_label_size)




class WaveletAnalyzerWindow(QWidget):

    def __init__(self, signal, dt, T_min, T_max, position, signal_id, step_num, v_max, time_unit):
        super().__init__()
        
        self.signal_id = signal_id
        self.signal = signal
        self.v_max = v_max
        self.time_unit = time_unit

        self.periods = np.linspace(T_min, T_max, step_num)
        
        print (self.periods[-1])
        
        # Plot input signal
        self.tvec = np.arange(0,len(signal)*dt,dt)

        # no ridge yet
        self.ridge = None
        self.ridge_data = None
        self.power_thresh = None
        self.rs_win_len = None
        self.rsmoothing = None
        self._has_ridge = False # no plotted ridge

        # no anneal parameters yet
        self.anneal_pars = None

        #=============Compute Spectrum=============================================
        self.modulus, self.wlet = wl.compute_spectrum(self.signal, dt, self.periods)
        #==========================================================================


        # Wavelet ridge-readout results
        self.ResultWindows = {}
        self.w_offset = 0

        self.initUI(position)
        
    def initUI(self, position):
        self.setWindowTitle('WaveletAnalyzer - '+str(self.signal_id))
        self.setGeometry(510+position,80+position,600,700)
        
        # Wavelet and signal plot
        self.waveletPlot = SpectrumCanvas()
        main_frame = QWidget()
        self.waveletPlot.setParent(main_frame)
        ntb = NavigationToolbar(self.waveletPlot, main_frame) # full toolbar

        #-------------plot the wavelet power spectrum---------------------------
        self.waveletPlot.plot_signal_modulus(self.tvec, self.signal,self.modulus,self.periods, self.v_max, self.time_unit)
        #-----------------------------------------------------------------------

        
        #Ridge detection options box 
        ridge_opt_box = QGroupBox("Ridge detection")
        ridge_opt_layout = QGridLayout()
        ridge_opt_box.setLayout(ridge_opt_layout)
 
        #Start ridge detection
        maxRidgeButton = QPushButton('Detect maximum ridge', self)
        maxRidgeButton.clicked.connect(self.do_maxRidge_detection)

        annealRidgeButton = QPushButton('Set up ridge\nfrom annealing', self)
        annealRidgeButton.clicked.connect(self.set_up_anneal)

        # not-needed.. ridge auto-updates with power threshold!
        # drawRidgeButton = QPushButton('(Re-)Draw ridge', self)
        # drawRidgeButton.clicked.connect(self.draw_ridge)


        power_label = QLabel("Power threshold: ")
        power_thresh_edit = QLineEdit()
        power_thresh_edit.setValidator(posfloatV)

        smooth_label = QLabel("Ridge smoothing: ")
        ridge_smooth_edit = QLineEdit()
        ridge_smooth_edit.setValidator(posfloatV)

        
        plotResultsButton = QPushButton('Plot Results', self)
        plotResultsButton.clicked.connect(self.ini_plot_readout)

        self.cb_coi = QCheckBox('COI', self)
        self.cb_coi.stateChanged.connect(self.draw_coi)

        ridge_opt_layout.addWidget(maxRidgeButton,0,0,1,1)
        ridge_opt_layout.addWidget(annealRidgeButton,1,0)

        ridge_opt_layout.addWidget(power_label,0,1)
        ridge_opt_layout.addWidget(power_thresh_edit,0,2)
        
        ridge_opt_layout.addWidget(smooth_label,1,1)
        ridge_opt_layout.addWidget(ridge_smooth_edit,1,2)
        
        
        # ridge_opt_layout.addWidget(drawRidgeButton,1,3) # not needed anymore?!
        ridge_opt_layout.addWidget(self.cb_coi,0,3)
        ridge_opt_layout.addWidget(plotResultsButton,1,3)
        
        
        # put everything together
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.waveletPlot)
        main_layout.addWidget(ntb)
        main_layout.addWidget(ridge_opt_box)
        self.setLayout(main_layout)

        # initialize line edits
                
        power_thresh_edit.textChanged[str].connect(self.qset_power_thresh)
        power_thresh_edit.insert('0.0') # initialize with 0

        ridge_smooth_edit.setValidator(QIntValidator(bottom = 0,top = 9999999999))
        ridge_smooth_edit.textChanged[str].connect(self.qset_ridge_smooth)
        ridge_smooth_edit.insert('0') # initialize with 0

        self.show()

    def qset_power_thresh(self, text):

        # catch empty line edit
        if not text:
            return
        text = text.replace(',','.')
        power_thresh = float(text)
        self.power_thresh = power_thresh
            
        if DEBUG:
            print('power thresh set to: ',self.power_thresh)

        # update the plot on the fly
        if self._has_ridge:
            self.draw_ridge()

        
    def qset_ridge_smooth(self, text):

        # text = text.replace(',','.')

        # catch empty line edit
        if not text:
            return
        
        rsmooth = int(text)
        
        # make an odd window length
        if rsmooth == 0:
            self.rsmoothing = False
        elif rsmooth < 3:
            self.rs_win_len = 3
            self.rsmoothing = True
        elif rsmooth > 3 and rsmooth%2 == 0:
            self.rs_win_len = rsmooth + 1
            self.rsmoothing = True
        else:
            self.rs_win_len = rsmooth
            self.rsmoothing = True

        # update the plot on the fly
        if self._has_ridge:
            self.draw_ridge()
            
        if DEBUG:
            print('ridge smooth win_len set to: ', self.rs_win_len)

    
    def do_maxRidge_detection(self):        

        ridge_y = wl.get_maxRidge(self.modulus)
        self.ridge = ridge_y

        if not np.any(ridge_y):
            self.e = Error('No ridge found..check spectrum!','Ridge detection error')
            return

        self._has_ridge = True
        self.draw_ridge() # ridge_data made here


    def draw_ridge(self):

        ''' makes also the ridge_data !! '''

        if not self._has_ridge:
            self.e = Error('Run a ridge detection first!','No Ridge')
            return

        ridge_data = wl.make_ridge_data(self.ridge,self.modulus,self.wlet,self.periods,self.tvec,Thresh = self.power_thresh, smoothing = self.rsmoothing, win_len = self.rs_win_len)

        # plot the ridge
        ax_spec = self.waveletPlot.axs[1] # the spectrum

        # already has a plotted ridge
        if ax_spec.lines:
            ax_spec.lines = [] # remove old ridge line
            self.cb_coi.setCheckState(0)
            
        ax_spec.plot(ridge_data['time'],ridge_data['periods'],'o',color = 'crimson',alpha = 0.6,ms = 2)
        
        # refresh the canvas
        self.waveletPlot.draw()
        
        self.ridge_data = ridge_data
        
    def set_up_anneal(self):

        ''' Spawns a new AnnealConfigWindow '''

        if DEBUG:
            print('set_up_anneal called')

        # is bound to parent Wavelet Window 
        self.ac = AnnealConfigWindow(self)
        self.ac.initUI(self.periods)

        
    def do_annealRidge_detection(self, anneal_pars):

        ''' Gets called from the AnnealConfigWindow '''
        
        if anneal_pars is None:
            self.noValues = Error('No parameters set for\nsimulated annealing!','No Parameters')
            return

        # todo add out-of-bounds parameter check in config window
        ini_per = anneal_pars['ini_per']
        ini_T = anneal_pars['ini_T']
        Nsteps = int(anneal_pars['Nsteps'])
        max_jump = int(anneal_pars['max_jump'])
        curve_pen = anneal_pars['curve_pen']

        # get modulus index of initial straight line ridge
        y0 = np.where(self.periods < ini_per)[0][-1]

        ridge_y, cost = wl.find_ridge_anneal(self.modulus, y0, ini_T, Nsteps, mx_jump = max_jump,curve_pen = curve_pen)
        
        self.ridge = ridge_y

        # draw the ridge and make ridge_data
        self._has_ridge = True
        self.draw_ridge()


    def draw_coi(self):

        ax_spec = self.waveletPlot.axs[1] # the spectrum axis
        
        # COI desired?
        if bool( self.cb_coi.checkState() ):
            # compute slope of COI
            coi_m = wl.Morlet_COI(self.periods)

            # Plot the COI
            N_2 = int(len(self.signal)/2.)

            # ascending left side
            ax_spec.plot(self.tvec[:N_2+1], coi_m * self.tvec[:N_2+1], 'k-.', alpha = 0.3)

            # descending right side
            ax_spec.plot(self.tvec[N_2:], coi_m * (self.tvec[-1]- self.tvec[N_2:]),'k-.',alpha = 0.3)

        else:
            ax_spec.lines = [] # remove coi
            if self._has_ridge:
                self.draw_ridge() # re-draw ridge


        # refresh the canvas
        self.waveletPlot.draw()
        
    def ini_plot_readout(self):
        
        if not self._has_ridge:
            self.e = Error('Do a ridge detection first!','No Ridge')
            return

        self.ResultWindows[self.w_offset] = WaveletReadoutWindow(self.signal_id,
                                                                 self.ridge_data,
                                                                 time_unit = self.time_unit,
                                                                 pos_offset = self.w_offset)
        self.w_offset += 20
            
class SpectrumCanvas(FigureCanvas):
    def __init__(self, parent=None): #, width=6, height=3, dpi=100):
        self.fig, self.axs = plt.subplots(2,1,gridspec_kw = {'height_ratios':[1, 2.5]}, sharex = True)
        
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)
        
        FigureCanvas.setSizePolicy(self,
                QSizePolicy.Expanding,
                QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

        
    def plot_signal_modulus(self,tvec, signal, modulus, periods, v_max, time_unit):
        
        # self.fig.clf() # not needed as only once initialised?!
        sig_ax = self.axs[0]
        mod_ax = self.axs[1]

        # Plot Signal

        sig_ax.plot(tvec, signal, color = 'black', lw = .75, alpha = 0.7)
        sig_ax.plot(tvec, signal, '.', color = 'black', ms = 3., alpha = 0.7)
        sig_ax.set_ylabel('signal [a.u.]', fontsize = label_size) 
        sig_ax.ticklabel_format(style='sci',axis='y',scilimits=(0,0))
        sig_ax.tick_params(axis = 'both',labelsize = tick_label_size)
        
        # Plot Wavelet Power Spectrum
        
        im = mod_ax.imshow(modulus[::-1], cmap = 'viridis', vmax = v_max,extent = (tvec[0],tvec[-1],periods[0],periods[-1]),aspect = 'auto')
        mod_ax.set_ylim( (periods[0],periods[-1]) )
        mod_ax.set_xlim( (tvec[0],tvec[-1]) )
        mod_ax.grid(axis = 'y',color = '0.6', lw = 1., alpha = 0.5) # vertical grid lines

        min_power = modulus.min()
        cb_ticks = [np.ceil(min_power),v_max]
        cb = self.fig.colorbar(im,ax = mod_ax,orientation='horizontal',fraction = 0.08,shrink = .6, pad = 0.25)
        cb.set_ticks(cb_ticks)
        cb.ax.set_xticklabels(cb.ax.get_xticklabels(), fontsize=tick_label_size)
        #cb.set_label('$|\mathcal{W}_{\Psi}(t,T)|^2$',rotation = '0',labelpad = 5,fontsize = 15)
        cb.set_label('Wavelet power',rotation = '0',labelpad = -5,fontsize = 10)

        mod_ax.set_xlabel('time (' + time_unit + ')', fontsize = label_size) 
        mod_ax.set_ylabel('period (' + time_unit + ')', fontsize = label_size)
        mod_ax.tick_params(axis = 'both',labelsize = tick_label_size)
        plt.subplots_adjust(bottom = 0.11, right=0.95,left = 0.13,top = 0.95)
        self.fig.tight_layout()

class AnnealConfigWindow(QWidget):

    # the signal for the anneal parameters
    signal = pyqtSignal('PyQt_PyObject')


    def __init__(self, parent = None):
        
        super().__init__()
        # get properly initialized in set_up_anneal
        self.parentWaveletWindow = parent 

    def initUI(self, periods):
        self.setWindowTitle('Simulated Annealing')
        self.setGeometry(310,330,350,200)

        config_grid = QGridLayout()
        
        ini_per = QLineEdit(str(int(np.mean(periods)))) # start at middle of period interval
        ini_T = QLineEdit('10')
        Nsteps = QLineEdit('5000')
        max_jump = QLineEdit('3')
        curve_pen = QLineEdit('0')

        # for easy readout and emission
        self.edits = {'ini_per' : ini_per, 'ini_T' : ini_T, 'Nsteps' : Nsteps, \
                      'max_jump' : max_jump, 'curve_pen' : curve_pen}
        
        per_ini_lab = QLabel('Initial period guess')
        T_ini_lab = QLabel('Initial temperature')
        Nsteps_lab = QLabel('Number of iterations')
        max_jump_lab = QLabel('Maximal jumping distance')
        curve_pen_lab = QLabel('Curvature cost')

        per_ini_lab.setWordWrap(True)
        T_ini_lab.setWordWrap(True) 
        Nsteps_lab.setWordWrap(True) 
        max_jump_lab.setWordWrap(True)
        curve_pen_lab.setWordWrap(True)
  
        OkButton = QPushButton("Run!", self)
        OkButton.clicked.connect(self.read_emit_parameters)


        # 1st column
        config_grid.addWidget( per_ini_lab, 0,0,1,1)
        config_grid.addWidget( ini_per, 0,1,1,1)
        config_grid.addWidget( T_ini_lab, 1,0,1,1)
        config_grid.addWidget( ini_T, 1,1,1,1)
        config_grid.addWidget( curve_pen_lab, 2,0,1,1)
        config_grid.addWidget( curve_pen, 2,1,1,1)

        # 2nd column
        config_grid.addWidget( Nsteps_lab, 0,2,1,1)
        config_grid.addWidget( Nsteps, 0,3,1,1)
        config_grid.addWidget( max_jump_lab, 1,2,1,1)
        config_grid.addWidget( max_jump, 1,3,1,1)
        config_grid.addWidget( OkButton, 2,3,1,1)
        
        # set main layout
        self.setLayout(config_grid)
        self.show()


    def read_emit_parameters(self):
        
        anneal_pars = {}
        for par_key in self.edits:
            edit = self.edits[par_key]
            anneal_pars[par_key] = float(edit.text())

        if DEBUG:
            print('Anneal pars:', anneal_pars )
        self.parentWaveletWindow.do_annealRidge_detection(anneal_pars)
        # send to WaveletAnalyzer Window
        # self.signal.emit(anneal_pars)

class WaveletReadoutWindow(QWidget):

    def __init__(self, signal_id, ridge_data, time_unit, pos_offset = 0):
        super().__init__()

        self.signal_id = signal_id
        
        self.RCanvas = ReadoutCanvas()
        self.RCanvas.plot_readout(ridge_data, time_unit)
        
        self.initUI( pos_offset )

        self.ridge_data = ridge_data

    def initUI(self, position):
        
        self.setWindowTitle('Wavelet Results - ' + str(self.signal_id) )
        self.setGeometry(700 + position,260 + position,550,800)

        main_frame = QWidget()
        
        # embed the plotting canvas
        self.RCanvas.setParent(main_frame)
        ntb = NavigationToolbar(self.RCanvas, main_frame)
        
        main_layout = QGridLayout()
        main_layout.addWidget(self.RCanvas,0,0,9,1)
        main_layout.addWidget(ntb,10,0,1,1)

        # add the save Button
        SaveButton = QPushButton('Save Results', self)
        SaveButton.clicked.connect(self.save_out)

        button_layout_h = QHBoxLayout()
        button_layout_h.addWidget(SaveButton)
        button_layout_h.addStretch(1)        
        main_layout.addLayout(button_layout_h,11,0,1,1)
        
        self.setLayout(main_layout)
        self.show()

    def save_out(self):
        
        if DEBUG:
            print('saving out')

        dialog = QFileDialog()
        options = QFileDialog.Options()

        #----------------------------------------------------------
        default_name = 'TFAres_' + str(self.signal_id)
        format_filter = "Text File (*.txt);; CSV ( *.csv);; Excel (*.xlsx)"
        #-----------------------------------------------------------
        file_name, sel_filter = dialog.getSaveFileName(self,"Save as",
                                              default_name,
                                              format_filter,
                                              '(*.txt)',
                                              options=options)

        # dialog cancelled
        if not file_name:
            return
        
        file_ext = file_name.split('.')[-1]

        if DEBUG:
            print('selected filter:',sel_filter)
            print('out-path:',file_name)
            print('extracted extension:', file_ext)
            print('ridge data keys:', self.ridge_data.keys())

        
        if file_ext not in ['txt','csv','xlsx']:
            self.e = Error("Ouput format not supported..\n" +
                           "Please append .txt, .csv or .xlsx\n" +
                           "to the file name!",
                           "Unknown format")
            return
        
            
        df_out = pd.DataFrame()
        # add everything to data frame
        for key in self.ridge_data:
            df_out[key] = self.ridge_data[key]            

        # the write out calls
        float_format = '%.2f' # still old style :/
            
        if file_ext == 'txt':
            df_out.to_csv(file_name, index = False,
                          sep = '\t',
                          float_format = float_format
            )

        elif file_ext == 'csv':
            df_out.to_csv(file_name, index = False,
                          sep = ',',
                          float_format = float_format
            )

        elif file_ext == 'xlsx':
            df_out.to_excel(file_name, index = False,
                          float_format = float_format
            )

        else:
            if DEBUG:
                print("Something went wrong during save out..")
            return
        if DEBUG:
            print('Saved!')
        

class ReadoutCanvas(FigureCanvas):

    def __init__(self):
        
        self.fig, self.axs = plt.subplots(3,1, sharex = True)

        FigureCanvas.__init__(self, self.fig)

        FigureCanvas.setSizePolicy(self,
                QSizePolicy.Expanding,
                QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def plot_readout(self, ridge_data, time_unit):

        ps = ridge_data['periods']
        phases = ridge_data['phase']
        powers = ridge_data['power']
        tvec = ridge_data['time']

        self.fig.subplots_adjust(top = 0.98, left = 0.18)
        ax1 = self.axs[0]
        ax2 = self.axs[1]
        ax3 = self.axs[2]



        ax1.plot(tvec,ps, alpha = 0.8)
        ax1.set_ylabel('period (min)', fontsize = label_size)
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


        
### end Wavelets

class SyntheticSignalGenerator(QWidget):
    ''' 
    tvec: array containing the time vector
    signal: array containing the signal or 'synthetic' if synthetic signal shall be used
    default_para_dic: dictonary containing default parameters for synthetic signal creation


    '''
    # Added a signal, that emits signal name, tvec and signal values
    DataTransfer = pyqtSignal('PyQt_PyObject')  #########################

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
        self.setGeometry(300,300,450,550) #???

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
        
        self.DataTransfer.emit(['synthetic signal1_{}'.format(pdic),signal])
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
        #self.setParent(parent)
        
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

class TimeSeriesViewerCanvas(FigureCanvas):

    # dpi != 100 looks wierd?!
    def __init__(self, parent=None, width=4, height=3, dpi=100):
        self.fig1 = Figure(figsize=(width,height), dpi=dpi)
        self.fig1.clf()
        #self.axes.set_xlabel('time')

        #if not signal:
        #    raise ValueError('No time or signal supplied') ###gen_func

        FigureCanvas.__init__(self, self.fig1)
        self.setParent(parent)
        
        # print ('Time Series Size', FigureCanvas.sizeHint(self))
        FigureCanvas.setSizePolicy(self,
                QSizePolicy.Expanding,
                QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        
    def mpl_update(self, tvec, signal,trend, plot_raw, plot_trend, plot_detrended,time_unit, clear = True):
        self.fig1.clf()
        ax1 = self.fig1.add_subplot(111)

        if DEBUG:
            print('mpl update called with {}, {}'.format(tvec[:10], signal[:10]))

        if clear:
            ax1.cla()
            
        if plot_raw:
            ax1.plot(tvec,signal,lw = 1.5, color = 'royalblue',alpha = 0.8)
            
        if trend is not None and plot_trend:
            ax1.plot(tvec,trend,color = 'orange',lw = 1.5)
            
        if trend is not None and plot_detrended:
            ax2 = ax1.twinx()
            ax2.plot(tvec, signal - trend,'-', color = 'k',lw = 1.5, alpha = 0.6) 
            ax2.set_ylabel('detrended', fontsize = label_size)
            ax2.ticklabel_format(style='sci',axis='y',scilimits=(0,0))
    
        ax1.set_xlabel('time (' + time_unit + ')', fontsize = label_size) 
        ax1.set_ylabel(r'signal', fontsize = label_size) 
        ax1.ticklabel_format(style='sci',axis='y',scilimits=(0,0))
        ax1.tick_params(axis = 'both',labelsize = tick_label_size)

        self.fig1.subplots_adjust(bottom = 0.15,left = 0.15, right = 0.85)

        self.draw()
        self.show()        

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

    if DEBUG:
        print(
            '''
            ----------------
            DEBUG enabled!!
            ---------------
            ''')

        screen = app.primaryScreen()
        print('Screen: %s' % screen.name())
        size = screen.size()
        print('Size: %d x %d' % (size.width(), size.height()))
        rect = screen.availableGeometry()
        print('Available: %d x %d' % (rect.width(), rect.height()))

    window = MainWindow()

    ### to trigger instant loading:
    # window.Load_andViewerIni()
    # window.load()
    sys.exit(app.exec_())
        
