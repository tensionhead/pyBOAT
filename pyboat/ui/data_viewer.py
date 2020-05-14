import sys, os
import numpy as np

from PyQt5.QtWidgets import QCheckBox, QTableView, QComboBox, QFileDialog, QAction, QMainWindow, QApplication, QLabel, QLineEdit, QPushButton, QMessageBox, QSizePolicy, QWidget, QVBoxLayout, QHBoxLayout, QDialog, QGroupBox, QFormLayout, QGridLayout, QTabWidget, QTableWidget

from PyQt5.QtGui import QDoubleValidator, QIntValidator
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

import pandas as pd

from pyboat.ui.util import MessageWindow, PandasModel, posfloatV, posintV
from pyboat.ui.analysis import mkTimeSeriesCanvas, FourierAnalyzer, WaveletAnalyzer

import pyboat
from pyboat import plotting as pl

# --- monkey patch label sizes good for ui ---
pl.tick_label_size = 12
pl.label_size = 14

#from tfa_lib import wavelets as wl


class DataViewer(QWidget):
        
    def __init__(self, data, debug = False):
        super().__init__()

        # this is the data table
        self.df = data
        
        self.anaWindows = {}
        self.w_position = 0 # analysis window position offset

        self.debug = debug

        # this variable tracks the selected trajectory
        # -> DataFrame column name!
        self.signal_id= None # no signal id initially selected
        
        self.raw_signal = None # no signal initial array
        self.dt = None # gets initialized from the UI -> qset_dt
        self.T_c = None # gets initialized from the UI -> qset_T_c
        self.L = None # gets initialized from the UI -> qset_L        
        self.tvec = None # gets initialized by vector_prep
        self.time_unit = None # gets initialized by qset_time_unit

        
        # get updated with dt in -> qset_dt
        self.periodV = QDoubleValidator(bottom = 1e-16, top = 1e16)
        self.envelopeV = QDoubleValidator(bottom = 3, top = 9999999)

        
        self.initUI()
            
    #===============UI=======================================

    def initUI(self):

        self.setWindowTitle(f'DataViewer - {self.df.name}')
        self.setGeometry(80,300,900,650)
        
        self.tsCanvas = mkTimeSeriesCanvas()
        main_frame = QWidget()
        self.tsCanvas.setParent(main_frame)
        ntb = NavigationToolbar(self.tsCanvas, main_frame) # full toolbar

        
        # the table instance,
        # self.df from load_data before init_UI gets called
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
        SignalBox.setToolTip('..or just click on a signal in the table!')
                        

        main_layout_v =QVBoxLayout() # The whole Layout
        #Data selction drop-down
        dataLabel = QLabel('Select Signal', self)
        
        dt_label= QLabel('Sampling Interval:')
        dt_edit = QLineEdit()
        dt_edit.setToolTip('How much time in between two recordings?')                
        dt_edit.setMinimumSize(70,0) # no effect :/
        dt_edit.setValidator(posfloatV)
                
        unit_label= QLabel('Time Unit:')
        unit_edit = QLineEdit(self)
        unit_edit.setToolTip('Changes only the axis labels!')                
        unit_edit.setMinimumSize(70,0)        


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

        ## detrending parameter
        
        self.T_c_edit = QLineEdit()
        self.T_c_edit.setMaximumWidth(70)
        self.T_c_edit.setValidator(posfloatV)
        self.T_c_edit.setToolTip('..in time units, e.g. 120min')
        
        
        sinc_options_box = QGroupBox('Sinc Detrending')
        sinc_options_layout = QGridLayout()
        sinc_options_layout.addWidget(QLabel('Cut-off Period:'),0,0)
        sinc_options_layout.addWidget(self.T_c_edit,0,1)
        sinc_options_box.setLayout(sinc_options_layout)

        ## Amplitude envelope parameter
        L_edit = QLineEdit()
        L_edit.setMaximumWidth(70)
        L_edit.setValidator(self.envelopeV)
        L_edit.setToolTip('..in time units, e.g. 120min')
        
        
        envelope_options_box = QGroupBox('Amplitude Envelope')
        envelope_options_layout = QGridLayout()
        envelope_options_layout.addWidget(QLabel('Window Size:'),0,0)
        envelope_options_layout.addWidget(L_edit,0,1)
        envelope_options_box.setLayout(envelope_options_layout)


        # plot options box
        plot_options_box = QGroupBox('Plotting Options')
        plot_options_layout = QGridLayout()
        
        self.cb_raw = QCheckBox('Raw Signal', self)
        self.cb_trend = QCheckBox('Trend', self)
        self.cb_detrend = QCheckBox('Detrended Signal', self)
        self.cb_envelope = QCheckBox('Envelope', self)
        
        plotButton = QPushButton('Refresh Plot', self)
        plotButton.clicked.connect(self.doPlot)

        saveButton = QPushButton('Save Filter Results', self)
        saveButton.clicked.connect(self.save_out_trend)
        saveButton.setToolTip('Writes the trend and the detrended signal into a file')
        
        
        ## checkbox layout
        plot_options_layout.addWidget(self.cb_raw,0,0)
        plot_options_layout.addWidget(self.cb_trend,0,1)
        plot_options_layout.addWidget(self.cb_detrend,1,0)
        plot_options_layout.addWidget(self.cb_envelope,1,1)        
        plot_options_layout.addWidget(plotButton,2,0)
        plot_options_layout.addWidget(saveButton,2,1,1,1)
        plot_options_box.setLayout(plot_options_layout)
                
        ## checkbox signal set and change
        self.cb_raw.toggle()
        
        self.cb_raw.stateChanged.connect(self.toggle_raw)
        self.cb_trend.stateChanged.connect(self.toggle_trend)
        self.cb_detrend.stateChanged.connect(self.toggle_trend)
        self.cb_envelope.stateChanged.connect(self.toggle_envelope)
        
        #Ploting box/Canvas area
        plot_box = QGroupBox('Signal and Trend')
        plot_layout = QVBoxLayout()
        plot_layout.addWidget(self.tsCanvas)
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
        tabs.addTab(tab1,"Wavelet Analysis")
        tabs.addTab(tab2,"Fourier Transform")
 
        ## Create first tab
        tab1.parameter_box = QFormLayout()
        
        ## for wavlet params, button, etc.
        self.T_min = QLineEdit()
        self.T_min.setToolTip('This is the lower period limit')        
        self.step_num = QLineEdit()
        self.step_num.insert('200')
        self.step_num.setToolTip('Increase this number for more resolution')             
        self.T_max = QLineEdit()
        self.T_max.setToolTip('This is the upper period limit')
        
        self.p_max = QLineEdit()
        self.p_max.setToolTip('Enter a fixed value for all signals or leave blank for automatic scaling')
        
        #self.p_max.insert(str(20)) # leave blank
        
        T_min_lab = QLabel('Smallest period')
        step_lab = QLabel('Number of periods')
        T_max_lab = QLabel('Highest  period')
        p_max_lab = QLabel('Expected maximal power')
        
        T_min_lab.setWordWrap(True)
        step_lab.setWordWrap(True)
        T_max_lab.setWordWrap(True)
        p_max_lab.setWordWrap(True)
        
        
        wletButton = QPushButton('Analyze Signal', self)
        wletButton.setStyleSheet("background-color: lightblue")
        wletButton.setToolTip('Start the wavelet analysis!')        
        wletButton.clicked.connect(self.run_wavelet_ana)

        batchButton = QPushButton('Run for All', self)
        batchButton.clicked.connect(self.run_wavelets_batch)
        batchButton.setToolTip('Runs the analysis for all signals in the table')
        
        ## add  button to layout
        wlet_button_layout_h = QHBoxLayout()

        wlet_button_layout_h.addStretch(0)
        wlet_button_layout_h.addWidget(wletButton)
        wlet_button_layout_h.addStretch(0)        
        wlet_button_layout_h.addWidget(batchButton)        
        wlet_button_layout_h.addStretch(0)
        
        self.cb_use_detrended = QCheckBox('Use Detrended Signal', self)
        
        # self.cb_use_detrended.stateChanged.connect(self.toggle_use)
        self.cb_use_detrended.setChecked(True) # detrend by default

        self.cb_use_envelope = QCheckBox('Normalize with Envelope', self)
        self.cb_use_envelope.setChecked(False) # no envelope by default

        
        ## Add Wavelet analyzer options to tab1.parameter_box layout
        
        tab1.parameter_box.addRow(T_min_lab,self.T_min)
        tab1.parameter_box.addRow(step_lab, self.step_num)
        tab1.parameter_box.addRow(T_max_lab,self.T_max)
        tab1.parameter_box.addRow(p_max_lab, self.p_max)
        tab1.parameter_box.addRow(self.cb_use_detrended)
        tab1.parameter_box.addRow(self.cb_use_envelope)        
        tab1.parameter_box.addRow(wlet_button_layout_h)
        
        tab1.setLayout(tab1.parameter_box)

        # fourier button
        fButton = QPushButton('Analyze Signal', self)
        ## add  button to layout
        f_button_layout_h = QHBoxLayout()
        fButton.clicked.connect(self.run_fourier_ana)
        f_button_layout_h.addStretch(0)
        f_button_layout_h.addWidget(fButton)

        # fourier detrended switch
        self.cb_use_detrended2 = QCheckBox('Use Detrended Signal', self)
        self.cb_use_detrended2.setChecked(True) # detrend by default

        self.cb_use_envelope2 = QCheckBox('Normalize with Envelope', self)
        self.cb_use_envelope2.setChecked(False) 
        
        # fourier period or frequency view
        self.cb_FourierT = QCheckBox('Show Frequencies', self)
        self.cb_FourierT.setChecked(False) # show periods per default 

        ## Create second tab
        tab2.parameter_box = QFormLayout()
        #tab2.parameter_box.addRow(T_min_lab,self.T_min)
        #tab2.parameter_box.addRow(T_max_lab,self.T_max)
        tab2.parameter_box.addRow(self.cb_use_detrended2)
        tab2.parameter_box.addRow(self.cb_use_envelope2)        
        tab2.parameter_box.addRow(self.cb_FourierT)
        tab2.parameter_box.addRow(f_button_layout_h)
        tab2.setLayout(tab2.parameter_box)
        
        
        #Add tabs to Vbox
        ana_box.addWidget(tabs)
        #set layout of ana_widget (will be added to options layout)
        # as ana_box (containing actual layout)
        ana_widget.setLayout(ana_box)
        
        # Fix size of table_widget containing parameter boxes
        # -> it's all done via column stretches of
        # the GridLayout below
        # size_pol= QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        # ana_widget.setSizePolicy(size_pol)
        
        #==========Plot and Options Layout=======================================
        # Merge Plotting canvas and options
        plot_and_options = QWidget()
        layout = QGridLayout()
        plot_and_options.setLayout(layout)

        layout.addWidget(plot_box, 0,0,4,1)
        layout.addWidget(sinc_options_box, 0,5,1,1)
        layout.addWidget(envelope_options_box, 0,6,1,1)        
        layout.addWidget(plot_options_box, 1,5,1,2)
        layout.addWidget(ana_widget, 2,5,2,2)

        # plotting-canvas column stretch <-> 1st (0th) column
        layout.setColumnStretch(0,1) # plot should stretch
        layout.setColumnMinimumWidth(0,360) # plot should not get too small

        layout.setColumnStretch(1,0) # options shouldn't stretch
        layout.setColumnStretch(2,0) # options shouldn't stretch

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

        self.T_c_edit.textChanged[str].connect(self.qset_T_c)
        L_edit.textChanged[str].connect(self.qset_L)
        
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
        if self.debug:
            print('table column number clicked:',col_nr)
        signal_id = self.df.columns[col_nr] # DataFrame column name
        self.select_signal_and_Plot(signal_id)


    # when clicked on the header
    def Header_select(self,index):
        # recieves index
        col_nr = index
        self.SignalBox.setCurrentIndex(col_nr + 1)

        if self.debug:
            print('table column number clicked:',col_nr)
            
        signal_id = self.df.columns[col_nr] # DataFrame column name
        self.select_signal_and_Plot(signal_id)
        
    # the signal to work on, connected to selection box
    def select_signal_and_Plot(self, text):
        self.signal_id = text
        succ =  self.vector_prep(self.signal_id) # fix a raw_signal + time vector
        if not succ: # error handling done in data_prep
            print('Could not load', self.signal_id)
            return
        self.set_initial_periods()
        self.set_initial_T_c()
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

        if self.debug:
            print ('old state:',self.cb_trend.isChecked())
            
        if state == Qt.Checked:
            # user warning - no effect without T_c set
            if not self.T_c:
                self.NoTrend = MessageWindow('Specify a cut-off period!','Missing value')
                

        # signal selected?
        if self.signal_id:
            self.doPlot()
        
    def toggle_envelope (self, state):
        if state == Qt.Checked:

            # user warning - no effect without L set
            if not self.L:
                self.NoEnvelope = MessageWindow('Specify a sliding window size!','Missing value')

        # signal selected?
        if self.signal_id:
            self.doPlot()
            
    #connected to unit_edit
    def qset_time_unit(self,text):
        self.time_unit = text #self.unit_edit.text()
        if self.debug:
            print('time unit changed to:',text)


    # connected to dt_edit 
    def qset_dt(self, text):

        # checking the input is done automatically via .setValidator!
        # check,str_val,_ = posfloatV.validate(t,  0) # pos argument not used
        t = text.replace(',','.')
        try:
            self.dt = float(t)
            self.set_initial_periods(force = True)
            self.set_initial_T_c(force = True)
            # update  Validators
            self.periodV = QDoubleValidator(bottom = 2*self.dt,top = 1e16)
            self.envelopeV = QDoubleValidator(bottom = 3 * self.dt, top = self.df.shape[0] * self.dt)

            # refresh plot if a is signal selected
            if self.signal_id:
                self.doPlot()
            

        # empty input
        except ValueError:
            if self.debug:
                print('dt ValueError',text)
            pass 
       
        if self.debug:
            print('dt set to:',self.dt)


    # connected to T_c_edit
    def qset_T_c(self, text):

        # value checking done by validator, accepts also comma '1,1' !
        tc = text.replace(',','.')
        try:
            self.T_c = float(tc)

        # empty line edit
        except ValueError:
            if self.debug:
                print('T_c ValueError',text)
            pass

        if self.debug:
            print('T_c set to:',self.T_c)

    # connected to T_c_edit
    def qset_L(self, text):

        # value checking done by validator, accepts also comma '1,1' !
        L = text.replace(',','.')
        try:
            L = int(L) 

        # empty line edit
        except ValueError:
            if self.debug:
                print('L ValueError',text)
            pass

        # transform to sampling intervals
        self.L = int( L / self.dt )
                        
        if self.debug:
            print('L set to:',self.L)
            
        
    def set_initial_periods(self, force = False):

        '''
        rewrite value if force is True
        '''

        if self.debug:
            print('set_initial_periods called')

        # check if a T_min was already entered
        # or rewrite is enforced
        if not bool(self.T_min.text()) or force:             
            self.T_min.clear()
            self.T_min.insert(str(2*self.dt)) # Nyquist
        
        if np.any(self.raw_signal): # check if raw_signal already selected
            # check if a T_max was already entered            
            if not bool(self.T_max.text()) or force: 
                # default is 1/4 the observation time
                self.T_max.clear()
                T_max_ini = self.dt * 1/4 * len(self.raw_signal)
                if self.dt > 0.1:
                    T_max_ini = int(T_max_ini)                
                self.T_max.insert(str(T_max_ini))

    def set_initial_T_c(self, force = False):
        if self.debug:
            print('set_initial_T_c called')

        if np.any(self.raw_signal): # check if raw_signal already selected
            if not bool(self.T_c_edit.text()) or force: # check if a T_c was already entered
                # default is 1.5 * T_max -> 3/8 the observation time
                self.T_c_edit.clear()
                # this will trigger qset_T_c and updates the variable
                T_c_ini = self.dt * 3/8 * len(self.raw_signal)
                if self.dt > 0.1:
                    T_c_ini = int(T_c_ini)
                else:
                    T_c_ini = np.round(T_c_ini,3)
                self.T_c_edit.insert(str(T_c_ini))
                
            

    # retrieve and check set wavelet parameters
    def set_wlet_pars (self):

        # period validator
        vali = self.periodV

        # read all the LineEdits:
        
        text = self.T_min.text()
        T_min = text.replace(',','.')
        check,_,_ = vali.validate(T_min, 0)
        if self.debug:
            print('Min periodValidator output:',check, 'value:',T_min)
        if check == 0:
            self.OutOfBounds = MessageWindow("Wavelet periods out of bounds!","Value Error")
            return False
        self.T_min_value = float(T_min)

        step_num = self.step_num.text()
        check,_,_ = posintV.validate(step_num, 0)
        if self.debug:
            print('# Periods posintValidator:',check, 'value:', step_num)
        if check == 0:
            self.OutOfBounds = MessageWindow("The Number of periods must be a positive integer!","Value Error")
            return False
        self.step_num_value = int(step_num)
        
        text = self.T_max.text()
        
        T_max = text.replace(',','.')
        check,_,_ = vali.validate(T_max, 0)
        if self.debug:
            print('Max periodValidator output:',check)
            print(f'Max period value: {self.T_max.text()}')
        if check == 0 or check == 1:
            self.OutOfBounds = MessageWindow("Wavelet highest period out of bounds!","Value Error")
            return False
        self.T_max_value = float(T_max)

        text = self.p_max.text()
        p_max = text.replace(',','.')
        check,_,_ = posfloatV.validate(p_max, 0) # checks for positive float
        if check == 0:
            self.OutOfBounds = MessageWindow("Powers are positive!", "Value Error")
            return False

        # check for empty string:
        if p_max:
            self.p_max_value = float(p_max)
        else:
            self.p_max_value = None
        
        # success!
        return True
        
    def vector_prep(self, signal_id):
        ''' 
        prepares raw signal vector (NaN removal) and
        corresponding time vector 
        '''
        if self.debug:
            print('preparing', signal_id)

        # checks for empty signal_id string
        if signal_id:
            self.raw_signal = self.df[signal_id]

            # remove NaNs
            self.raw_signal =self.raw_signal[~np.isnan(self.raw_signal)]
            self.tvec =np.arange(0,len(self.raw_signal), step = 1) * self.dt
            return True # success
            
        else:
            self.NoSignalSelected = MessageWindow('Please select a signal!','No Signal')
            return False
        
    def calc_trend(self):

        ''' Uses maximal sinc window size '''
        
        trend = pyboat.sinc_smooth(raw_signal = self.raw_signal,T_c = self.T_c, dt = self.dt)
        return trend

    def calc_envelope(self):

        if self.L < 3:
            self.OutOfBounds = MessageWindow(f"Minimum sliding\nwindow size is {3*self.dt}{self.time_unit} !","Value Error")
            self.L = None
            return

        if self.L > self.df.shape[0]:
            maxL  = self.df.shape[0] * self.dt
            self.OutOfBounds = MessageWindow(f"Maximum sliding window\nsize is {maxL:.2f} {self.time_unit}!","Value Error")
            self.L = None
            return

        
        # cut of frequency set?!
        if self.T_c:
            if self.debug:
                print('calculating envelope for detrended signal', self.L, self.T_c)
            
            trend = self.calc_trend()            
            signal = self.raw_signal - trend
            envelope = pyboat.sliding_window_amplitude(signal, window_size = self.L)
            
            if self.cb_detrend.isChecked():
                return envelope

            # fits on the original signal!
            else:
                return envelope + trend

        # otherwise add the mean
        else:
            if self.debug:
                print('calculating envelope for raw signal', self.L)
            
            mean = self.raw_signal.mean()
            envelope = pyboat.sliding_window_amplitude(self.raw_signal, window_size = self.L)            
            return envelope + mean
    
        
    def doPlot(self):
        
        # update raw_signal and tvec
        succ = self.vector_prep(self.signal_id) # error handling done here
        
        if not succ:
            return False

        if self.debug:
            print("called Plotting [raw] [trend] [detrended] [envelope]",
                  self.cb_raw.isChecked(),
                  self.cb_trend.isChecked(),
                  self.cb_detrend.isChecked(),
                  self.cb_envelope.isChecked(),)
            
        # check if trend is needed
        if self.T_c and (self.cb_trend.isChecked() or self.cb_detrend.isChecked()):
            if self.debug:
                print("Calculating trend with T_c = ", self.T_c)
            trend = self.calc_trend()
                
        else:
            trend = None

        # envelope calculation
        if self.L and self.cb_envelope.isChecked():
            if self.debug:
                print("Calculating envelope with L = ", self.L)
            envelope = self.calc_envelope()
                
        else:
            envelope = None
            
        
        self.tsCanvas.fig1.clf()

        ax1 = pl.mk_signal_ax(self.time_unit, fig = self.tsCanvas.fig1)
        self.tsCanvas.fig1.add_axes(ax1)
        

        if self.debug:
            print(f'plotting signal and trend with {self.tvec[:10]}, {self.raw_signal[:10]}')
            
        if self.cb_raw.isChecked():
            pl.draw_signal(ax1, time_vector = self.tvec, signal = self.raw_signal)
            
        if trend is not None and self.cb_trend.isChecked():
            pl.draw_trend(ax1, time_vector = self.tvec, trend = trend)
                
        if trend is not None and self.cb_detrend.isChecked():
            ax2 = pl.draw_detrended(ax1, time_vector = self.tvec,
                                    detrended = self.raw_signal - trend)
            ax2.legend(fontsize = pl.tick_label_size)
        if envelope is not None and not self.cb_detrend.isChecked():
            pl.draw_envelope(ax1, time_vector = self.tvec, envelope = envelope)
            
        # plot on detrended axis
        if envelope is not None and self.cb_detrend.isChecked():
            pl.draw_envelope(ax2, time_vector = self.tvec, envelope = envelope)
            ax2.legend(fontsize = pl.tick_label_size)
            
        self.tsCanvas.fig1.subplots_adjust(bottom = 0.15,left = 0.15, right = 0.85)

        # add a simple legend
        ax1.legend(fontsize = pl.tick_label_size)

        self.tsCanvas.draw()
        self.tsCanvas.show()        
        

    def run_wavelet_ana(self):
        ''' run the Wavelet Analysis '''

        if not np.any(self.raw_signal):
            self.NoSignalSelected = MessageWindow('Please select a signal first!','No Signal')
            return False
        
        succ = self.set_wlet_pars() # Error handling done there
        if not succ:
            if self.debug:
                print('Wavelet parameters could not be set!')
            return False

        # move to set_wlet_pars?!
        if self.step_num_value > 1000:
            
            choice = QMessageBox.question(self, 'Too much periods?: ',
                                            'High number of periods: Do you want to continue?',
                                            QMessageBox.Yes | QMessageBox.No)
            if choice == QMessageBox.Yes:
                pass
            else:
                return

        # detrend for the analysis?
        if self.cb_use_detrended.isChecked() and not self.T_c:
            self.NoTrend = MessageWindow('Detrending parameter not set,\n' +
                                 'specify a cut-off period!','No Trend')
            return
        
        elif self.cb_use_detrended.isChecked():
            trend = self.calc_trend()
            signal= self.raw_signal - trend
        else:
            signal= self.raw_signal

        # amplitude normalization is downstram of detrending!
        if self.cb_use_envelope.isChecked() and not self.L:
            self.NoTrend = MessageWindow('Envelope parameter not set,\n' +
                                 'specify a sliding window size!','No Envelope')
            return

        elif self.cb_use_envelope.isChecked():
            signal = pyboat.normalize_with_envelope(signal, self.L)
        
            
        self.w_position += 20
        
        self.anaWindows[self.w_position] = WaveletAnalyzer(signal=signal,
                                                           dt=self.dt,
                                                           T_min= self.T_min_value,
                                                           T_max= self.T_max_value,
                                                           position= self.w_position,
                                                           signal_id =self.signal_id,
                                                           step_num= self.step_num_value,
                                                           p_max = self.p_max_value,
                                                           time_unit= self.time_unit,
                                                           DEBUG = self.debug)

    def run_wavelets_batch(self):

        '''
        Takes ui wavelet settings and batch processes all loaded trajectories

        No power thresholding supported atm
        '''

        if self.debug:
            print('started batch processing..')

        # reads the settings from the ui input
        succ = self.set_wlet_pars() # Error handling done there
        if not succ:
            return
        
        if self.step_num_value > 1000:
            
            choice = QMessageBox.question(self, 'Too much periods?: ',
                                            'High number of periods: Do you want to continue?',
                                            QMessageBox.Yes | QMessageBox.No)
            if choice == QMessageBox.Yes:
                pass
            else:
                return

        periods = np.linspace(self.T_min_value, self.T_max_value, self.step_num_value)        

        if self.cb_use_detrended.isChecked() and not self.T_c:
            self.NoTrend = MessageWindow('Detrending parameter not set,\n' +
                                 'specify a cut-off period!','No Trend')
            return

        if self.cb_use_envelope.isChecked() and not self.L:
            self.NoTrend = MessageWindow('Envelope parameter not set,\n' +
                                 'specify a sliding window size!','No Envelope')
            return
        

        # --- get output directory ---

        dialog = QFileDialog()
        dialog.setFileMode(QFileDialog.DirectoryOnly);
        dialog.setOption(QFileDialog.ShowDirsOnly, False);

        dir_name = dialog.getExistingDirectory(self,"Select a folder to save the results",
                                              os.getenv('HOME'))


        # dialog cancelled
        if not dir_name:
            return
        
        if self.debug:
            print('Batch output name:', dir_name)
        
        # --- Processing starts ------------------------------------------
        Nproc = 0
        # loop over columns - trajectories
        for signal_id in self.df:

            # log to terminal
            print(f'processing {signal_id}..')

            # sets self.raw_signal
            succ = self.vector_prep(signal_id)
            
            # ui silently pass over..
            if not succ:
                print(f"Can't process signal {signal_id}..")
                continue

            # detrend?!            
            if self.cb_use_detrended.isChecked():
                trend = self.calc_trend()
                signal = self.raw_signal - trend
            else:
                signal = self.raw_signal

            # amplitude normalization?
            if self.cb_use_envelope.isChecked():
                signal = pyboat.normalize_with_envelope(signal, self.L)
            else:
                signal = self.raw_signal
                
            # compute the spectrum
            modulus, wlet = pyboat.compute_spectrum(signal, self.dt, periods)
            # get maximum ridge
            ridge = pyboat.get_maxRidge(modulus)
            # generate time vector
            tvec = np.arange(0, len(signal)) * self.dt
            # evaluate along the ridge            
            ridge_results = pyboat.eval_ridge(ridge, wlet, signal, periods, tvec)

            # add the signal to the results
            ridge_results['signal'] = signal

            # add the trend and detrended trajectory

            if self.cb_use_detrended.isChecked():
                ridge_results['trend'] = pd.Series(trend)                
            
            # -- write output --
            out_path = os.path.join(dir_name, signal_id + '_wres.csv')            
            ridge_results.to_csv( out_path, index = False)
            print(f'written results to {out_path}')

            Nproc += 1
            
        print('batch processing done!')
        msg = f'Processed {Nproc} signals!\n ..saved results to {dir_name}'
        self.msg = MessageWindow(msg,'Finished' )
        
    def run_fourier_ana(self):
        if not np.any(self.raw_signal):
            self.NoSignalSelected = MessageWindow('Please select a signal first!','No Signal')
            return False

        # shift new analyser windows 
        self.w_position += 20

        if self.cb_use_detrended2.isChecked() and not self.T_c:                
            self.NoTrend = MessageWindow('Detrending not set, can not use detrended signal!','No Trend')
            return
        
        elif self.cb_use_detrended2.isChecked():
            trend = self.calc_trend()
            signal= self.raw_signal- trend
        else:
            signal= self.raw_signal

        if self.cb_use_envelope2.isChecked() and not self.L:
            self.NoTrend = MessageWindow('Envelope parameter not set,\n' +
                                 'specify a sliding window size!','No Envelope')
            return

        elif self.cb_use_envelope2.isChecked():
            signal = pyboat.normalize_with_envelope(signal, self.L)
        
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
            self.NoTrend = MessageWindow('Detrending parameter not set,\n' + 
                                 'specify a cut-off period!','No Trend')
            return

        if not np.any(self.raw_signal):
            self.NoSignalSelected = MessageWindow('Please select a signal first!','No Signal')
            return 

        if self.debug:
            print('saving trend out')

        #-------calculate trend and detrended signal------------
        trend = self.calc_trend()
        dsignal= self.raw_signal - trend

        # add everything to a pandas data frame
        data = np.array([self.raw_signal,trend,dsignal]).T # stupid pandas..
        columns = ['raw', 'trend', 'detrended']
        df_out = pd.DataFrame(data = data, columns = columns)
        #------------------------------------------------------

        if self.debug:
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

        if self.debug:
            print('selected filter:',sel_filter)
            print('out-path:',file_name)
            print('extracted extension:', file_ext)
        
        if file_ext not in ['txt','csv','xlsx']:
            self.e = MessageWindow("Ouput format not supported..\n" +
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
            if self.debug:
                print("Something went wrong during save out..")
            return
        if self.debug:
            print('Saved!')

