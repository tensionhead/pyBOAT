import os
import numpy as np

from PyQt5.QtWidgets import (
    QCheckBox,
    QMessageBox,
    QTableView,
    QComboBox,
    QFileDialog,
    QMainWindow,
    QLabel,
    QLineEdit,
    QPushButton,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QFormLayout,
    QGridLayout,
    QTabWidget,
)

from PyQt5.QtGui import QDoubleValidator, QRegExpValidator
from PyQt5.QtCore import Qt, QSettings, QRegExp
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

import pandas as pd

from pyboat.ui.util import (
    PandasModel,
    posfloatV,
    posintV,
    default_par_dict
)
from pyboat.ui.analysis import mkTimeSeriesCanvas, FourierAnalyzer, WaveletAnalyzer
from pyboat.ui.batch_process import BatchProcessWindow

import pyboat
from pyboat import plotting as pl

# --- monkey patch label sizes to better fit the ui ---
pl.tick_label_size = 12
pl.label_size = 14


class DataViewer(QMainWindow):
    
    def __init__(self, data, pos_offset, debug=False):
        super().__init__()

        # this is the data table
        self.df = data

        self.anaWindows = {} # allows for multiple open analysis windows
        self.w_position = 0  # analysis window position offset

        self.debug = debug

        # this variable tracks the selected trajectory
        # -> DataFrame column name!
        self.signal_id = None  # no signal id initially selected

        self.raw_signal = None  # no signal initial array
        self.dt = None  # gets initialized from the UI -> qset_dt
        self.tvec = None  # gets initialized by vector_prep
        self.time_unit = None  # gets initialized by qset_time_unit

        # get updated with dt in -> qset_dt
        self.periodV = QDoubleValidator(bottom=1e-16, top=1e16)
        self.envelopeV = QDoubleValidator(bottom=3, top=9999999)

        self.initUI(pos_offset)

    # ===========    UI    ================================

    def initUI(self, pos_offset):

        self.setWindowTitle(f"DataViewer - {self.df.name}")
        self.setGeometry(80 + pos_offset, 300 + pos_offset, 900, 650)

        # for the status bar
        main_widget = QWidget()
        self.statusBar()
                
        self.tsCanvas = mkTimeSeriesCanvas()
        main_frame = QWidget()
        self.tsCanvas.setParent(main_frame)
        ntb = NavigationToolbar(self.tsCanvas, main_frame)  # full toolbar

        # the table instance,
        DataTable = QTableView()
        model = PandasModel(self.df)
        DataTable.setModel(model)
        DataTable.setSelectionBehavior(2)  # columns only
        DataTable.clicked.connect(self.Table_select)  # magically transports QModelIndex
        # so that it also works for header selection
        header = DataTable.horizontalHeader()  # returns QHeaderView
        header.sectionClicked.connect(
            self.Header_select
        )  # magically transports QModelIndex

        # size policy for DataTable, not needed..?!
        # size_pol= QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # DataTable.setSizePolicy(size_pol)

        # the signal selection box
        SignalBox = QComboBox(self)
        SignalBox.setStatusTip("..or just click directly on a signal in the table!")

        main_layout_v = QVBoxLayout()  # The whole Layout
        # Data selction drop-down
        dataLabel = QLabel("Select Signal", self)

        dt_label = QLabel("Sampling Interval:")
        self.dt_edit = QLineEdit()
        self.dt_edit.setStatusTip("How much time in between two recordings?")
        self.dt_edit.setMinimumSize(70, 0)  # no effect :/
        self.dt_edit.setValidator(posfloatV)

        unit_label = QLabel("Time Unit:")
        self.unit_edit = QLineEdit(self)
        self.unit_edit.setStatusTip("Changes only the axis labels..")
        self.unit_edit.setMinimumSize(70, 0)

        top_bar_box = QWidget()
        top_bar_layout = QHBoxLayout()

        top_bar_layout.addWidget(dataLabel)
        top_bar_layout.addWidget(SignalBox)
        top_bar_layout.addStretch(0)
        top_bar_layout.addWidget(dt_label)
        top_bar_layout.addWidget(self.dt_edit)
        top_bar_layout.addStretch(0)
        top_bar_layout.addWidget(unit_label)
        top_bar_layout.addWidget(self.unit_edit)
        top_bar_layout.addStretch(0)
        top_bar_box.setLayout(top_bar_layout)

        top_and_table = QGroupBox("Settings and Data")
        top_and_table_layout = QVBoxLayout()
        top_and_table_layout.addWidget(top_bar_box)
        top_and_table_layout.addWidget(DataTable)
        top_and_table.setLayout(top_and_table_layout)
        main_layout_v.addWidget(top_and_table)

        ## detrending parameter

        self.T_c_edit = QLineEdit()
        self.T_c_edit.setMaximumWidth(70)
        self.T_c_edit.setValidator(posfloatV)
        self.T_c_edit.setStatusTip("..in time units, e.g. 120min")

        sinc_options_box = QGroupBox("Sinc Detrending")
        sinc_options_layout = QGridLayout()
        sinc_options_layout.addWidget(QLabel("Cut-off Period:"), 0, 0)
        sinc_options_layout.addWidget(self.T_c_edit, 0, 1)
        sinc_options_box.setLayout(sinc_options_layout)

        ## Amplitude envelope parameter
        self.wsize_edit = QLineEdit()
        self.wsize_edit.setMaximumWidth(70)
        self.wsize_edit.setValidator(self.envelopeV)
        self.wsize_edit.setStatusTip("..in time units, e.g. 120min")

        envelope_options_box = QGroupBox("Amplitude Envelope")
        envelope_options_layout = QGridLayout()
        envelope_options_layout.addWidget(QLabel("Window Size:"), 0, 0)
        envelope_options_layout.addWidget(self.wsize_edit, 0, 1)
        envelope_options_box.setLayout(envelope_options_layout)

        # plot options box
        plot_options_box = QGroupBox("Plotting Options")
        plot_options_layout = QGridLayout()

        self.cb_raw = QCheckBox("Raw Signal", self)
        self.cb_raw.setStatusTip("Plots the raw unfiltered signal")
        
        self.cb_trend = QCheckBox("Trend", self)
        self.cb_trend.setStatusTip("Plots the sinc filtered signal")
        
        self.cb_detrend = QCheckBox("Detrended Signal", self)
        self.cb_detrend.setStatusTip("Plots the signal after trend subtraction (detrending)")
        
        self.cb_envelope = QCheckBox("Envelope", self)
        self.cb_envelope.setStatusTip("Plots the estimated amplitude envelope")

        plotButton = QPushButton("Refresh Plot", self)
        plotButton.setStatusTip("Updates the plot with the new filter parameter values")    
        plotButton.clicked.connect(self.doPlot)

        saveButton = QPushButton("Save Filter Results", self)
        saveButton.clicked.connect(self.save_out_trend)
        saveButton.setStatusTip("Writes the trend and the detrended signal into a file")

        ## checkbox layout
        plot_options_layout.addWidget(self.cb_raw, 0, 0)
        plot_options_layout.addWidget(self.cb_trend, 0, 1)
        plot_options_layout.addWidget(self.cb_detrend, 1, 0)
        plot_options_layout.addWidget(self.cb_envelope, 1, 1)
        plot_options_layout.addWidget(plotButton, 2, 0)
        plot_options_layout.addWidget(saveButton, 2, 1, 1, 1)
        plot_options_box.setLayout(plot_options_layout)

        ## checkbox signal set and change
        self.cb_raw.toggle()

        self.cb_raw.stateChanged.connect(self.toggle_raw)
        self.cb_trend.stateChanged.connect(self.toggle_trend)
        self.cb_detrend.stateChanged.connect(self.toggle_trend)
        self.cb_envelope.stateChanged.connect(self.toggle_envelope)

        # Ploting box/Canvas area
        plot_box = QGroupBox("Signal and Trend")
        plot_layout = QVBoxLayout()
        plot_layout.addWidget(self.tsCanvas)
        plot_layout.addWidget(ntb)
        plot_box.setLayout(plot_layout)

        # Analyzer box with tabs
        ana_widget = QGroupBox("Analysis")
        ana_box = QVBoxLayout()

        ## Initialize tab scresen
        tabs = QTabWidget()
        tab1 = QWidget()
        tab2 = QWidget()

        ## Add tabs
        tabs.addTab(tab1, "Wavelet Analysis")
        tabs.addTab(tab2, "Fourier Transform")

        ## Create first tab
        tab1.parameter_box = QFormLayout()

        ## for wavlet params, button, etc.
        self.Tmin_edit = QLineEdit()
        self.Tmin_edit.setStatusTip("This is the lower period limit")
        self.nT_edit = QLineEdit()
        self.nT_edit.setValidator(QRegExpValidator(QRegExp('[0-9]+')))        
        self.nT_edit.setStatusTip("Increase this number for more spectral resolution")
        self.Tmax_edit = QLineEdit()
        self.Tmax_edit.setStatusTip("This is the upper period limit")

        self.pow_max_edit = QLineEdit()
        self.pow_max_edit.setStatusTip(
            "Enter a fixed value or leave blank for automatic wavelet power scaling"
        )

        Tmin_lab = QLabel("Lowest period")
        step_lab = QLabel("Number of periods")
        Tmax_lab = QLabel("Highest  period")
        pow_max_lab = QLabel("Expected maximal power")

        Tmin_lab.setWordWrap(True)
        step_lab.setWordWrap(True)
        Tmax_lab.setWordWrap(True)
        pow_max_lab.setWordWrap(True)

        wletButton = QPushButton("Analyze Signal", self)
        wletButton.setStyleSheet("background-color: lightblue")
        wletButton.setStatusTip("Opens the wavelet analysis..")
        wletButton.clicked.connect(self.run_wavelet_ana)

        batchButton = QPushButton("Analyze All..", self)
        batchButton.clicked.connect(self.run_batch)
        batchButton.setStatusTip("Starts the batch processing")

        ## add  button to layout
        wlet_button_layout_h = QHBoxLayout()

        wlet_button_layout_h.addStretch(0)
        wlet_button_layout_h.addWidget(wletButton)
        wlet_button_layout_h.addStretch(0)
        wlet_button_layout_h.addWidget(batchButton)
        wlet_button_layout_h.addStretch(0)

        self.cb_use_detrended = QCheckBox("Use Detrended Signal", self)
        # self.cb_use_detrended.stateChanged.connect(self.toggle_trend)
        self.cb_use_detrended.setChecked(True)  # detrend by default

        self.cb_use_envelope = QCheckBox("Normalize with Envelope", self)
        self.cb_use_envelope.stateChanged.connect(self.toggle_envelope)        
        self.cb_use_envelope.setChecked(False)  # no envelope by default

        ## Add Wavelet analyzer options to tab1.parameter_box layout

        tab1.parameter_box.addRow(Tmin_lab, self.Tmin_edit)
        tab1.parameter_box.addRow(Tmax_lab, self.Tmax_edit)
        tab1.parameter_box.addRow(step_lab, self.nT_edit)        
        tab1.parameter_box.addRow(pow_max_lab, self.pow_max_edit)
        tab1.parameter_box.addRow(self.cb_use_detrended)
        tab1.parameter_box.addRow(self.cb_use_envelope)
        tab1.parameter_box.addRow(wlet_button_layout_h)

        tab1.setLayout(tab1.parameter_box)

        # fourier button
        fButton = QPushButton("Analyze Signal", self)
        ## add  button to layout
        f_button_layout_h = QHBoxLayout()
        fButton.clicked.connect(self.run_fourier_ana)
        f_button_layout_h.addStretch(0)
        f_button_layout_h.addWidget(fButton)

        # fourier detrended switch
        self.cb_use_detrended2 = QCheckBox("Use Detrended Signal", self)
        self.cb_use_detrended2.setChecked(True)  # detrend by default

        self.cb_use_envelope2 = QCheckBox("Normalize with Envelope", self)
        self.cb_use_envelope2.setChecked(False)

        # fourier period or frequency view
        self.cb_FourierT = QCheckBox("Show Frequencies", self)
        self.cb_FourierT.setChecked(False)  # show periods per default

        ## Create second tab
        tab2.parameter_box = QFormLayout()
        # tab2.parameter_box.addRow(Tmin_lab,self.Tmin)
        # tab2.parameter_box.addRow(Tmax_lab,self.Tmax)
        tab2.parameter_box.addRow(self.cb_use_detrended2)
        tab2.parameter_box.addRow(self.cb_use_envelope2)
        tab2.parameter_box.addRow(self.cb_FourierT)
        tab2.parameter_box.addRow(f_button_layout_h)
        tab2.setLayout(tab2.parameter_box)

        # Add tabs to Vbox
        ana_box.addWidget(tabs)
        # set layout of ana_widget (will be added to options layout)
        # as ana_box (containing actual layout)
        ana_widget.setLayout(ana_box)

        # Fix size of table_widget containing parameter boxes
        # -> it's all done via column stretches of
        # the GridLayout below
        # size_pol= QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        # ana_widget.setSizePolicy(size_pol)

        # ==========Plot and Options Layout=================================
        # Merge Plotting canvas and options
        plot_and_options = QWidget()
        layout = QGridLayout()
        plot_and_options.setLayout(layout)

        layout.addWidget(plot_box, 0, 0, 4, 1)
        layout.addWidget(sinc_options_box, 0, 5, 1, 1)
        layout.addWidget(envelope_options_box, 0, 6, 1, 1)
        layout.addWidget(plot_options_box, 1, 5, 1, 2)
        layout.addWidget(ana_widget, 2, 5, 2, 2)

        # plotting-canvas column stretch <-> 1st (0th) column
        layout.setColumnStretch(0, 1)  # plot should stretch
        layout.setColumnMinimumWidth(0, 360)  # plot should not get too small

        layout.setColumnStretch(1, 0)  # options shouldn't stretch
        layout.setColumnStretch(2, 0)  # options shouldn't stretch

        # ==========Main Layout=======================================
        main_layout_v.addWidget(plot_and_options)  # is a VBox

        # populate signal selection box
        SignalBox.addItem("")  # empty initial selector

        for col in self.df.columns:
            SignalBox.addItem(col)

        # connect to plotting machinery
        SignalBox.activated[str].connect(self.select_signal_and_Plot)
        # to modify current index by table selections        
        self.SignalBox = SignalBox  

        # --- connect some parameter fields ---
        
        self.dt_edit.textChanged[str].connect(self.qset_dt)
        self.unit_edit.textChanged[str].connect(self.qset_time_unit)
        
        # --- initialize parameter fields from settings ---

        self.load_settings()
                
        main_widget.setLayout(main_layout_v)
        self.setCentralWidget(main_widget)        
        self.show()

        # trigger initial plot?!
        # self.select_signal_and_Plot(self.df.columns[0])

    # when clicked into the table
    def Table_select(self, qm_index):
        # recieves QModelIndex
        col_nr = qm_index.column()
        self.SignalBox.setCurrentIndex(col_nr + 1)
        if self.debug:
            print("table column number clicked:", col_nr)
        signal_id = self.df.columns[col_nr]  # DataFrame column name
        self.select_signal_and_Plot(signal_id)

    # when clicked on the header
    def Header_select(self, index):
        # recieves index
        col_nr = index
        self.SignalBox.setCurrentIndex(col_nr + 1)

        if self.debug:
            print("table column number clicked:", col_nr)

        signal_id = self.df.columns[col_nr]  # DataFrame column name
        self.select_signal_and_Plot(signal_id)

    # the signal to work on, connected to selection box
    def select_signal_and_Plot(self, text):
        self.signal_id = text
        succ = self.vector_prep(self.signal_id)  # fix a raw_signal + time vector
        if not succ:  # error handling done in data_prep
            print("Could not load", self.signal_id)
            return
        self.set_initial_periods()
        self.set_initial_T_c()
        self.doPlot()

    # probably all the toggle state variables are not needed -> read out checkboxes directly
    def toggle_raw(self, state):
        if state == Qt.Checked:
            self.plot_raw = True
        else:
            self.plot_raw = False

        # signal selected?
        if self.signal_id:
            self.doPlot()

    def toggle_trend(self, state):

        if self.debug:
            print("new state:", self.cb_trend.isChecked())

        # trying to plot the trend
        if state == Qt.Checked:
            T_c = self.get_T_c(self.T_c_edit)
            if not T_c:
                self.cb_trend.setChecked(False)
                # self.cb_use_detrended.setChecked(False)
                
        # signal selected?
        if self.signal_id:
            self.doPlot()

    def toggle_envelope(self, state):
                         
        # signal selected?
        if self.signal_id:
            # trying to plot the envelope
            if state == Qt.Checked:                
                window_size = self.get_wsize(self.wsize_edit)
                if not window_size:
                    self.cb_envelope.setChecked(False)
                    self.cb_use_envelope.setChecked(False)
            
            self.doPlot()

    # connected to unit_edit
    def qset_time_unit(self, text):
        self.time_unit = text  # self.unit_edit.text()
        if self.debug:
            print("time unit changed to:", text)

    # connected to dt_edit
    def qset_dt(self, text):

        '''
        Triggers the rewrite of the initial periods and
        cut-off period T_c
        '''

        # checking the input is done automatically via .setValidator!
        # check,str_val,_ = posfloatV.validate(t,  0) # pos argument not used
        t = text.replace(",", ".")
        try:
            self.dt = float(t)
            self.set_initial_periods(force=False)
            self.set_initial_T_c(force=False)
            # update  Validators
            self.periodV = QDoubleValidator(bottom=2 * self.dt, top=1e16)
            self.envelopeV = QDoubleValidator(
                bottom=3 * self.dt, top=self.df.shape[0] * self.dt
            )

            # refresh plot if a is signal selected
            if self.signal_id:
                self.doPlot()

        # empty input
        except ValueError:
            if self.debug:
                print("dt ValueError", text)
            pass

        if self.debug:
            print("dt set to:", self.dt)

    def get_T_c(self, T_c_edit):

        '''
        Uses self.T_c_edit, argument just for clarity. Checks
        for empty input, this function only gets called when
        a detrending operation is requested. Hence, an empty
        QLineEdit will display a user warning and return nothing..
        '''

        # value checking done by validator, accepts also comma '1,1' !
        tc = T_c_edit.text().replace(",", ".")        
        try:
            T_c = float(tc)
            if self.debug:
                print("T_c set to:", T_c)
            return T_c
        
        # empty line edit
        except ValueError:

            msgBox = QMessageBox()
            msgBox.setWindowTitle('Missing Parameter')
            msgBox.setText(
                'Detrending parameter not set, specify a cut-off period!')
            msgBox.exec()
                        
            if self.debug:
                print("T_c ValueError", tc)
            return None
        
    def get_wsize(self, wsize_edit):

        # value checking done by validator, accepts also comma '1,1' !
        window_size = wsize_edit.text().replace(",", ".")
        try:
            window_size = float(window_size)

        # empty line edit
        except ValueError:

            msgBox = QMessageBox()
            msgBox.setWindowTitle('Missing Parameter')            
            msgBox.setText(
                'Amplitude envelope parameter not set, specify a sliding window size!')

            msgBox.exec()
            
            if self.debug:
                print("window_size ValueError", window_size)
            return None

        if window_size / self.dt < 4:

            msgBox = QMessageBox()
            msgBox.setWindowTitle("Out of Bounds")
            msgBox.setText(
                f'''Minimal sliding window size for envelope estimation is {4*self.dt} {self.time_unit}!''')
            msgBox.exec()

            return None

        if window_size / self.dt > self.df.shape[0]:
            max_window_size = self.df.shape[0] * self.dt

            msgBox = QMessageBox()
            msgBox.setWindowTitle("Out of Bounds")
            msgBox.setText(
                f"Maximal sliding window size for envelope estimation is {max_window_size:.2f} {self.time_unit}!")
            msgBox.exec()
            
            return None
        
        if self.debug:
            print("window_size set to:", window_size)
            
        return window_size

    def set_initial_periods(self, force=False):

        """
        rewrite value if force is True
        """

        if self.debug:
            print("set_initial_periods called")

        # check if a Tmin was already entered
        # or rewrite if enforced
        if not bool(self.Tmin_edit.text()) or force:
            self.Tmin_edit.clear()
            self.Tmin_edit.insert(str(2 * self.dt))  # Nyquist

        if np.any(self.raw_signal):  # check if raw_signal already selected
            # check if a Tmax was already entered
            if not bool(self.Tmax_edit.text()) or force:
                # default is 1/4 the observation time
                self.Tmax_edit.clear()
                Tmax_ini = self.dt * 1 / 4 * len(self.raw_signal)
                if self.dt > 0.1:
                    Tmax_ini = int(Tmax_ini)
                self.Tmax_edit.insert(str(Tmax_ini))

    def set_initial_T_c(self, force=False):
        if self.debug:
            print("set_initial_T_c called")

        if np.any(self.raw_signal):  # check if raw_signal already selected
            # check if a T_c was already entered
            if (
                not bool(self.T_c_edit.text()) or force
            ):  
                # default is 1.5 * Tmax -> 3/8 the observation time
                self.T_c_edit.clear()
                
                T_c_ini = self.dt * 3 / 8 * len(self.raw_signal)
                if self.dt > 0.1:
                    T_c_ini = int(T_c_ini)
                else:
                    T_c_ini = np.round(T_c_ini, 3)

                self.T_c_edit.insert(str(T_c_ini))

    def set_wlet_pars(self):

        '''
        Retrieves and checks the set wavelet parameters
        of the 'Analysis' input box reading the following
        QLineEdits:

        self.Tmin_edit
        self.Tmax_edit
        self.nT_edit
        self.pow_max_edit

        Further the checkboxes regarding detrending and amplitude
        normalization are evaluated. And

        self.get_wsize()
        self.get_T_c()

        are called if needed. 

        Returns
        -------

        wlet_pars : dictionary holding the retrieved parameters,
                    window_size and T_c are set to None if no amplitude
                    normalization or detrending operation should be done

        '''

        wlet_pars = {}

        # period validator
        vali = self.periodV

        # -- read all the QLineEdits --

        text = self.Tmin_edit.text()
        text = text.replace(",", ".")
        check, _, _ = vali.validate(text, 0)
        if self.debug:
            print("Min periodValidator output:", check, "value:", text)
        if check == 0:

            msgBox = QMessageBox()
            msgBox.setText("Lowest period out of bounds!")
            msgBox.exec()

            return False

        Tmin = float(text)

        if Tmin < 2 * self.dt:

            Tmin = 2 * self.dt
            self.Tmin_edit.clear()
            self.Tmin_edit.insert(str(Tmin))
            
            msgBox = QMessageBox()
            msgBox.setWindowTitle('Warning')
            msg = f"Lowest period set to Nyquist limit: {Tmin} {self.time_unit}!"
            msgBox.setText(msg)
            msgBox.exec()
        
        wlet_pars['Tmin'] = Tmin
                
        step_num = self.nT_edit.text()
        check, _, _ = posintV.validate(step_num, 0)
        if self.debug:
            print("# Periods posintValidator:", check, "value:", step_num)
        if check == 0:

            msgBox = QMessageBox()
            msgBox.setText(
                "The Number of periods must be a positive integer!")
            msgBox.exec()
            return False

        wlet_pars['step_num'] = int(step_num)
        if int(step_num) > 1000:

            choice = QMessageBox.question(
                self,
                "Too much periods?: ",
                "High number of periods: Do you want to continue?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if choice == QMessageBox.Yes:
                pass
            else:
                return False        
        
        text = self.Tmax_edit.text()
        Tmax = text.replace(",", ".")
        check, _, _ = vali.validate(Tmax, 0)
        if self.debug:
            print("Max periodValidator output:", check)
            print(f"Max period value: {self.Tmax_edit.text()}")
        if check == 0 or check == 1:

            msgBox = QMessageBox()
            msgBox.setText("Highest periods out of bounds!")
            msgBox.exec()
            
            return False
        wlet_pars['Tmax'] = float(Tmax)

        text = self.pow_max_edit.text()
        pow_max = text.replace(",", ".")
        check, _, _ = posfloatV.validate(pow_max, 0)  # checks for positive float
        if check == 0:

            msgBox = QMessageBox()
            msgBox.setText("Maximal power must be positive!")
            msgBox.exec()

            return False

        # check for empty string:
        if pow_max:
            wlet_pars['pow_max'] = float(pow_max)
        else:
            wlet_pars['pow_max'] = None

        # -- the checkboxes --
            
        # detrend for the analysis?
        if self.cb_use_detrended.isChecked():
            T_c = self.get_T_c(self.T_c_edit)
            if T_c is None:
                return False # abort settings            
            wlet_pars['T_c'] = T_c
        else:
            # indicates no detrending requested
            wlet_pars['T_c'] = None
            
        # amplitude normalization is downstram of detrending!
        if self.cb_use_envelope.isChecked():
            window_size = self.get_wsize(self.wsize_edit)        
            if window_size is None:
                return False # abort settings                        
            wlet_pars['window_size'] = window_size
        else:
            # indicates no ampl. normalization
            wlet_pars['window_size'] = None
                    
        # success!
        return wlet_pars

    def vector_prep(self, signal_id):
        """ 
        prepares raw signal vector (NaN removal) and
        corresponding time vector 
        """
        if self.debug:
            print("preparing", signal_id)

        # checks for empty signal_id string
        if signal_id:
            raw_signal = self.df[signal_id]

            NaNswitches = np.sum( np.diff( np.isnan(raw_signal) ) )
            if NaNswitches > 1:
                print(f'Warning, non-contiguous NaN region found in {signal_id}!')

                msgBox = QMessageBox()
                msgBox.setText(
                    '''
                    Non contiguous regions of missing values 
                    encountered, using linear interpolation. 

                    Try 'Import..' from the main menu 
                    to interpolate missing values for all signals!
                    ''')                
                msgBox.exec()
                
                self.raw_signal = pyboat.core.interpolate_NaNs(raw_signal)
            else:
                # remove contiguous (like trailing) NaN region
                self.raw_signal = raw_signal[~np.isnan(raw_signal)]

            self.tvec = np.arange(0, len(self.raw_signal), step=1) * self.dt
            return True  # success

        else:

            msgBox = QMessageBox()
            msgBox.setText("Please select a signal!")
            msgBox.exec()
            return False

    def calc_trend(self):

        """ Uses maximal sinc window size """
        
        T_c = self.get_T_c(self.T_c_edit)
        if not T_c:
            return
        if self.debug:
            print("Calculating trend with T_c = ", T_c)
        
        trend = pyboat.sinc_smooth(raw_signal=self.raw_signal, T_c = T_c, dt=self.dt)
        return trend

    def calc_envelope(self):

        window_size = self.get_wsize(self.wsize_edit)
        if not window_size:
            return
        if self.debug:
            print("Calculating envelope with window_size = ", window_size)
            

        # cut of frequency set?!
        if self.get_T_c(self.T_c_edit):

            trend = self.calc_trend()
            if trend is None:
                return
            
            signal = self.raw_signal - trend
            envelope = pyboat.sliding_window_amplitude(signal,
                                                       window_size,
                                                       dt = self.dt)

            if self.cb_detrend.isChecked():
                return envelope

            # fits on the original signal!
            else:
                return envelope + trend

        # otherwise add the mean
        else:
            if self.debug:
                print("calculating envelope for raw signal", window_size)

            mean = self.raw_signal.mean()
            envelope = pyboat.sliding_window_amplitude(
                self.raw_signal, window_size = window_size, dt = self.dt
            )
            return envelope + mean

    def doPlot(self):

        '''
        Checks the checkboxes for trend and envelope..
        '''

        # update raw_signal and tvec
        succ = self.vector_prep(self.signal_id)  # error handling done here

        if not succ:
            return False

        if self.debug:
            print(
                "called Plotting [raw] [trend] [detrended] [envelope]",
                self.cb_raw.isChecked(),
                self.cb_trend.isChecked(),
                self.cb_detrend.isChecked(),
                self.cb_envelope.isChecked(),
            )

        # check if trend is needed
        if self.cb_trend.isChecked() or self.cb_detrend.isChecked():
            trend = self.calc_trend()
            if trend is None:
                return
        else:
            trend = None

        # envelope calculation
        if self.cb_envelope.isChecked():
            envelope = self.calc_envelope()
            if envelope is None:
                return

        else:
            envelope = None

        self.tsCanvas.fig1.clf()

        ax1 = pl.mk_signal_ax(self.time_unit, fig=self.tsCanvas.fig1)
        self.tsCanvas.fig1.add_axes(ax1)

        if self.debug:
            print(
                f"plotting signal and trend with {self.tvec[:10]}, {self.raw_signal[:10]}"
            )

        if self.cb_raw.isChecked():
            pl.draw_signal(ax1, time_vector=self.tvec, signal=self.raw_signal)

        if trend is not None and self.cb_trend.isChecked():
            pl.draw_trend(ax1, time_vector=self.tvec, trend=trend)

        if trend is not None and self.cb_detrend.isChecked():
            ax2 = pl.draw_detrended(
                ax1, time_vector=self.tvec, detrended=self.raw_signal - trend
            )
            ax2.legend(fontsize=pl.tick_label_size, loc='lower left')
        if envelope is not None and not self.cb_detrend.isChecked():
            pl.draw_envelope(ax1, time_vector=self.tvec, envelope=envelope)

        # plot on detrended axis
        if envelope is not None and self.cb_detrend.isChecked():
            pl.draw_envelope(ax2, time_vector=self.tvec, envelope=envelope)
            ax2.legend(fontsize=pl.tick_label_size)

        self.tsCanvas.fig1.subplots_adjust(bottom=0.15, left=0.15, right=0.85)

        # add a simple legend
        ax1.legend(fontsize=pl.tick_label_size)

        self.tsCanvas.draw()
        self.tsCanvas.show()

    def run_wavelet_ana(self):
        """ run the Wavelet Analysis """

        if not np.any(self.raw_signal):

            msgBox = QMessageBox()
            msgBox.setWindowTitle("No Signal")
            msgBox.setText("Please select a signal first!")
            msgBox.exec()

            return False

        wlet_pars = self.set_wlet_pars()  # Error handling done there
        if not wlet_pars:
            if self.debug:
                print("Wavelet parameters could not be set!")
            return False

        if self.cb_use_detrended.isChecked():
            trend = self.calc_trend()
            signal = self.raw_signal - trend
        else:
            signal = self.raw_signal

        if self.cb_use_envelope.isChecked():
            window_size = self.get_wsize(self.wsize_edit)
            signal = pyboat.normalize_with_envelope(signal, window_size, dt = self.dt)

        self.w_position += 20
        
        self.anaWindows[self.w_position] = WaveletAnalyzer(
            signal=signal,
            dt=self.dt,
            Tmin = wlet_pars['Tmin'],
            Tmax = wlet_pars['Tmax'],
            pow_max = wlet_pars['pow_max'],
            step_num=wlet_pars['step_num'],
            position=self.w_position,
            signal_id=self.signal_id,
            time_unit=self.time_unit,            
            DEBUG=self.debug,
        )

    def run_batch(self):

        """
        Takes the ui wavelet settings and 
        spwans the batch processing Widget
        """

        # reads the wavelet analysis settings from the ui input
        wlet_pars = self.set_wlet_pars()  # Error handling done there
        if not wlet_pars:
            return

        if self.debug:
            print(f'Started batch processing with {wlet_pars}')

        # Spawning the batch processing config widget
        # is bound to parent Wavelet Window 
        self.bc = BatchProcessWindow(self, self.debug)
        self.bc.initUI(wlet_pars)

        return
        print("batch processing done!")
        
    def run_fourier_ana(self):
        if not np.any(self.raw_signal):

            msgBox = QMessageBox()
            msgBox.setWindowTitle("No Signal")
            msgBox.setText("Please select a signal first!")
            msgBox.exec()
            return False

        # shift new analyser windows
        self.w_position += 20

        if self.cb_use_detrended2.isChecked():
            trend = self.calc_trend()
            signal = self.raw_signal - trend
        else:
            signal = self.raw_signal

        if self.cb_use_envelope2.isChecked():
            window_size = self.get_wsize(self.wsize_edit)
            signal = pyboat.normalize_with_envelope(signal, window_size, self.dt)

        # periods or frequencies?
        if self.cb_FourierT.isChecked():
            show_T = False
        else:
            show_T = True

        self.anaWindows[self.w_position] = FourierAnalyzer(
            signal=signal,
            dt=self.dt,
            signal_id=self.signal_id,
            position=self.w_position,
            time_unit=self.time_unit,
            show_T=show_T,
        )

    def save_out_trend(self):

        if not np.any(self.raw_signal):

            msgBox = QMessageBox()
            msgBox.setWindowTitle("No Signal")
            msgBox.setText("Please select a signal first!")
            msgBox.exec()
            return

        if self.debug:
            print("saving trend out")

        # -------calculate trend and detrended signal------------
        trend = self.calc_trend()
        dsignal = self.raw_signal - trend

        # add everything to a pandas data frame
        data = np.array([self.raw_signal, trend, dsignal]).T  # stupid pandas..
        columns = ["raw", "trend", "detrended"]
        df_out = pd.DataFrame(data=data, columns=columns)
        # ------------------------------------------------------

        if self.debug:
            print("df_out", df_out[:10])
            print("trend", trend[:10])
        dialog = QFileDialog()
        options = QFileDialog.Options()

        settings = QSettings()        
        # ----------------------------------------------------------
        base_name = str(self.signal_id).replace(' ', '-')
        dir_path = settings.value('dir_name', os.path.curdir)

        default_name = os.path.join(dir_path,
                                    base_name + '_trend')
        format_filter = "Text File (*.txt);; csv ( *.csv);; MS Excel (*.xlsx)"
        # -----------------------------------------------------------
        file_name, sel_filter = dialog.getSaveFileName(
            self, "Save as", directory=default_name, filter=format_filter, options=options
        )
        # dialog cancelled
        if not file_name:
            return

        file_ext = file_name.split(".")[-1]

        if self.debug:
            print("selected filter:", sel_filter)
            print("out-path:", file_name)
            print("extracted extension:", file_ext)

        if file_ext not in ["txt", "csv", "xlsx"]:

            msgBox = QMessageBox()
            msgBox.setWindowTitle("Unknown File Format")
            msgBox.setText(
                "Please append .txt, .csv or .xlsx to the file name!")
            msgBox.exec()            
            return

        # ------the write out calls to pandas----------------

        # defaults to 3 decimals
        float_format = settings.value('float_format', '%.3f')

        if file_ext == "txt":
            df_out.to_csv(file_name, index=False, sep="\t", float_format=float_format)

        elif file_ext == "csv":
            df_out.to_csv(file_name, index=False, sep=",", float_format=float_format)

        elif file_ext == "xlsx":
            df_out.to_excel(file_name, index=False, float_format=float_format)

        else:
            if self.debug:
                print("Something went wrong during save out..")
            return
        if self.debug:
            print("Saved!")

    def load_settings(self):
        
        settings = QSettings()

        # map parameter keys to edits
        key_to_edit = {
            'dt' : self.dt_edit,
            'time_unit' : self.unit_edit,
            'cut_off' : self.T_c_edit,
            'window_size' : self.wsize_edit,
            'Tmin' : self.Tmin_edit,
            'Tmax' : self.Tmax_edit,
            'nT' : self.nT_edit,
            'pow_max' : self.pow_max_edit,
            'float_format' : None,
            'graphics_format' : None
        }

        # load defaults from dict or restore values
        for key, value in default_par_dict.items():
            val = settings.value(key, value)
            edit = key_to_edit[key]
            # some fields might be left empty for dynamic defaults
            if edit and (val is not None):
                edit.clear() # to be on the safe side
                edit.insert(str(val))
