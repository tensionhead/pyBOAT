from PyQt5.QtWidgets import (
    QCheckBox,
    QMainWindow,
    QLabel,
    QLineEdit,
    QPushButton,
    QMessageBox,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QFormLayout,
    QGridLayout,
    QTabWidget,
    QSpacerItem,
)

from PyQt5.QtGui import QDoubleValidator, QIntValidator
from PyQt5.QtCore import Qt

from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

import pyboat
from pyboat.ui.util import MessageWindow, posfloatV, posintV, floatV, set_max_width
from pyboat.ui.analysis import mkTimeSeriesCanvas, FourierAnalyzer, WaveletAnalyzer
from pyboat import plotting as pl
from pyboat import ssg  # the synthetic signal generator
import numpy as np

# --- monkey patch label sizes good for ui ---
pl.tick_label_size = 12
pl.label_size = 14


class SynthSignalGen(QMainWindow):

    """
    This is basically a clone of the
    DataViewer, but instead of the
    imported data, controls for
    generating a synthetic signal
    are provided
    """

    def __init__(self, debug=False):
        super().__init__()

        self.anaWindows = {}
        self.w_position = 0  # analysis window position offset

        self.debug = debug

        self.raw_signal = None
        self.Nt = 500  # needs to be set here for initial plot..
        self.dt = 1  # .. to create the default signal
        self.tvec = None  # gets initialized by vector_prep
        self.time_unit = None  # gets initialized by qset_time_unit

        # get updated with dt in -> qset_dt
        self.periodV = QDoubleValidator(bottom=1e-16, top=1e16)
        self.envelopeV = QDoubleValidator(bottom=3, top=9999999)

        self.initUI()

    # ===============UI=======================================

    def initUI(self):

        self.setWindowTitle(f"Synthetic Signal Generator")
        self.setGeometry(80, 300, 900, 650)

        main_widget = QWidget()
        self.statusBar()

        self.tsCanvas = mkTimeSeriesCanvas()
        main_frame = QWidget()
        self.tsCanvas.setParent(main_frame)
        ntb = NavigationToolbar(self.tsCanvas, main_frame)  # full toolbar

        main_layout_v = QVBoxLayout()  # The whole Layout

        # --- the synthesizer controls ---

        # width of the input fields
        iwidth = 50

        dt_label = QLabel("Sampling Interval:")
        dt_edit = QLineEdit()
        set_max_width(dt_edit, iwidth)
        dt_edit.setValidator(posfloatV)
        dt_edit.textChanged[str].connect(self.qset_dt)

        unit_label = QLabel("Time Unit:")
        unit_edit = QLineEdit(self)
        set_max_width(unit_edit, iwidth)
        unit_edit.textChanged[str].connect(self.qset_time_unit)
        unit_edit.insert("min")  # standard time unit is minutes

        Nt_label = QLabel("# Samples")
        self.Nt_edit = QLineEdit()
        self.Nt_edit.setStatusTip(
            "Number of data points, minimum is 10, maximum is 25 000!"
        )
        set_max_width(self.Nt_edit, iwidth)
        self.Nt_edit.setValidator(QIntValidator(bottom=10, top=25000))


        # --- the basic settings box ---
        basics_box = QGroupBox("Basics")
        basics_box_layout = QVBoxLayout()
        basics_box_layout.setSpacing(2.5)
        basics_box.setLayout(basics_box_layout)

        basics_box_layout.addWidget(Nt_label)
        basics_box_layout.addWidget(self.Nt_edit)        
        basics_box_layout.addWidget(dt_label)
        basics_box_layout.addWidget(dt_edit)
        basics_box_layout.addWidget(unit_label)
        basics_box_layout.addWidget(unit_edit)
        
        basics_box_layout.addStretch(0)

        # --- chirp 1 ---

        T11_label = QLabel("Initial Period")
        self.T11_edit = QLineEdit()
        self.T11_edit.setStatusTip("Period at the beginning of the signal")
        set_max_width(self.T11_edit, iwidth)
        self.T11_edit.setValidator(posfloatV)
        self.T11_edit.insert(str(50))  # initial period of chirp 1

        T12_label = QLabel("Final Period")
        self.T12_edit = QLineEdit()
        self.T12_edit.setStatusTip("Period at the end of the signal")
        set_max_width(self.T12_edit, iwidth)
        self.T12_edit.setValidator(posfloatV)
        self.T12_edit.insert(str(150))  # initial period of chirp 1

        A1_label = QLabel("Amplitude")
        self.A1_edit = QLineEdit()
        self.A1_edit.setStatusTip("The amplitude :)")
        set_max_width(self.A1_edit, iwidth)
        self.A1_edit.setValidator(posfloatV)
        self.A1_edit.insert(str(1))  # initial amplitude

        # --- the chirp 1 box ---
        self.chirp1_box = QGroupBox("Oscillator I")
        self.chirp1_box.setCheckable(True)
        chirp1_box_layout = QVBoxLayout()
        chirp1_box_layout.setSpacing(2.5)
        self.chirp1_box.setLayout(chirp1_box_layout)

        chirp1_box_layout.addWidget(T11_label)
        chirp1_box_layout.addWidget(self.T11_edit)
        chirp1_box_layout.addWidget(T12_label)
        chirp1_box_layout.addWidget(self.T12_edit)
        chirp1_box_layout.addWidget(A1_label)
        chirp1_box_layout.addWidget(self.A1_edit)
        chirp1_box_layout.addStretch(0)

        # --- chirp 2 ---
        # can be used to simulate a trend :)

        T21_label = QLabel("Initial Period")
        self.T21_edit = QLineEdit()
        self.T21_edit.setStatusTip("Period at the beginning of the signal")
        set_max_width(self.T21_edit, iwidth)
        self.T21_edit.setValidator(posfloatV)
        self.T21_edit.insert(str(1000))  # initial period of chirp 1

        T22_label = QLabel("Final Period")
        self.T22_edit = QLineEdit()
        self.T22_edit.setStatusTip("Period at the end of the signal")
        set_max_width(self.T22_edit, iwidth)
        self.T22_edit.setValidator(posfloatV)
        self.T22_edit.insert(str(1000))  # initial period of chirp 1

        A2_label = QLabel("Amplitude")
        self.A2_edit = QLineEdit()
        self.A2_edit.setStatusTip("Can be negative to induce different trends..")
        set_max_width(self.A2_edit, iwidth)
        self.A2_edit.setValidator(floatV)
        self.A2_edit.insert(str(2))  # initial amplitude

        # --- the chirp 2 box ---
        self.chirp2_box = QGroupBox("Oscillator II")
        self.chirp2_box.setCheckable(True)
        chirp2_box_layout = QVBoxLayout()
        self.chirp2_box.setLayout(chirp2_box_layout)
        chirp2_box_layout.setSpacing(2.5)

        chirp2_box_layout.addWidget(T21_label)
        chirp2_box_layout.addWidget(self.T21_edit)
        chirp2_box_layout.addWidget(T22_label)
        chirp2_box_layout.addWidget(self.T22_edit)
        chirp2_box_layout.addWidget(A2_label)
        chirp2_box_layout.addWidget(self.A2_edit)
        chirp2_box_layout.addStretch(0)

        # --- the AR1 box ---
        self.noise_box = QGroupBox("Noise")
        self.noise_box.setStatusTip("Adds colored AR(1) noise to the signal")
        self.noise_box.setCheckable(True)
        self.noise_box.setChecked(False)
        noise_box_layout = QVBoxLayout()
        self.noise_box.setLayout(noise_box_layout)

        alpha_label = QLabel("AR(1) parameter")
        self.alpha_edit = QLineEdit(parent=self.noise_box)
        self.alpha_edit.setStatusTip("0 is white noise, must be smaller than 1!")
        set_max_width(self.alpha_edit, iwidth)
        # does not work as expected :/
        V1 = QDoubleValidator(bottom = 0, top = 0.99)
        # V1.setNotation(QDoubleValidator.StandardNotation)
        # V1.setDecimals(5)
        # V1.setRange(0.0, 0.99)
        self.alpha_edit.setValidator(V1)
        self.alpha_edit.insert("0.2")  # initial AR(1)

        d_label = QLabel("Noise Strength")
        self.d_edit = QLineEdit(parent=self.noise_box)
        self.d_edit.setStatusTip("Relative to oscillator amplitudes..")
        set_max_width(self.d_edit, iwidth)
        self.d_edit.setValidator(QDoubleValidator(bottom=0, top=999999))
        self.d_edit.insert("0.5")  # initial noise strength

        noise_box_layout.addWidget(alpha_label)
        noise_box_layout.addWidget(self.alpha_edit)
        noise_box_layout.addWidget(d_label)
        noise_box_layout.addWidget(self.d_edit)
        noise_box_layout.addStretch(0)

        # --- Amplitude envelope ---

        tau_label = QLabel("Decay Time")
        self.tau_edit = QLineEdit()
        self.tau_edit.setStatusTip(
            "Time after which the signal decayed to around a third of the initial amplitude"
        )
        set_max_width(self.tau_edit, iwidth)
        self.tau_edit.setValidator(posfloatV)
        self.tau_edit.insert("500")  # initial decay constant

        # --- the Envelope box ---
        self.env_box = QGroupBox("Exponential Envelope")
        self.env_box.setCheckable(True)
        env_box_layout = QVBoxLayout()
        env_box_layout.setSpacing(2.5)
        self.env_box.setLayout(env_box_layout)

        env_box_layout.addWidget(tau_label)
        env_box_layout.addWidget(self.tau_edit)
        env_box_layout.addStretch(0)

        # --- the create signal button
        ssgButton = QPushButton("Synthesize Signal", self)
        ssgButton.clicked.connect(self.create_signal)
        ssgButton.setStatusTip(
            "Click again with same settings for different noise realizations"
        )
        ssgButton.setStyleSheet("background-color: orange")

        ssgButton_box = QWidget()
        ssgButton_box_layout = QHBoxLayout()
        ssgButton_box.setLayout(ssgButton_box_layout)
        #ssgButton_box_layout.addStretch(0)
        ssgButton_box_layout.addWidget(ssgButton)
        #ssgButton_box_layout.addStretch(0)

        control_grid = QWidget()
        control_grid_layout = QGridLayout()
        control_grid.setLayout(control_grid_layout)

        control_grid_layout.addWidget(basics_box, 0, 0, 4, 1)
        control_grid_layout.addWidget(self.chirp1_box, 0, 1, 4, 1)
        control_grid_layout.addWidget(self.chirp2_box, 0, 2, 4, 1)
        control_grid_layout.addWidget(self.noise_box, 0, 3, 4, 1)
        control_grid_layout.addWidget(self.env_box, 0, 4, 1, 1)
        control_grid_layout.addItem(QSpacerItem(2, 15), 1, 4)
        control_grid_layout.addWidget(ssgButton_box, 2, 4)
        control_grid_layout.addItem(QSpacerItem(2, 15), 3, 4)

        controls = QWidget()
        controls_layout = QVBoxLayout()
        controls.setLayout(controls_layout)
        controls_layout.addWidget(control_grid)

        main_layout_v.addWidget(controls)

        ## detrending parameter

        self.T_c_edit = QLineEdit()
        self.T_c_edit.setMaximumWidth(70)
        self.T_c_edit.setValidator(posfloatV)

        sinc_options_box = QGroupBox("Sinc Detrending")
        sinc_options_layout = QGridLayout()
        sinc_options_layout.addWidget(QLabel("Cut-off Period:"), 0, 0)
        sinc_options_layout.addWidget(self.T_c_edit, 0, 1)
        sinc_options_box.setLayout(sinc_options_layout)

        ## Amplitude envelope parameter
        self.L_edit = QLineEdit()
        self.L_edit.setMaximumWidth(70)
        self.L_edit.setValidator(self.envelopeV)

        envelope_options_box = QGroupBox("Amplitude Envelope")
        envelope_options_layout = QGridLayout()
        envelope_options_layout.addWidget(QLabel("Window Size:"), 0, 0)
        envelope_options_layout.addWidget(self.L_edit, 0, 1)
        envelope_options_box.setLayout(envelope_options_layout)

        # plot options box
        plot_options_box = QGroupBox("Plotting Options")
        plot_options_layout = QGridLayout()

        self.cb_raw = QCheckBox("Raw Signal", self)
        self.cb_trend = QCheckBox("Trend", self)
        self.cb_detrend = QCheckBox("Detrended Signal", self)
        self.cb_envelope = QCheckBox("Envelope", self)

        plotButton = QPushButton("Refresh Plot", self)
        plotButton.clicked.connect(self.doPlot)

        ## checkbox layout
        plot_options_layout.addWidget(self.cb_raw, 0, 0)
        plot_options_layout.addWidget(self.cb_trend, 0, 1)
        plot_options_layout.addWidget(self.cb_detrend, 1, 0)
        plot_options_layout.addWidget(self.cb_envelope, 1, 1)
        plot_options_layout.addWidget(plotButton, 2, 0)
        plot_options_box.setLayout(plot_options_layout)

        ## checkbox signal set and change
        self.cb_raw.toggle()
        self.cb_trend.toggle()

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
        self.T_min = QLineEdit()
        self.T_min.setStatusTip("This is the lower period limit")
        self.step_num = QLineEdit()
        self.step_num.insert("200")
        self.step_num.setStatusTip("Increase this number for more resolution")
        self.T_max = QLineEdit()
        self.T_max.setStatusTip("This is the upper period limit")

        self.p_max = QLineEdit()
        self.p_max.setStatusTip(
            "Enter a fixed value for all signals or leave blank for automatic scaling"
        )

        # self.p_max.insert(str(20)) # leave blank

        T_min_lab = QLabel("Lowest period")
        step_lab = QLabel("Number of periods")
        T_max_lab = QLabel("Highest  period")
        p_max_lab = QLabel("Expected maximal power")

        T_min_lab.setWordWrap(True)
        step_lab.setWordWrap(True)
        T_max_lab.setWordWrap(True)
        p_max_lab.setWordWrap(True)

        wletButton = QPushButton("Analyze Signal", self)
        wletButton.clicked.connect(self.run_wavelet_ana)
        wletButton.setStatusTip("Start the wavelet analysis!")
        wletButton.setStyleSheet("background-color: lightblue")

        ## add  button to layout
        wlet_button_layout_h = QHBoxLayout()

        wlet_button_layout_h.addStretch(0)
        wlet_button_layout_h.addWidget(wletButton)
        wlet_button_layout_h.addStretch(0)

        self.cb_use_detrended = QCheckBox("Use Detrended Signal", self)

        # self.cb_use_detrended.stateChanged.connect(self.toggle_use)
        self.cb_use_detrended.setChecked(True)  # detrend by default

        self.cb_use_envelope = QCheckBox("Normalize with Envelope", self)
        self.cb_use_envelope.setChecked(False)  # no envelope by default

        ## Add Wavelet analyzer options to tab1.parameter_box layout

        tab1.parameter_box.addRow(T_min_lab, self.T_min)
        tab1.parameter_box.addRow(T_max_lab, self.T_max)
        tab1.parameter_box.addRow(step_lab, self.step_num)        
        tab1.parameter_box.addRow(p_max_lab, self.p_max)
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
        # tab2.parameter_box.addRow(T_min_lab,self.T_min)
        # tab2.parameter_box.addRow(T_max_lab,self.T_max)
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

        # ==========Plot and Options Layout=======================================
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
        main_layout_v.setStretch(0, 0)  # controls shouldn't stretch
        main_layout_v.setStretch(1, 5)  # plot should stretch

        # --- initialize parameter fields only now ---

        # this will trigger setting initial periods
        self.Nt_edit.insert(str(self.Nt))  # initial signal length
        dt_edit.insert(str(self.dt))  # initial sampling interval is 1

        # create the default signal
        self.create_signal()

        main_widget.setLayout(main_layout_v)
        self.setCentralWidget(main_widget)        
        self.show()
 
    # probably all the toggle state variables are not needed -> read out checkboxes directly
    def toggle_raw(self, state):
        if state == Qt.Checked:
            self.plot_raw = True
        else:
            self.plot_raw = False

        if self.raw_signal is not None:
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
        if np.any(self.raw_signal):
            self.doPlot()

    def toggle_envelope(self, state):

        # trying to plot the envelope
        if state == Qt.Checked:                
            L = self.get_L(self.L_edit)
            if not L:
                self.cb_envelope.setChecked(False)
                self.cb_use_envelope.setChecked(False)
                 
         
        # signal selected?
        if np.any(self.raw_signal):
            self.doPlot()

    # connected to unit_edit
    def qset_time_unit(self, text):
        self.time_unit = text  # self.unit_edit.text()
        if self.debug:
            print("time unit changed to:", text)

    # connected to dt_edit
    def qset_dt(self, text):

        # checking the input is done automatically via .setValidator!
        # check,str_val,_ = posfloatV.validate(t,  0) # pos argument not used
        t = text.replace(",", ".")
        try:
            self.dt = float(t)
            self.set_initial_periods(force=True)
            self.set_initial_T_c(force=True)
            # update  Validators
            self.periodV = QDoubleValidator(bottom=2 * self.dt, top=1e16)
            self.envelopeV = QDoubleValidator(bottom=3 * self.dt, top=self.Nt * self.dt)

            # refresh plot if a is signal selected
            if self.raw_signal is not None:
                self.doPlot()

        # empty input, keeps old value!
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
            self.NoTrend = MessageWindow(
                "Detrending parameter not set,\n" + "specify a cut-off period!",
                "No Trend",
            )
            
            if self.debug:
                print("T_c ValueError", tc)
            return None
        
    def get_L(self, L_edit):

        # value checking done by validator, accepts also comma '1,1' !
        L = L_edit.text().replace(",", ".")
        try:
            L = int(L)

        # empty line edit
        except ValueError:
            self.NoTrend = MessageWindow(
                "Envelope parameter not set,\n" + "specify a sliding window size!",
                "No Envelope",
            )
            
            if self.debug:
                print("L ValueError", L)
            return None
        
        if self.debug:
            print("L set to:", L)
        return L
            
    def set_initial_periods(self, force=False):

        """
        rewrite value if force is True
        """

        if self.debug:
            print("set_initial_periods called")

        # check if a T_min was already entered
        # or rewrite is enforced
        if not bool(self.T_min.text()) or force:
            self.T_min.clear()
            self.T_min.insert(str(2 * self.dt))  # Nyquist

        if np.any(self.raw_signal):  # check if raw_signal already selected
            # check if a T_max was already entered
            if not bool(self.T_max.text()) or force:

                # default is 1/4 the observation time
                self.T_max.clear()
                T_max_ini = self.dt * 1 / 4 * self.Nt
                if self.dt > 0.1:
                    T_max_ini = int(T_max_ini)
                self.T_max.insert(str(T_max_ini))

    def set_initial_T_c(self, force=False):

        if self.debug:
            print("set_initial_T_c called")

        if np.any(self.raw_signal):  # check if raw_signal already selected
            if (
                not bool(self.T_c_edit.text()) or force
            ):  # check if a T_c was already entered
                # default is 1.5 * T_max -> 3/8 the observation time
                self.T_c_edit.clear()
                # this will trigger qset_T_c and updates the variable
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

        self.Tmin
        self.Tmax
        self.step_num
        self.pmax

        Further the checkboxes regarding detrending and amplitude
        normalization are evaluated. And

        self.get_L()
        self.get_T_c()

        are called if needed. These respective parameters are set to False
        if the opereation in question is not requested.

        Returns
        -------

        wlet_pars : dictionary holding the retrieved parameters,
                    L and T_c are set to None if no amplitude
                    normalization or detrending operation should be done

        '''

        wlet_pars = {}

        # period validator
        vali = self.periodV

        # -- read all the QLineEdits --

        text = self.T_min.text()
        T_min = text.replace(",", ".")
        check, _, _ = vali.validate(T_min, 0)
        if self.debug:
            print("Min periodValidator output:", check, "value:", T_min)
        if check == 0:
            self.OutOfBounds = MessageWindow(
                "Wavelet periods out of bounds!", "Value Error"
            )
            return False

        wlet_pars['T_min'] = float(T_min)
                
        step_num = self.step_num.text()
        check, _, _ = posintV.validate(step_num, 0)
        if self.debug:
            print("# Periods posintValidator:", check, "value:", step_num)
        if check == 0:
            self.OutOfBounds = MessageWindow(
                "The Number of periods must be a positive integer!", "Value Error"
            )
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
        
        text = self.T_max.text()
        T_max = text.replace(",", ".")
        check, _, _ = vali.validate(T_max, 0)
        if self.debug:
            print("Max periodValidator output:", check)
            print(f"Max period value: {self.T_max.text()}")
        if check == 0 or check == 1:
            self.OutOfBounds = MessageWindow(
                "Wavelet highest period out of bounds!", "Value Error"
            )
            return False
        wlet_pars['T_max'] = float(T_max)

        text = self.p_max.text()
        p_max = text.replace(",", ".")
        check, _, _ = posfloatV.validate(p_max, 0)  # checks for positive float
        if check == 0:
            self.OutOfBounds = MessageWindow("Powers are positive!", "Value Error")
            return False

        # check for empty string:
        if p_max:
            wlet_pars['p_max'] = float(p_max)
        else:
            wlet_pars['p_max'] = None

        # -- the checkboxes --
            
        # detrend for the analysis?
        if self.cb_use_detrended.isChecked():
            T_c = self.get_T_c(self.T_c_edit)
            if T_c is None:
                return False # abort settings            
            wlet_pars['T_c'] = T_c
        else:
            # indicates no detrending requested
            wlet_pars['T_c'] = False
            
        # amplitude normalization is downstram of detrending!
        if self.cb_use_envelope.isChecked():
            L = self.get_L(self.L_edit)        
            if L is None:
                return False # abort settings                        
            wlet_pars['L'] = L
        else:
            # indicates no ampl. normalization
            wlet_pars['L'] = False
                    
        # success!
        return wlet_pars
                
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

        L = self.get_L(self.L_edit)
        if not L:
            return
        if self.debug:
            print("Calculating envelope with L = ", L)
            
        if L / self.dt < 4:
            self.OutOfBounds = MessageWindow(
                f"Minimum sliding window size is {4*self.dt}{self.time_unit} !",
                "Value Error",
            )
            L = None
            return

        if L / self.dt > self.raw_signal.size:
            maxL = self.raw_signal.size * self.dt
            self.OutOfBounds = MessageWindow(
                f"Maximum sliding window size is {maxL:.2f} {self.time_unit}!",
                "Value Error",
            )
            L = None
            return

        # cut of frequency set?!
        if self.get_T_c(self.T_c_edit):

            trend = self.calc_trend()
            if trend is None:
                return
            
            signal = self.raw_signal - trend
            envelope = pyboat.sliding_window_amplitude(signal,
                                                       window_size = L,
                                                       dt = self.dt)

            if self.cb_detrend.isChecked():
                return envelope

            # fits on the original signal!
            else:
                return envelope + trend

        # otherwise add the mean
        else:
            if self.debug:
                print("calculating envelope for raw signal", L)

            mean = self.raw_signal.mean()
            envelope = pyboat.sliding_window_amplitude(
                self.raw_signal, window_size = L, dt = self.dt
            )
            return envelope + mean

    def create_signal(self):

        """
        Retrieves all paramters from the synthesizer controls
        and calls the ssg. All line edits have validators set,
        however we have to check for intermediate empty inputs..

        This sets

        self.raw_signal

        according to the synthesizer controls.
        """

        # the signal components
        components = []
        weights = []

        # number of sample points
        if not self.Nt_edit.hasAcceptableInput():
            self.OutOfBounds = MessageWindow(
                "Minimum number of sample points is 10!", "Value Error"
            )
            return

        self.Nt = int(self.Nt_edit.text())
        if self.debug:
            print("Nt changed to:", self.Nt)

        self.set_initial_periods(force=False)
        self.set_initial_T_c(force=False)

        T_edits = [self.T11_edit, self.T12_edit, self.T21_edit, self.T22_edit]

        # check for valid inputs
        for T_e in T_edits:

            if self.debug:
                print(f"Is enabled: {T_e.isEnabled()}")
                print(f"Checking T_edits: {T_e.text()}")
                print(f"Validator output: {T_e.hasAcceptableInput()}")

            if not T_e.isEnabled():
                continue

            if not T_e.hasAcceptableInput():
                self.OutOfBounds = MessageWindow(
                    "All periods must be greater than 0!", "Value Error"
                )
                return

        # envelope before chirp creation
        if self.env_box.isChecked():

            if not self.tau_edit.hasAcceptableInput():
                self.OutOfBounds = MessageWindow(
                    "Missing envelope parameter!", "Value Error"
                )
                return

            tau = float(self.tau_edit.text()) / self.dt
            env = ssg.create_exp_envelope(tau, self.Nt)
            if self.debug:
                print(f"Creating the envelope with tau = {tau}")

        else:
            env = 1  # no envelope

        if self.chirp1_box.isChecked():

            if not self.A1_edit.hasAcceptableInput():
                self.OutOfBounds = MessageWindow(
                    "Set an amplitude for oscillator1!", "Value Error"
                )
                return

            # the periods
            T11 = float(self.T11_edit.text()) / self.dt
            T12 = float(self.T12_edit.text()) / self.dt
            A1 = float(self.A1_edit.text())
            chirp1 = ssg.create_chirp(T11, T12, self.Nt)
            components.append(env * chirp1)
            weights.append(A1)

        if self.chirp2_box.isChecked():

            if not self.A2_edit.hasAcceptableInput():
                self.OutOfBounds = MessageWindow(
                    "Set an amplitude for oscillator2!", "Value Error"
                )
                return

            T21 = float(self.T21_edit.text()) / self.dt
            T22 = float(self.T22_edit.text()) / self.dt
            A2 = float(self.A2_edit.text())
            chirp2 = ssg.create_chirp(T21, T22, self.Nt)
            components.append(env * chirp2)
            weights.append(A2)

        # noise
        if self.noise_box.isChecked():


            try:
                # QDoubleValidator is a screw up..
                alpha = float(self.alpha_edit.text())
            # empty input
            except ValueError:
                self.OutOfBounds = MessageWindow(
                    "Missing AR(1) alpha parameter!", "Value Error"
                )
                return
            
            if not 0 <= alpha < 1:
                self.OutOfBounds = MessageWindow(
                    "AR1 parameter must be between 0 and <1!", "Value Error"
                )
                return

            # Noise amplitude
            try:
                d = float(self.d_edit.text())

            except ValueError:
                self.OutOfBounds = MessageWindow(
                    "Missing Noise Strength parameter!", "Value Error"
                )
                return

            d = float(self.d_edit.text())
            noise = ssg.ar1_sim(alpha, self.Nt)
            components.append(noise)
            weights.append(d)

        if len(components) == 0:
            self.OutOfBounds = MessageWindow(
                "Activate at least one signal component!", "No Signal"
            )
            return

        signal = ssg.assemble_signal(components, weights)

        # ----------------------------------------
        self.raw_signal = signal
        self.tvec = self.dt * np.arange(self.Nt)
        # ---------------------------------------

        if self.debug:
            print("created synth. signal:", self.raw_signal[:10])

        # plot right away
        self.set_initial_periods()
        self.set_initial_T_c()
        self.doPlot()

    def doPlot(self):

        if self.raw_signal is None:
            self.NoSignal = MessageWindow("Please create a signal first!", "No Signal")

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
            ax2.legend(fontsize=pl.tick_label_size, loc='lower left')
        self.tsCanvas.fig1.subplots_adjust(bottom=0.15, left=0.15, right=0.85)
        # add a simple legend
        ax1.legend(fontsize=pl.tick_label_size)

        self.tsCanvas.draw()
        self.tsCanvas.show()

    def run_wavelet_ana(self):
        
        """ run the Wavelet Analysis on the synthetic signal """

        if not np.any(self.raw_signal):
            self.NoSignalSelected = MessageWindow(
                "Please create a signal first!", "No Signal"
            )
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
            L = self.get_L(self.L_edit)
            signal = pyboat.normalize_with_envelope(signal, L, dt = self.dt)

        self.w_position += 20

        
        self.anaWindows[self.w_position] = WaveletAnalyzer(
            signal=signal,
            dt=self.dt,
            Tmin = wlet_pars['T_min'],
            Tmax = wlet_pars['T_max'],
            pow_max = wlet_pars['p_max'],
            step_num=wlet_pars['step_num'],
            position=self.w_position,
            signal_id='Synthetic Signal',
            time_unit=self.time_unit,            
            DEBUG=self.debug,
        )

    def run_fourier_ana(self):
        if not np.any(self.raw_signal):
            self.NoSignalSelected = MessageWindow(
                "Please select a signal first!", "No Signal"
            )
            return False

        # shift new analyser windows
        self.w_position += 20


        if self.cb_use_detrended2.isChecked():
            trend = self.calc_trend()
            if trend is not None:
                signal = self.raw_signal - trend
            else:
                return
        else:
            signal = self.raw_signal


        if self.cb_use_envelope2.isChecked():
            L = self.get_L(self.L_edit)
            if L is not None:                
                signal = pyboat.normalize_with_envelope(signal, L, self.dt)
            else:
                return
            
        # periods or frequencies?
        if self.cb_FourierT.isChecked():
            show_T = False
        else:
            show_T = True

        self.anaWindows[self.w_position] = FourierAnalyzer(
            signal=signal,
            dt=self.dt,
            signal_id="Synthetic Signal",
            position=self.w_position,
            time_unit=self.time_unit,
            show_T=show_T,
        )


# little helper function


