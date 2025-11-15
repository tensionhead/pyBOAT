from PyQt6.QtWidgets import (
    QLabel,
    QLineEdit,
    QPushButton,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QGridLayout,
    QSpacerItem,
    QAbstractSpinBox,
    QSplitter
)

from PyQt6.QtCore import Qt

import numpy as np
from pyboat import plotting as pl
from pyboat import ssg  # the synthetic signal generator

from .util import (
    set_max_width,
    spawn_warning_box,
    is_dark_color_scheme,
    create_spinbox,
    mk_spinbox_unit_slot,
)
from . import analysis_parameters as ap
from . import style
from .data_viewer import DataViewerBase

# --- monkey patch label sizes good for ui ---
pl.tick_label_size = 12
pl.label_size = 14


class SynthSignalGen(DataViewerBase, ap.SettingsManager):

    """
    This is basically a clone of the
    DataViewer, but instead of the
    imported data, controls for
    generating a synthetic signal
    are provided.
    """

    def __init__(self, parent):
        super().__init__(0, parent)
        self.is_ssg: bool = True
        self.signal_id = "Synthetic signal"
        self.initUI()
        self._parameter_widgets = {
            "dt": self.dt_spin,
            "time_unit": self.unit_edit,
        }
        self._restore_settings()
        self.installEventFilter(self)
    # ===============UI=======================================

    def initUI(self, pos_offset=0):

        self.setWindowTitle(f"Synthetic Signal Generator")
        self.restore_geometry()
        super().initUI(pos_offset)

    def _create_synthesizer_controls(self):

        connect_to_create: list[QWidget] = []
        connect_to_unit: list[QAbstractSpinBox] = []

        # width of the input fields
        iwidth = 100

        dt_label = QLabel("Sampling Interval")
        dt_spin = create_spinbox(10, step=1, minimum=.1, maximum=1000., double=True)
        set_max_width(dt_spin, iwidth)
        connect_to_create.append(dt_spin)
        connect_to_unit.append(dt_spin)
        self.dt_spin = dt_spin

        unit_label = QLabel("Time Unit")
        unit_edit = QLineEdit(self)
        set_max_width(unit_edit, iwidth)
        self.unit_edit = unit_edit

        Nt_label = QLabel("# Samples")
        self.Nt_spin = create_spinbox(
            250, 10, 25_000, step = 25,
            status_tip="Number of data points, minimum is 10, maximum is 25 000"
        )
        connect_to_create.append(self.Nt_spin)
        set_max_width(self.Nt_spin, iwidth)

        # --- the basic settings box ---
        basics_box = QGroupBox("Basics")
        basics_box_layout = QVBoxLayout()
        basics_box_layout.setSpacing(2)
        basics_box.setLayout(basics_box_layout)

        basics_box_layout.addWidget(Nt_label)
        basics_box_layout.addWidget(self.Nt_spin)
        basics_box_layout.addWidget(dt_label)
        basics_box_layout.addWidget(dt_spin)
        basics_box_layout.addWidget(unit_label)
        basics_box_layout.addWidget(unit_edit)

        basics_box_layout.addStretch(0)

        # --- chirp 1 ---

        T11_label = QLabel("Initial Period")
        self.T11_spin = create_spinbox(
            start_value=20,
            minimum=1,
            maximum=1_000,
            double=False,
            status_tip="Period at the beginning of the signal")
        set_max_width(self.T11_spin, iwidth)
        connect_to_create.append(self.T11_spin)
        connect_to_unit.append(self.T11_spin)

        T12_label = QLabel("Final Period")
        self.T12_spin = create_spinbox(
            start_value=40,
            minimum=1,
            maximum=1_000,
            double=False,
            status_tip="Period at the end of the signal")
        set_max_width(self.T12_spin, iwidth)
        connect_to_create.append(self.T12_spin)
        connect_to_unit.append(self.T12_spin)

        A1_label = QLabel("Amplitude")
        self.A1_spin = create_spinbox(
            1,
            minimum=1,
            maximum=100,
            double=False)
        self.A1_spin.setStatusTip("The amplitude :)")
        connect_to_create.append(self.A1_spin)
        set_max_width(self.A1_spin, iwidth)

        # --- the chirp 1 box ---
        self.chirp1_box = QGroupBox("Oscillator I")
        self.chirp1_box.setCheckable(True)
        connect_to_create.append(self.chirp1_box)
        chirp1_box_layout = QVBoxLayout()
        chirp1_box_layout.setSpacing(2)
        self.chirp1_box.setLayout(chirp1_box_layout)

        chirp1_box_layout.addWidget(T11_label)
        chirp1_box_layout.addWidget(self.T11_spin)
        chirp1_box_layout.addWidget(T12_label)
        chirp1_box_layout.addWidget(self.T12_spin)
        chirp1_box_layout.addWidget(A1_label)
        chirp1_box_layout.addWidget(self.A1_spin)
        chirp1_box_layout.addStretch(0)

        # --- chirp 2 ---
        # can be used to simulate a trend :)

        T21_label = QLabel("Initial Period")
        self.T21_spin = create_spinbox(
            80, 1, 10_000, step=10, double=False,
            status_tip="Period at the beginning of the signal")
        set_max_width(self.T21_spin, iwidth)
        connect_to_create.append(self.T21_spin)
        connect_to_unit.append(self.T21_spin)

        T22_label = QLabel("Final Period")
        self.T22_spin = create_spinbox(
            700, 1, 10_000, step=25, double=False,
            status_tip="Period at the end of the signal")
        set_max_width(self.T22_spin, iwidth)
        connect_to_create.append(self.T22_spin)
        connect_to_unit.append(self.T22_spin)

        A2_label = QLabel("Amplitude")
        self.A2_spin = create_spinbox(4, -100, 100, double=False)
        self.A2_spin.setStatusTip("The amplitude :)")
        connect_to_create.append(self.A2_spin)
        set_max_width(self.A2_spin, iwidth)

        # --- the chirp 2 box ---
        self.chirp2_box = QGroupBox("Oscillator II (Trend)")
        self.chirp2_box.setCheckable(True)
        connect_to_create.append(self.chirp2_box)
        chirp2_box_layout = QVBoxLayout()
        self.chirp2_box.setLayout(chirp2_box_layout)
        chirp2_box_layout.setSpacing(2)

        chirp2_box_layout.addWidget(T21_label)
        chirp2_box_layout.addWidget(self.T21_spin)
        chirp2_box_layout.addWidget(T22_label)
        chirp2_box_layout.addWidget(self.T22_spin)
        chirp2_box_layout.addWidget(A2_label)
        chirp2_box_layout.addWidget(self.A2_spin)
        chirp2_box_layout.addStretch(0)

        # --- the AR1 box ---
        self.noise_box = QGroupBox("Noise")
        self.noise_box.setStatusTip("Adds colored AR(1) noise to the signal")
        self.noise_box.setCheckable(True)
        self.noise_box.setChecked(True)
        connect_to_create.append(self.noise_box)
        noise_box_layout = QVBoxLayout()
        self.noise_box.setLayout(noise_box_layout)

        alpha_label = QLabel("AR(1) parameter")
        self.alpha_spin = create_spinbox(0.2, 0, .99, step=0.1, double=True)
        self.alpha_spin.setDecimals(2)
        self.alpha_spin.setStatusTip("0 is white noise, must be smaller than 1!")
        set_max_width(self.alpha_spin, iwidth)
        connect_to_create.append(self.alpha_spin)

        d_label = QLabel("Noise Strength")
        self.d_spin = create_spinbox(0.5, step=0.1, minimum=0, maximum=1000, double=True)
        self.d_spin.setStatusTip("Relative to oscillator amplitudes")
        connect_to_create.append(self.d_spin)
        set_max_width(self.d_spin, iwidth)

        noise_box_layout.addWidget(alpha_label)
        noise_box_layout.addWidget(self.alpha_spin)
        noise_box_layout.addWidget(d_label)
        noise_box_layout.addWidget(self.d_spin)
        noise_box_layout.addStretch(0)

        # --- Amplitude envelope ---

        tau_label = QLabel("Decay Time")
        self.tau_spin = create_spinbox(
            300, 10, 25_000, step=25,
            status_tip="Time after which the signal "
            "decayed to around a third of "
            "the initial amplitude"
        )
        self.tau_spin.setAccelerated(True)
        connect_to_create.append(self.tau_spin)
        connect_to_unit.append(self.tau_spin)
        set_max_width(self.tau_spin, iwidth)

        # --- the Envelope box ---
        self.env_box = QGroupBox("Exponential Envelope")
        self.env_box.setCheckable(True)
        connect_to_create.append(self.env_box)
        env_box_layout = QVBoxLayout()
        env_box_layout.setSpacing(2)
        self.env_box.setLayout(env_box_layout)

        env_box_layout.addWidget(tau_label)
        env_box_layout.addWidget(self.tau_spin)
        env_box_layout.addStretch(0)

        # --- the create signal button
        ssgButton = QPushButton("Synthesize Signal", self)
        ssgButton.clicked.connect(self.create_signal)
        ssgButton.setStatusTip(
            "Resynthesize for different realizations with same settings"
        )

        ssgButton_box = QWidget()
        ssgButton_box_layout = QHBoxLayout()
        ssgButton_box.setLayout(ssgButton_box_layout)
        # ssgButton_box_layout.addStretch(0)
        ssgButton_box_layout.addWidget(ssgButton)
        # ssgButton_box_layout.addStretch(0)

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

        # --- initialize parameter fields only now ---

        # this will trigger setting initial periods
        dt_spin.setValue(10)  # initial sampling interval is 1

        # connect unit edit
        for spin in connect_to_unit:
            unit_edit.textChanged[str].connect(mk_spinbox_unit_slot(spin))
        unit_edit.insert("min")  # standard time unit is minutes
        unit_edit.textChanged[str].connect(self.doPlot)

        # now connect the input fields
        for w in connect_to_create:
            if isinstance(w, QAbstractSpinBox):
                w.valueChanged.connect(self.create_signal)
            if isinstance(w, QGroupBox):
                w.toggled.connect(self.create_signal)

        return controls

    def _compose_init_main_layout(self) -> QSplitter:

        controls = self._create_synthesizer_controls()
        plot_and_parameters = self._create_plot_parameter_area()
        self.dt_spin.textChanged[str].connect(self.qset_dt)
        self.dt_spin.setValue(10)

        # vertical splitter between data table and plot + options
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(controls)
        splitter.addWidget(plot_and_parameters)

        # create the default signal
        self.create_signal()

        return splitter

    @property
    def Nt(self) -> int:
        return self.Nt_spin.value()

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
        if not self.Nt_spin.hasAcceptableInput():
            msgBox = spawn_warning_box(self, "Value Error",
                                       "Minimum number of sample points is 10!")
            msgBox.exec()
            return

        # envelope before chirp creation
        if self.env_box.isChecked():

            if not self.tau_spin.hasAcceptableInput():
                msgBox = spawn_warning_box(self,
                                           "Missing Value",
                                           "Missing envelope decay time parameter!")
                msgBox.exec()
                return

            tau = self.tau_spin.value() / self.dt
            env = ssg.create_exp_envelope(tau, self.Nt)
            if self.debug:
                print(f"Creating the envelope with tau = {tau}")

        else:
            env = 1  # no envelope

        if self.chirp1_box.isChecked():

            # the periods
            T11 = self.T11_spin.value() / self.dt
            T12 = self.T12_spin.value() / self.dt
            A1 = self.A1_spin.value()
            chirp1 = ssg.create_chirp(T11, T12, self.Nt)
            components.append(env * chirp1)
            weights.append(A1)

        if self.chirp2_box.isChecked():

            T21 = self.T21_spin.value() / self.dt
            T22 = self.T22_spin.value() / self.dt
            A2 = self.A2_spin.value()
            chirp2 = ssg.create_chirp(T21, T22, self.Nt)
            components.append(env * chirp2)
            weights.append(A2)

        # noise
        if self.noise_box.isChecked():

            alpha = self.alpha_spin.value()
            d = self.d_spin.value()
            noise = ssg.ar1_sim(alpha, self.Nt)
            components.append(noise)
            weights.append(d)

        if len(components) == 0:
            msgBox = spawn_warning_box(self,
                                       "No Signal",
                                       "Activate at least one signal component!")
            msgBox.exec()
            return

        signal = ssg.assemble_signal(components, weights)

        # ----------------------------------------
        self.raw_signal = signal
        self.tvec = self.dt * np.arange(self.Nt)
        # ---------------------------------------
        # plot right away
        self.wavelet_tab.set_auto_periods()
        self.sinc_envelope.set_auto_T_c()
        self.sinc_envelope.set_auto_wsize()
        self.doPlot()
        # only triggers when an analysis is opened
        self.reanalyze_signal()
