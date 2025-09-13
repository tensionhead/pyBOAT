"""The analysis parameter widgets for the data viewer and SSG"""

from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np

from PyQt6 import QtWidgets
from PyQt6.QtCore import Qt, QSettings

from pyboat.ui import style
from pyboat.ui.util import create_spinbox, default_par_dict, mk_spinbox_unit_slot, is_dark_color_scheme

if TYPE_CHECKING:
    from pyboat.ui.data_viewer import DataViewer


class SincEnvelopeOptions(QtWidgets.QWidget):
    """
    The group boxes to set sinc detrending and and amplitude
    envelope window parameters.
    """

    def __init__(self, parent: DataViewer):
        super().__init__(parent)

        self._dv = parent
        self.restored_T_c: bool = False
        self.restored_w_size: bool = False

        self._setup_UI()
        self._connect_to_dataviewer()

    def _connect_to_dataviewer(self):
        for spin in [self.T_c_spin, self.wsize_spin]:
            self._dv.unit_edit.textChanged.connect(mk_spinbox_unit_slot(spin))

    def _setup_UI(self):

        ## detrending parameter
        self.T_c_spin = create_spinbox(
            1,
            step=1.0,
            status_tip="Sinc filter cut-off period, e.g. 120min",
            double=True,
        )
        # replot when changing values
        self.T_c_spin.valueChanged.connect(self._dv.doPlot)

        self.cb_use_detrended = QtWidgets.QCheckBox("Detrend", self)
        self.cb_use_detrended.toggled.connect(self.T_c_spin.setEnabled)
        self.cb_use_detrended.setChecked(True)  # detrend by default

        sinc_options_box = QtWidgets.QGroupBox("Sinc Detrending")
        sinc_options_box.setStyleSheet("QGroupBox {font-weight:normal;}")
        sinc_options_layout = QtWidgets.QGridLayout()
        sinc_options_layout.addWidget(QtWidgets.QLabel("Cut-off Period:"), 0, 0)
        sinc_options_layout.addWidget(self.T_c_spin, 0, 1)
        sinc_options_layout.addWidget(self.cb_use_detrended, 1, 0, 1, 2)
        sinc_options_box.setLayout(sinc_options_layout)

        ## Amplitude envelope parameter
        self.wsize_spin = create_spinbox(
            1,
            step=1.0,
            status_tip="Window size for emplitude envelope estimation",
            double=True,
        )

        # replot when changing values
        self.wsize_spin.valueChanged.connect(self._dv.doPlot)


        self.cb_use_envelope = QtWidgets.QCheckBox("Normalize", self)
        self.wsize_spin.setEnabled(False)  # disable by default
        self.cb_use_envelope.toggled.connect(self.wsize_spin.setEnabled)
        self.cb_use_envelope.setChecked(False)  # no envelope by default

        envelope_options_box = QtWidgets.QGroupBox("Amplitude Envelope")
        envelope_options_box.setStyleSheet("QGroupBox {font-weight:normal;}")
        envelope_options_layout = QtWidgets.QGridLayout()
        envelope_options_layout.addWidget(QtWidgets.QLabel("Window Size:"), 0, 0)
        envelope_options_layout.addWidget(self.wsize_spin, 0, 1)
        envelope_options_layout.addWidget(self.cb_use_envelope, 1, 0, 1, 2)
        envelope_options_box.setLayout(envelope_options_layout)

        # main layout of widget
        sinc_envelope_layout = QtWidgets.QHBoxLayout()
        sinc_envelope_layout.addWidget(sinc_options_box)
        sinc_envelope_layout.addWidget(envelope_options_box)
        self.setLayout(sinc_envelope_layout)

        self._load_settings()

    @property
    def do_detrend(self) -> bool:
        return self.cb_use_detrended.isChecked()

    @property
    def do_normalize(self) -> bool:
        return self.cb_use_envelope.isChecked()

    def get_T_c(self) -> float | None:
        if self.do_detrend:
            return self.T_c_spin.value()
        return None

    def get_wsize(self) -> float | None:
        if self.do_normalize:
            return self.wsize_spin.value()
        return None

        if window_size / self._dv.dt < 4:

            msgBox = QtWidgets.QMessageBox(parent=self)
            msgBox.setWindowTitle("Out of Bounds")
            msgBox.setText(
                f"Minimal sliding window size for envelope estimation"
                f"is {4 * self._dv.dt} {self._dv.time_unit}!"
            )
            msgBox.exec()

            self._dv.cb_envelope.setChecked(False)
            return None

        if window_size / self._dv.dt > self._dv.df.shape[0]:
            max_window_size = self._dv.df.shape[0] * self._dv.dt

            msgBox = QtWidgets.QMessageBox(parent=self)
            msgBox.setWindowTitle("Out of Bounds")
            msgBox.setText(
                "Maximal sliding window size for envelope estimation "
                f"is {max_window_size:.2f} {self._dv.time_unit}!"
            )
            msgBox.exec()

            return None

        return window_size

    def _load_settings(self):

        settings = QSettings()
        settings.beginGroup("user-settings")
        # load defaults from dict or restore values
        val = settings.value("cut_off", default_par_dict["cut_off"])
        if val:
            self.T_c_spin.setValue(float(val))
            self.restored_T_c = True
        # load defaults from dict or restore values
        val = settings.value("window_size", default_par_dict["window_size"])
        if val:
            self.wsize_spin.setValue(float(val))
            self.restored_T_c = True

    def set_auto_T_c(self, force=False):
        """
        Set the initial cut off period to a sensitive default,
        depending on dt and signal length.
        """

        assert self._dv.dt is not None
        
        if np.any(self._dv.raw_signal):
            # check if a T_c was already entered
            if not self.restored_T_c or force:
                # default is 1.5 * Tmax -> 3/8 the observation time
                T_c_ini = self._dv.dt * 3 / 8 * len(self._dv.raw_signal)
                if self._dv.dt % 1 == 0.:
                    T_c_ini = int(T_c_ini)
                else:
                    T_c_ini = np.round(T_c_ini, 3)

                self.T_c_spin.setValue(T_c_ini)


class WaveletTab(QtWidgets.QFormLayout):
    """Tmin, Tmax and nT widgets plus analyze buttons"""


    def __init__(self, dv: DataViewer):
        super().__init__()

        self._dv = dv

        self._spins: dict[str, QtWidgets.QSpinBox | QtWidgets.QDoubleSpinBox] = {}
        self._restored: list[str] = []

        self._setup_UI()
        self._load_settings()
        self._connect_to_unit()


    def _setup_UI(self):

        Tmin_spin = create_spinbox(
            1.,  # value gets filled via set_initial_periods
            minimum=0.1,
            maximum=9999,
            step=1.0,
            status_tip="This is the lower period limit of the spectrum",
            double=True,
        )
        self._spins["Tmin"] = Tmin_spin
        # replot when changing values
        Tmin_spin.valueChanged.connect(self._dv.reanalyze_signal)

        Tmax_spin = create_spinbox(
            1.,  # value gets filled via set_initial_periods
            minimum=0.2,
            maximum=10_000,
            step=1.0,
            status_tip="This is the upper period limit of the spectrum",
            double=True,
        )
        self._spins["Tmax"] = Tmax_spin
        Tmax_spin.valueChanged.connect(self._dv.reanalyze_signal)

        nT_spin = create_spinbox(
            100,
            minimum=10,
            maximum=1_000,
            step=10,
            status_tip="Increase this number for more spectral resolution",
            double=False,
        )
        self._spins["nT"] = nT_spin
        nT_spin.valueChanged.connect(self._dv.reanalyze_signal)

        Tmin_lab = QtWidgets.QLabel("Lowest period")
        step_lab = QtWidgets.QLabel("Number of periods")
        Tmax_lab = QtWidgets.QLabel("Highest  period")

        Tmin_lab.setWordWrap(True)
        step_lab.setWordWrap(True)
        Tmax_lab.setWordWrap(True)

        # -- Buttons --
        wletButton = QtWidgets.QPushButton("Analyze Signal")
        if is_dark_color_scheme():
            wletButton.setStyleSheet(f"background-color: {style.dark_primary}")
        else:
            wletButton.setStyleSheet(f"background-color: {style.light_primary}")
        wletButton.setStatusTip("Opens the wavelet analysis..")
        wletButton.clicked.connect(self._dv.run_wavelet_ana)

        batchButton = QtWidgets.QPushButton("Analyze All..")
        batchButton.clicked.connect(self._dv.run_batch)
        batchButton.setStatusTip("Starts a batch processing with the selected Wavelet parameters")

        self.addRow(Tmin_lab, Tmin_spin)
        self.addRow(Tmax_lab, Tmax_spin)
        self.addRow(step_lab, nT_spin)
        self.addRow(batchButton, wletButton)

    def _connect_to_unit(self):
        for spin in self._spins.values():
            self._dv.unit_edit.textChanged.connect(mk_spinbox_unit_slot(spin))

    def set_auto_periods(self, force=False) -> None:

        """
        Determine period range from signal length and sampling interval.
        """

        if not np.any(self._dv.raw_signal):
            return
        assert self._dv.raw_signal is not None
        assert self._dv.dt is not None

        Tmin_spin = self._spins['Tmin']
        Tmax_spin = self._spins['Tmax']

        # check if a Tmin was restored from settings
        # or rewrite if enforced
        if 'Tmin' not in self._restored or force:
            Tmin_spin.setValue(2 * self._dv.dt)  # Nyquist
            Tmin_spin.setMinimum(2 * self._dv.dt)
        if 'Tmax' not in self._restored or force:
            # default is 1/4 the observation time
            Tmax_ini = self._dv.dt * 1 / 4 * len(self._dv.raw_signal)
            if self._dv.dt % 1 == 0.:
                Tmax_ini = int(Tmax_ini)
            Tmax_spin.setValue(Tmax_ini)


    def _load_settings(self):

        settings = QSettings()
        settings.beginGroup("user-settings")

        # load defaults from dict or restore values
        for widget_name, spin in self._spins.items():
            val = settings.value(widget_name, default_par_dict[widget_name])
            if val is not None:
                if widget_name == "nT":
                    spin.setValue(int(val))
                else:
                    spin.setValue(float(val))
                self._restored.append(widget_name)

    def _get_parameters(self) -> dict:
        return {name: spin.value() for name, spin in self._spins.items()}

    def assemble_wlet_pars(self) -> dict:
        wlet_pars = self._get_parameters()
        # TODO: shall that come back?
        wlet_pars["pow_max"] = None
        return wlet_pars
