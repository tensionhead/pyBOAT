"""The analysis parameter widgets for the data viewer and SSG"""

from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np

from PyQt6 import QtWidgets
from PyQt6.QtCore import Qt, QSettings

from pyboat.ui.util import create_spinbox, default_par_dict

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
        
    def _setup_UI(self):

        ## detrending parameter
        self.T_c_spin = create_spinbox(
            1,
            step=1.0,
            status_tip="Since filter cut-off period, e.g. 120min",
            double=True,
        )
        # replot when changing values
        self.T_c_spin.valueChanged.connect(self._dv.doPlot)
        
        self.cb_use_detrended = QtWidgets.QCheckBox("Detrend", self)
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

    def detrend(self) -> bool:
        return self.cb_use_detrended.isEnabled()

    def get_T_c(self) -> float:
        return self.T_c_spin.value()

    def normalize(self) -> bool:
        return self.cb_use_envelope.isEnabled()

    def get_wsize(self) -> float:
        window_size = self.wsize_spin.value()
        
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
                f"Maximal sliding window size for envelope estimation is {max_window_size:.2f} {self.time_unit}!"
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

    def set_initial_T_c(self, force=False):
        """
        Set the initial cut off period to a sensitive default,
        depending on dt and signal length/
        """
        
        if np.any(self._dv.raw_signal):
            # check if a T_c was already entered
            if not self.restored_T_c or force:
                # default is 1.5 * Tmax -> 3/8 the observation time
                T_c_ini = self._dv.dt * 3 / 8 * len(self._dv.raw_signal)
                if self._dv.dt > 0.1:
                    T_c_ini = int(T_c_ini)
                else:
                    T_c_ini = np.round(T_c_ini, 3)

                self.T_c_spin.setValue(T_c_ini)


