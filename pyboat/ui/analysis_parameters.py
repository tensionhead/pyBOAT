"""The analysis parameter widgets for the data viewer and SSG"""

from __future__ import annotations
from typing import TYPE_CHECKING, Literal
from functools import partial
from logging import getLogger

import numpy as np
from PyQt6 import QtWidgets, QtCore
from PyQt6.QtCore import QSettings, QSignalBlocker

from pyboat.ui import style
from pyboat.ui.util import create_spinbox, mk_spinbox_unit_slot, spawn_warning_box

from pyboat.ui.defaults import default_par_dict

if TYPE_CHECKING:
    from pyboat.ui.data_viewer import DataViewer

logger = getLogger(__name__)

WidgetName = str
ParameterName = str


class SettingsManager:

    # maps `default_par_dict` keys to widgets
    _parameter_widgets: dict[ParameterName, QtWidgets.QWidget]

    def __init__(self):
        pass

    def _restore_settings(self):

        settings = QSettings()
        settings.beginGroup("user-settings")

        for name, widget in self._parameter_widgets.items():
            value = settings.value(name, default_par_dict[name])
            if isinstance(widget, QtWidgets.QLineEdit):
                if value is not None:
                    widget.setText(value)
                else:
                    widget.clear()
            if isinstance(widget, QtWidgets.QSpinBox):
                if value is not None:
                    widget.setValue(int(value))
            if isinstance(widget, QtWidgets.QDoubleSpinBox):
                if value is not None:
                    widget.setValue(float(value))

    def _save_parameters(self):
        settings = QSettings()
        settings.beginGroup("user-settings")

        for name, widget in self._parameter_widgets.items():
            if isinstance(widget, QtWidgets.QLineEdit):
                value = widget.text()
            elif isinstance(widget, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)):
                value = widget.value()
            else:
                assert False, f"Unknow widget type {type(widget)}"
            settings.setValue(name, value)

    def eventFilter(self, _source, event) -> bool:
        if event.type() == QtCore.QEvent.Type.Close:
            self._save_parameters()
        return False


class SincEnvelopeOptions(QtWidgets.QWidget, SettingsManager):
    """
    The group boxes to set sinc detrending and and amplitude
    envelope window parameters.
    """

    T_c_spin: QtWidgets.QDoubleSpinBox
    wsize_spin: QtWidgets.QDoubleSpinBox

    def __init__(self, parent: DataViewer):
        QtWidgets.QWidget.__init__(self, parent)
        SettingsManager.__init__(self)

        self._dv = parent
        self.restored_T_c: bool = False
        self.restored_wsize: bool = False

        self._setup_UI()
        self._connect_to_dataviewer()

        self._parameter_widgets = {
            "cut_off_period": self.T_c_spin,
            "window_size": self.wsize_spin
        }
        self._restore_settings()

    def _connect_to_dataviewer(self):
        for spin in [self.T_c_spin, self.wsize_spin]:
            self._dv.unit_edit.textChanged.connect(mk_spinbox_unit_slot(spin))
            # replot when changing values
            spin.valueChanged.connect(self._dv.reanalyze_signal)
            spin.valueChanged.connect(self._dv.doPlot)

        # to catch an initial change by the user
        self.T_c_spin.valueChanged.connect(partial(self._changed_by_user_input, "T_c"))
        self.wsize_spin.valueChanged.connect(partial(self._changed_by_user_input, "wsize"))

        self.sinc_options_box.toggled.connect(
            self._dv.toggle_trend
        )
        self.sinc_options_box.toggled.connect(
            self._dv.rb_detrend.setEnabled
        )
        self.envelope_options_box.toggled.connect(
            self._dv.doPlot
        )

    def _setup_UI(self):

        ## detrending parameter
        self.T_c_spin = create_spinbox(
            1,
            step=1.0,
            status_tip="Sinc filter cut-off period, removes slower signal components",
            double=True,
        )


        sinc_options_box = QtWidgets.QGroupBox("Sinc Detrending")
        sinc_options_box.setCheckable(True)
        sinc_options_box.setChecked(True)  # detrend by default
        sinc_options_box.toggled.connect(self.T_c_spin.setEnabled)
        sinc_options_box.toggled.connect(self._dv.reanalyze_signal)

        sinc_options_box.setStyleSheet("QGroupBox {font-weight:normal;}")
        sinc_options_layout = QtWidgets.QGridLayout()
        sinc_options_layout.addWidget(QtWidgets.QLabel("Cut-off period"), 0, 0)
        sinc_options_layout.addWidget(self.T_c_spin, 0, 1)
        sinc_options_box.setLayout(sinc_options_layout)
        self.sinc_options_box = sinc_options_box

        ## Amplitude envelope parameter
        self.wsize_spin = create_spinbox(
            1,
            step=1.0,
            status_tip="Window size for amplitude envelope estimation, should be at least one period",
            double=True,
        )

        self.wsize_spin.setEnabled(False)  # disable by default

        envelope_options_box = QtWidgets.QGroupBox("Amplitude Envelope")
        envelope_options_box.setStyleSheet("QGroupBox {font-weight:normal;}")
        envelope_options_box.setCheckable(True)
        envelope_options_box.setChecked(False)  # no amplitude normalization by default
        envelope_options_box.toggled.connect(self.wsize_spin.setEnabled)
        envelope_options_box.toggled.connect(self._dv.reanalyze_signal)

        envelope_options_layout = QtWidgets.QGridLayout()
        envelope_options_layout.addWidget(QtWidgets.QLabel("Window size"), 0, 0)
        envelope_options_layout.addWidget(self.wsize_spin, 0, 1)
        envelope_options_box.setLayout(envelope_options_layout)
        self.envelope_options_box = envelope_options_box

        # main layout of widget
        sinc_envelope_layout = QtWidgets.QHBoxLayout()
        sinc_envelope_layout.addWidget(sinc_options_box)
        sinc_envelope_layout.addWidget(envelope_options_box)
        self.setLayout(sinc_envelope_layout)

    @property
    def do_detrend(self) -> bool:
        return self.sinc_options_box.isChecked()

    @property
    def do_normalize(self) -> bool:
        return self.envelope_options_box.isChecked()

    def get_T_c(self) -> float | None:
        if self.do_detrend:
            return self.T_c_spin.value()
        return None

    def _changed_by_user_input(self, spin_name: Literal["T_c", "wsize"]):
        # to block dynamic defaults once the user changed the value
        if spin_name == 'T_c':
            self.restored_T_c = True
        if spin_name == 'wsize':
            self.restored_wsize = True

    def get_wsize(self) -> float | None:
        if not self.do_normalize:
            return None

        return self.wsize_spin.value()

    def set_auto_T_c(self, force=False):
        """
        Set the initial cut off period to a sensitive default,
        depending on dt and signal length.
        """

        if not np.any(self._dv.raw_signal):
            return
        assert self._dv.raw_signal is not None
        assert self._dv.dt is not None
        # check if a T_c was already entered
        if not self.restored_T_c or force:
            # default is 1.5 * Tmax -> 3/8 the observation time
            T_c_ini = self._dv.dt * 3 / 8 * len(self._dv.raw_signal)
            if self._dv.dt % 1 == 0.:
                T_c_ini = int(T_c_ini)
            else:
                T_c_ini = np.round(T_c_ini, 3)
            with QSignalBlocker(self.T_c_spin):
                self.T_c_spin.setValue(T_c_ini)
            logger.debug("Set auto T_c=%s, with force=%s, dt=%s and signal length %s",
                         T_c_ini,
                         force,
                         self._dv.dt,
                         len(self._dv.raw_signal)
                         )
        # set maximal cut off period to 5 times the signal length
        self.T_c_spin.setMaximum(10 * self._dv.dt * len(self._dv.raw_signal))
        # set minimum to Nyquist
        self.T_c_spin.setMinimum(2 * self._dv.dt)


    def set_auto_wsize(self, force=False):
        """
        Set the initial window size to a sensitive default,
        depending on dt and signal length.
        """

        if not np.any(self._dv.raw_signal):
            return
        assert self._dv.raw_signal is not None
        assert self._dv.dt is not None

        if not self.restored_wsize or force:
            # default is 1/4th of the observation time
            wsize_ini = self._dv.dt * len(self._dv.raw_signal) / 4
            if self._dv.dt % 1 == 0.:
                wsize_ini = int(wsize_ini)
            else:
                wsize_ini = np.round(wsize_ini, 3)
            with QSignalBlocker(self.wsize_spin):
                self.wsize_spin.setValue(wsize_ini)

            logger.debug("Set auto wsize=%s, with force=%s, dt=%s and signal length %s",
                         wsize_ini,
                         force,
                         self._dv.dt,
                         len(self._dv.raw_signal)
                         )

        # set maximal window size to the signal length
        self.wsize_spin.setMaximum(self._dv.dt * len(self._dv.raw_signal))
        # set minimum to 4 times the sample interval
        self.wsize_spin.setMinimum(4 * self._dv.dt)


class WaveletTab(QtWidgets.QFormLayout, SettingsManager):
    """Tmin, Tmax and nT widgets plus analyze buttons"""

    def __init__(self, dv: DataViewer):
        QtWidgets.QFormLayout.__init__(self)
        SettingsManager.__init__(self)

        self._dv = dv

        self._spins: dict[WidgetName, QtWidgets.QSpinBox | QtWidgets.QDoubleSpinBox] = {}
        self._restored: list[WidgetName] = []

        self._setup_UI()
        self._parameter_widgets = {name: self._spins[name] for name in ["Tmin", "Tmax", "nT"]}
        self._restore_settings()
        self._connect_to_unit()


    def _changed_by_user(self, name: WidgetName):
        """Catch first user input to disable dynamic defaults"""
        self._restored.append(name)

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
        Tmin_spin.valueChanged.connect(partial(self._changed_by_user, name='Tmin'))

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
        Tmax_spin.valueChanged.connect(partial(self._changed_by_user, name='Tmax'))

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

        self.addRow(Tmin_lab, Tmin_spin)
        self.addRow(Tmax_lab, Tmax_spin)
        self.addRow(step_lab, nT_spin)

    def _connect_to_unit(self):
        self._dv.unit_edit.textChanged.connect(
            mk_spinbox_unit_slot(self._spins["Tmin"])
        )
        self._dv.unit_edit.textChanged.connect(
            mk_spinbox_unit_slot(self._spins["Tmax"])
        )

    def set_auto_periods(self, force=False) -> None:

        """
        Determine period range from signal length and sampling interval.
        """
        if self._dv.raw_signal is None:
            return
        assert self._dv.raw_signal is not None
        assert self._dv.dt is not None

        Tmin_spin = self._spins['Tmin']
        Tmax_spin = self._spins['Tmax']

        with QSignalBlocker(Tmin_spin):
            Tmin_spin.setMinimum(2 * self._dv.dt)
        with QSignalBlocker(Tmax_spin):
            Tmax_spin.setMinimum(3 * self._dv.dt)
        # check if a Tmin was restored from settings
        # or rewrite if enforced
        if 'Tmin' not in self._restored or force:
            with QSignalBlocker(Tmin_spin):
                Tmin_spin.setValue(2 * self._dv.dt)  # Nyquist
                logger.debug(
                    "Set auto Tmin=%s, with force=%s, dt=%s and signal length %s",
                    2 * self._dv.dt,
                    force,
                    self._dv.dt,
                    len(self._dv.raw_signal)
                )
        if 'Tmax' not in self._restored or force:
            # default is 1/4 the observation time
            Tmax_ini = self._dv.dt * 1 / 4 * len(self._dv.raw_signal)
            if self._dv.dt % 1 == 0.:
                Tmax_ini = int(Tmax_ini)
            with QSignalBlocker(Tmax_spin):
                Tmax_spin.setValue(Tmax_ini)

                logger.debug(
                    "Set auto Tmax=%s, with force=%s, dt=%s and signal length %s",
                    Tmax_ini,
                    force,
                    self._dv.dt,
                    len(self._dv.raw_signal)
                )

    def _get_parameters(self) -> dict:
        return {name: spin.value() for name, spin in self._spins.items()}

    def assemble_wlet_pars(self) -> dict:
        wlet_pars = self._get_parameters()
        return wlet_pars
