from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import TYPE_CHECKING

from PyQt6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QMainWindow,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QWidget,
    QHBoxLayout,
    QGroupBox,
    QGridLayout,
    QSpacerItem,
    QMessageBox,
)

from PyQt6.QtCore import QSettings, Qt, QTimer

from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import pandas as pd

from pyboat import core
from pyboat import plotting as pl
from pyboat.ui.util import (
    mkGenericCanvas,
    selectFilter, is_dark_color_scheme,
    write_df, StoreGeometry,
    WAnalyzerParams, create_spinbox
)
from pyboat.ui import style
from pyboat.ui.defaults import debounce_ms

if TYPE_CHECKING:
    from .data_viewer import DataViewer
    from pandas import DataFrame

FormatFilter = "csv ( *.csv);; MS Excel (*.xlsx);; Text File (*.txt)"


class mkTimeSeriesCanvas(FigureCanvas):

    # dpi != 100 looks wierd?!
    def __init__(self, parent=None, width=4, height=3, dpi=100):

        self.fig1 = Figure(figsize=(width, height), dpi=dpi)

        FigureCanvas.__init__(self, self.fig1)
        self.setParent(parent)

        # print ('Time Series Size', FigureCanvas.sizeHint(self))
        FigureCanvas.setSizePolicy(self, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        FigureCanvas.updateGeometry(self)


class FourierAnalyzer(StoreGeometry, QMainWindow):
    def __init__(self, signal, dt, signal_id, position, time_unit, show_T, parent=None):
        StoreGeometry.__init__(self, pos=(510 + position, 520 + position), size=(520, 600))
        QMainWindow.__init__(self, parent=parent)

        self.time_unit = time_unit
        self.show_T = show_T

        # --- calculate Fourier spectrum ------------------
        self.fft_freqs, self.fpower = core.compute_fourier(signal, dt)
        # -------------------------------------------------

        self.initUI(signal_id, position)

    def initUI(self, signal_id: str, position_offset: int):

        self.setWindowTitle("Fourier spectrum " + signal_id)
        self.restore_geometry(position_offset)

        main_frame = QWidget()
        self.fCanvas = mkFourierCanvas()
        self.fCanvas.setParent(main_frame)
        ntb = NavigationToolbar(self.fCanvas, main_frame)

        # plot it
        ax = pl.mk_Fourier_ax(self.fCanvas.fig, self.time_unit, self.show_T)
        pl.Fourier_spec(ax, self.fft_freqs, self.fpower, self.show_T)
        self.fCanvas.fig.subplots_adjust(left=0.15)
        # self.fCanvas.fig.tight_layout()

        main_layout = QGridLayout()
        main_layout.addWidget(self.fCanvas, 0, 0, 9, 1)
        main_layout.addWidget(ntb, 10, 0, 1, 1)

        main_frame.setLayout(main_layout)
        self.setCentralWidget(main_frame)
        self.show()


class mkFourierCanvas(FigureCanvas):
    def __init__(self):
        self.fig, self.ax = plt.subplots(1, 1)

        FigureCanvas.__init__(self, self.fig)

        FigureCanvas.setSizePolicy(self, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        FigureCanvas.updateGeometry(self)


class WaveletAnalyzer(StoreGeometry, QMainWindow):
    def __init__(
        self,
        wanalyzer_params: WAnalyzerParams,
        position: int,
        signal_id: str,
        time_unit: str,
        dv: DataViewer,
    ) -> None:
        QMainWindow.__init__(self, parent=dv)
        pos=(dv.geometry().x() + position, dv.geometry().y() + position)
        StoreGeometry.__init__(self, pos=pos, size=(620, 750))

        self._wp = wanalyzer_params
        self.signal_id = signal_id
        self.time_unit: str = time_unit

        self._dv = dv

        # no ridge yet
        self.ridge: np.ndarray | None = None
        self.ridge_data: DataFrame | None = None
        self.power_thresh: int = 0
        self.rsmoothing: int | None = None
        self._has_ridge = False  # no plotted ridge

        # no anneal parameters yet
        self.anneal_pars = None

        # Wavelet ridge-readout results
        self.ResultWindows = {}
        self.w_offset = 0

        self._replot_timer = QTimer(self)
        self._replot_timer.setInterval(debounce_ms)
        self._replot_timer.setSingleShot(True)

        self.initUI(position)

        # throttle reanalyze
        self._replot_timer.timeout.connect(self._update_plot)


    def reanalyze(self, wp: WAnalyzerParams, new_signal: bool = False):
        """Recompute and update signal and spectrum plot"""
        params = wp.asdict()
        if not new_signal:
            # recover original signal
            params["raw_signal"] = self._wp.raw_signal
        self._wp = WAnalyzerParams(**params)
        self._compute_spectrum()
        self._update_plot()  # updates also ridge readout window

    def _compute_spectrum(self):
        """Compute the wavelet spectrum"""

        # == Compute Spectrum ==
        self.modulus, self.wlet = core.compute_spectrum(
            self._wp.filtered_signal, self._wp.dt, self._wp.periods
        )

    def initUI(self, position_offset: int):
        self.setWindowTitle("Wavelet Spectrum - " + str(self.signal_id))
        self.restore_geometry(position_offset)

        main_widget = QWidget()
        self.statusBar()

        # Wavelet and signal plot
        self.wCanvas = mkWaveletCanvas()
        main_frame = QWidget()
        self.wCanvas.setParent(main_frame)
        ntb = NavigationToolbar(self.wCanvas, main_frame)  # full toolbar

        # -------------plot the wavelet power spectrum---------------------------

        self._compute_spectrum()
        # creates the ax and attaches it to the widget figure
        axs = pl.mk_signal_modulus_ax(self.time_unit, fig=self.wCanvas.fig)
        pl.plot_signal_modulus(
            axs,
            time_vector=self._wp.tvec,
            signal=self._wp.filtered_signal,
            modulus=self.modulus,
            periods=self._wp.periods,
            p_max=self._wp.max_power,
        )

        self.wCanvas.fig.subplots_adjust(bottom=0.11, right=0.95, left=0.13, top=0.95)
        self.wCanvas.fig.tight_layout()

        # --- Spectrum plotting options ---

        spectrum_opt_box = QGroupBox("Spectrum Plotting Options")
        spectrum_opt_box.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Maximum)

        spectrum_opt_layout = QHBoxLayout()
        spectrum_opt_layout.setSpacing(10)
        spectrum_opt_box.setLayout(spectrum_opt_layout)

        # uppler limit of the colormap <-> imshow(...,vmax = pmax)
        pmax_label = QLabel("Maximal Power:")
        # retrieve initial power value, axs[1] is the spectrum
        pmin, pmax = axs[1].images[0].get_clim()

        self.pmax_spin = create_spinbox(
            int(pmax) if pmax > 1 else 1,
            minimum=1,
            maximum=1_000,
            step=1,
            status_tip="Change upper limit of the wavelet power color map",
            double=False
        )
        self.pmax_spin.setMaximumWidth(80)
        self.pmax_spin.valueChanged.connect(self._replot)

        equalize_powers_button = QPushButton("Set for all", self)
        equalize_powers_button.setStatusTip(
            "Set this maximal power for all opened wavelet spectra"
            )
        equalize_powers_button.clicked.connect(self._equalize_powers)
        if is_dark_color_scheme():
            equalize_powers_button.setStyleSheet(f"background-color: {style.dark_accent}")
        else:
            equalize_powers_button.setStyleSheet(f"background-color: {style.light_accent}")


        self.cb_coi = QCheckBox("COI", self)
        self.cb_coi.setStatusTip("Draws the cone of influence onto the spectrum")
        self.cb_coi.stateChanged.connect(self.draw_coi)

        # ridge_opt_layout.addWidget(drawRidgeButton,1,3) # not needed anymore?!
        spectrum_opt_layout.addWidget(pmax_label)
        spectrum_opt_layout.addWidget(self.pmax_spin)
        spectrum_opt_layout.addStretch(0)
        spectrum_opt_layout.addWidget(equalize_powers_button)
        spectrum_opt_layout.addStretch(0)
        spectrum_opt_layout.addWidget(self.cb_coi)

        # --- Time average -> asymptotic Fourier ---

        time_av_box = QGroupBox("Time Averaging")

        estimFourierButton = QPushButton("Estimate Fourier", self)
        estimFourierButton.clicked.connect(self.ini_average_spec)
        estimFourierButton.setStatusTip("Shows time averaged Wavelet power spectrum")

        time_av_layout = QHBoxLayout()
        # time_av_layout.addStretch()
        time_av_layout.addWidget(estimFourierButton)
        # time_av_layout.addStretch()

        time_av_box.setLayout(time_av_layout)

        # --- Ridge detection and options --

        ridge_opt_box = QGroupBox("Ridge Detection")
        ridge_opt_box.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Maximum)

        ridge_opt_layout = QGridLayout()
        ridge_opt_box.setLayout(ridge_opt_layout)

        # Start ridge detection
        maxRidgeButton = QPushButton("Detect Maximum Ridge", self)
        maxRidgeButton.setStatusTip("Traces the time-consecutive power maxima")
        if is_dark_color_scheme():
            maxRidgeButton.setStyleSheet(f"background-color: {style.dark_primary}")
        else:
            maxRidgeButton.setStyleSheet(f"background-color: {style.light_primary}")


        maxRidgeButton.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        maxRidgeButton.clicked.connect(self.do_maxRidge_detection)

        power_label = QLabel("Ridge threshold")
        power_label.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)

        power_thresh_spin = create_spinbox(
            0,
            minimum=0,
            maximum=1_000,
            step=1.,
            status_tip="Threshold for the traced wavelet power maxima ",
            double=True
        )
        power_thresh_spin.valueChanged.connect(self._replot)
        self.power_thresh_spin = power_thresh_spin

        smooth_label = QLabel("Ridge smoothing")
        ridge_smooth_spin = create_spinbox(
            0,
            minimum=0,
            maximum=100,
            step=1,
            status_tip="Savitzky-Golay smoothing (k=3) of the ridge time series",
            double=False
        )
        ridge_smooth_spin.valueChanged.connect(self.qset_ridge_smooth)
        # Plot Results
        plotResultsButton = QPushButton("Plot Ridge Readout", self)
        maxRidgeButton.setStatusTip("Traces the time-consecutive power maxima")

        plotResultsButton.setStatusTip(
            "Shows instantaneous period, phase, power and amplitude along the ridge"
        )
        # plotResultsButton.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        plotResultsButton.clicked.connect(self.ini_plot_readout)

        ridge_opt_layout.addWidget(maxRidgeButton, 0, 0, 1, 1)
        ridge_opt_layout.addWidget(plotResultsButton, 1, 0, 1, 1)

        ridge_opt_layout.addWidget(power_label, 0, 1)
        ridge_opt_layout.addWidget(power_thresh_spin, 0, 2)

        ridge_opt_layout.addWidget(smooth_label, 1, 1)
        ridge_opt_layout.addWidget(ridge_smooth_spin, 1, 2)

        # for spacing
        rtool_box = QWidget()
        rtool_layout = QHBoxLayout()
        rtool_box.setLayout(rtool_layout)
        rtool_layout.addWidget(ridge_opt_box)
        rtool_layout.addStretch(0)

        # --- put everything together ---

        main_layout = QGridLayout()
        # main_layout.setSpacing(0)
        main_layout.addWidget(self.wCanvas, 0, 0, 4, 5)
        main_layout.addWidget(ntb, 5, 0, 1, 5)
        main_layout.addWidget(spectrum_opt_box, 6, 0, 1, 3)
        main_layout.addWidget(time_av_box, 6, 3, 1, 1)
        main_layout.addItem(QSpacerItem(2, 15), 6, 5)
        main_layout.addWidget(ridge_opt_box, 7, 0, 1, 5)

        # set stretching (resizing) behavior
        main_layout.setRowStretch(0, 1)  # plot should stretch
        main_layout.setRowMinimumHeight(0, 300)  # plot should not get squeezed too much

        main_layout.setRowStretch(5, 0)  # options shouldn't stretch
        main_layout.setRowStretch(6, 0)  # options shouldn't stretch
        main_layout.setRowStretch(7, 0)  # options shouldn't stretch

        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        self.show()

    def qset_ridge_smooth(self, rsmooth: int):

        """
        rsmooth is the window size for
        the savgol filter
        """

        # make an odd window length
        if rsmooth == 0:
            self.rsmoothing = None
        elif rsmooth < 5:
            self.rsmoothing = 5
        elif rsmooth > 5 and rsmooth % 2 == 0:
            self.rsmoothing = rsmooth + 1
        else:
            self.rsmoothing = rsmooth

        # update the plot on the fly
        if self._has_ridge:
            self.draw_ridge()

    def do_maxRidge_detection(self):

        ridge_y = core.get_maxRidge_ys(self.modulus)
        self.ridge = ridge_y

        if not np.any(ridge_y):
            msgBox = QMessageBox()
            msgBox.setWindowTitle("Ridge detection error")
            msgBox.setText("No ridge found..check spectrum!")
            msgBox.exec()

            return

        self._has_ridge = True
        self.draw_ridge()  # ridge_data made here

    def draw_ridge(self):
        """ makes also the ridge_data !! """

        if not self._has_ridge:

            msgBox = QMessageBox()
            msgBox.setWindowTitle("No Ridge")
            msgBox.setText("Do a ridge detection first!")
            msgBox.exec()

        ridge_data = core.eval_ridge(
            self.ridge,
            self.wlet,
            self._wp.filtered_signal,
            self._wp.periods,
            self._wp.tvec,
            power_thresh=self.power_thresh_spin.value(),
            smoothing_wsize=self.rsmoothing
        )

        # plot the ridge
        ax_spec = self.wCanvas.fig.axes[1]  # the spectrum

        # already has a plotted ridge
        for line in ax_spec.lines:
            line.remove()  # remove old ridge line and COI
        self.cb_coi.setCheckState(Qt.CheckState.Unchecked)

        pl.draw_Wavelet_ridge(ax_spec, ridge_data, marker_size=1.5)

        # refresh the canvas
        self.wCanvas.draw()

        self.ridge_data = ridge_data

    def _equalize_powers(self):
        """Set the maximal power in all opened WaveletAnalyzer windows """

        pmax = self.pmax_spin.value()

        for aw in self._dv.anaWindows:
            # don't replot for **this** WaveletAnalyzer
            if aw is not self:
                aw.pmax_spin.setValue(pmax)

    def _replot(self) -> None:
        self._replot_timer.start()

    def _update_plot(self, _=None):
        """
        Replots the entire spectrum canvas
        with the current maximal power.
        """

        # remove the old plot
        self.wCanvas.fig.clf()

        # retrieve new pmax value
        pmax = self.pmax_spin.value()
        # creates the ax and attaches it to the widget figure
        axs = pl.mk_signal_modulus_ax(self.time_unit, fig=self.wCanvas.fig)

        pl.plot_signal_modulus(
            axs,
            time_vector=self._wp.tvec,
            signal=self._wp.filtered_signal,
            modulus=self.modulus,
            periods=self._wp.periods,
            p_max=pmax,
        )

        # redraw COI if checkbox is checked
        self.draw_coi()

        # re-draw ridge
        if self._has_ridge:
            self.do_maxRidge_detection()
            self.draw_ridge()
            if rw := self.ResultWindows.get(self.w_offset - 30):  # 30px is the increment
                rw.plot_readout(self.ridge_data)

        # refresh the canvas
        self.wCanvas.draw()
        self.wCanvas.show()

    def draw_coi(self):

        """
        Draws the COI on the spectrum.
        Also redraws the ridge!
        """

        ax_spec = self.wCanvas.fig.axes[1]  # the spectrum axis

        # COI desired?
        if self.cb_coi.isChecked():
            # draw on the spectrum
            pl.draw_COI(ax_spec, self._wp.tvec)

        else:
            for line in ax_spec.lines:
                line.remove()  # removes coi and ridge!
            if self._has_ridge:
                self.draw_ridge()  # re-draw ridge

        # refresh the canvas
        self.wCanvas.draw()

    def ini_plot_readout(self):

        if not self._has_ridge:
            msgBox = QMessageBox()
            msgBox.setWindowTitle("No Ridge")
            msgBox.setText("Do a ridge detection first!")
            msgBox.exec()
            return

        if not np.any(self.ridge_data):
            msgBox = QMessageBox()
            msgBox.setWindowTitle("No Ridge")
            msgBox.setText("Empty ridge - reduce ridge power threshold!")
            msgBox.exec()
            return

        # to keep the line shorter..
        wo = self.w_offset
        self.ResultWindows[wo] = WaveletReadoutWindow(
            self.signal_id,
            self.ridge_data,
            time_unit=self.time_unit,
            draw_coi=self.cb_coi.isChecked(),
            pos_offset=self.w_offset,
            parent=self,
        )
        self.w_offset += 30

    def ini_average_spec(self):

        self.avWspecWindow = AveragedWaveletWindow(self.w_offset, parent=self)

    def closeEvent(self, event):
        """Removes itself from the analyzer stack of the DataViewer."""
        self._dv.anaWindows.remove(self)
        event.accept()


class mkWaveletCanvas(FigureCanvas):
    def __init__(self, parent=None):  # , width=6, height=3, dpi=100):

        # dpi changes fontsize, so bette leave it as is..
        self.fig = Figure(dpi=100)

        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        FigureCanvas.updateGeometry(self)


class WaveletReadoutWindow(StoreGeometry, QMainWindow):
    def __init__(
        self,
        signal_id,
        ridge_data,
        time_unit,
        draw_coi,
        parent,
        pos_offset=0,
        DEBUG=False,
    ):
        StoreGeometry.__init__(self, pos=(700 + pos_offset, 260 + pos_offset),size=(750, 500))
        QMainWindow.__init__(self, parent=parent)

        self.signal_id = signal_id

        self.draw_coi = draw_coi
        self.ridge_data = ridge_data
        self.time_unit = time_unit

        # creates self.rCanvas and plots the results
        self.initUI(pos_offset)

        self.DEBUG = DEBUG

    def initUI(self, pos_offset):

        self.setWindowTitle("Wavelet Results - " + str(self.signal_id))
        self.restore_geometry(pos_offset)

        # embed the plotting canvas

        self.rCanvas = mkReadoutCanvas()
        main_frame = QWidget()
        self.rCanvas.setParent(main_frame)
        ntb = NavigationToolbar(self.rCanvas, main_frame)

        # --- plot the wavelet results ---------
        self.plot_readout(
            self.ridge_data,
            draw_coi=self.draw_coi
        )
        # messes things up here :/
        # self.rCanvas.fig.tight_layout()

        main_layout = QGridLayout()
        main_layout.addWidget(self.rCanvas, 0, 0, 9, 1)
        main_layout.addWidget(ntb, 10, 0, 1, 1)

        # add the save Button
        SaveButton = QPushButton("Save Results", self)
        if is_dark_color_scheme():
            SaveButton.setStyleSheet(f"background-color: {style.dark_primary}")
        else:
            SaveButton.setStyleSheet(f"background-color: {style.light_primary}")

        SaveButton.clicked.connect(self.save_out)

        button_layout_h = QHBoxLayout()
        button_layout_h.addStretch(1)
        button_layout_h.addWidget(SaveButton)
        button_layout_h.addStretch(1)
        main_layout.addLayout(button_layout_h, 11, 0, 1, 1)

        main_frame.setLayout(main_layout)
        self.setCentralWidget(main_frame)
        self.show()

    def plot_readout(self, ridge_data: pd.DataFrame, draw_coi: bool = False):
        """Can be also used to re-plot in an existing readout window instance"""

        # empty ridge can happen with thresholding
        if not np.any(ridge_data):
            return

        self.rCanvas.fig.clf()
        pl.plot_readout(
            ridge_data,
            self.time_unit,
            fig=self.rCanvas.fig,
            draw_coi=draw_coi,
        )
        self.rCanvas.fig.subplots_adjust(
            wspace=0.3, left=0.1, top=0.98, right=0.95, bottom=0.15
        )
        self.rCanvas.draw()
        self.ridge_data = ridge_data

    def save_out(self):

        dialog = QFileDialog()
        dialog.setDefaultSuffix("csv")
        # retrieve or initialize directory path
        settings = QSettings()
        dir_path = settings.value("dir_name", os.path.curdir)
        data_format = settings.value("default-settings/data_format", "csv")

        # ----------------------------------------------------------
        base_name = str(self.signal_id).replace(" ", "-")
        default_name = os.path.join(dir_path, base_name + "_ridgeRO.")
        default_name += data_format
        # -----------------------------------------------------------
        file_name, sel_filter = dialog.getSaveFileName(
            self,
            "Save ridge readout as",
            default_name,
            FormatFilter,
            selectFilter[data_format],
        )

        # dialog cancelled
        if not file_name:
            return

        if self.DEBUG:
            print("selected filter:", sel_filter)
            print("out-path:", file_name)
            print("ridge data keys:", self.ridge_data.keys())

        write_df(self.ridge_data, file_name)


class mkReadoutCanvas(FigureCanvas):
    def __init__(self):

        self.fig = Figure(figsize=(8.5, 7), dpi=100)

        FigureCanvas.__init__(self, self.fig)

        FigureCanvas.setSizePolicy(self, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        FigureCanvas.updateGeometry(self)


class AveragedWaveletWindow(StoreGeometry, QMainWindow):
    # `parent` is a `WaveletAnalyzer` instance
    def __init__(self, pos_offset: int, parent):

        StoreGeometry.__init__(self, pos=(510 + pos_offset, 80 + pos_offset), size=(550, 400))
        QMainWindow.__init__(self, parent=parent)

        # --- calculate time averaged power spectrum <-> Fourier estimate ---
        self.avWspec = np.sum(parent.modulus, axis=1) / parent.modulus.shape[1]
        # -------------------------------------------------------------------

        # the Wavelet analysis window spawning *this* Widget
        self.parentWA = parent
        self.initUI()

    def initUI(self):

        self.setWindowTitle(f"Fourier Spectrum Estimate - {self.parentWA.signal_id}")
        self.restore_geometry()

        main_frame = QWidget()
        pCanvas = mkGenericCanvas()
        pCanvas.setParent(main_frame)
        ntb = NavigationToolbar(pCanvas, main_frame)

        # plot it
        pCanvas.fig.clf()

        pl.averaged_Wspec(
            self.avWspec,
            self.parentWA._wp.periods,
            time_unit=self.parentWA.time_unit,
            fig=pCanvas.fig,
        )

        pCanvas.fig.subplots_adjust(left=0.15, bottom=0.17)

        main_layout = QGridLayout()
        main_layout.addWidget(pCanvas, 0, 0, 9, 1)
        main_layout.addWidget(ntb, 10, 0, 1, 1)

        # add the save Button
        SaveButton = QPushButton("Save Results", self)
        SaveButton.clicked.connect(self.save_out)

        button_layout_h = QHBoxLayout()
        button_layout_h.addStretch(1)
        button_layout_h.addWidget(SaveButton)
        button_layout_h.addStretch(1)
        main_layout.addLayout(button_layout_h, 11, 0, 1, 1)

        main_frame.setLayout(main_layout)
        self.setCentralWidget(main_frame)
        self.show()

    def save_out(self):

        df_out = pd.DataFrame(data=self.avWspec, columns=["power"])
        df_out.index = self.parentWA.periods
        df_out.index.name = "periods"

        dialog = QFileDialog()
        dialog.setDefaultSuffix("csv")
        # retrieve or initialize directory path
        settings = QSettings()
        dir_path = settings.value("dir_name", os.path.curdir)
        data_format = settings.value("default-settings/data_format", "csv")

        # --------------------------------------------------------
        base_name = str(self.parentWA.signal_id).replace(" ", "-")
        default_name = os.path.join(dir_path, base_name + "_avWavelet.")
        default_name += data_format
        # -----------------------------------------------------------

        file_name, sel_filter = dialog.getSaveFileName(
            self,
            "Save averaged Wavelet spectrum",
            default_name,
            FormatFilter,
            selectFilter[data_format],
        )

        # dialog cancelled
        if not file_name:
            return

        write_df(df_out, file_name)
