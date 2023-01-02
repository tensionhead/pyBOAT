import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os.path import expanduser

from PyQt5.QtWidgets import (
    QCheckBox,
    QMessageBox,
    QFileDialog,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QWidget,
    QGroupBox,
    QGridLayout,
    QProgressBar,
    QMainWindow,
)
from PyQt5.QtGui import QIntValidator
from PyQt5.QtCore import QSettings, Qt
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from pyboat.ui.util import posfloatV, mkGenericCanvas, spawn_warning_box

import pyboat
from pyboat import plotting as pl
from pyboat import ensemble_measures as em


# --- resolution of the output plots ---
DPI = 250


class BatchProcessWindow(QMainWindow):

    """
    The parent is a DataViewer instance holding the
    data as a DataFrame, and other global properties:

    parent.df
    parent.dt
    parent.time_unit

    """

    def __init__(self, DEBUG, parent=None):

        super().__init__(parent=parent)

        # the DataViewer spawning *this* Widget
        self.parentDV = parent
        self.debug = DEBUG

    def initUI(self, wlet_pars):

        """
        Gets called from the parent DataViewer
        """

        self.setWindowTitle("Batch Processing")
        self.setGeometry(310, 330, 600, 200)

        # from the DataViewer
        self.wlet_pars = wlet_pars

        # for the status bar
        main_widget = QWidget()
        self.statusBar()

        main_layout = QGridLayout()


        # -- Plotting Options --

        plotting_options = QGroupBox("Show Summary Statistics")

        self.cb_plot_ens_dynamics = QCheckBox("Ensemble Dynamics")
        self.cb_plot_ens_dynamics.setStatusTip(
            "Show period, amplitude and phase median and quartiles over time"
        )
        self.cb_plot_ens_dynamics.setChecked(True)

        self.cb_plot_global_spec = QCheckBox("Global Wavelet Spectrum")
        self.cb_plot_global_spec.setStatusTip("Ensemble averaged Wavelet spectrum")

        self.cb_plot_Fourier_dis = QCheckBox("Global Fourier Estimate")
        self.cb_plot_Fourier_dis.setStatusTip(
            "Ensemble median and quartiles of the time averaged Wavelet spectra"
        )

        self.cb_power_hist = QCheckBox("Ridge Power Histogram")
        self.cb_power_hist.setStatusTip(
            "Show time- and frequency averaged distribution of ridge powers"
        )

        lo = QGridLayout()
        lo.addWidget(self.cb_plot_ens_dynamics, 0, 0)
        lo.addWidget(self.cb_plot_global_spec, 1, 0)
        lo.addWidget(self.cb_plot_Fourier_dis, 2, 0)
        lo.addWidget(self.cb_power_hist, 3, 0)
        plotting_options.setLayout(lo)

        # -- Ridge Analysis Options --

        ridge_options = QGroupBox("Ridge Detection Options")

        thresh_label = QLabel("Ridge Threshold:")
        thresh_edit = QLineEdit()
        thresh_edit.setValidator(posfloatV)
        thresh_edit.insert("0")
        thresh_edit.setMaximumWidth(60)
        thresh_edit.setStatusTip(
            "Ridge points below that power value will be filtered out "
        )
        self.thresh_edit = thresh_edit

        smooth_label = QLabel("Ridge Smoothing:")
        smooth_edit = QLineEdit()
        smooth_edit.setMaximumWidth(60)
        smooth_edit.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        smooth_edit.setValidator(QIntValidator(bottom=3, top=99999999))
        smooth_edit.setStatusTip(
            """Savitkzy-Golay window size, leave blank for no smoothing"""
        )
        self.smooth_edit = smooth_edit

        ridge_options_layout = QGridLayout()
        ridge_options_layout.addWidget(thresh_label, 0, 0)
        ridge_options_layout.addWidget(thresh_edit, 0, 1)
        ridge_options_layout.addWidget(smooth_label, 1, 0)
        ridge_options_layout.addWidget(smooth_edit, 1, 1)
        ridge_options.setLayout(ridge_options_layout)

        # --- Export Path ---

        path_options = QGroupBox("Set Export Directory")

        # defaults to HOME or former working dir
        # retrieve or initialize directory path
        settings = QSettings()
        dir_path = settings.value("dir_name", expanduser("~"))
        self.OutPath_edit = QLineEdit(dir_path)

        PathButton = QPushButton("Select Path..")
        PathButton.setMaximumWidth(100)
        PathButton.clicked.connect(self.select_export_dir)

        lo = QGridLayout()
        lo.addWidget(PathButton, 0, 0)
        lo.addWidget(self.OutPath_edit, 1, 0)
        path_options.setLayout(lo)

        # -- Export Results --

        self.cb_filtered_sigs = QCheckBox("Filtered Signals")
        self.cb_filtered_sigs.setStatusTip(
            "Saves detrended and amplitude normalized signals to disc as csv's"
        )

        self.cb_specs = QCheckBox("Wavelet Spectra with ridges")
        self.cb_specs.setStatusTip("Saves the individual wavelet spectra as images")

        self.cb_specs_noridge = QCheckBox("Wavelet Spectra w/o ridges")
        self.cb_specs_noridge.setStatusTip(
            "Saves the individual wavelet spectra without the ridges as images"
        )

        self.cb_readout = QCheckBox("Ridge Readouts")
        self.cb_readout.setStatusTip(
            "Saves one analysis result per signal to disc as csv"
        )

        self.cb_readout_plots = QCheckBox("Ridge Readout Plots")
        self.cb_readout_plots.setStatusTip("Saves the individual readout plots to disc")
        self.cb_sorted_powers = QCheckBox("Sorted Average Powers")
        self.cb_sorted_powers.setStatusTip(
            "Saves the time-averaged ridge powers in descending order"
        )
        self.cb_save_ensemble_dynamics = QCheckBox("Ensemble Dynamics")
        self.cb_save_ensemble_dynamics.setStatusTip(
            "Saves period, amplitude, power and phase summary statistics to csv files"
        )

        self.cb_save_Fourier_dis = QCheckBox("Global Fourier Estimate")
        self.cb_save_Fourier_dis.setStatusTip(
            "Saves median and quartiles of the ensemble Fourier power spectral distribution"
        )

        export_data = QGroupBox("Export Data")
        export_data.setStatusTip("Creates csv files")

        # export data layout

        lo = QGridLayout()
        lo.setSpacing(0)

        lo.addWidget(self.cb_filtered_sigs, 0, 0)
        lo.addWidget(self.cb_readout, 1, 0)
        lo.addWidget(self.cb_sorted_powers, 2, 0)
        lo.addWidget(self.cb_save_ensemble_dynamics, 3, 0)
        lo.addWidget(self.cb_save_Fourier_dis, 4, 0)

        export_data.setLayout(lo)

        self.export_data = export_data

        # --- Export figures ---

        export_figs = QGroupBox("Export Figures")
        export_figs.setStatusTip("Creates images on disc")

        lo = QGridLayout()
        lo.setSpacing(0)

        lo.addWidget(self.cb_specs, 0, 0)
        lo.addWidget(self.cb_specs_noridge, 1, 0)
        lo.addWidget(self.cb_readout_plots, 2, 0)

        export_figs.setLayout(lo)
        self.export_figs = export_figs


        # -- Progress and Run --
        Nsignals = self.parentDV.df.shape[1]

        RunButton = QPushButton(f"Analyze {Nsignals} Signals!", self)
        RunButton.setStyleSheet("background-color: orange")
        RunButton.clicked.connect(self.run_batch)
        # RunButton.setMaximumWidth(60)


        # the progress bar
        self.progress = QProgressBar(self)
        self.progress.setRange(0, Nsignals - 1)
        # self.progress.setGeometry(0,0, 300, 20)
        self.progress.setMinimumWidth(200)

        process_box = QGroupBox("Run with Settings")
        process_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        lo = QGridLayout()
        lo.addWidget(RunButton, 0, 0)
        lo.addWidget(self.progress, 0, 1)
        process_box.setLayout(lo)

        # -- main layout --

        main_layout.addWidget(ridge_options, 0, 0, 1, 1)
        main_layout.addWidget(plotting_options, 1, 0, 1, 1)
        main_layout.addWidget(path_options, 2, 0, 1, 1)
        main_layout.addWidget(process_box, 3, 0, 1, 1)
        main_layout.addWidget(export_data, 0, 1, 2, 1)
        main_layout.addWidget(export_figs, 2, 1, 2, 1)

        # set main layout
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        self.show()

    def run_batch(self):

        """
        Retrieve all batch settings and loop over the signals
        present in the parentDV

        """

        dataset_name = self.parentDV.df.name
        # just rebind the name
        dt = self.parentDV.dt
        time_unit = self.parentDV.time_unit

        OutPath = self.get_OutPath()
        # if user cleared then do nothing
        # should not be reached due to validator
        if OutPath is None:
            return

        # check signal lengths
        lens = []
        # for the global modulus normalization
        norm_vec = np.ones(self.parentDV.df.shape[0])
        for signal_id in self.parentDV.df:
            signal = self.parentDV.df[signal_id]
            start, end = signal.first_valid_index(), signal.last_valid_index()
            # intermediate NaNs get interpolated in `vector_prep`
            norm_vec[start:end + 1] += 1
            lens.append(end - start + 1)

        if min(lens) != max(lens):
            tt = ("Signals with different lengths found!\n"
                  "pyBOAT can still process the ensemble, but\n"
                  "consider trimming for more consistent results\n\n"
                  f"Shortest signal: {min(lens) * dt:.2f} {time_unit}\n"
                  f"Longest signal: {max(lens) * dt:.2f} {time_unit}\n"
                  "Do you want to continue?"
                  )

            choice = QMessageBox.question(
                self,
                "Warning",
                tt,
                QMessageBox.Yes | QMessageBox.No,
            )
            if choice == QMessageBox.Yes:
                pass
            else:
                # abort batch processing
                return

        # TODO: parallelize
        ridge_results, df_fouriers, global_modulus = self.do_the_loop(norm_vec)

        # check for empty ridge_results
        if not ridge_results:

            msgBox = QMessageBox(parent=self)
            msgBox.setWindowTitle("No Results")
            msgBox.setText("All ridges below threshold.. no results!")
            msgBox.exec()

            return

        settings = QSettings()
        float_format = settings.value("float_format", "%.3f")

        # --- compute the time-averaged powers ---

        if self.cb_power_hist.isChecked() or self.cb_sorted_powers.isChecked():

            powers_series = em.average_power_distribution(
                ridge_results.values(), ridge_results.keys(), exclude_coi=True
            )

        if self.cb_power_hist.isChecked():
            # plot the distribution
            self.pdw = PowerHistogramWindow(powers_series, dataset_name=dataset_name, parent=self)

        # save out the sorted average powers
        if self.cb_sorted_powers.isChecked():
            fname = os.path.join(OutPath, f"{dataset_name}_sorted-powers.csv")
            powers_series.to_csv(
                fname, sep=",", float_format=float_format, index=True, header=False
            )

        # --- compute summary statistics over time ---

        if (
            self.cb_plot_ens_dynamics.isChecked()
            or self.cb_save_ensemble_dynamics.isChecked()
        ):
            # res is a tuple of  DataFrames, one each for
            # periods, amplitude, power and phase
            res = em.get_ensemble_dynamics(ridge_results.values())

        if self.cb_plot_ens_dynamics.isChecked():
            self.edw = EnsembleDynamicsWindow(
                res,
                dt=self.parentDV.dt,
                time_unit=self.parentDV.time_unit,
                dataset_name=dataset_name,
                parent=self
            )

        if (
            self.cb_save_ensemble_dynamics.isChecked()
        ):
            # create time axis, all DataFrames have same number of rows
            tvec = np.arange(res[0].shape[0]) * self.parentDV.dt
            for obs, df in zip(["periods", "amplitudes", "powers", "phasesR"], res):
                fname = os.path.join(OutPath, f"{dataset_name}_{obs}.csv")
                df.index = tvec
                df.index.name = "time"
                df.to_csv(fname, sep=",", float_format=float_format)

        # --- Fourier Distribution Outputs ---

        if self.cb_plot_Fourier_dis.isChecked():

            self.fdw = FourierDistributionWindow(
                df_fouriers, self.parentDV.time_unit, dataset_name, parent=self
            )

        if self.cb_save_Fourier_dis.isChecked():

            fname = os.path.join(OutPath, f"{dataset_name}_global-fourier-estimate.csv")

            # save out median and quartiles of Fourier powers
            df_fdis = pd.DataFrame(index=df_fouriers.index)
            df_fdis["Median"] = df_fouriers.median(axis=1)
            df_fdis["Mean"] = df_fouriers.mean(axis=1)
            df_fdis["Q1"] = df_fouriers.quantile(q=0.25, axis=1)
            df_fdis["Q3"] = df_fouriers.quantile(q=0.75, axis=1)

            df_fdis.to_csv(fname, sep=",", float_format=float_format)

        # --- Global Wavelet Spectrum ---
        if self.cb_plot_global_spec.isChecked():

            self.gspec = GlobalSpectrumWindow(
                global_modulus, self.parentDV.time_unit, dataset_name, parent=self
            )

        if self.debug:
            print(list(ridge_results.items())[:2])

    def get_thresh(self):

        """
        Reads the self.thresh_edit
        A Validator is set..
        """

        thresh_str = self.thresh_edit.text().replace(",", ".")
        try:
            thresh = float(thresh_str)
            if self.debug:
                print("thresh set to:", thresh)
            return thresh

        # empty line edit is interpreted as no thresholding required
        except ValueError:
            if self.debug:
                print("thresh ValueError", thresh_str)
            return 0

    def get_ridge_smooth(self):

        """
        Reads the self.smooth_edit
        A Validator is set..
        """

        rs = self.smooth_edit.text()
        try:
            rsmooth = int(rs)
            if self.debug:
                print("rsmooth set to:", rs)

            # make an odd window length
            if rsmooth == 0:
                return None
            elif rsmooth < 5:
                return 5
            elif rsmooth > 5 and rsmooth % 2 == 0:
                return rsmooth + 1
            else:
                return rsmooth

        # empty line edit is interpreted as no ridge smoothing required
        except ValueError:
            if self.debug:
                print("No rsmooth", rs)
            return None

    def get_OutPath(self):

        """
        Reads the self.OutPath_edit
        There is no validator but an os.path
        check is done!
        """

        path = self.OutPath_edit.text()

        if not os.path.isdir(path):

            msgBox = QMessageBox()
            msgBox.setWindowTitle("Invalid export path")
            msgBox.setText("Specified path is not a valid directory..")
            msgBox.exec()

            return None
        return path

    def select_export_dir(self):

        dialog = QFileDialog()
        dialog.setFileMode(QFileDialog.DirectoryOnly)
        dialog.setOption(QFileDialog.ShowDirsOnly, False)

        # retrieve or initialize directory path
        settings = QSettings()
        dir_path = settings.value("dir_name", expanduser("~"))

        dir_name = dialog.getExistingDirectory(
            self, "Select a folder to save the results", dir_path
        )

        # dialog cancelled
        if not dir_name:
            return

        if self.debug:
            print("Batch output name:", dir_name)

        self.OutPath_edit.setText(dir_name)

    def do_the_loop(self, norm_vec):

        """
        Uses the explicitly parsed self.wlet_pars
        to control signal analysis settings.

        Takes general analysis Parameters

        self.parentDV.dt
        self.parentDV.time_unit

        and the DataFrame

        self.parentDV.df

        from the parent DataViewer.
        Reads additional settings from this Batch Process Window.

        """

        EmptyRidge = 0

        OutPath = self.get_OutPath()

        periods = np.linspace(
            self.wlet_pars["Tmin"], self.wlet_pars["Tmax"], self.wlet_pars["step_num"]
        )

        # retrieve batch settings
        power_thresh = self.get_thresh()
        rsmooth = self.get_ridge_smooth()

        # results get stored here
        ridge_results = {}
        df_fouriers = pd.DataFrame(index=periods)
        df_fouriers.index.name = "period"
        # ensemble averaged wavelet spectrum
        global_modulus = np.zeros((len(periods), len(norm_vec)))

        for i, signal_id in enumerate(self.parentDV.df):

            # log to terminal
            print(f"processing {signal_id}..")

            # sets parentDV.raw_signal and parentDV.tvec
            succ, start, end = self.parentDV.vector_prep(signal_id)
            # ui silently passes over..
            if not succ:
                print(f"Warning, can't process signal {signal_id}..")
                continue

            # detrend?!
            if self.parentDV.cb_use_detrended.isChecked():
                trend = self.parentDV.calc_trend()
                signal = self.parentDV.raw_signal - trend
            else:
                signal = self.parentDV.raw_signal

            # amplitude normalization?
            if self.parentDV.cb_use_envelope.isChecked():
                if self.debug:
                    print("Calculating envelope with L=", self.wlet_pars["window_size"])
                signal = pyboat.normalize_with_envelope(
                    signal, self.wlet_pars["window_size"], self.parentDV.dt
                )

            # compute the spectrum
            modulus, wlet = pyboat.compute_spectrum(signal, self.parentDV.dt, periods)
            global_modulus[:, start:end + 1] += modulus
            # get maximum ridge
            ridge = pyboat.get_maxRidge_ys(modulus)
            # generate time vector
            tvec = np.arange(len(signal)) * self.parentDV.dt
            # evaluate along the ridge
            ridge_data = pyboat.eval_ridge(
                ridge,
                wlet,
                signal,
                periods,
                tvec,
                power_thresh,
                smoothing_wsize=rsmooth,
            )

            # from ridge thresholding..
            if ridge_data.empty:
                EmptyRidge += 1
            else:
                ridge_results[signal_id] = ridge_data

            # time average the spectrum, all have shape len(periods)!
            averaged_Wspec = np.mean(modulus, axis=1)
            df_fouriers[signal_id] = averaged_Wspec

            # -- Save out individual results --
            settings = QSettings()
            float_format = settings.value("float_format", "%.3f")
            graphics_format = settings.value("graphics_format", "png")

            if self.cb_filtered_sigs.isChecked():

                signal_df = pd.DataFrame()
                signal_df["signal"] = signal
                signal_df.index = tvec
                signal_df.index.name = "time"

                fname = os.path.join(OutPath, f"{signal_id}_filtered.csv")
                if self.debug:
                    print(f"Saving filtered signal to {fname}")
                signal_df.to_csv(
                    fname, sep=",", float_format=float_format, index=True, header=True
                )

            if self.cb_specs.isChecked():

                # plot spectrum and ridge
                ax_sig, ax_spec = pl.mk_signal_modulus_ax(self.parentDV.time_unit)
                pl.plot_signal_modulus(
                    (ax_sig, ax_spec),
                    tvec,
                    signal,
                    modulus,
                    periods,
                    p_max=self.wlet_pars["pow_max"],
                )
                pl.draw_Wavelet_ridge(ax_spec, ridge_data)
                plt.tight_layout()
                fname = os.path.join(OutPath, f"{signal_id}_wspec.{graphics_format}")
                if self.debug:
                    print(f"Plotting and saving spectrum {signal_id} to {fname}")
                plt.savefig(fname, dpi=DPI)
                plt.close()

            if self.cb_specs_noridge.isChecked():

                # plot spectrum without ridge
                ax_sig, ax_spec = pl.mk_signal_modulus_ax(self.parentDV.time_unit)
                pl.plot_signal_modulus(
                    (ax_sig, ax_spec),
                    tvec,
                    signal,
                    modulus,
                    periods,
                    p_max=self.wlet_pars["pow_max"],
                )
                plt.tight_layout()
                fname = os.path.join(OutPath, f"{signal_id}_wspecNR.{graphics_format}")
                if self.debug:
                    print(f"Plotting and saving spectrum {signal_id} to {fname}")
                plt.savefig(fname, dpi=DPI)
                plt.close()

            if (
                self.cb_readout_plots.isChecked()
                and not ridge_data.empty
            ):

                pl.plot_readout(ridge_data)
                fname = os.path.join(OutPath, f"{signal_id}_readout.{graphics_format}")
                if self.debug:
                    print(f"Plotting and saving {signal_id} to {fname}")
                plt.savefig(fname, dpi=DPI)
                plt.close()

            if self.cb_readout.isChecked() and not ridge_data.empty:

                fname = os.path.join(OutPath, f"{signal_id}_readout.csv")
                if self.debug:
                    print(f"Saving ridge readout to {fname}")
                ridge_data.to_csv(
                    fname, sep=",", float_format=float_format, index=False
                )

            self.progress.setValue(i)

        if EmptyRidge > 0:

            msg = f"{EmptyRidge} ridge readouts entirely below threshold.."
            msgBox = QMessageBox()
            msgBox.setWindowTitle("Discarded Ridges")
            msgBox.setText(msg)
            msgBox.exec()

        # apply normalization
        global_modulus /= norm_vec
        # convert to DataFrame to attach periods
        global_modulus = pd.DataFrame(global_modulus, index=periods)
        return ridge_results, df_fouriers, global_modulus


class PowerHistogramWindow(QWidget):
    def __init__(self, powers, dataset_name, parent=None):

        super().__init__(parent=parent)

        # to spawn as extra window from parent
        self.setWindowFlags(Qt.Window)
        self.powers = powers
        self.initUI(dataset_name)

    def initUI(self, dataset_name):

        self.setWindowTitle(f"Ridge Power Histogram - {dataset_name}")
        self.setGeometry(410, 220, 550, 400)

        pCanvas = mkGenericCanvas()
        pCanvas.setParent(self)
        ntb = NavigationToolbar(pCanvas, self)

        # plot it
        pCanvas.fig.clf()
        pl.power_distribution(self.powers, fig=pCanvas.fig)
        pCanvas.fig.subplots_adjust(left=0.15, bottom=0.17)

        main_layout = QGridLayout()
        main_layout.addWidget(pCanvas, 0, 0, 9, 1)
        main_layout.addWidget(ntb, 10, 0, 1, 1)

        self.setLayout(main_layout)
        self.show()


class EnsembleDynamicsWindow(QWidget):
    def __init__(self, ensemble_results, dt, time_unit, dataset_name="", parent=None):

        super().__init__(parent=parent)

        # to spawn as extra window from parent
        self.setWindowFlags(Qt.Window)
        self.time_unit = time_unit
        self.dt = dt
        # period, amplitude and phase
        self.results = ensemble_results

        self.initUI(dataset_name)

    def initUI(self, dataset_name):

        self.setWindowTitle(f"Ensemble Dynamics - {dataset_name}")
        self.setGeometry(210, 80, 700, 480)

        Canvas = mkGenericCanvas()
        Canvas.setParent(self)
        ntb = NavigationToolbar(Canvas, self)

        Canvas.fig.clf()
        pl.ensemble_dynamics(
            *self.results, dt=self.dt, time_unit=self.time_unit, fig=Canvas.fig
        )
        Canvas.fig.subplots_adjust(
            wspace=0.3, left=0.1, top=0.98, right=0.95, bottom=0.15
        )
        main_layout = QGridLayout()
        main_layout.addWidget(Canvas, 0, 0, 9, 1)
        main_layout.addWidget(ntb, 10, 0, 1, 1)

        self.setLayout(main_layout)
        self.show()


class FourierDistributionWindow(QWidget):
    def __init__(self, df_fouriers, time_unit, dataset_name="", parent=None):

        super().__init__(parent=parent)

        # to spawn as extra window from parent
        self.setWindowFlags(Qt.Window)

        self.time_unit = time_unit
        # time averaged wavelet spectra + period index
        self.df_fouriers = df_fouriers

        self.initUI(dataset_name)

    def initUI(self, dataset_name):

        self.setWindowTitle(f"Fourier Power Median + Q1, Q3 - {dataset_name}")
        self.setGeometry(510, 230, 550, 400)

        Canvas = mkGenericCanvas()
        Canvas.setParent(self)
        ntb = NavigationToolbar(Canvas, self)

        Canvas.fig.clf()
        pl.Fourier_distribution(
            self.df_fouriers, time_unit=self.time_unit, fig=Canvas.fig
        )

        Canvas.fig.subplots_adjust(
            wspace=0.3, left=0.15, top=0.98, right=0.95, bottom=0.15
        )
        main_layout = QGridLayout()
        main_layout.addWidget(Canvas, 0, 0, 9, 1)
        main_layout.addWidget(ntb, 10, 0, 1, 1)

        self.setLayout(main_layout)
        self.show()


class GlobalSpectrumWindow(QWidget):
    def __init__(self, modulus, time_unit, dataset_name="", parent=None):

        super().__init__(parent=parent)

        # to spawn as extra window from parent
        self.setWindowFlags(Qt.Window)

        self.time_unit = time_unit

        # global Wavelet spectrum
        self.modulus = modulus
        self.tvec = np.arange(0, modulus.shape[1]) * parent.parentDV.dt
        self.pow_max = parent.wlet_pars["pow_max"]

        self.initUI(dataset_name)

    def initUI(self, dataset_name):

        self.setWindowTitle(f"Global Wavelet Spectrum - {dataset_name}")
        self.setGeometry(410, 360, 700, 500)

        Canvas = mkGenericCanvas()
        Canvas.setParent(self)
        ntb = NavigationToolbar(Canvas, self)
        Canvas.fig.clf()

        # creates the ax and attaches it to the widget figure
        ax = pl.mk_modulus_ax(time_unit=self.time_unit, fig=Canvas.fig)
        pl.plot_modulus(ax,
                        self.tvec,
                        self.modulus.to_numpy(),
                        periods=self.modulus.index,
                        p_max=self.pow_max)

        Canvas.fig.subplots_adjust(
            wspace=0.2, left=0.15, top=0.98, right=0.95, bottom=0.1
        )
        main_layout = QGridLayout()
        main_layout.addWidget(Canvas, 0, 0, 9, 1)
        main_layout.addWidget(ntb, 10, 0, 1, 1)

        self.setLayout(main_layout)
        self.show()
