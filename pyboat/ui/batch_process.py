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
    QHBoxLayout,
    QGroupBox,
    QGridLayout,
    QProgressBar,
    QSpacerItem,
    QFrame,
    QMainWindow,
)
from PyQt5.QtGui import QIntValidator
from PyQt5.QtCore import QSettings
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from pyboat.ui.util import posfloatV, mkGenericCanvas

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

    def __init__(self, parent, DEBUG):

        super().__init__()

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

        # -- Ridge Analysis Options --

        ridge_options = QGroupBox("Ridge Detection")

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
            """Savitkzy-Golay window size for smoothing the ridge,
            leave blank for no smoothing"""
        )
        self.smooth_edit = smooth_edit

        ridge_options_layout = QGridLayout()
        ridge_options_layout.addWidget(thresh_label, 0, 0)
        ridge_options_layout.addWidget(thresh_edit, 0, 1)
        ridge_options_layout.addWidget(smooth_label, 1, 0)
        ridge_options_layout.addWidget(smooth_edit, 1, 1)
        ridge_options.setLayout(ridge_options_layout)

        # -- Plotting Options --

        plotting_options = QGroupBox("Summary Statistics")
        self.cb_power_dis = QCheckBox("Ridge Power Distribution")
        self.cb_power_dis.setStatusTip(
            "Show time-averaged distribution of ridge powers"
        )
        self.cb_plot_ens_dynamics = QCheckBox("Ensemble Dynamics")
        self.cb_plot_ens_dynamics.setStatusTip(
            "Show period, amplitude and phase distribution over time"
        )
        self.cb_plot_Fourier_dis = QCheckBox("Fourier Spectra Distribution")
        self.cb_plot_Fourier_dis.setStatusTip(
            "Ensemble power distribution of the time averaged Wavelet spectra"
        )

        lo = QGridLayout()
        lo.addWidget(self.cb_plot_ens_dynamics, 0, 0)
        lo.addWidget(self.cb_plot_Fourier_dis, 1, 0)
        lo.addWidget(self.cb_power_dis, 2, 0)
        plotting_options.setLayout(lo)

        # -- Save Out Results --

        export_options = QGroupBox("Export Results")
        export_options.setStatusTip("Creates various figures and csv's")
        export_options.setCheckable(True)
        export_options.setChecked(False)

        self.cb_filtered_sigs = QCheckBox("Filtered Signals")
        self.cb_filtered_sigs.setStatusTip(
            "Saves detrended and amplitude normalized signals to disc as csv's"
        )
        
        self.cb_specs = QCheckBox("Wavelet Spectra")
        self.cb_specs.setStatusTip(
            "Saves the individual wavelet spectra as images"
        )

        self.cb_specs_noridge = QCheckBox("Wavelet Spectra w/o ridges")
        self.cb_specs_noridge.setStatusTip(
            "Saves the individual wavelet spectra without the ridges as images"
        )

        self.cb_readout = QCheckBox("Ridge Readouts")
        self.cb_readout.setStatusTip("Saves one analysis result per signal to disc as csv")

        self.cb_readout_plots = QCheckBox("Ridge Readout Plots")
        self.cb_readout_plots.setStatusTip(
            "Saves the individual readout plots to disc"
        )
        self.cb_sorted_powers = QCheckBox("Sorted Average Powers")
        self.cb_sorted_powers.setStatusTip(
            "Saves the time-averaged ridge powers in descending order"
        )
        self.cb_save_ensemble_dynamics = QCheckBox("Ensemble Dynamics")
        self.cb_save_ensemble_dynamics.setStatusTip(
            "Separately saves period, amplitude, power and phase summary statistics to a csv file"
        )

        self.cb_save_Fourier_dis = QCheckBox("Fourier Distribution")
        self.cb_save_Fourier_dis.setStatusTip(
            "Saves median and quartiles of the ensemble Fourier power spectral distribution"
        )

        # defaults to HOME
        self.OutPath_edit = QLineEdit(expanduser("~"))

        PathButton = QPushButton("Select Path..")
        PathButton.setMaximumWidth(100)
        PathButton.clicked.connect(self.select_export_dir)

        line1 = QFrame()
        line1.setFrameShape(QFrame.HLine)
        line1.setFrameShadow(QFrame.Sunken)

        line2 = QFrame()
        line2.setFrameShape(QFrame.HLine)
        line2.setFrameShadow(QFrame.Sunken)

        lo = QGridLayout()
        lo.setSpacing(1.5)

        lo.addWidget(self.cb_filtered_sigs, 0, 0)
        
        lo.addWidget(self.cb_specs, 1, 0)
        lo.addWidget(self.cb_specs_noridge, 2, 0)

        lo.addWidget(self.cb_readout, 3, 0)
        lo.addWidget(self.cb_readout_plots, 4, 0)
        # lo.addWidget(line1, 3,0)
        lo.addWidget(self.cb_sorted_powers, 5, 0)
        lo.addWidget(self.cb_save_ensemble_dynamics, 6, 0)
        lo.addWidget(self.cb_save_Fourier_dis, 7, 0)
        # lo.addWidget(line2, 6,0)
        lo.addWidget(PathButton, 8, 0)
        lo.addWidget(self.OutPath_edit, 9, 0)
        export_options.setLayout(lo)
        self.export_options = export_options

        # -- Progress and Run --
        Nsignals = self.parentDV.df.shape[1]

        RunButton = QPushButton(f"Run for {Nsignals} Signals!", self)
        RunButton.setStyleSheet("background-color: orange")
        RunButton.clicked.connect(self.run_batch)
        # RunButton.setMaximumWidth(60)

        # the progress bar
        self.progress = QProgressBar(self)
        self.progress.setRange(0, Nsignals - 1)
        # self.progress.setGeometry(0,0, 300, 20)
        self.progress.setMinimumWidth(200)

        # nsig_label = QLabel(f'{Nsignals} Signals')

        process_box = QGroupBox("Processing")
        process_box.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        lo = QHBoxLayout()
        lo.addWidget(self.progress)
        lo.addItem(QSpacerItem(30, 2))
        # lo.addStretch(0)
        lo.addWidget(RunButton)
        lo.addStretch(0)
        process_box.setLayout(lo)

        # -- main layout --

        main_layout.addWidget(plotting_options, 0, 0, 1, 1)
        main_layout.addWidget(ridge_options, 1, 0, 1, 1)
        main_layout.addWidget(export_options, 0, 1, 2, 1)
        main_layout.addWidget(process_box, 2, 0, 1, 2)

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

        if self.export_options.isChecked():
            OutPath = self.get_OutPath()
            if OutPath is None:
                return

        # TODO: parallelize
        ridge_results, df_fouriers = self.do_the_loop()

        # check for empty ridge_results
        if not ridge_results:
            
            msgBox = QMessageBox()
            msgBox.setWindowTitle("No Results")
            msgBox.setText("All ridges below threshold.. no results!")
            msgBox.exec()

            return

        settings = QSettings()
        float_format = settings.value('float_format', '%.3f')

        # --- compute the time-averaged powers ---

        if self.cb_power_dis.isChecked() or self.cb_sorted_powers.isChecked():

            powers_series = em.average_power_distribution(
                ridge_results.values(), ridge_results.keys(), exclude_coi=True
            )

        if self.cb_power_dis.isChecked():
            # plot the distribution
            self.pdw = PowerDistributionWindow(powers_series, dataset_name=dataset_name)

        # save out the sorted average powers
        if self.export_options.isChecked() and self.cb_sorted_powers.isChecked():
            fname = os.path.join(OutPath, f"{dataset_name}_ridge-powers.csv")
            powers_series.to_csv(fname, sep=",", float_format=float_format,
                                 index=True, header=False)

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
            )

        if (
            self.export_options.isChecked()
            and self.cb_save_ensemble_dynamics.isChecked()
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
                df_fouriers, self.parentDV.time_unit, dataset_name
            )

        if self.export_options.isChecked() and self.cb_save_Fourier_dis.isChecked():

            fname = os.path.join(OutPath, f"{dataset_name}_fourier-distribution.csv")

            # save out median and quartiles of Fourier powers
            df_fdis = pd.DataFrame(index=df_fouriers.index)
            df_fdis["Median"] = df_fouriers.median(axis=1)
            df_fdis["Mean"] = df_fouriers.mean(axis=1)
            df_fdis["Q1"] = df_fouriers.quantile(q=0.25, axis=1)
            df_fdis["Q3"] = df_fouriers.quantile(q=0.75, axis=1)

            df_fdis.to_csv(fname, sep=",", float_format=float_format)

        if self.debug:
            print(list(ridge_results.items())[:2])

        Nsignals = len(self.parentDV.df.columns)
        msg = f"Processed {Nsignals} signals!"
        msgBox = QMessageBox()
        msgBox.setWindowTitle("Batch processing done")
        msgBox.setText(msg)
        msgBox.exec()
            
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

        dir_name = dialog.getExistingDirectory(
            self, "Select a folder to save the results", expanduser("~")
        )

        # dialog cancelled
        if not dir_name:
            return

        if self.debug:
            print("Batch output name:", dir_name)

        self.OutPath_edit.setText(dir_name)

    def do_the_loop(self):

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

        if self.export_options.isChecked():
            OutPath = self.get_OutPath()
            if OutPath is None:
                return

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

        for i, signal_id in enumerate(self.parentDV.df):

            # log to terminal
            print(f"processing {signal_id}..")

            # sets parentDV.raw_signal and parentDV.tvec
            succ = self.parentDV.vector_prep(signal_id)
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
            float_format = settings.value('float_format', '%.3f')
            graphics_format = settings.value('graphics_format', 'png')
            
            exbox_checked = self.export_options.isChecked()
            
            if exbox_checked and self.cb_filtered_sigs.isChecked():

                signal_df = pd.DataFrame()
                signal_df['signal'] = signal
                signal_df.index = tvec
                signal_df.index.name = 'time'
                                
                fname = os.path.join(OutPath, f"{signal_id}_filtered.csv")
                if self.debug:
                    print(f"Saving filtered signal to {fname}")
                signal_df.to_csv(fname, sep=",", float_format=float_format,
                                 index=True, header=True)
                         
            if exbox_checked and self.cb_specs.isChecked():

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

            if exbox_checked and self.cb_specs_noridge.isChecked():

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

            if exbox_checked and self.cb_readout_plots.isChecked() and not ridge_data.empty:

                pl.plot_readout(ridge_data)
                fname = os.path.join(OutPath, f"{signal_id}_readout.{graphics_format}")
                if self.debug:
                    print(f"Plotting and saving {signal_id} to {fname}")
                plt.savefig(fname, dpi=DPI)
                plt.close()

            if exbox_checked and self.cb_readout.isChecked() and not ridge_data.empty:

                fname = os.path.join(OutPath, f"{signal_id}_readout.csv")
                if self.debug:
                    print(f"Saving ridge readout to {fname}")
                ridge_data.to_csv(fname, sep=",", float_format=float_format,
                                  index=False)

            self.progress.setValue(i)

        if EmptyRidge > 0:

            msg =  f"{EmptyRidge} ridge readouts entirely below threshold.."
            msgBox = QMessageBox()
            msgBox.setWindowTitle("Discarded Ridges")
            msgBox.setText(msg)
            msgBox.exec()

        return ridge_results, df_fouriers


class PowerDistributionWindow(QWidget):
    def __init__(self, powers, dataset_name, parent=None):
        super().__init__()

        # --- calculate average powers ------------------
        self.powers = powers
        # -------------------------------------------------

        self.initUI(dataset_name)

    def initUI(self, dataset_name):

        self.setWindowTitle(f"Ridge Power Distribution - {dataset_name}")
        self.setGeometry(410, 220, 550, 400)

        main_frame = QWidget()
        pCanvas = mkGenericCanvas()
        pCanvas.setParent(main_frame)
        ntb = NavigationToolbar(pCanvas, main_frame)

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
    def __init__(self, ensemble_results, dt, time_unit, dataset_name=""):
        super().__init__()

        self.time_unit = time_unit
        self.dt = dt
        # period, amplitude and phase
        self.results = ensemble_results

        self.initUI(dataset_name)

    def initUI(self, dataset_name):

        self.setWindowTitle(f"Ensemble Dynamics - {dataset_name}")
        self.setGeometry(210, 80, 700, 480)

        main_frame = QWidget()
        Canvas = mkGenericCanvas()
        Canvas.setParent(main_frame)
        ntb = NavigationToolbar(Canvas, main_frame)

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
    def __init__(self, df_fouriers, time_unit, dataset_name=""):
        super().__init__()

        self.time_unit = time_unit
        self.df_fouriers = df_fouriers

        self.initUI(dataset_name)

    def initUI(self, dataset_name):

        self.setWindowTitle(f"Fourier Power Distribution - {dataset_name}")
        self.setGeometry(510, 330, 550, 400)

        main_frame = QWidget()
        Canvas = mkGenericCanvas()
        Canvas.setParent(main_frame)
        ntb = NavigationToolbar(Canvas, main_frame)

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
