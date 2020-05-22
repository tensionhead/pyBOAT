import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os.path import expanduser

from PyQt5.QtWidgets import QCheckBox, QComboBox, QFileDialog, QAction, QLabel, QLineEdit, QPushButton, QMessageBox, QSizePolicy, QWidget, QVBoxLayout, QHBoxLayout, QDialog, QGroupBox, QGridLayout, QProgressBar, QSpacerItem, QFrame
from PyQt5.QtGui import QDoubleValidator, QIntValidator, QScreen
from PyQt5.QtCore import Qt, pyqtSignal

from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from pyboat.ui.util import MessageWindow, posfloatV, posintV

import pyboat
from pyboat import plotting as pl
from pyboat import ensemble_measures as em

class BatchProcessWindow(QWidget):

    '''
    The parent is a DataViewer instance holding the
    data as a DataFrame, and other global properties:
    
    parent.df
    parent.dt
    parent.time_unit

    '''

    def __init__(self, parent, DEBUG):
        
        super().__init__()

        # the DataViewer spawning *this* Widget
        self.parentDV = parent
        self.debug = DEBUG

    def initUI(self, wlet_pars):

        '''
        Gets called from the parent DataViewer
        '''
        
        self.setWindowTitle('Batch Processing')
        self.setGeometry(310,330,600,200)

        # from the DataViewer
        self.wlet_pars = wlet_pars
        
        main_layout = QGridLayout()

        # -- Ridge Analysis Options --
        
        ridge_options = QGroupBox('Ridge Extraction Options')        

        thresh_label = QLabel("Ridge Threshold:")
        thresh_edit = QLineEdit()
        thresh_edit.setValidator(posfloatV)
        thresh_edit.insert('0')
        thresh_edit.setMaximumWidth(60)
        thresh_edit.setToolTip('Ridge points below that power value will be filtered out ')
        self.thresh_edit = thresh_edit
        
        
        smooth_label = QLabel("Ridge Smoothing:")        
        smooth_edit = QLineEdit()
        smooth_edit.setMaximumWidth(60)        
        smooth_edit.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)        
        smooth_edit.setValidator( QIntValidator(bottom = 3, top = 99999999) )
        tt = 'Savitkzy-Golay window size for smoothing the ridge,\nleave blank for no smoothing'
        smooth_edit.setToolTip(tt)
        self.smooth_edit = smooth_edit

        ridge_options_layout = QGridLayout()
        ridge_options_layout.addWidget(thresh_label, 0,0)
        ridge_options_layout.addWidget(thresh_edit, 0,1)
        ridge_options_layout.addWidget(smooth_label, 1,0)
        ridge_options_layout.addWidget(smooth_edit, 1,1)
        ridge_options.setLayout(ridge_options_layout)

        # -- Plotting Options --

        plotting_options = QGroupBox('Summary Statistics')
        self.cb_power_dis = QCheckBox('Ensemble Power Distribution')
        self.cb_power_dis.setToolTip('Show time-averaged wavelet power of the ensemble')
        self.cb_plot_ens_dynamics = QCheckBox('Ensemble Dynamics')
        self.cb_plot_ens_dynamics.setToolTip('Show period, amplitude and phase distribution over time')
        lo = QGridLayout()
        lo.addWidget(self.cb_power_dis,0,0)
        lo.addWidget(self.cb_plot_ens_dynamics,1,0)
        plotting_options.setLayout(lo)

        # -- Save Out Results --
        
        export_options = QGroupBox('Export Results')        
        export_options.setToolTip('Saves also the summary statistics..')
        export_options.setCheckable(True)
        export_options.setChecked(False)        
        self.cb_specs = QCheckBox('Wavelet Spectra')
        self.cb_specs.setToolTip("Saves the individual wavelet spectra as images (png's)")

        self.cb_readout = QCheckBox('Ridge Readouts')
        self.cb_readout.setToolTip('Saves one data frame per signal to disc as csv')

        self.cb_readout_plots = QCheckBox('Ridge Readout Plots')
        self.cb_readout_plots.setToolTip("Saves the individual readout plots to disc as png's")
        self.cb_sorted_powers = QCheckBox('Sorted Average Powers')
        self.cb_sorted_powers.setToolTip("Saves the time-averaged powers in descending order")
        self.cb_save_ensemble_dynamics = QCheckBox('Ensemble Dynamics')
        self.cb_save_ensemble_dynamics.setToolTip("Saves each period, amplitude and phase summary statistics to a csv table")
        
        home = expanduser("~")
        OutPath_label = QLabel('Export to:')
        self.OutPath_edit = QLineEdit(home)
        if self.debug:
            self.OutPath_edit.setText(home + '/Desktop/wres')

        PathButton = QPushButton('Select Path..')
        PathButton.setMaximumWidth(100)        
        PathButton.clicked.connect(self.select_export_dir)

        line1 = QFrame()
        line1.setFrameShape(QFrame.HLine)
        line1.setFrameShadow(QFrame.Sunken);

        line2 = QFrame()
        line2.setFrameShape(QFrame.HLine)
        line2.setFrameShadow(QFrame.Sunken);
        
        lo = QGridLayout()
        lo.setSpacing(1.5)
        lo.addWidget(self.cb_specs,0,0)
        lo.addWidget(self.cb_readout,1,0)
        lo.addWidget(self.cb_readout_plots,2,0)
        #lo.addWidget(line1, 3,0)        
        lo.addWidget(self.cb_sorted_powers,4,0)
        lo.addWidget(self.cb_save_ensemble_dynamics,5,0)
        #lo.addWidget(line2, 6,0)        
        lo.addWidget(PathButton,7,0)                
        lo.addWidget(self.OutPath_edit,8,0)
        export_options.setLayout(lo)
        self.export_options = export_options       
        
        # -- Progress and Run --
        Nsignals = self.parentDV.df.shape[1]
        
        RunButton = QPushButton(f"Run for {Nsignals} Signals!", self)
        RunButton.setStyleSheet("background-color: orange")        
        RunButton.clicked.connect(self.run_batch)
        #RunButton.setMaximumWidth(60)
        
        # the progress bar
        self.progress = QProgressBar(self)
        self.progress.setRange(0, Nsignals-1)
        #self.progress.setGeometry(0,0, 300, 20)
        self.progress.setMinimumWidth(200)
        

        # nsig_label = QLabel(f'{Nsignals} Signals')
        
        process_box = QGroupBox('Processing')
        process_box.setSizePolicy(QSizePolicy.Maximum,QSizePolicy.Maximum)        
        lo = QHBoxLayout()
        lo.addWidget(self.progress)
        lo.addItem(QSpacerItem(30, 2))
        # lo.addStretch(0)
        lo.addWidget(RunButton)
        lo.addStretch(0)        
        process_box.setLayout(lo)

        # -- main layout --
        
        main_layout.addWidget(ridge_options, 0, 0, 1, 1)
        main_layout.addWidget(plotting_options, 1, 0, 1, 1)
        main_layout.addWidget(export_options, 0, 1, 2, 1)
        main_layout.addWidget(process_box, 2, 0, 1, 2)
                
        # set main layout
        self.setLayout(main_layout)
        self.show()


    def run_batch(self):

        '''
        Retrieve all batch settings and loop over the signals
        present in the parentDV

        '''

        if self.export_options.isChecked():
            OutPath = self.get_OutPath()
            if OutPath is None:
                return

        # is a dictionary holding the ridge-data
        # for each signal and the signal_id as key
        ridge_results = self.do_the_loop()

        # compute the time-averaged powers
        if self.cb_power_dis.isChecked() or self.cb_sorted_powers.isChecked():
            
            powers = em.average_power_distribution(ridge_results.values())
            powers_series = pd.Series(index = ridge_results.keys(),
                                  data = powers)
            # sort by power, descending
            powers_series.sort_values(
                ascending = False,
                inplace = True)

        if self.cb_power_dis.isChecked():            
            # plot the distribution
            self.pdw = PowerDistributionWindow(powers_series)

        # save out the sorted average powers
        if self.cb_sorted_powers.isChecked():
            df_name = self.parentDV.df.name
            fname = f'{OutPath}/average_powers_{df_name}.csv'
            powers_series.to_csv(fname, sep = ',', index = True, header = False)
        # compute summary statistics over time
        if self.cb_plot_ens_dynamics.isChecked() or self.cb_save_ensemble_dynamics.isChecked():
            # res is a tuple of  DataFrames, one each for
            # periods, amplitude and phase
            res = em.get_ensemble_dynamics(ridge_results.values())

        if self.cb_plot_ens_dynamics.isChecked():            
            self.edw = EnsembleDynamicsWindow(res,
                                      dt = self.parentDV.dt,
                                      time_unit = self.parentDV.time_unit)
            
        if self.cb_save_ensemble_dynamics.isChecked():
            # create time axis, all DataFrames have same number of rows
            tvec = np.arange(res[0].shape[0]) * self.parentDV.dt
            dataset_name = self.parentDV.df.name
            for obs, df in zip(['periods', 'amplitudes', 'phasesR'], res):
                fname = f'{OutPath}/{obs}_{dataset_name}.csv'
                df.index = tvec
                df.index.name = 'time'
                df.to_csv(fname, sep = ',', float_format = '%.3f')

        if self.debug:
            print(list(ridge_results.items())[:2])


    def get_thresh(self):

        '''
        Reads the self.thresh_edit 
        A Validator is set..
        '''

        thresh_str = self.thresh_edit.text().replace(",", ".")        
        try:
            thresh  = float(thresh_str)
            if self.debug:
                print("thresh set to:", thresh)
            return thresh
        
        # empty line edit is interpreted as no thresholding required
        except ValueError:            
            if self.debug:
                print("thresh ValueError", thresh_str)
            return 0

    def get_ridge_smooth(self):

        
        '''
        Reads the self.smooth_edit 
        A Validator is set..
        '''

        rs = self.smooth_edit.text()
        try:
            rsmooth  = int(rs)
            if self.debug:
                print("rsmooth set to:", rs)

            # make an odd window length
            if rsmooth == 0:
                return None
            elif rsmooth < 5:
                return 5
            elif rsmooth > 5 and rsmooth%2 == 0:
                return rsmooth + 1
            else:                                
                return rsmooth
        
        # empty line edit is interpreted as no ridge smoothing required
        except ValueError:            
            if self.debug:
                print("No rsmooth", rs)
            return None
        
    def get_OutPath(self):

        '''
        Reads the self.OutPath_edit 
        There is no validator but an os.path
        check is done!
        '''

        path = self.OutPath_edit.text()

        if not os.path.isdir(path):
            self.e = MessageWindow("Specified path is not a valid directory..",
                           "Invalid export path")
            return None
        return path
                            
    def select_export_dir(self):

        dialog = QFileDialog()
        dialog.setFileMode(QFileDialog.DirectoryOnly)
        dialog.setOption(QFileDialog.ShowDirsOnly, False)

        dir_name = dialog.getExistingDirectory(
            self, "Select a folder to save the results", os.getenv("HOME")
        )

        # dialog cancelled
        if not dir_name:
            return

        if self.debug:
            print("Batch output name:", dir_name)

        self.OutPath_edit.setText(dir_name)

    def do_the_loop(self):
        
        '''
        Uses the explicitly parsed self.wlet_pars 
        to control signal analysis settings.

        Takes general analysis Parameters

        self.parentDV.dt
        self.parentDV.time_unit

        and the DataFrame

        self.parentDV.df

        from the parent DataViewer.
        Reads additional settings from this Batch Process Window.

        '''

        if self.export_options.isChecked():
            OutPath = self.get_OutPath()
            if OutPath is None:
                return

        periods = np.linspace(
            self.wlet_pars['T_min'],
            self.wlet_pars['T_max'],
            self.wlet_pars['step_num'])

        # retrieve batch settings
        power_thresh = self.get_thresh()
        rsmooth = self.get_ridge_smooth()
        
        ridge_results = {}
        for i, signal_id in enumerate(self.parentDV.df):

            # log to terminal
            print(f"processing {signal_id}..")

            # sets parentDV.raw_signal and parentDV.tvec 
            succ = self.parentDV.vector_prep(signal_id)
            # ui silently passes over..
            if not succ:
                print(f"Can't process signal {signal_id}..")
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
                    print('Calculating envelope with L=',self.wlet_pars['L'])
                signal = pyboat.normalize_with_envelope(signal, self.wlet_pars['L'])
                
            # compute the spectrum
            modulus, wlet = pyboat.compute_spectrum(signal, self.parentDV.dt, periods )
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
                smoothing_wsize = rsmooth)
            ridge_results[signal_id] = (ridge_data)

            # -- Save out individual results --
            
            if self.cb_specs.isChecked():
                # plot spectrum and ridge
                ax_sig, ax_spec = pl.mk_signal_modulus_ax(self.parentDV.time_unit)
                pl.plot_signal_modulus((ax_sig, ax_spec),
                                       tvec, signal,
                                       modulus,
                                       periods,
                                       p_max = self.wlet_pars['p_max'])
                pl.draw_Wavelet_ridge(ax_spec, ridge_data)
                plt.tight_layout()
                fname = f'{OutPath}/{signal_id}_wspec.png'
                if self.debug:
                    print(f'Plotting and saving {signal_id} to {fname}')                
                plt.savefig(fname)
                plt.close()

            if self.cb_readout_plots.isChecked():
                pl.plot_readout(ridge_data)
                fname = f'{OutPath}/{signal_id}_readout.png'
                if self.debug:
                    print(f'Plotting and saving {signal_id} to {fname}')                
                plt.savefig(fname)
                plt.close()
                
            if self.cb_readout.isChecked():
                fname = f'{OutPath}/{signal_id}_readout.csv'
                if self.debug:
                    print(f'Saving ridge reatout to {fname}')
                ridge_data.to_csv(fname, sep = ',', float_format = '%.3f', index = False)                

            self.progress.setValue(i)
            
        return ridge_results


class PowerDistributionWindow(QWidget):
    def __init__(self, powers, parent = None):
        super().__init__()

                
        # --- calculate average powers ------------------
        self.powers = powers
        # -------------------------------------------------                
        
        self.initUI()

    def initUI(self):

        self.setWindowTitle('Ridge Power Distribution ')
        self.setGeometry(510,80,550,400)

        main_frame = QWidget()
        pCanvas = mkGenericCanvas()        
        pCanvas.setParent(main_frame)
        ntb = NavigationToolbar(pCanvas, main_frame)
        
        # plot it
        pCanvas.fig.clf()
        pl.plot_power_distribution(self.powers, fig = pCanvas.fig)
        pCanvas.fig.subplots_adjust(left = 0.15, bottom = 0.17)
        
        main_layout = QGridLayout()
        main_layout.addWidget(pCanvas,0,0,9,1)
        main_layout.addWidget(ntb,10,0,1,1)
        
        self.setLayout(main_layout)
        self.show()

class EnsembleDynamicsWindow(QWidget):
    def __init__(self, ensemble_results, dt, time_unit):
        super().__init__()

        self.time_unit = time_unit
        self.dt = dt
        # period, amplitude and phase
        self.results = ensemble_results
        
        self.initUI()

    def initUI(self):

        self.setWindowTitle('Ensemble Dynamics')
        self.setGeometry(510,80,480,700)

        main_frame = QWidget()
        Canvas = mkGenericCanvas()        
        Canvas.setParent(main_frame)
        ntb = NavigationToolbar(Canvas, main_frame)        

        Canvas.fig.clf()
        pl.plot_ensemble_dynamics(*self.results,
                                  dt = self.dt,
                                  time_unit = self.time_unit,
                                  fig = Canvas.fig)
        Canvas.fig.subplots_adjust(left = 0.2, bottom = 0.1)
        main_layout = QGridLayout()
        main_layout.addWidget(Canvas,0,0,9,1)
        main_layout.addWidget(ntb,10,0,1,1)
        
        self.setLayout(main_layout)
        self.show()


class mkGenericCanvas(FigureCanvas):
    def __init__(self):
        self.fig, self.ax = plt.subplots(1,1)

        FigureCanvas.__init__(self, self.fig)

        FigureCanvas.setSizePolicy(self,
                QSizePolicy.Expanding,
                QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        

        
