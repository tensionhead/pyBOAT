import os
import numpy as np
import matplotlib.pyplot as plt

from PyQt5.QtWidgets import (
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
    QMessageBox
)

from PyQt5.QtGui import QIntValidator
from PyQt5.QtCore import pyqtSignal, QSettings

from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from pyboat import core
from pyboat import plotting as pl
from pyboat.ui.util import posfloatV, mkGenericCanvas


class mkTimeSeriesCanvas(FigureCanvas):

    # dpi != 100 looks wierd?!
    def __init__(self, parent=None, width=4, height=3, dpi=100):

        self.fig1 = Figure(figsize=(width, height), dpi=dpi)

        FigureCanvas.__init__(self, self.fig1)
        self.setParent(parent)

        # print ('Time Series Size', FigureCanvas.sizeHint(self))
        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)


class FourierAnalyzer(QWidget):
    def __init__(self, signal, dt, signal_id, position, time_unit, show_T, parent=None):
        super().__init__()

        self.time_unit = time_unit
        self.show_T = show_T

        # --- calculate Fourier spectrum ------------------
        self.fft_freqs, self.fpower = core.compute_fourier(signal, dt)
        # -------------------------------------------------

        self.initUI(position, signal_id)

    def initUI(self, position, signal_id):

        self.setWindowTitle("Fourier spectrum " + signal_id)
        self.setGeometry(510 + position, 80 + position, 520, 600)
        
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

        self.setLayout(main_layout)
        self.show()


class mkFourierCanvas(FigureCanvas):
    def __init__(self):
        self.fig, self.ax = plt.subplots(1, 1)

        FigureCanvas.__init__(self, self.fig)

        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)


class WaveletAnalyzer(QMainWindow):
    def __init__(
        self,
        signal,
        dt,
        Tmin,
        Tmax,
        position,
        signal_id,
        step_num,
        pow_max,
        time_unit,
        DEBUG=False,
    ):

        super().__init__()

        self.DEBUG = DEBUG

        self.signal_id = signal_id
        self.signal = signal
        self.pow_max = pow_max
        self.time_unit = time_unit

        self.periods = np.linspace(Tmin, Tmax, step_num)

        # generate time vector
        self.tvec = np.arange(0, len(signal)) * dt

        # no ridge yet
        self.ridge = None
        self.ridge_data = None
        self.power_thresh = None
        self.rsmoothing = None
        self._has_ridge = False  # no plotted ridge

        # no anneal parameters yet
        self.anneal_pars = None

        # =============Compute Spectrum========================================
        self.modulus, self.wlet = core.compute_spectrum(
            self.signal,
            dt,
            self.periods)
        # =====================================================================

        # Wavelet ridge-readout results
        self.ResultWindows = {}
        self.w_offset = 0

        self.initUI(position)

    def initUI(self, position):
        self.setWindowTitle("Wavelet Spectrum - " + str(self.signal_id))
        self.setGeometry(510 + position, 80 + position, 620, 750)

        main_widget = QWidget()
        self.statusBar()        
        
        # Wavelet and signal plot
        self.wCanvas = mkWaveletCanvas()
        main_frame = QWidget()
        self.wCanvas.setParent(main_frame)
        ntb = NavigationToolbar(self.wCanvas, main_frame)  # full toolbar

        # -------------plot the wavelet power spectrum---------------------------

        # creates the ax and attaches it to the widget figure
        axs = pl.mk_signal_modulus_ax(self.time_unit, fig=self.wCanvas.fig)

        pl.plot_signal_modulus(
            axs,
            time_vector=self.tvec,
            signal=self.signal,
            modulus=self.modulus,
            periods=self.periods,
            p_max=self.pow_max,
        )

        self.wCanvas.fig.subplots_adjust(bottom=0.11, right=0.95, left=0.13, top=0.95)
        self.wCanvas.fig.tight_layout()

        # --- Spectrum plotting options ---

        spectrum_opt_box = QGroupBox("Spectrum Plotting Options")
        spectrum_opt_box.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)

        spectrum_opt_layout = QHBoxLayout()
        spectrum_opt_layout.setSpacing(10)
        spectrum_opt_box.setLayout(spectrum_opt_layout)

        # uppler limit of the colormap <-> imshow(...,vmax = pmax)
        pmax_label = QLabel("Maximal Power:")
        self.pmax_edit = QLineEdit()
        self.pmax_edit.setStatusTip("Sets upper power limit of the spectrum")

        self.pmax_edit.setMaximumWidth(80)
        self.pmax_edit.setValidator(posfloatV)

        # retrieve initial power value, axs[1] is the spectrum
        pmin, pmax = axs[1].images[0].get_clim()
        self.pmax_edit.insert(f"{pmax:.0f}")

        RePlotButton = QPushButton("Update Plot", self)
        RePlotButton.setStatusTip("Rescales the color map of the spectrum with the new max value")
        RePlotButton.clicked.connect(self.update_plot)

        self.cb_coi = QCheckBox("COI", self)
        self.cb_coi.setStatusTip("Draws the cone of influence onto the spectrum")
        self.cb_coi.stateChanged.connect(self.draw_coi)

        # ridge_opt_layout.addWidget(drawRidgeButton,1,3) # not needed anymore?!
        spectrum_opt_layout.addWidget(pmax_label)
        spectrum_opt_layout.addWidget(self.pmax_edit)
        spectrum_opt_layout.addStretch(0)
        spectrum_opt_layout.addWidget(RePlotButton)
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
        ridge_opt_box.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)

        ridge_opt_layout = QGridLayout()
        ridge_opt_box.setLayout(ridge_opt_layout)

        # Start ridge detection
        maxRidgeButton = QPushButton("Detect Maximum Ridge", self)
        maxRidgeButton.setStatusTip("Finds the time-consecutive power maxima")
        
        maxRidgeButton.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        maxRidgeButton.clicked.connect(self.do_maxRidge_detection)

        # remove annealing, too slow.. not well implemented
        # annealRidgeButton = QPushButton('Set up ridge\nfrom annealing', self)
        # annealRidgeButton.clicked.connect(self.set_up_anneal)

        power_label = QLabel("Ridge Threshold:")
        power_label.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)

        power_thresh_edit = QLineEdit()
        power_thresh_edit.setStatusTip(
            "Sets the minimal power value required to be considered part of the ridge"
        )
        power_thresh_edit.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        power_thresh_edit.setMinimumSize(60, 0)
        power_thresh_edit.setValidator(posfloatV)

        smooth_label = QLabel("Ridge Smoothing:")
        ridge_smooth_edit = QLineEdit()
        ridge_smooth_edit.setStatusTip("Savitzky-Golay smoothing (k=3) of the ridge in time")
        ridge_smooth_edit.setMinimumSize(60, 0)
        ridge_smooth_edit.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        ridge_smooth_edit.setValidator(QIntValidator(bottom=3, top=len(self.signal)))

        # Plot Results
        plotResultsButton = QPushButton("Plot Ridge Readout", self)
        plotResultsButton.setStatusTip("Shows instantaneous period, phase, power and amplitude along the ridge")
        # plotResultsButton.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        plotResultsButton.clicked.connect(self.ini_plot_readout)

        ridge_opt_layout.addWidget(maxRidgeButton, 0, 0, 1, 1)
        ridge_opt_layout.addWidget(plotResultsButton, 1, 0, 1, 1)
        # ridge_opt_layout.addWidget(annealRidgeButton,1,0)

        ridge_opt_layout.addWidget(power_label, 0, 1)
        ridge_opt_layout.addWidget(power_thresh_edit, 0, 2)

        ridge_opt_layout.addWidget(smooth_label, 1, 1)
        ridge_opt_layout.addWidget(ridge_smooth_edit, 1, 2)

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

        # initialize line edits

        power_thresh_edit.textChanged[str].connect(self.qset_power_thresh)
        power_thresh_edit.insert("0.0")  # initialize with 0

        ridge_smooth_edit.textChanged[str].connect(self.qset_ridge_smooth)
        ridge_smooth_edit.insert("0")  # initialize with 0

        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)        
        self.show()

    def qset_power_thresh(self, text):

        # catch empty line edit
        if not text:
            return
        text = text.replace(",", ".")

        try:
            power_thresh = float(text)
            self.power_thresh = power_thresh
        # no parsable input
        except ValueError:
            return

        if self.DEBUG:
            print("power thresh set to: ", self.power_thresh)

        # update the plot on the fly
        if self._has_ridge:
            self.draw_ridge()

    def qset_ridge_smooth(self, text):

        '''
        rsmooth is the window size for
        the savgol filter
        '''

        # catch empty line edit
        if not text:
            return

        text = text.replace(',', '.')
        try:
            rsmooth = float(text)
            rsmooth = int(text)            
        # no parsable input
        except ValueError:
            return

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

        if self.DEBUG:
            print("ridge smooth win_len set to: ", self.rsmoothing)

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
            self.signal,
            self.periods,
            self.tvec,
            power_thresh=self.power_thresh,
            smoothing_wsize=self.rsmoothing,
        )

        # plot the ridge
        ax_spec = self.wCanvas.fig.axes[1]  # the spectrum

        # already has a plotted ridge
        if ax_spec.lines:
            ax_spec.lines = []  # remove old ridge line
            self.cb_coi.setCheckState(0)  # remove COI

        pl.draw_Wavelet_ridge(ax_spec, ridge_data, marker_size=1.5)

        # refresh the canvas
        self.wCanvas.draw()

        self.ridge_data = ridge_data

    def update_plot(self):

        """
        Replots the entire spectrum canvas 
        with a new maximal power.
        """

        # remove the old plot
        self.wCanvas.fig.clf()

        # retrieve new pmax value
        text = self.pmax_edit.text()
        text = text.replace(",", ".")
        pmax = float(text)  # pmax_edit has a positive float validator

        if self.DEBUG:
            print(f"new pmax value {pmax}")

        # creates the ax and attaches it to the widget figure
        axs = pl.mk_signal_modulus_ax(self.time_unit, fig=self.wCanvas.fig)

        pl.plot_signal_modulus(
            axs,
            time_vector=self.tvec,
            signal=self.signal,
            modulus=self.modulus,
            periods=self.periods,
            p_max=pmax,
        )

        # redraw COI if checkbox is checked
        self.draw_coi()

        # re-draw ridge
        if self._has_ridge:
            self.draw_ridge()

        # refresh the canvas
        self.wCanvas.draw()
        self.wCanvas.show()

    def set_up_anneal(self):

        """ Spawns a new AnnealConfigWindow 
        deactivated for the public version..!
        """

        if self.DEBUG:
            print("set_up_anneal called")

        # is bound to parent Wavelet Window
        self.ac = AnnealConfigWindow(self, self.DEBUG)
        self.ac.initUI(self.periods)

    def do_annealRidge_detection(self, anneal_pars):

        """ Gets called from the AnnealConfigWindow 
        deactivated for the public version..!
        """

        if anneal_pars is None:
            return

        # todo add out-of-bounds parameter check in config window
        ini_per = anneal_pars["ini_per"]
        ini_T = anneal_pars["ini_T"]
        Nsteps = int(anneal_pars["Nsteps"])
        max_jump = int(anneal_pars["max_jump"])
        curve_pen = anneal_pars["curve_pen"]

        # get modulus index of initial straight line ridge
        y0 = np.where(self.periods < ini_per)[0][-1]

        ridge_y, cost = core.find_ridge_anneal(
            self.modulus, y0, ini_T, Nsteps, mx_jump=max_jump, curve_pen=curve_pen
        )

        self.ridge = ridge_y

        # draw the ridge and make ridge_data
        self._has_ridge = True
        self.draw_ridge()

    def draw_coi(self):

        """
        Draws the COI on the spectrum.
        Also redraws the ridge!
        """

        ax_spec = self.wCanvas.fig.axes[1]  # the spectrum axis

        # COI desired?
        if bool(self.cb_coi.checkState()):

            # draw on the spectrum
            pl.draw_COI(ax_spec, self.tvec)

        else:
            ax_spec.lines = []  # remove coi, and ridge?!
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

        # to keep the line shorter..
        wo = self.w_offset
        self.ResultWindows[wo] = WaveletReadoutWindow(
            self.signal_id,
            self.ridge_data,
            time_unit=self.time_unit,
            draw_coi=self.cb_coi.isChecked(),
            pos_offset=self.w_offset,
            DEBUG=self.DEBUG,
        )
        self.w_offset += 20

    def ini_average_spec(self):

        self.avWspecWindow = AveragedWaveletWindow(self.modulus, parent=self)


class mkWaveletCanvas(FigureCanvas):
    def __init__(self, parent=None):  # , width=6, height=3, dpi=100):

        # dpi changes fontsize, so bette leave it as is..
        self.fig = Figure(dpi=100)

        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)


class WaveletReadoutWindow(QWidget):
    def __init__(
        self, signal_id, ridge_data, time_unit, draw_coi, pos_offset=0, DEBUG=False
    ):
        super().__init__()

        self.signal_id = signal_id

        self.draw_coi = draw_coi
        self.ridge_data = ridge_data
        self.time_unit = time_unit

        # creates self.rCanvas and plots the results
        self.initUI(pos_offset)

        self.DEBUG = DEBUG

    def initUI(self, position):

        self.setWindowTitle("Wavelet Results - " + str(self.signal_id))
        self.setGeometry(700 + position, 260 + position, 750, 500)

        # embed the plotting canvas

        self.rCanvas = mkReadoutCanvas()
        main_frame = QWidget()
        self.rCanvas.setParent(main_frame)
        ntb = NavigationToolbar(self.rCanvas, main_frame)

        # --- plot the wavelet results ---------
        pl.plot_readout(
            self.ridge_data,
            self.time_unit,
            fig=self.rCanvas.fig,
            draw_coi=self.draw_coi,
        )
        self.rCanvas.fig.subplots_adjust(
            wspace=0.3, left=0.1, top=0.98, right=0.95, bottom=0.15
        )

        # messes things up here :/
        # self.rCanvas.fig.tight_layout()

        main_layout = QGridLayout()
        main_layout.addWidget(self.rCanvas, 0, 0, 9, 1)
        main_layout.addWidget(ntb, 10, 0, 1, 1)

        # add the save Button
        SaveButton = QPushButton("Save Results", self)
        SaveButton.clicked.connect(self.save_out)

        button_layout_h = QHBoxLayout()
        button_layout_h.addStretch(1)
        button_layout_h.addWidget(SaveButton)
        button_layout_h.addStretch(1)
        main_layout.addLayout(button_layout_h, 11, 0, 1, 1)

        self.setLayout(main_layout)
        self.show()

    def save_out(self):

        dialog = QFileDialog()
        options = QFileDialog.Options()

        # ----------------------------------------------------------
        base_name = str(self.signal_id).replace(' ', '-')
        default_name = os.path.join(os.path.expanduser('~'),  base_name + '_ridgeRO')
        format_filter = "Text File (*.txt);; csv ( *.csv);; MS Excel (*.xlsx)"
        # -----------------------------------------------------------
        file_name, sel_filter = dialog.getSaveFileName(
            self, "Save ridge readout as", default_name, format_filter, "(*.txt)", options=options
        )

        # dialog cancelled
        if not file_name:
            return

        file_ext = file_name.split(".")[-1]

        if self.DEBUG:
            print("selected filter:", sel_filter)
            print("out-path:", file_name)
            print("extracted extension:", file_ext)
            print("ridge data keys:", self.ridge_data.keys())

        if file_ext not in ["txt", "csv", "xlsx"]:

            msgBox = QMessageBox()
            msgBox.setWindowTitle("Unknown File Format")
            msgBox.setText(
                "Please append .txt, .csv or .xlsx to the file name!")
            msgBox.exec()            

            return

        # the write out calls
        settings = QSettings()
        float_format = settings.value('float_format', '%.3f')

        if file_ext == "txt":
            self.ridge_data.to_csv(
                file_name, index=False, sep="\t", float_format=float_format
            )

        elif file_ext == "csv":
            self.ridge_data.to_csv(
                file_name, index=False, sep=",", float_format=float_format
            )

        elif file_ext == "xlsx":
            self.ridge_data.to_excel(file_name, index=False, float_format=float_format)

        else:
            if self.DEBUG:
                print("Something went wrong during save out..")
            return
        if self.DEBUG:
            print("Saved!")


class mkReadoutCanvas(FigureCanvas):
    def __init__(self):

        self.fig = Figure(figsize=(8.5, 7), dpi=100)

        FigureCanvas.__init__(self, self.fig)

        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)


class AveragedWaveletWindow(QWidget):
    def __init__(self, modulus, parent):
        super().__init__()

        # --- calculate time averaged power spectrum <-> Fourier estimate ---
        self.avWspec = np.sum(modulus, axis=1) / modulus.shape[1]
        # -------------------------------------------------------------------

        # the Wavelet analysis window spawning *this* Widget
        self.parentWA = parent

        self.initUI()

    def initUI(self):

        self.setWindowTitle(f"Fourier Spectrum Estimate - {self.parentWA.signal_id}")
        self.setGeometry(510, 80, 550, 400)

        main_frame = QWidget()
        pCanvas = mkGenericCanvas()
        pCanvas.setParent(main_frame)
        ntb = NavigationToolbar(pCanvas, main_frame)

        # plot it
        pCanvas.fig.clf()
        
        pl.averaged_Wspec(
            self.avWspec,
            self.parentWA.periods,
            time_unit=self.parentWA.time_unit,
            fig=pCanvas.fig)
        
        pCanvas.fig.subplots_adjust(left=0.15, bottom=0.17)

        main_layout = QGridLayout()
        main_layout.addWidget(pCanvas, 0, 0, 9, 1)
        main_layout.addWidget(ntb, 10, 0, 1, 1)

        self.setLayout(main_layout)
        self.show()


# --- Not used in the public version.. ---


class AnnealConfigWindow(QWidget):

    """
    Not used in the public version..
    """

    # the signal for the anneal parameters
    signal = pyqtSignal("PyQt_PyObject")

    def __init__(self, parent, DEBUG):

        super().__init__()
        # get properly initialized in set_up_anneal
        self.parentWaveletWindow = parent
        self.DEBUG = DEBUG

    def initUI(self, periods):
        self.setWindowTitle("Ridge from Simulated Annealing")
        self.setGeometry(310, 330, 350, 200)

        config_grid = QGridLayout()

        ini_per = QLineEdit(
            str(int(np.mean(periods)))
        )  # start at middle of period interval
        ini_T = QLineEdit("10")
        Nsteps = QLineEdit("5000")
        max_jump = QLineEdit("3")
        curve_pen = QLineEdit("0")

        # for easy readout and emission
        self.edits = {
            "ini_per": ini_per,
            "ini_T": ini_T,
            "Nsteps": Nsteps,
            "max_jump": max_jump,
            "curve_pen": curve_pen,
        }

        per_ini_lab = QLabel("Initial period guess")
        T_ini_lab = QLabel("Initial temperature")
        Nsteps_lab = QLabel("Number of iterations")
        max_jump_lab = QLabel("Maximal jumping distance")
        curve_pen_lab = QLabel("Curvature cost")

        per_ini_lab.setWordWrap(True)
        T_ini_lab.setWordWrap(True)
        Nsteps_lab.setWordWrap(True)
        max_jump_lab.setWordWrap(True)
        curve_pen_lab.setWordWrap(True)

        OkButton = QPushButton("Run!", self)
        OkButton.clicked.connect(self.read_emit_parameters)

        # 1st column
        config_grid.addWidget(per_ini_lab, 0, 0, 1, 1)
        config_grid.addWidget(ini_per, 0, 1, 1, 1)
        config_grid.addWidget(T_ini_lab, 1, 0, 1, 1)
        config_grid.addWidget(ini_T, 1, 1, 1, 1)
        config_grid.addWidget(curve_pen_lab, 2, 0, 1, 1)
        config_grid.addWidget(curve_pen, 2, 1, 1, 1)

        # 2nd column
        config_grid.addWidget(Nsteps_lab, 0, 2, 1, 1)
        config_grid.addWidget(Nsteps, 0, 3, 1, 1)
        config_grid.addWidget(max_jump_lab, 1, 2, 1, 1)
        config_grid.addWidget(max_jump, 1, 3, 1, 1)
        config_grid.addWidget(OkButton, 2, 3, 1, 1)

        # set main layout
        self.setLayout(config_grid)
        self.show()

    def read_emit_parameters(self):

        anneal_pars = {}
        for par_key in self.edits:
            edit = self.edits[par_key]
            anneal_pars[par_key] = float(edit.text())

        if self.DEBUG:
            print("Anneal pars:", anneal_pars)
        self.parentWaveletWindow.do_annealRidge_detection(anneal_pars)
        # send to WaveletAnalyzer Window
        # self.signal.emit(anneal_pars)
