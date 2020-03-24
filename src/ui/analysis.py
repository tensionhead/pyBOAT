import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PyQt5.QtWidgets import QCheckBox, QTableView, QComboBox, QFileDialog, QAction, QMainWindow, QApplication, QLabel, QLineEdit, QPushButton, QMessageBox, QSizePolicy, QWidget, QVBoxLayout, QHBoxLayout, QDialog, QGroupBox, QFormLayout, QGridLayout, QTabWidget, QTableWidget
from PyQt5.QtGui import QDoubleValidator, QIntValidator, QScreen
from PyQt5.QtCore import Qt, pyqtSignal

from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from ui.util import MessageWindow, posfloatV, posintV
from tfa_lib import core as wl
from tfa_lib import plotting as pl


class mkTimeSeriesCanvas(FigureCanvas):

    # dpi != 100 looks wierd?!
    def __init__(self, parent=None, width=4, height=3, dpi=100):
        
        self.fig1 = Figure(figsize=(width,height), dpi=dpi)

        FigureCanvas.__init__(self, self.fig1)
        self.setParent(parent)
        
        # print ('Time Series Size', FigureCanvas.sizeHint(self))
        FigureCanvas.setSizePolicy(self,
                QSizePolicy.Expanding,
                QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)


class FourierAnalyzer(QWidget):
    def __init__(self, signal, dt,
                 signal_id, position,
                 time_unit, show_T, parent = None):
        super().__init__()


        self.time_unit = time_unit
        self.show_T = show_T
                
        # --- calculate Fourier spectrum ------------------
        self.fft_freqs, self.fpower = wl.compute_fourier(signal, dt)
        # -------------------------------------------------                
        
        self.initUI(position, signal_id)

    def initUI(self, position, signal_id):

        self.setWindowTitle('Fourier spectrum ' + signal_id)
        self.setGeometry(510+position,80+position,550,600)

        main_frame = QWidget()
        self.fCanvas = mkFourierCanvas()        
        self.fCanvas.setParent(main_frame)
        ntb = NavigationToolbar(self.fCanvas, main_frame)
        

        # plot it
        ax = pl.mk_Fourier_ax(self.fCanvas.fig, self.time_unit, self.show_T)
        pl.Fourier_spec(ax, self.fft_freqs, self.fpower, self.show_T)

        
        main_layout = QGridLayout()
        main_layout.addWidget(self.fCanvas,0,0,9,1)
        main_layout.addWidget(ntb,10,0,1,1)
        
        self.setLayout(main_layout)
        self.show()

class mkFourierCanvas(FigureCanvas):
    def __init__(self):
        self.fig, self.ax = plt.subplots(1,1)

        FigureCanvas.__init__(self, self.fig)

        FigureCanvas.setSizePolicy(self,
                QSizePolicy.Expanding,
                QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        

class WaveletAnalyzer(QWidget):

    def __init__(self, signal, dt, T_min, T_max, position,
                 signal_id, step_num, v_max, time_unit, DEBUG = False):
        
        super().__init__()

        self.DEBUG = DEBUG
        
        self.signal_id = signal_id
        self.signal = signal
        self.v_max = v_max
        self.time_unit = time_unit

        self.periods = np.linspace(T_min, T_max, step_num)
        
        print (self.periods[-1])
        
        # generate time vector
        self.tvec = np.arange(0, len(signal)) * dt

        # no ridge yet
        self.ridge = None
        self.ridge_data = None
        self.power_thresh = None
        self.rsmoothing = None
        self._has_ridge = False # no plotted ridge

        # no anneal parameters yet
        self.anneal_pars = None

        #=============Compute Spectrum=============================================
        self.modulus, self.wlet = wl.compute_spectrum(self.signal, dt, self.periods)
        #==========================================================================


        # Wavelet ridge-readout results
        self.ResultWindows = {}
        self.w_offset = 0

        self.initUI(position)
        
    def initUI(self, position):
        self.setWindowTitle('Wavelet Spectrum - '+str(self.signal_id))
        self.setGeometry(510+position,80+position,600,700)
        
        # Wavelet and signal plot
        self.wCanvas = mkWaveletCanvas()
        main_frame = QWidget()
        self.wCanvas.setParent(main_frame)
        ntb = NavigationToolbar(self.wCanvas, main_frame) # full toolbar

        #-------------plot the wavelet power spectrum---------------------------

        # creates the ax and attaches it to the widget figure
        axs = pl.mk_signal_modulus_ax(self.time_unit, fig = self.wCanvas.fig)
        
        pl.plot_signal_modulus(axs, time_vector = self.tvec, signal = self.signal,
                               modulus = self.modulus, periods = self.periods,
                               v_max = self.v_max)

        self.wCanvas.fig.subplots_adjust(bottom = 0.11, right=0.95,left = 0.13,top = 0.95)
        self.wCanvas.fig.tight_layout()
        # attach the axs for later reference (ridge plotting and so on..)        
        self.wCanvas.axs = self.wCanvas.fig.axes 
        #-----------------------------------------------------------------------

        
        #Ridge detection options box 
        ridge_opt_box = QGroupBox("Ridge detection")
        ridge_opt_layout = QGridLayout()
        ridge_opt_box.setLayout(ridge_opt_layout)
 
        #Start ridge detection
        maxRidgeButton = QPushButton('Detect maximum ridge', self)
        maxRidgeButton.clicked.connect(self.do_maxRidge_detection)

        annealRidgeButton = QPushButton('Set up ridge\nfrom annealing', self)
        annealRidgeButton.clicked.connect(self.set_up_anneal)

        # not-needed.. ridge auto-updates with power threshold!
        # drawRidgeButton = QPushButton('(Re-)Draw ridge', self)
        # drawRidgeButton.clicked.connect(self.draw_ridge)


        power_label = QLabel("Power threshold: ")
        power_thresh_edit = QLineEdit()
        power_thresh_edit.setValidator(posfloatV)

        smooth_label = QLabel("Ridge smoothing: ")
        ridge_smooth_edit = QLineEdit()
        ridge_smooth_edit.setValidator(posfloatV)

        
        plotResultsButton = QPushButton('Plot Results', self)
        plotResultsButton.clicked.connect(self.ini_plot_readout)

        self.cb_coi = QCheckBox('COI', self)
        self.cb_coi.stateChanged.connect(self.draw_coi)

        ridge_opt_layout.addWidget(maxRidgeButton,0,0,1,1)
        ridge_opt_layout.addWidget(annealRidgeButton,1,0)

        ridge_opt_layout.addWidget(power_label,0,1)
        ridge_opt_layout.addWidget(power_thresh_edit,0,2)
        
        ridge_opt_layout.addWidget(smooth_label,1,1)
        ridge_opt_layout.addWidget(ridge_smooth_edit,1,2)
        
        
        # ridge_opt_layout.addWidget(drawRidgeButton,1,3) # not needed anymore?!
        ridge_opt_layout.addWidget(self.cb_coi,0,3)
        ridge_opt_layout.addWidget(plotResultsButton,1,3)
        
        
        # put everything together
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.wCanvas)
        main_layout.addWidget(ntb)
        main_layout.addWidget(ridge_opt_box)
        self.setLayout(main_layout)

        # initialize line edits
                
        power_thresh_edit.textChanged[str].connect(self.qset_power_thresh)
        power_thresh_edit.insert('0.0') # initialize with 0

        ridge_smooth_edit.setValidator(QIntValidator(bottom = 0,top = 9999999999))
        ridge_smooth_edit.textChanged[str].connect(self.qset_ridge_smooth)
        ridge_smooth_edit.insert('0') # initialize with 0

        self.show()

    def qset_power_thresh(self, text):

        # catch empty line edit
        if not text:
            return
        text = text.replace(',','.')
        power_thresh = float(text)
        self.power_thresh = power_thresh
            
        if self.DEBUG:
            print('power thresh set to: ',self.power_thresh)

        # update the plot on the fly
        if self._has_ridge:
            self.draw_ridge()

        
    def qset_ridge_smooth(self, text):

        # text = text.replace(',','.')

        # catch empty line edit
        if not text:
            return
        
        rsmooth = int(text)
        
        # make an odd window length
        if rsmooth == 0:
            self.rsmoothing = None
        elif rsmooth < 5:
            self.rsmoothing = 5
        elif rsmooth > 5 and rsmooth%2 == 0:
            self.rsmoothing = rsmooth + 1
        else:
            self.rsmoothing = rsmooth

        # update the plot on the fly
        if self._has_ridge:
            self.draw_ridge()
            
        if self.DEBUG:
            print('ridge smooth win_len set to: ', self.rsmoothing)

    
    def do_maxRidge_detection(self):        

        ridge_y = wl.get_maxRidge(self.modulus)
        self.ridge = ridge_y

        if not np.any(ridge_y):
            self.e = MessageWindow('No ridge found..check spectrum!','Ridge detection error')
            return

        self._has_ridge = True
        self.draw_ridge() # ridge_data made here


    def draw_ridge(self):

        ''' makes also the ridge_data !! '''

        if not self._has_ridge:
            self.e = MessageWindow('Run a ridge detection first!','No Ridge')
            return

        ridge_data = wl.eval_ridge(self.ridge, self.wlet, self.signal,
                                   self.periods,self.tvec,
                                   power_thresh = self.power_thresh,
                                   smoothing = self.rsmoothing)


        # plot the ridge
        ax_spec = self.wCanvas.axs[1] # the spectrum

        # already has a plotted ridge
        if ax_spec.lines:
            ax_spec.lines = [] # remove old ridge line
            self.cb_coi.setCheckState(0)
            
        pl.draw_Wavelet_ridge(ax_spec, ridge_data, marker_size = 2)
        
        # refresh the canvas
        self.wCanvas.draw()
        
        self.ridge_data = ridge_data
        
    def set_up_anneal(self):

        ''' Spawns a new AnnealConfigWindow '''

        if self.DEBUG:
            print('set_up_anneal called')

        # is bound to parent Wavelet Window 
        self.ac = AnnealConfigWindow(self, self.DEBUG)
        self.ac.initUI(self.periods)

        
    def do_annealRidge_detection(self, anneal_pars):

        ''' Gets called from the AnnealConfigWindow '''
        
        if anneal_pars is None:
            self.noValues = MessageWindow('No parameters set for\nsimulated annealing!','No Parameters')
            return

        # todo add out-of-bounds parameter check in config window
        ini_per = anneal_pars['ini_per']
        ini_T = anneal_pars['ini_T']
        Nsteps = int(anneal_pars['Nsteps'])
        max_jump = int(anneal_pars['max_jump'])
        curve_pen = anneal_pars['curve_pen']

        # get modulus index of initial straight line ridge
        y0 = np.where(self.periods < ini_per)[0][-1]

        ridge_y, cost = wl.find_ridge_anneal(self.modulus, y0, ini_T, Nsteps,
                                             mx_jump = max_jump, curve_pen = curve_pen)
        
        self.ridge = ridge_y

        # draw the ridge and make ridge_data
        self._has_ridge = True
        self.draw_ridge()


    def draw_coi(self):

        ax_spec = self.wCanvas.axs[1] # the spectrum axis
        
        # COI desired?
        if bool( self.cb_coi.checkState() ):
            
            # compute slope of COI
            coi_m = wl.Morlet_COI()

            pl.draw_COI(ax_spec, self.tvec, coi_m, alpha = 0.35)
            
        else:
            ax_spec.lines = [] # remove coi, and ridge?!
            if self._has_ridge:
                self.draw_ridge() # re-draw ridge

        # refresh the canvas
        self.wCanvas.draw()
        
    def ini_plot_readout(self):
        
        if not self._has_ridge:
            self.e = MessageWindow('Do a ridge detection first!','No Ridge')
            return

        self.ResultWindows[self.w_offset] = WaveletReadoutWindow(self.signal_id,
                                                                 self.ridge_data,
                                                                 time_unit = self.time_unit,
                                                                 pos_offset = self.w_offset,
                                                                 DEBUG = self.DEBUG)
        self.w_offset += 20
            
class mkWaveletCanvas(FigureCanvas):
    
    def __init__(self, parent=None): #, width=6, height=3, dpi=100):

        self.fig = Figure(dpi = 100)

        # self.fig, self.axs = plt.subplots(2,1,
        # gridspec_kw = {'height_ratios':[1, 2.5]}, sharex = True)

        
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)
        
        FigureCanvas.setSizePolicy(self,
                QSizePolicy.Expanding,
                QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

                

class AnnealConfigWindow(QWidget):

    # the signal for the anneal parameters
    signal = pyqtSignal('PyQt_PyObject')


    def __init__(self, parent, DEBUG):
        
        super().__init__()
        # get properly initialized in set_up_anneal
        self.parentWaveletWindow = parent
        self.DEBUG = DEBUG

    def initUI(self, periods):
        self.setWindowTitle('Ridge from Simulated Annealing')
        self.setGeometry(310,330,350,200)

        config_grid = QGridLayout()
        
        ini_per = QLineEdit(str(int(np.mean(periods)))) # start at middle of period interval
        ini_T = QLineEdit('10')
        Nsteps = QLineEdit('5000')
        max_jump = QLineEdit('3')
        curve_pen = QLineEdit('0')

        # for easy readout and emission
        self.edits = {'ini_per' : ini_per, 'ini_T' : ini_T, 'Nsteps' : Nsteps, \
                      'max_jump' : max_jump, 'curve_pen' : curve_pen}
        
        per_ini_lab = QLabel('Initial period guess')
        T_ini_lab = QLabel('Initial temperature')
        Nsteps_lab = QLabel('Number of iterations')
        max_jump_lab = QLabel('Maximal jumping distance')
        curve_pen_lab = QLabel('Curvature cost')

        per_ini_lab.setWordWrap(True)
        T_ini_lab.setWordWrap(True) 
        Nsteps_lab.setWordWrap(True) 
        max_jump_lab.setWordWrap(True)
        curve_pen_lab.setWordWrap(True)
  
        OkButton = QPushButton("Run!", self)
        OkButton.clicked.connect(self.read_emit_parameters)


        # 1st column
        config_grid.addWidget( per_ini_lab, 0,0,1,1)
        config_grid.addWidget( ini_per, 0,1,1,1)
        config_grid.addWidget( T_ini_lab, 1,0,1,1)
        config_grid.addWidget( ini_T, 1,1,1,1)
        config_grid.addWidget( curve_pen_lab, 2,0,1,1)
        config_grid.addWidget( curve_pen, 2,1,1,1)

        # 2nd column
        config_grid.addWidget( Nsteps_lab, 0,2,1,1)
        config_grid.addWidget( Nsteps, 0,3,1,1)
        config_grid.addWidget( max_jump_lab, 1,2,1,1)
        config_grid.addWidget( max_jump, 1,3,1,1)
        config_grid.addWidget( OkButton, 2,3,1,1)
        
        # set main layout
        self.setLayout(config_grid)
        self.show()


    def read_emit_parameters(self):
        
        anneal_pars = {}
        for par_key in self.edits:
            edit = self.edits[par_key]
            anneal_pars[par_key] = float(edit.text())

        if self.DEBUG:
            print('Anneal pars:', anneal_pars )
        self.parentWaveletWindow.do_annealRidge_detection(anneal_pars)
        # send to WaveletAnalyzer Window
        # self.signal.emit(anneal_pars)

class WaveletReadoutWindow(QWidget):

    def __init__(self, signal_id, ridge_data, time_unit, pos_offset = 0, DEBUG = False):
        super().__init__()
        
        self.signal_id = signal_id


        self.ridge_data = ridge_data
        self.time_unit = time_unit

        # creates self.rCanvas and plots the results
        self.initUI( pos_offset )
        
        self.DEBUG = DEBUG


    def initUI(self, position):
        
        self.setWindowTitle('Wavelet Results - ' + str(self.signal_id) )
        self.setGeometry(700 + position,260 + position,750,500)

        # embed the plotting canvas
        
        self.rCanvas = mkReadoutCanvas()        
        main_frame = QWidget()
        self.rCanvas.setParent(main_frame)
        ntb = NavigationToolbar(self.rCanvas, main_frame)

        # --- plot the wavelet results ---------
        pl.plot_readout(self.ridge_data, self.time_unit,
                        fig = self.rCanvas.fig)        
        self.rCanvas.fig.subplots_adjust(wspace = 0.3, left = 0.1, top = 0.98,
                        right = 0.95, bottom = 0.15)

        # messes things up here :/
        # self.rCanvas.fig.tight_layout()
                
        main_layout = QGridLayout()
        main_layout.addWidget(self.rCanvas,0,0,9,1)
        main_layout.addWidget(ntb,10,0,1,1)

        # add the save Button
        SaveButton = QPushButton('Save Results', self)
        SaveButton.clicked.connect(self.save_out)

        button_layout_h = QHBoxLayout()
        button_layout_h.addWidget(SaveButton)
        button_layout_h.addStretch(1)        
        main_layout.addLayout(button_layout_h,11,0,1,1)
        
        self.setLayout(main_layout)
        self.show()

    def save_out(self):
        
        dialog = QFileDialog()
        options = QFileDialog.Options()

        #----------------------------------------------------------
        default_name = os.getenv('HOME') + '/TFAres_' + str(self.signal_id)
        format_filter = "Text File (*.txt);; CSV ( *.csv);; Excel (*.xlsx)"
        #-----------------------------------------------------------
        file_name, sel_filter = dialog.getSaveFileName(self,"Save as",
                                              default_name,
                                              format_filter,
                                              '(*.txt)',
                                              options=options)

        # dialog cancelled
        if not file_name:
            return
        
        file_ext = file_name.split('.')[-1]

        if self.DEBUG:
            print('selected filter:',sel_filter)
            print('out-path:',file_name)
            print('extracted extension:', file_ext)
            print('ridge data keys:', self.ridge_data.keys())

        
        if file_ext not in ['txt','csv','xlsx']:
            self.e = MessageWindow("Ouput format not supported..\n" +
                           "Please append .txt, .csv or .xlsx\n" +
                           "to the file name!",
                           "Unknown format")
            return
        
            

        # the write out calls
        float_format = '%.2f' # still old style :/
            
        if file_ext == 'txt':
            self.ridge_data.to_csv(file_name, index = False,
                          sep = '\t',
                          float_format = float_format
            )

        elif file_ext == 'csv':
            self.ridge_data.to_csv(file_name, index = False,
                          sep = ',',
                          float_format = float_format
            )

        elif file_ext == 'xlsx':
            self.ridge_data.to_excel(file_name, index = False,
                          float_format = float_format
            )

        else:
            if self.DEBUG:
                print("Something went wrong during save out..")
            return
        if self.DEBUG:
            print('Saved!')
        

class mkReadoutCanvas(FigureCanvas):

    def __init__(self):
        
        self.fig = Figure(figsize = (8.5,7), dpi = 100)

        FigureCanvas.__init__(self, self.fig)

        FigureCanvas.setSizePolicy(self,
                QSizePolicy.Expanding,
                QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        
