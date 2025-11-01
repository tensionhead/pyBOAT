import os
import numpy as np

from PyQt6.QtWidgets import (
    QCheckBox,
    QMessageBox,
    QTableView,
    QComboBox,
    QFileDialog,
    QMainWindow,
    QLabel,
    QLineEdit,
    QPushButton,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QFormLayout,
    QGridLayout,
    QTabWidget,
    QAbstractItemView,
    QDoubleSpinBox,
    QSpinBox,
    QSplitter,
    QApplication,
    QRadioButton,
    QButtonGroup
)


SpinBox = QSpinBox | QDoubleSpinBox

from PyQt6.QtCore import Qt, QSettings, QTimer, QPoint
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

import pandas as pd

from pyboat.ui.util import (
    PandasModel,
    selectFilter,
    spawn_warning_box,
    is_dark_color_scheme,
    StoreGeometry,
    create_spinbox,
    mk_spinbox_unit_slot,
    WAnalyzerParams
)
from pyboat.ui.analyzer import mkTimeSeriesCanvas, FourierAnalyzer, WaveletAnalyzer
from pyboat.ui.batch_process import BatchProcessWindow
from pyboat.ui import style
from pyboat.ui import analysis_parameters as ap

from pyboat.ui.defaults import default_par_dict, debounce_ms

import pyboat
from pyboat import plotting as pl

# --- monkey patch label sizes to better fit the ui ---
pl.tick_label_size = 12
pl.label_size = 14

# same for all FileDialogs
FormatFilter = "csv ( *.csv);; MS Excel (*.xlsx);; Text File (*.txt)"
Position = int


class AnalyzerStack:
    """Stack of analyzer instances/windows"""

    delta: Position = 30  # new instance shift in pixels
    def __init__(self):
        self.w_position: Position = 0  # analysis window position offset
        self._stack: list[WaveletAnalyzer] = []

    def push(self, ana: WaveletAnalyzer) -> None:
        self._stack.append(ana)
        self.shift()

    def remove(self, ana: WaveletAnalyzer) -> None:
        if not self._stack:
            return
        self._stack.remove(ana)
        self.w_position -= self.delta

    def last(self) -> WaveletAnalyzer | None:
        if self:
            return self._stack[-1]
        return None

    def shift(self):
        self.w_position += self.delta


    def __bool__(self) -> bool:
        return bool(self._stack)



class DataViewer(StoreGeometry, QMainWindow):
    def __init__(self, data, pos_offset, parent, debug=False):
        StoreGeometry.__init__(self, pos=(80 + pos_offset, 300 + pos_offset), size=(900, 650))
        QMainWindow.__init__(self, parent=parent)

        # this is the data table
        self.df = data

        self.anaWindows: AnalyzerStack = AnalyzerStack()

        self.debug = debug

        # this variable tracks the selected trajectory
        # -> DataFrame column name!
        self.signal_id = None  # no signal id initially selected

        self.raw_signal: np.ndarray | None = None  # no signal initial array
        self.dt: float | None = None  # gets initialized from the UI -> qset_dt
        self.tvec: np.ndarray | None = None  # gets initialized by vector_prep
        self.time_unit: str | None  = None  # gets initialized by qset_time_unit

        self._ra_timer = QTimer(self)
        self._ra_timer.setInterval(debounce_ms)
        self._ra_timer.setSingleShot(True)

        self.initUI(pos_offset)

        # throttle reanalyze
        self._ra_timer.timeout.connect(self.re_wavelet_ana)

    # ===========    UI    ================================

    def initUI(self, pos_offset):

        # spinboxes which get the unit suffix
        connect_to_unit: list[SpinBox] = []

        self.setWindowTitle(f"DataViewer - {self.df.name}")
        self.restore_geometry(pos_offset)

        central_widget = QWidget()
        # for the status bar
        self.statusBar()

        plot_frame = QWidget()
        self.tsCanvas = mkTimeSeriesCanvas()
        self.tsCanvas.setParent(plot_frame)
        ntb = NavigationToolbar(self.tsCanvas, plot_frame)  # full matplotlib toolbar

        # the table instance,
        DataTable = QTableView()
        model = PandasModel(self.df)
        DataTable.setModel(model)
        DataTable.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectColumns)  # columns only
        DataTable.clicked.connect(self.Table_select)  # magically transports QModelIndex
        # so that it also works for header selection
        header = DataTable.horizontalHeader()  # returns QHeaderView
        header.sectionClicked.connect(
            self.Header_select
        )  # magically transports QModelIndex

        # the signal selection box
        SignalBox = QComboBox(self)
        SignalBox.setStatusTip("..or just click directly on a signal in the table!")

        main_layout_v = QVBoxLayout()  # The whole Layout
        # Data selction drop-down
        dataLabel = QLabel("Select Signal", self)

        dt_label = QLabel("Sampling Interval:")

        self.dt_spin = create_spinbox(1, step=1, minimum=.1, double=True)
        self.dt_spin.setStatusTip("How much time in between two recordings?")
        connect_to_unit.append(self.dt_spin)

        unit_label = QLabel("Time Unit:")
        self.unit_edit = QLineEdit(self)
        self.unit_edit.setStatusTip("Changes only the axis labels..")
        self.unit_edit.setMinimumSize(70, 0)

        # == Top row and data table ==

        top_bar_box = QWidget()
        top_bar_layout = QHBoxLayout()

        top_bar_layout.addWidget(dataLabel)
        top_bar_layout.addWidget(SignalBox)
        top_bar_layout.addStretch(0)
        top_bar_layout.addWidget(dt_label)
        top_bar_layout.addWidget(self.dt_spin)
        top_bar_layout.addStretch(0)
        top_bar_layout.addWidget(unit_label)
        top_bar_layout.addWidget(self.unit_edit)
        top_bar_layout.addStretch(0)
        top_bar_box.setLayout(top_bar_layout)

        top_and_table = QGroupBox()
        top_and_table_layout = QVBoxLayout()
        top_and_table_layout.addWidget(top_bar_box)
        top_and_table_layout.addWidget(DataTable)
        top_and_table.setLayout(top_and_table_layout)

        # == Plot frame/Canvas area ==

        plot_box = QWidget()
        plot_layout = QVBoxLayout()
        plot_layout.addWidget(self.tsCanvas)
        plot_layout.addWidget(ntb)
        plot_box.setLayout(plot_layout)

        # plot options box
        plot_options_box = QGroupBox("Plotting Options")
        plot_options_box.setStyleSheet("""QGroupBox {font-weight:bold;}""")
        plot_options_layout = QGridLayout()

        self.rb_raw = QRadioButton("Raw Signal")
        self.rb_raw.setStatusTip("Plot the raw unfiltered signal")

        self.rb_detrend = QRadioButton("Detrended Signal")
        self.rb_detrend.setStatusTip(
            "Plots the signal after trend subtraction (detrending)"
        )

        group = QButtonGroup()
        group.addButton(self.rb_raw)
        group.addButton(self.rb_detrend)

        saveButton = QPushButton("Save Filter Results", self)
        saveButton.clicked.connect(self.save_out_trend)
        saveButton.setStatusTip("Writes the trend and the detrended signal into a file")

        ## checkbox layout
        plot_options_layout.addWidget(self.rb_raw, 0, 0)
        plot_options_layout.addWidget(self.rb_detrend, 0, 1)
        plot_options_layout.addWidget(saveButton, 0, 2, 1, 1)
        plot_options_box.setLayout(plot_options_layout)

        ## checkbox signal set and change
        self.rb_raw.toggle()

        self.rb_raw.toggled.connect(self.toggle_raw)
        self.rb_detrend.toggled.connect(self.toggle_trend)

        # == Wavelet parameters ==

        # filter and amplitude normalization
        self.sinc_envelope = ap.SincEnvelopeOptions(self)

        # Analyzer box with tabs
        ana_box = QGroupBox("Frequency Analysis")
        ana_box.setStyleSheet("QGroupBox {font-weight: bold; }")
        # setStyleSheet("QGroupBox::title{font:bold;}")
        ana_layout = QVBoxLayout()

        ## Initialize tab scresen
        tabs = QTabWidget()
        tab1 = QWidget()
        tab2 = QWidget()

        ## Add tabs
        tabs.addTab(tab1, "Wavelet Analysis")
        tabs.addTab(tab2, "Fourier Transform")

        ## Create first tab
        self.wavelet_tab = ap.WaveletTab(self)
        tab1.setLayout(self.wavelet_tab)

        # -- Wavelet Buttons --
        wletButton = QPushButton("Analyze Signal")
        if is_dark_color_scheme():
            wletButton.setStyleSheet(f"background-color: {style.dark_primary}")
        else:
            wletButton.setStyleSheet(f"background-color: {style.light_primary}")
        wletButton.setStatusTip("Opens the wavelet analysis..")
        wletButton.clicked.connect(self.new_wavelet_ana)

        batchButton = QPushButton("Analyze All..")
        batchButton.clicked.connect(self.run_batch)
        batchButton.setStatusTip("Starts batch processing with the current parameters")

        wbutton_layout_h = QHBoxLayout()
        wbutton_layout_h.addWidget(batchButton)
        wbutton_layout_h.addStretch(0)
        wbutton_layout_h.addWidget(wletButton)

        # fourier button
        fButton = QPushButton("Fourier Transform", self)

        ## add  button to layout
        f_button_layout_h = QHBoxLayout()
        fButton.clicked.connect(self.run_fourier_ana)
        f_button_layout_h.addStretch(0)
        f_button_layout_h.addWidget(fButton)

        # fourier detrended switch
        self.cb_use_detrended2 = QCheckBox("Use Detrended Signal", self)
        self.cb_use_detrended2.setChecked(True)  # detrend by default

        self.cb_use_envelope2 = QCheckBox("Normalize with Envelope", self)
        self.cb_use_envelope2.setChecked(False)

        # fourier period or frequency view
        self.cb_FourierT = QCheckBox("Show Frequencies", self)
        self.cb_FourierT.setChecked(False)  # show periods per default

        ## Create second Fourier tab
        tab2.parameter_box = QFormLayout()
        # tab2.parameter_box.addRow(Tmin_lab,self.Tmin)
        # tab2.parameter_box.addRow(Tmax_lab,self.Tmax)
        tab2.parameter_box.addRow(self.cb_use_detrended2)
        tab2.parameter_box.addRow(self.cb_use_envelope2)
        tab2.parameter_box.addRow(self.cb_FourierT)
        tab2.parameter_box.addRow(f_button_layout_h)
        tab2.setLayout(tab2.parameter_box)

        # Add tabs to Vbox
        ana_layout.addWidget(self.sinc_envelope)
        ana_layout.addWidget(tabs)
        ana_layout.addLayout(wbutton_layout_h)
        ana_box.setLayout(ana_layout)

        # = Combine Plot and Options ==

        # merge plot and analysis options
        options = QWidget()
        options_layout = QGridLayout()
        options.setLayout(options_layout)
        options_layout.addWidget(plot_options_box, 1, 0, 1, 1)
        options_layout.addWidget(ana_box, 2, 0, 1, 1)

        # fix width of options -> only plot should stretch
        # options.setFixedWidth(int(options.sizeHint().width() * 0.98))

        plot_and_options = QWidget()
        lower_layout = QHBoxLayout()
        plot_and_options.setLayout(lower_layout)
        lower_layout.addWidget(plot_box, stretch=10)
        lower_layout.addWidget(options, stretch=1)

        # == Main Layout ==

        # vertical splitter between data table and plot + options
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(top_and_table)
        splitter.addWidget(plot_and_options)
        main_layout_v.addWidget(splitter)

        # populate signal selection box
        SignalBox.addItem("")  # empty initial selector

        for col in self.df.columns:
            SignalBox.addItem(col)

        # connect to plotting machinery
        SignalBox.textActivated[str].connect(self.select_signal_and_Plot)
        # to modify current index by table selections
        self.SignalBox = SignalBox

        # --- connect some parameter fields ---

        self.dt_spin.valueChanged.connect(self.qset_dt)
        # propagate initial value
        self.qset_dt()
        self.unit_edit.textChanged[str].connect(self.qset_time_unit)

        # connect unit edit
        for spin in connect_to_unit:
            self.unit_edit.textChanged[str].connect(mk_spinbox_unit_slot(spin))
        self.unit_edit.insert("min")  # standard time unit is minutes

        # --- initialize parameter fields from settings ---

        self.load_settings()

        central_widget.setLayout(main_layout_v)
        self.setCentralWidget(central_widget)
        self.show()

        # trigger initial plot
        # select 1st signal
        self.Table_select(DataTable.indexAt(QPoint(0, 0)))

    def reanalyze_signal(self) -> None:
        self._ra_timer.start()

    # when clicked into the table
    def Table_select(self, qm_index):
        # recieves QModelIndex
        col_nr = qm_index.column()
        self.SignalBox.setCurrentIndex(col_nr + 1)
        if self.debug:
            print("table column number clicked:", col_nr)
        signal_id = self.df.columns[col_nr]  # DataFrame column name
        self.select_signal_and_Plot(signal_id)

    # when clicked on the header
    def Header_select(self, index):
        # recieves index
        col_nr = index
        self.SignalBox.setCurrentIndex(col_nr + 1)

        if self.debug:
            print("table column number clicked:", col_nr)

        signal_id = self.df.columns[col_nr]  # DataFrame column name
        self.select_signal_and_Plot(signal_id)

    # the signal to work on, connected to selection box
    def select_signal_and_Plot(self, text) -> None:
        self.signal_id = text
        succ, _, _ = self.vector_prep(self.signal_id)  # fix a raw_signal + time vector
        if not succ:  # error handling done in data_prep
            print("Could not load", self.signal_id)
            return
        self.wavelet_tab.set_auto_periods()
        self.sinc_envelope.set_auto_T_c()
        self.sinc_envelope.set_auto_wsize()
        self.doPlot()

    # probably all the toggle state variables are not needed -> read out checkboxes directly
    def toggle_raw(self, checked: bool):
        if checked:
            self.plot_raw = True
            # untoggle the detrended cb
            self.rb_detrend.setChecked(False)
        else:
            self.plot_raw = False

        # signal selected?
        if self.signal_id:
            self.doPlot()

    def toggle_trend(self, _: bool):
        # signal selected?
        if self.signal_id:
            self.doPlot()

    # connected to unit_edit
    def qset_time_unit(self, text):
        self.time_unit = text  # self.unit_edit.text()
        if self.debug:
            print("time unit changed to:", text)

    # connected to dt_spin
    def qset_dt(self):
        """
        Triggers the rewrite of the initial periods and
        cut-off period T_c
        """
        t = self.dt_spin.value()
        self.dt = float(t)
        self.wavelet_tab.set_auto_periods(force=True)
        self.sinc_envelope.set_auto_T_c(force=True)
        self.sinc_envelope.set_auto_wsize(force=True)
        # refresh plot if a is signal selected
        if self.signal_id:
            self.doPlot()

        if self.debug:
            print("dt set to:", self.dt)

    def vector_prep(self, signal_id):
        """
        prepares raw signal vector (NaN removal) and
        corresponding time vector
        """
        if self.debug:
            print("preparing", signal_id)

        # checks for empty signal_id string
        if signal_id:
            a = self.df[signal_id]
            # remove contiguous (like trailing) NaN regions
            start, end = a.first_valid_index(), a.last_valid_index()
            a = a.loc[start:end]
            raw_signal = np.array(a)

            # catch intermediate (non-trailing) NaNs
            NaNswitches = np.sum(np.diff(np.isnan(raw_signal)))
            if NaNswitches > 0:

                msgBox = spawn_warning_box(
                    self,
                    text=("Non contiguous regions of missing data samples (NaN) "
                          f"encountered for '{signal_id}', using linear interpolation.\n"
                          "Try 'Import..' from the main menu "
                          "to interpolate missing values for all signals at once!"
                          ),
                    title="Found missing samples")
                msgBox.exec()

                raw_signal = pyboat.core.interpolate_NaNs(raw_signal)
                # inject into DataFrame on the fly, re-adding trailing NaNs
                self.df[signal_id] = pd.Series(raw_signal)

            # set attribute
            self.raw_signal = raw_signal

            self.tvec = np.arange(0, len(self.raw_signal), step=1) * self.dt
            return True, start, end  # success

        else:

            msgBox = QMessageBox(parent=self)
            msgBox.setText("Please select a signal!")
            msgBox.exec()
            return False, None, None

    def calc_trend(self) -> np.ndarray | None:
        """ Uses maximal sinc window size """

        T_c = self.sinc_envelope.get_T_c()
        if not T_c:
            return None
        if self.debug:
            print("Calculating trend with T_c = ", T_c)

        trend = pyboat.sinc_smooth(raw_signal=self.raw_signal,
                                   T_c=T_c,
                                   dt=self.dt)
        return trend

    def calc_envelope(self) -> np.ndarray | None:

        assert self.raw_signal is not None
        window_size = self.sinc_envelope.get_wsize()
        if not window_size:
            return None
        if self.debug:
            print("Calculating envelope with window_size = ", window_size)

        # cut of frequency set and detrended plot activated?
        if self.sinc_envelope.do_detrend:
            trend = self.calc_trend()
            signal = self.raw_signal - trend
        else:
            signal = self.raw_signal
        envelope = pyboat.sliding_window_amplitude(signal,
                                                   window_size,
                                                   dt=self.dt)
        return envelope

    def doPlot(self):

        """
        Checks the checkboxes for trend and envelope..
        """

        # update raw_signal and tvec
        succ, _, _ = self.vector_prep(self.signal_id)  # error handling done here

        if not succ:
            return False

        trend = self.calc_trend()
        envelope = self.calc_envelope()

        self.tsCanvas.fig1.clf()

        ax1 = pl.mk_signal_ax(self.time_unit, fig=self.tsCanvas.fig1)
        ax2 = None
        self.tsCanvas.fig1.add_axes(ax1)

        if self.rb_raw.isChecked():
            pl.draw_signal(ax1, time_vector=self.tvec, signal=self.raw_signal)

        if trend is not None:
            # plot trend only on raw signal
            if self.rb_raw.isChecked():
                pl.draw_trend(ax1, time_vector=self.tvec, trend=trend)

            else:
                ax2 = pl.draw_detrended(
                    ax1, time_vector=self.tvec, detrended=self.raw_signal - trend
                )
                ax2.legend(fontsize=pl.tick_label_size)
        # can not plot detrended signal without trend
        elif self.rb_detrend.isChecked():
            spawn_warning_box(
                self,
                "Nothing to plot",
                "Can not plot detrended signal without sinc detrending..\nReverting to plot raw signal!"
            ).exec()
            # triggers replot
            self.rb_raw.setChecked(True)
            return

        if envelope is not None:
            # add envelope to trend if available
            if self.rb_raw.isChecked():
                envelope = envelope + trend if trend is not None else envelope
                pl.draw_envelope(ax1, time_vector=self.tvec, envelope=envelope)

            # plot on detrended axis if existing
            else:
                pl.draw_envelope(ax2, time_vector=self.tvec, envelope=envelope)
                ax2.legend(fontsize=pl.tick_label_size)


        self.tsCanvas.fig1.subplots_adjust(bottom=0.15, left=0.15, right=0.85)

        # add a simple legend
        ax1.legend(fontsize=pl.tick_label_size)

        self.tsCanvas.draw()
        self.tsCanvas.show()

    def _get_analyzer_params(self) -> WAnalyzerParams:

        assert self.dt is not None
        assert self.raw_signal is not None

        wlet_pars = self.wavelet_tab.assemble_wlet_pars()
        periods = np.linspace(wlet_pars["Tmin"], wlet_pars["Tmax"], wlet_pars["nT"])

        return WAnalyzerParams(
            self.dt,
            self.raw_signal,
            periods=periods,
            max_power=wlet_pars["pow_max"],
            T_c=self.sinc_envelope.get_T_c(),
            window_size=self.sinc_envelope.get_wsize()
        )


    def re_wavelet_ana(self):

        if not self.anaWindows:
            return

        wp = self._get_analyzer_params()

        # renalyze either active analyzer window or last create analysis
        active = QApplication.activeWindow()
        if isinstance(active, WaveletAnalyzer):
            active.reanalyze(wp)
        else:
            self.anaWindows.last().reanalyze(wp)

    def new_wavelet_ana(self):
        """ run the Wavelet Analysis """

        if not np.any(self.raw_signal):
            # should not happen..
            msgBox = spawn_warning_box(self, "No Signal", "Please select a signal first!")
            msgBox.exec()
            return False

        assert self.dt is not None
        assert self.raw_signal is not None

        self.anaWindows.push(
            WaveletAnalyzer(
                self._get_analyzer_params(),
                position=self.anaWindows.w_position,
                signal_id=self.signal_id,
                time_unit=self.time_unit,
                dv=self,
            )
        )

    def run_batch(self):

        """
        Takes the ui wavelet settings and
        spwans the batch processing Widget
        """

        # reads the wavelet analysis settings from the ui input
        wlet_pars = self.wavelet_tab.assemble_wlet_pars()
        wlet_pars["window_size"] = self.sinc_envelope.get_wsize()

        if self.debug:
            print(f"Started batch processing with {wlet_pars}")

        # Spawning the batch processing config widget
        # is bound to parent Wavelet Window
        self.bc = BatchProcessWindow(self.debug, parent=self)
        self.bc.initUI(wlet_pars)

    def run_fourier_ana(self):
        if not np.any(self.raw_signal):

            msgBox = spawn_warning_box(self, "No Signal",
                                       text="Please select a signal first!")
            msgBox.exec()
            return False

        if self.sinc_envelope.do_detrend:
            trend = self.calc_trend()
            signal = self.raw_signal - trend
        else:
            signal = self.raw_signal

        if self.sinc_envelope.do_normalize:
            window_size = self.sinc_envelope.get_wsize()
            signal = pyboat.normalize_with_envelope(signal, window_size, dt=self.dt)

        w_position = self.anaWindows.w_position + 20

        # periods or frequencies?
        if self.cb_FourierT.isChecked():
            show_T = False
        else:
            show_T = True

        FourierAnalyzer(
            signal=signal,
            dt=self.dt,
            signal_id=self.signal_id,
            position=self.anaWindows.w_position,
            time_unit=self.time_unit,
            show_T=show_T,
            parent=self,
        )
        # to keep Analyzer window positions shifting
        self.anaWindows.shift()

    def save_out_trend(self):

        if not np.any(self.raw_signal):

            msgBox = QMessageBox()
            msgBox.setWindowTitle("No Signal")
            msgBox.setText("Please select a signal first!")
            msgBox.exec()
            return

        if self.debug:
            print("saving trend out")

        # -------calculate trend and detrended signal------------
        trend = self.calc_trend()
        dsignal = self.raw_signal - trend

        # add everything to a pandas data frame
        data = np.array([self.raw_signal, trend, dsignal]).T  # stupid pandas..
        columns = ["raw", "trend", "detrended"]
        df_out = pd.DataFrame(data=data, columns=columns)
        # ------------------------------------------------------

        if self.debug:
            print("df_out", df_out[:10])
            print("trend", trend[:10])
        dialog = QFileDialog()

        settings = QSettings()
        # ----------------------------------------------------------
        base_name = str(self.signal_id).replace(" ", "-")
        dir_path = settings.value("dir_name", os.path.curdir)
        data_format = settings.value("data_format", "csv")
        default_name = os.path.join(dir_path, base_name + "_trend.")
        default_name += data_format
        # -----------------------------------------------------------
        file_name, sel_filter = dialog.getSaveFileName(
            self, "Save as", default_name, FormatFilter, selectFilter[data_format]
        )
        # dialog cancelled
        if not file_name:
            return

        file_ext = file_name.split(".")[-1]

        if self.debug:
            print("selected filter:", sel_filter)
            print("out-path:", file_name)
            print("extracted extension:", file_ext)

        if file_ext not in ["txt", "csv", "xlsx"]:

            msgBox = QMessageBox()
            msgBox.setWindowTitle("Unknown File Format")
            msgBox.setText("Please append .txt, .csv or .xlsx to the file name!")
            msgBox.exec()
            return

        # ------the write out calls to pandas----------------

        # defaults to 3 decimals
        float_format = settings.value("float_format", "%.3f")

        if file_ext == "txt":
            df_out.to_csv(file_name, index=False, sep="\t", float_format=float_format)

        elif file_ext == "csv":
            df_out.to_csv(file_name, index=False, sep=",", float_format=float_format)

        elif file_ext == "xlsx":
            df_out.to_excel(file_name, index=False, float_format=float_format)

        else:
            if self.debug:
                print("Something went wrong during save out..")
            return
        if self.debug:
            print("Saved!")

    def load_settings(self):

        settings = QSettings()
        settings.beginGroup("user-settings")

        # map parameter keys to edits
        key_to_edit = {
            "dt": self.dt_spin,
            "time_unit": self.unit_edit,
            "float_format": None,
            "graphics_format": None,
            "data_format": None,
        }

        # load defaults from dict or restore values
        for key, value in default_par_dict.items():
            if key not in key_to_edit:
                continue
            val = settings.value(key, value)
            edit = key_to_edit[key]
            # some fields might be left empty for dynamic defaults
            if edit and (val is not None):
                edit.clear()  # to be on the safe side
                if isinstance(edit, (QSpinBox, QDoubleSpinBox)):
                    edit.setValue(float(val))
                else:
                    edit.insert(str(val))
