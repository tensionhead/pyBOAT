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
    QSizePolicy
)


SpinBox = QSpinBox | QDoubleSpinBox

from PyQt6.QtGui import QDoubleValidator, QRegularExpressionValidator
from PyQt6.QtCore import Qt, QSettings, QRegularExpression, QPoint
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

import pandas as pd

from pyboat.ui.util import (
    PandasModel,
    default_par_dict,
    selectFilter,
    spawn_warning_box,
    is_dark_color_scheme,
    StoreGeometry,
    create_spinbox,
    mk_spinbox_unit_slot,
)
from pyboat.ui.analysis import mkTimeSeriesCanvas, FourierAnalyzer, WaveletAnalyzer
from pyboat.ui.batch_process import BatchProcessWindow
from pyboat.ui import style
from pyboat.ui import analysis_parameters as ap

import pyboat
from pyboat import plotting as pl

# --- monkey patch label sizes to better fit the ui ---
pl.tick_label_size = 12
pl.label_size = 14

# same for all FileDialogs
FormatFilter = "csv ( *.csv);; MS Excel (*.xlsx);; Text File (*.txt)"


class DataViewer(StoreGeometry, QMainWindow):
    def __init__(self, data, pos_offset, parent, debug=False):
        StoreGeometry.__init__(self, pos=(80 + pos_offset, 300 + pos_offset), size=(900, 650))
        QMainWindow.__init__(self, parent=parent)

        # this is the data table
        self.df = data

        self.anaWindows = {}  # allows for multiple open analysis windows
        self.w_position = 0  # analysis window position offset

        self.debug = debug

        # this variable tracks the selected trajectory
        # -> DataFrame column name!
        self.signal_id = None  # no signal id initially selected

        self.raw_signal: np.ndarray | None = None  # no signal initial array
        self.dt: float | None = None  # gets initialized from the UI -> qset_dt
        self.tvec: np.ndarray | None = None  # gets initialized by vector_prep
        self.time_unit: str | None  = None  # gets initialized by qset_time_unit

        # get updated with dt in -> qset_dt
        self.periodV = QDoubleValidator(bottom=1e-16, top=1e16, decimals=2)

        self.initUI(pos_offset)

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

        self.cb_raw = QCheckBox("Raw Signal", self)
        self.cb_raw.setStatusTip("Plots the raw unfiltered signal")

        self.cb_trend = QCheckBox("Trend", self)
        self.cb_trend.setStatusTip("Plots the sinc filtered signal")

        self.cb_detrend = QCheckBox("Detrended Signal", self)
        self.cb_detrend.setStatusTip(
            "Plots the signal after trend subtraction (detrending)"
        )

        self.cb_envelope = QCheckBox("Envelope", self)
        self.cb_envelope.setStatusTip("Plots the estimated amplitude envelope")

        plotButton = QPushButton("Refresh Plot", self)
        plotButton.setStatusTip("Updates the plot with the new filter parameter values")
        plotButton.clicked.connect(self.doPlot)

        saveButton = QPushButton("Save Filter Results", self)
        saveButton.clicked.connect(self.save_out_trend)
        saveButton.setStatusTip("Writes the trend and the detrended signal into a file")

        ## checkbox layout
        plot_options_layout.addWidget(self.cb_raw, 0, 0)
        plot_options_layout.addWidget(self.cb_trend, 0, 1)
        plot_options_layout.addWidget(self.cb_detrend, 1, 0)
        plot_options_layout.addWidget(self.cb_envelope, 1, 1)
        plot_options_layout.addWidget(plotButton, 2, 0)
        plot_options_layout.addWidget(saveButton, 2, 1, 1, 1)
        plot_options_box.setLayout(plot_options_layout)

        ## checkbox signal set and change
        self.cb_raw.toggle()

        self.cb_raw.toggled.connect(self.toggle_raw)
        self.cb_trend.toggled.connect(self.toggle_trend)
        self.cb_detrend.toggled.connect(self.toggle_trend)
        self.cb_envelope.toggled.connect(self.doPlot)

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

        # fourier button
        fButton = QPushButton("Analyze Signal", self)
        if is_dark_color_scheme():
            fButton.setStyleSheet(f"background-color: {style.dark_primary}")
        else:
            fButton.setStyleSheet(f"background-color: {style.light_primary}")

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

        ## Create second tab
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

    def reanalyze_signal(self):
        print("REANNALYUZE")
        self.run_wavelet_ana()

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
        self.doPlot()

    # probably all the toggle state variables are not needed -> read out checkboxes directly
    def toggle_raw(self, checked: bool):
        if checked:
            self.plot_raw = True
            # untoggle the detrended cb
            self.cb_detrend.setChecked(False)
        else:
            self.plot_raw = False

        # signal selected?
        if self.signal_id:
            self.doPlot()

    def toggle_trend(self, checked: bool):

        # trying to plot the trend
        if checked:
            # don't plot raw and detrended together (trend is ok)
            if self.cb_detrend.isChecked():
                self.cb_raw.setChecked(False)

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

    def calc_trend(self):

        """ Uses maximal sinc window size """

        T_c = self.sinc_envelope.get_T_c()
        if not T_c:
            return
        if self.debug:
            print("Calculating trend with T_c = ", T_c)

        trend = pyboat.sinc_smooth(raw_signal=self.raw_signal,
                                   T_c=T_c,
                                   dt=self.dt)
        return trend

    def calc_envelope(self) -> np.ndarray | None:

        window_size = self.sinc_envelope.get_wsize()
        if not window_size:
            return None
        if self.debug:
            print("Calculating envelope with window_size = ", window_size)

        # cut of frequency set and detrended plot activated?
        if self.cb_detrend.isChecked():
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

        if self.debug:
            print(
                "called Plotting [raw] [trend] [detrended] [envelope]",
                self.cb_raw.isChecked(),
                self.cb_trend.isChecked(),
                self.cb_detrend.isChecked(),
                self.cb_envelope.isChecked(),
            )

        # check if trend is needed
        if self.cb_trend.isChecked() or self.cb_detrend.isChecked():
            trend = self.calc_trend()
        else:
            trend = None

        # envelope calculation
        if self.cb_envelope.isChecked():
            envelope = self.calc_envelope()
            if envelope is None:
                return
        else:
            envelope = None

        self.tsCanvas.fig1.clf()

        ax1 = pl.mk_signal_ax(self.time_unit, fig=self.tsCanvas.fig1)
        self.tsCanvas.fig1.add_axes(ax1)

        if self.debug:
            print(
                f"plotting signal and trend with {self.tvec[:10]}, {self.raw_signal[:10]}"
            )

        if self.cb_raw.isChecked():
            pl.draw_signal(ax1, time_vector=self.tvec, signal=self.raw_signal)

        if trend is not None and self.cb_trend.isChecked():
            pl.draw_trend(ax1, time_vector=self.tvec, trend=trend)

        if trend is not None and self.cb_detrend.isChecked():
            ax2 = pl.draw_detrended(
                ax1, time_vector=self.tvec, detrended=self.raw_signal - trend
            )
            ax2.legend(fontsize=pl.tick_label_size, loc="lower left")
        if envelope is not None and not self.cb_detrend.isChecked():
            pl.draw_envelope(ax1, time_vector=self.tvec, envelope=envelope)

        # plot on detrended axis
        if envelope is not None and self.cb_detrend.isChecked():
            pl.draw_envelope(ax2, time_vector=self.tvec, envelope=envelope)
            ax2.legend(fontsize=pl.tick_label_size)

        self.tsCanvas.fig1.subplots_adjust(bottom=0.15, left=0.15, right=0.85)

        # add a simple legend
        ax1.legend(fontsize=pl.tick_label_size)

        self.tsCanvas.draw()
        self.tsCanvas.show()

    def run_wavelet_ana(self):
        """ run the Wavelet Analysis """

        if not np.any(self.raw_signal):

            msgBox = spawn_warning_box(self, "No Signal", "Please select a signal first!")
            msgBox.exec()

            return False

        wlet_pars = self.wavelet_tab.assemble_wlet_pars()

        if self.sinc_envelope.get_T_c():
            trend = self.calc_trend()
            signal = self.raw_signal - trend
        else:
            signal = self.raw_signal

        if self.sinc_envelope.get_wsize():
            window_size = self.sinc_envelope.get_wsize()
            signal = pyboat.normalize_with_envelope(signal, window_size, dt=self.dt)

        self.w_position += 30

        self.anaWindows[self.w_position] = WaveletAnalyzer(
            signal=signal,
            dt=self.dt,
            Tmin=wlet_pars["Tmin"],
            Tmax=wlet_pars["Tmax"],
            pow_max=wlet_pars["pow_max"],
            step_num=wlet_pars["nT"],
            position=self.w_position,
            signal_id=self.signal_id,
            time_unit=self.time_unit,
            DEBUG=self.debug,
            parent=self,
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

        if self.sinc_envelope.detrend:
            trend = self.calc_trend()
            signal = self.raw_signal - trend
        else:
            signal = self.raw_signal

        if self.sinc_envelope.normalize:
            window_size = self.sinc_envelope.get_wsize()
            signal = pyboat.normalize_with_envelope(signal, window_size, dt=self.dt)

        # shift new analyser windows
        self.w_position += 20


        # periods or frequencies?
        if self.cb_FourierT.isChecked():
            show_T = False
        else:
            show_T = True

        self.anaWindows[self.w_position] = FourierAnalyzer(
            signal=signal,
            dt=self.dt,
            signal_id=self.signal_id,
            position=self.w_position,
            time_unit=self.time_unit,
            show_T=show_T,
            parent=self,
        )

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
