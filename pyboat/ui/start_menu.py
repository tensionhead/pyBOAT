#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
from PyQt6.QtWidgets import (
    QCheckBox,
    QMainWindow,
    QFileDialog,
    QApplication,
    QLabel,
    QLineEdit,
    QPushButton,
    QMessageBox,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QGridLayout,
    QComboBox,
)
from PyQt6.QtGui import QDesktopServices, QAction
from PyQt6.QtCore import QUrl, QSettings, QRegularExpression, Qt
from PyQt6.QtGui import QDoubleValidator, QRegularExpressionValidator


from pyboat.ui import util
from pyboat.ui.data_viewer import DataViewer
from pyboat.ui.synth_gen import SynthSignalGen
from pyboat import __version__

# matplotlib settings
from matplotlib import rc

rc("text", usetex=False)  # better for the UI


doc_url = "https://github.com/tensionhead/pyBOAT/blob/master/README.md"
gitter_url = "https://gitter.im/pyBOATbase/support"


class MainWindow(util.StoreGeometry, QMainWindow):
    def __init__(self, debug):

        util.StoreGeometry.__init__(self, pos=(80, 100), size=(200, 50))
        QMainWindow.__init__(self, parent=None)
        
        self.debug = debug

        self._convert_settings()

        self.nViewers = 0
        self.DataViewers = {}  # no Viewers yet
        self.initUI()

    def initUI(self):

        self.setFixedSize(300, 180)
        self.restore_geometry()
        self.setWindowTitle(f"pyBOAT {__version__}")

        # Actions for the menu - status bar
        main_widget = QWidget()
        # online help in lower left corner
        self.statusBar()

        quitAction = QAction("&Quit", self)
        quitAction.setShortcut("Ctrl+Q")
        quitAction.setStatusTip("Quit pyBOAT")
        quitAction.triggered.connect(self.close_application)

        openFile = QAction("&Load data", self)
        openFile.setShortcut("Ctrl+L")
        openFile.setStatusTip("Load data")
        openFile.triggered.connect(self.Load_and_init_Viewer)

        ImportMenu = QAction("&Import..", self)
        ImportMenu.setShortcut("Ctrl+I")
        ImportMenu.setStatusTip("Set import options")
        ImportMenu.triggered.connect(self.init_import_menu)

        go_to_settings = QAction("&Default Parameters", self)
        go_to_settings.setStatusTip("Change default analysis parameters")
        go_to_settings.triggered.connect(self.open_settings_menu)

        go_to_doc = QAction("&Documentation..", self)
        go_to_doc.triggered.connect(self.open_doc_link)

        go_to_gitter = QAction("&Online support..", self)
        go_to_gitter.triggered.connect(self.open_gitter_link)

        # --- the menu bar ---

        mainMenu = self.menuBar()
        mainMenu.setNativeMenuBar(False)

        fileMenu = mainMenu.addMenu("&File")
        fileMenu.addAction(openFile)
        fileMenu.addAction(ImportMenu)
        fileMenu.addAction(quitAction)

        settingsMenu = mainMenu.addMenu("&Settings")
        settingsMenu.addAction(go_to_settings)

        helpMenu = mainMenu.addMenu("&Help")
        helpMenu.addAction(go_to_doc)
        helpMenu.addAction(go_to_gitter)

        # --- Import Data ---

        load_box = QGroupBox("Import Data")
        load_box_layout = QVBoxLayout()

        openFileButton = QPushButton("Open", self)
        openFileButton.setStatusTip(
            "Load a data table with header"
        )
        if util.get_color_scheme() != Qt.ColorScheme.Light:
            openFileButton.setStyleSheet("background-color: darkblue")
        else:
            openFileButton.setStyleSheet("background-color: lightblue")

        openFileButton.clicked.connect(self.Load_and_init_Viewer)
        load_box_layout.addWidget(openFileButton)

        ImportButton = QPushButton("Import..", self)
        ImportButton.setStatusTip("Set import options")
        ImportButton.clicked.connect(self.init_import_menu)
        load_box_layout.addWidget(ImportButton)

        load_box.setLayout(load_box_layout)

        # --- SSG ---

        synsig_box = QGroupBox("Synthetic Signals")
        synsig_box_layout = QVBoxLayout()

        synsigButton = QPushButton("Start Generator", self)
        if util.get_color_scheme() != Qt.ColorScheme.Light:
            synsigButton.setStyleSheet("background-color: darkred")
        else:
            synsigButton.setStyleSheet("background-color: orange")
        synsigButton.setStatusTip("Start the synthetic signal generator")
        synsigButton.clicked.connect(self.init_synsig_generator)

        # quitButton = QPushButton("Quit", self)
        # quitButton.clicked.connect(self.close_application)
        # quitButton.setMaximumWidth(50)

        synsig_box_layout.addStretch(1)
        synsig_box_layout.addWidget(synsigButton)
        synsig_box_layout.addStretch(1)
        synsig_box.setLayout(synsig_box_layout)

        # --- Main Layout ---

        main_layout = QHBoxLayout()
        main_layout.addWidget(load_box)
        main_layout.addWidget(synsig_box)

        # main_layout.addWidget(quitButton,1,1)

        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        self.show()

    def init_synsig_generator(self):

        if self.debug:
            print("function init_synsig_generator called..")

        self.ssg = SynthSignalGen(parent=self, debug=self.debug)

    def close_application(self):

        # no confirmation window
        if self.debug:
            appc = QApplication.instance()
            appc.closeAllWindows()
            return

        choice = QMessageBox.question(
            self,
            "Quitting",
            "Do you want to exit pyBOAT?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if choice == QMessageBox.StandardButton.Yes:
            print("Quitting ...")
            # sys.exit()
            appc = QApplication.instance()
            appc.closeAllWindows()
        else:
            pass

    def Load_and_init_Viewer(self):

        if self.debug:
            print("function Viewer_Ini called")

        # retrieve or initialize directory path
        settings = QSettings()
        dir_path = settings.value("dir_name", os.path.curdir)

        # load a table directly
        df, err_msg, dir_path = util.load_data(dir_path, self.debug)

        if err_msg == "cancelled":
            return
        else:
            # save the last directory
            settings.setValue("dir_name", dir_path)

        if err_msg:

            msgBox = QMessageBox()
            msgBox.setWindowTitle("Data Import Error")
            msgBox.setIcon(QMessageBox.Critical)
            msgBox.setText(err_msg)
            msgBox.setDetailedText(
                """Have a look at the example data directory at\n
                github.com/tensionhead/pyBOAT"""
            )
            msgBox.exec()
            return

        self.nViewers += 1
        # initialize new DataViewer with the loaded data
        self.DataViewers[self.nViewers] = DataViewer(
            data=df, pos_offset=self.nViewers * 20, debug=self.debug, parent=self
        )

    def init_import_menu(self):

        self.imp_menu = ImportMenu(self, debug=self.debug)

    def open_settings_menu(self):

        self.settings_menu = SettingsMenu(self, debug=self.debug)

    def open_doc_link(self):

        QDesktopServices.openUrl(QUrl(doc_url))

    def open_gitter_link(self):

        QDesktopServices.openUrl(QUrl(gitter_url))

    def _convert_settings(self):
        """Convert QSettings from < 1.0"""
        old_settings = {}
        settings = QSettings()
        top_level = 'dir_name'
        # pre 1.0 did not have any top level groups
        if not settings.childGroups():
            # get existing settings
            for key in settings.allKeys():
                # remains top level key
                if key in top_level:
                    continue
                old_settings[key] = settings.value(key, util.default_par_dict[key])
                settings.remove(key)

        # create sub settings
        settings.beginGroup("default-settings")
        for key, value in old_settings.items():
            settings.setValue(key, value)
        settings.endGroup()

class ImportMenu(QMainWindow):
    def __init__(self, parent, debug=False):
        super().__init__(parent=parent)

        self.parent = parent
        self.debug = debug
        self.DataViewers = {}  # no Viewers yet

        self.initUI()

    def initUI(self):

        self.setWindowTitle("Data Import Options")
        self.setGeometry(120, 150, 150, 50)

        config_w = QWidget()
        config_grid = QGridLayout()
        config_w.setLayout(config_grid)

        self.cb_use_ext = QCheckBox("Separator from extension")
        self.cb_use_ext.toggle()
        self.cb_use_ext.setStatusTip(
            "Infer the column separator from the file extension like '.csv'"
        )
        self.cb_use_ext.stateChanged.connect(self.toggle_ext)
        self.sep_label = QLabel("Column separator")
        self.sep_label.setDisabled(True)
        self.sep_edit = QLineEdit()
        self.sep_edit.setDisabled(True)
        self.sep_edit.setStatusTip("Leave empty for automatic detection")

        self.cb_header = QCheckBox("No header row present", self)
        self.cb_header.setStatusTip("Assigns a numeric sequence as column names")
        self.cb_header.setChecked(False)

        self.cb_NaN = QCheckBox("Interpolate missing values")
        self.cb_NaN.setStatusTip("Linear interpolate in between missing values")

        NaN_label = QLabel("Set missing values character")
        self.NaN_edit = QLineEdit()
        tt = """
        The following values are interpreted as missing values per default:
        ‘NA’, ‘NULL’, ‘NaN’, ‘n/a’, ‘nan’, ‘null’
        ‘#N/A’, ‘#N/A N/A’, ‘#NA’, ‘-1.#IND’, ‘-1.#QNAN’,
        ‘-NaN’, ‘-nan’, ‘1.#IND’, ‘1.#QNAN’, ‘<NA>’, ‘N/A’,
        """
        self.NaN_edit.setToolTip(tt)
        self.NaN_edit.setStatusTip("Enter custom NaN symbol if needed, e.g. '#N'")

        config_grid.addWidget(self.cb_use_ext, 0, 0, 1, 2)
        config_grid.addWidget(self.sep_label, 1, 0)
        config_grid.addWidget(self.sep_edit, 1, 1)
        config_grid.addWidget(self.cb_header, 2, 0, 1, 2)
        config_grid.addWidget(self.cb_NaN, 3, 0, 1, 2)
        config_grid.addWidget(NaN_label, 4, 0)
        config_grid.addWidget(self.NaN_edit, 4, 1)

        OpenButton = QPushButton("Open", self)
        if util.get_color_scheme() != Qt.ColorScheme.Light:
            OpenButton.setStyleSheet("background-color: darkblue")
        else:
            OpenButton.setStyleSheet("background-color: lightblue")
        OpenButton.setStatusTip("Select an input file with the chosen settings..")
        OpenButton.clicked.connect(self.import_file)

        button_box = QHBoxLayout()
        button_box.addStretch(1)
        button_box.addWidget(OpenButton)
        button_box.addStretch(1)
        button_w = QWidget()
        button_w.setLayout(button_box)

        main_layout = QVBoxLayout()
        main_layout.addWidget(config_w)
        main_layout.addWidget(button_w)

        # set main layout
        main_frame = QWidget()
        main_frame.setLayout(main_layout)
        self.setCentralWidget(main_frame)
        self.statusBar()

        self.show()

    def toggle_ext(self):

        self.sep_label.setDisabled(self.cb_use_ext.isChecked())
        self.sep_edit.setDisabled(self.cb_use_ext.isChecked())

    def import_file(self):

        """
        Reads the values from the config grid
        and prepares kwargs for the pandas read_... functions

        NaN interpolation is done here if requested
        """

        kwargs = {}

        if self.cb_header.isChecked():
            header = None  # pandas will assign numeric column names
        else:
            header = "infer"

        if not self.cb_use_ext.isChecked():

            sep = self.sep_edit.text()
            # empty field
            if sep == "":
                sep = None
            kwargs["sep"] = sep
            if self.debug:
                print(f"Separator is {sep}, with type {type(sep)}")

        NaN_value = self.NaN_edit.text()
        # empty field
        if NaN_value == "":
            NaN_value = None
        if self.debug:
            print(f"NaN value is {NaN_value}, with type {type(NaN_value)}")

        # assemble key-words for pandas read_... functions
        kwargs["header"] = header
        kwargs["na_values"] = NaN_value

        if self.debug:
            print(f"kwargs for load_data: {kwargs}")

        # retrieve or initialize directory path
        settings = QSettings()
        dir_path = settings.value("dir_name", os.path.curdir)

        # -----------------------------------------------------
        df, err_msg, dir_path = util.load_data(dir_path, debug=self.debug, **kwargs)

        if err_msg != "cancelled":
            # save the last directory
            settings.setValue("dir_name", dir_path)
        elif err_msg == "cancelled":
            return

        if err_msg:
            msgBox = QMessageBox()
            msgBox.setWindowTitle("Data Import Error")
            msgBox.setIcon(QMessageBox.Critical)
            msgBox.setText(err_msg)
            msgBox.setDetailedText(
                """Have a look at the example data directory at\n
                github.com/tensionhead/pyBOAT"""
            )
            msgBox.exec()
            return
        # -----------------------------------------------------

        if self.cb_NaN.isChecked():

            N_NaNs = df.isna().to_numpy().sum()
            if N_NaNs > 0:
                msg = (f"Found {N_NaNs} missing values (NaNs) in total, "
                       "including leading and trailing NaNs.\n"
                       "linearly interpolating through intermittent NaNs..")
                msgBox = QMessageBox(parent=self)
                msgBox.setIcon(QMessageBox.Information)
                msgBox.setWindowTitle("NaNs detected")
                msgBox.setText(msg)
                msgBox.exec()

            else:
                msgBox = QMessageBox(parent=self)
                msgBox.setWindowTitle("NaN Interpolation")
                msgBox.setText("No missing values found!")
                msgBox.exec()

            name = df.name
            df = util.interpol_NaNs(df)
            df.name = name  # restore name

        # initialize new DataViewer with the loaded data
        self.parent.nViewers += 1
        self.parent.DataViewers[self.parent.nViewers] = DataViewer(
            data=df, pos_offset=self.parent.nViewers * 20, debug=self.debug, parent=self
        )

        self.close()


class SettingsMenu(QMainWindow):
    def __init__(self, parent, debug=False):
        super().__init__(parent=parent)

        self.parent = parent
        self.debug = debug
        self.DataViewers = {}  # no Viewers yet

        self.IniFormatFilter = ".ini ( *.ini)"
        self.initUI()

    def initUI(self):

        self.setWindowTitle("Change Default Parameters")
        self.setGeometry(400, 100, 550, 50)

        config_w = QWidget()
        config_grid = QGridLayout()
        config_w.setLayout(config_grid)

        dt_label = QLabel("Sampling Interval")
        self.dt_edit = QLineEdit()
        self.dt_edit.setValidator(QDoubleValidator(0, 99999, 3))
        self.dt_edit.setStatusTip("How much time in between two samples?")
        time_unit_label = QLabel("Time Unit")
        self.time_unit_edit = QLineEdit()
        self.time_unit_edit.setStatusTip("Sets the time unit label")

        cut_off_label = QLabel("Cut-off Period")
        self.cut_off_edit = QLineEdit()
        self.cut_off_edit.setValidator(QDoubleValidator(0, 99999, 3))
        tt = "Sinc filter, leave blank for pyBOATs dynamic defaults.."
        self.cut_off_edit.setStatusTip(tt)

        wsize_label = QLabel("Window Size")
        self.wsize_edit = QLineEdit()
        self.wsize_edit.setValidator(QDoubleValidator(0, 99999, 3))
        tt = "Amplitude normalization, leave blank for pyBOATs dynamic defaults.."
        self.wsize_edit.setStatusTip(tt)

        Tmin_label = QLabel("Lowest Period")
        self.Tmin_edit = QLineEdit()
        self.Tmin_edit.setValidator(QDoubleValidator(0, 99999, 3))
        self.Tmin_edit.setStatusTip("Leave blank for pyBOATs dynamic defaults..")
        Tmax_label = QLabel("Highest Period")
        self.Tmax_edit = QLineEdit()
        self.Tmax_edit.setValidator(QDoubleValidator(0, 99999, 3))
        self.Tmax_edit.setStatusTip("Leave blank for pyBOATs dynamic defaults..")
        nT_label = QLabel("Number of Periods")
        self.nT_edit = QLineEdit()
        self.nT_edit.setValidator(QRegularExpressionValidator(QRegularExpression("[0-9]+")))
        self.nT_edit.setStatusTip("Spectral resolution on the period axis")

        pow_max_label = QLabel("Maximal Power")
        self.pow_max_edit = QLineEdit()
        self.pow_max_edit.setValidator(QDoubleValidator(0, 99999, 3))
        self.pow_max_edit.setStatusTip(
            "Scales the wavelet colormap, leave blank for pyBOATs dynamic defaults.."
        )

        # 1st column

        config_grid.addWidget(dt_label, 0, 0)
        config_grid.addWidget(self.dt_edit, 0, 1)

        config_grid.addWidget(time_unit_label, 1, 0)
        config_grid.addWidget(self.time_unit_edit, 1, 1)

        config_grid.addWidget(cut_off_label, 2, 0)
        config_grid.addWidget(self.cut_off_edit, 2, 1)

        config_grid.addWidget(wsize_label, 3, 0)
        config_grid.addWidget(self.wsize_edit, 3, 1)

        # 2nd column

        config_grid.addWidget(Tmin_label, 0, 2)
        config_grid.addWidget(self.Tmin_edit, 0, 3)

        config_grid.addWidget(Tmax_label, 1, 2)
        config_grid.addWidget(self.Tmax_edit, 1, 3)

        config_grid.addWidget(nT_label, 2, 2)
        config_grid.addWidget(self.nT_edit, 2, 3)

        config_grid.addWidget(pow_max_label, 3, 2)
        config_grid.addWidget(self.pow_max_edit, 3, 3)

        CloseButton = QPushButton("Close", self)
        CloseButton.setStatusTip("Discards changes")
        CloseButton.clicked.connect(self.clicked_close)

        RevertButton = QPushButton("Clear", self)
        if util.get_color_scheme() != Qt.ColorScheme.Light:
            RevertButton.setStyleSheet("background-color: darkred")
        else:
            RevertButton.setStyleSheet("background-color: red")
        RevertButton.setStatusTip("Revert to dynamic defaults")
        RevertButton.clicked.connect(self.clicked_clear)

        LoadButton = QPushButton("Load", self)
        LoadButton.setStatusTip("Load from .ini file")
        LoadButton.clicked.connect(self.clicked_load)

        SaveButton = QPushButton("Save", self)
        SaveButton.setStatusTip("Save to custom .ini file")
        SaveButton.clicked.connect(self.clicked_save)

        OkButton = QPushButton("Set", self)
        OkButton.setStatusTip("Approve changes")
        if util.get_color_scheme() != Qt.ColorScheme.Light:
            OkButton.setStyleSheet("background-color: darkblue")
        else:
            OkButton.setStyleSheet("background-color: lightblue")
        OkButton.clicked.connect(self.clicked_set)

        button_box = QHBoxLayout()
        button_box.addWidget(RevertButton)
        button_box.addWidget(CloseButton)
        button_box.addStretch(1)
        button_box.addWidget(LoadButton)
        button_box.addWidget(SaveButton)
        button_box.addStretch(1)
        button_box.addWidget(OkButton)
        button_w = QWidget()
        button_w.setLayout(button_box)

        config_box = QGroupBox("Analysis")
        # no extra status bar
        # config_box.setStatusTip(
        # "Clear numeric boxes to revert to pyBOAT's dynamic defaults")

        config_box_layout = QVBoxLayout()
        config_box_layout.addWidget(config_w)
        config_box.setLayout(config_box_layout)

        # fmt options box

        fmt_label = QLabel("Number Format")
        self.fmt_dropdown = QComboBox()
        self.fmt_dropdown.setStatusTip("Use scientific for very large or small numbers")
        self.fmt_dropdown.addItem("Decimal")
        self.fmt_dropdown.addItem("Scientific")

        # self.fmt_dropdown.activated[str].connect(self.fmt_choice)

        graphics_label = QLabel("Graphics")
        self.graphics_dropdown = QComboBox()
        self.graphics_dropdown.setStatusTip("Graphics format for the batch processing")
        self.graphics_dropdown.addItem("png")
        self.graphics_dropdown.addItem("pdf")
        self.graphics_dropdown.addItem("svg")
        self.graphics_dropdown.addItem("jpg")

        data_label = QLabel("Data")
        self.data_dropdown = QComboBox()
        self.data_dropdown.setStatusTip("File format of the various tabular outputs")
        self.data_dropdown.addItem("csv")
        self.data_dropdown.addItem("xlsx")
        self.data_dropdown.addItem("txt")

        output_box = QGroupBox("Output")
        output_box_layout = QHBoxLayout()
        output_box_layout.addWidget(fmt_label)
        output_box_layout.addWidget(self.fmt_dropdown)
        output_box_layout.addStretch(1)
        output_box_layout.addWidget(graphics_label)
        output_box_layout.addWidget(self.graphics_dropdown)
        output_box_layout.addWidget(data_label)
        output_box_layout.addWidget(self.data_dropdown)

        output_box_layout.addStretch(5)
        output_box.setLayout(output_box_layout)

        # map parameter keys to edits
        self.key_to_edit = {
            "dt": self.dt_edit,
            "time_unit": self.time_unit_edit,
            "cut_off": self.cut_off_edit,
            "window_size": self.wsize_edit,
            "Tmin": self.Tmin_edit,
            "Tmax": self.Tmax_edit,
            "nT": self.nT_edit,
            "pow_max": self.pow_max_edit,
            "float_format": None,
            "graphics_format": None,
            "data_format": None,
        }

        # load parameters
        self.load_settings()

        # set main layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(config_box)
        main_layout.addWidget(output_box)
        main_layout.addWidget(button_w)

        main_frame = QWidget()
        main_frame.setLayout(main_layout)

        self.setCentralWidget(main_frame)

        self.statusBar()
        self.show()

    def clicked_set(self):

        choice = QMessageBox.question(
            self,
            "Apply settings ",
            f"Set current settings as new defaults?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if choice == QMessageBox.StandardButton.Yes:
            # write global settings
            self.set_or_save(path=None)
            self.close()
        else:
            return

    def set_or_save(self, path=None):

        """
        Retrieves all input fields
        and saves to QSettings
        """

        # create new global settings
        if path is None:
            settings = QSettings()
        # write to custom file
        else:
            settings = QSettings(path, QSettings.Format.IniFormat)
        settings.beginGroup("default-settings")
        for key, edit in self.key_to_edit.items():

            # setting key has no edit
            if not edit:
                continue

            if key == "time_unit":
                value = edit.text()  # the only string parameter
                settings.setValue(key, value)
                continue
            if key == "nT":
                value = int(edit.text())  # the only integer parameter
                settings.setValue(key, value)
                continue

            value = util.retrieve_double_edit(edit)
            # None is also fine!
            settings.setValue(key, value)

        # the output settings are strings
        if self.fmt_dropdown.currentText() == "Decimal":
            settings.setValue("float_format", "%.3f")
        elif self.fmt_dropdown.currentText() == "Scientific":
            settings.setValue("float_format", "%e")

        # we can take the items directly
        settings.setValue("graphics_format", self.graphics_dropdown.currentText())

        # we can take the items directly
        settings.setValue("data_format", self.data_dropdown.currentText())

        if self.debug:
            for key in settings.allKeys():
                print(f"Set: {key} -> {settings.value(key)}")

    def clicked_close(self):

        choice = QMessageBox.question(
            self,
            "Close settings",
            "Exit settings without saving?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if choice == QMessageBox.StandardButton.Yes:
            self.close()
        else:
            return

    def clicked_load(self):

        # retrieve or initialize directory path
        settings = QSettings()
        dir_path = settings.value("dir_name", os.path.curdir)

        file_name = QFileDialog.getOpenFileName(
            parent=self, caption="Load Settings from File", directory=dir_path,
            filter=self.IniFormatFilter
        )
        # dialog cancelled
        if not file_name:
            return

        self.load_settings(path=file_name[0])

    def clicked_save(self):

        # retrieve or initialize directory path
        settings = QSettings()
        dir_path = settings.value("dir_name", os.path.curdir)

        dialog = QFileDialog()
        file_name, sel_filter = dialog.getSaveFileName(
            self, "Save settings as", os.path.join(dir_path, "pyboat.ini"), self.IniFormatFilter)

        # dialog cancelled
        if not file_name:
            return

        self.set_or_save(path=file_name)

    def load_settings(self, path=None):

        # load (can be empty) system stored settings
        if path is None:
            settings = QSettings()
        # load from custom location
        else:
            settings = QSettings(path, QSettings.Format.IniFormat)

        settings.beginGroup("default-settings")
        # load defaults from dict or restore values
        for key, value in util.default_par_dict.items():
            val = settings.value(key, value)
            edit = self.key_to_edit[key]
            # some fields left empty for dynamic defaults
            if edit and (val is not None):
                edit.clear()
                edit.insert(str(val))
            elif edit and val is None:
                edit.clear()

        # load combo box defaults, only works via setting the index :/
        default = util.default_par_dict["float_format"]
        float_format = settings.value("float_format", default)
        map_to_ind = {default: 0, "%e": 1}
        self.fmt_dropdown.setCurrentIndex(map_to_ind[float_format])

        default = util.default_par_dict["graphics_format"]
        graphics_format = settings.value("graphics_format", default)
        map_to_ind = {default: 0, "pdf": 1, "svg": 2, "jpg": 3}
        self.graphics_dropdown.setCurrentIndex(map_to_ind[graphics_format])

        default = util.default_par_dict["data_format"]
        data_format = settings.value("data_format", default)
        map_to_ind = {default: 0, "xlsx": 1, "txt": 2}
        self.data_dropdown.setCurrentIndex(map_to_ind[data_format])

    def clicked_clear(self):

        settings = QSettings()

        choice = QMessageBox.question(
            self,
            "Revert settings",
            f"Revert to dynamic (empty) defaults?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if choice == QMessageBox.StandardButton.No:
            return

        # load defaults from dict
        for key, value in util.default_par_dict.items():
            if self.debug:
                print(f"Set: {key} -> {settings.value(key)}")
            settings.setValue(key, value)

        # to update the display
        self.load_settings()
