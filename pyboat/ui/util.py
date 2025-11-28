from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
from typing import TYPE_CHECKING, Callable
from dataclasses import dataclass, asdict

from PySide6.QtWidgets import (
    QFileDialog,
    QMessageBox,
    QLabel,
    QPushButton,
    QSizePolicy,
    QWidget,
    QGridLayout,
    QSpinBox,
    QDoubleSpinBox
)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PySide6.QtCore import Qt, QAbstractTableModel, QSettings, QPoint, QSize
from PySide6.QtGui import QDoubleValidator, QIntValidator, QGuiApplication

from pyboat.core import interpolate_NaNs, sinc_smooth, normalize_with_envelope
if TYPE_CHECKING:
    from PySide6.QtWidgets import QAbstractSpinBox
    from pandas import DataFrame
    from .data_viewer import DataViewer

# map data ouput format to QFileDialog Filter
selectFilter = {
    "csv": "csv ( *.csv)",
    "xlsx": "MS Excel (*.xlsx)",
    "txt": "Text File (*.txt)",
}


@dataclass
class WAnalyzerParams:
    dt: float
    raw_signal: np.ndarray
    periods: np.ndarray

    # could be removed?!
    max_power: float | None = None

    T_c: float | None = None
    window_size: float | None = None

    @property
    def tvec(self) -> np.ndarray:
        return np.arange(0, len(self.raw_signal)) * self.dt

    @property
    def filtered_signal(self) -> np.ndarray:
        """
        Returns preprocessed signal: detrended and amplitude normalized
        if respective parameters `T_c` and `wsize` are given
        """
        fsignal = self.raw_signal
        # first detrend
        if self.T_c is not None:
            fsignal = fsignal - sinc_smooth(
                raw_signal=self.raw_signal, T_c=self.T_c, dt=self.dt
            )
        # normalize amplitude
        if self.window_size is not None:
            fsignal = normalize_with_envelope(fsignal, self.window_size, self.dt)

        return fsignal

    def asdict(self) -> dict:
        return asdict(self)


def spawn_warning_box(parent: QWidget, title: str, text: str) -> QMessageBox:

    msgBox = QMessageBox(parent=parent)
    msgBox.setWindowTitle(title)
    msgBox.setIcon(QMessageBox.Icon.Warning)
    msgBox.setText(text)

    return msgBox


class MessageWindow(QWidget):

    """
    A generic window do display a message
    and an Ok button to close.. better to use QMessage box!
    """

    def __init__(self, message, title):
        super().__init__()
        self.message = message
        self.title = title
        self.initUI()

    def initUI(self):
        error = QLabel(self.message)
        self.setGeometry(300, 450, 220, 100)  # x,y, xlen, ylen
        self.setWindowTitle(self.title)
        okButton = QPushButton("OK", self)
        okButton.clicked.connect(self.close)
        main_layout_v = QGridLayout()
        main_layout_v.addWidget(error, 0, 0, 1, 3)
        main_layout_v.addWidget(okButton, 1, 1)
        self.setLayout(main_layout_v)
        self.show()


class mkGenericCanvas(FigureCanvas):
    def __init__(self):
        self.fig, self.ax = plt.subplots(1, 1)

        FigureCanvas.__init__(self, self.fig)

        FigureCanvas.setSizePolicy(self, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        FigureCanvas.updateGeometry(self)


def load_data(dir_path="./", debug=False, **kwargs):

    """
    This is the entry point to import the data

    **kwargs: keyword arguments for pandas.read_... functions
    to control things like *header*, *sep* and so on.

    If no 'sep' is present in kwargs, infer column separator from
    default extensions:

    'csv' : ','
    'tsv' : '\t'
    'txt' : '\s+'
    ['xlsx', 'xls'] : use pandas read_excel

    any other extension calls Python's csv.Sniffer in good faith ;)

    """

    err_msg1 = "Non-numeric values encountered in\n"
    err_msg2 = "Parsing errors encountered in\n"
    err_suffix = "\ncheck input file..!"

    # returns a list with 1 element, stand alone File Dialog
    file_names = QFileDialog.getOpenFileName(
        parent=None, caption="Open Tabular Data", directory=dir_path
    )

    if debug:
        print(file_names, type(file_names))

    file_name = file_names[0]
    file_ext = file_name.split(".")[-1]

    # check if cancelled -> Null string
    if file_name == "":
        return None, "cancelled", None

    print("Loading", file_ext, file_name)
    new_dir_path = os.path.dirname(file_name)
    # check if it's an excel file:
    if file_ext in ["xls", "xlsx"]:

        # not needed for reading excel sheets
        if "sep" in kwargs:
            del kwargs["sep"]

        try:
            raw_df = pd.read_excel(file_name, **kwargs)
            san_df = sanitize_df(raw_df, debug)
            if san_df is None:
                print("Error loading data..")
                return None, f"{err_msg1}{file_name}{err_suffix}", new_dir_path
            else:
                # attach a name for later reference in the DataViewer
                # strip off extension
                bname = os.path.basename(file_name)
                san_df.name = "".join(bname.split(".")[:-1])
                return san_df, "", os.path.dirname(file_name)  # empty error msg

        except (pd.errors.ParserError, csv.Error):
            return None, f"{err_msg2}{file_name}{err_suffix}", new_dir_path

    # infer table type from extension
    if "sep" not in kwargs:

        if file_ext == "csv":
            delimiter = ","
        elif file_ext == "tsv":
            delimiter = "\t"
        elif file_ext == "txt":
            delimiter = "\s+"
        else:
            # calls Python's inbuilt csv.Sniffer
            delimiter = None
            kwargs["engine"] = "python"

        kwargs["sep"] = delimiter

        if debug:
            print(f"Infered separator from extension: {delimiter}")
            print(f"Loading with kwargs: {kwargs}")
    else:
        if debug:
            print(f"Separator was given as: '{kwargs['sep']}'")

    try:
        raw_df = pd.read_table(file_name, **kwargs)
        if raw_df is None:
            print("Error loading data..")
            return None, f"{err_msg1}{file_name}{err_suffix}", new_dir_path

        # check that we have a standard RangeIndex,
        # things like MultiIndex point to a faulty header
        # (to few/to many columns)
        if debug:
            print("df columns:", raw_df.columns)
            print("df shape:", raw_df.shape)
            print("got df index:", type(raw_df.index))
        if isinstance(raw_df.index, pd.core.indexes.multi.MultiIndex):
            msg = (f"Can not parse table header in\n{file_name}\n"
                   "check number of columns vs. available column names!")
            return None, msg, new_dir_path

        san_df = sanitize_df(raw_df, debug)
        if san_df is None:
            print("Error sanitizing data..")
            return None, f"{err_msg1}{file_name}{err_suffix}", new_dir_path

        else:
            # attach a name for later reference in the DataViewer
            bname = os.path.basename(file_name)
            san_df.name = "".join(bname.split(".")[:-1])
            return san_df, "", new_dir_path  # empty error msg

    except pd.errors.ParserError:
        return None, f"{err_msg2}{file_name}{err_suffix}", new_dir_path


def sanitize_df(raw_df, debug=False):

    """
    Makes sure the DataFrame is in a form to
    work as a 'PandasModel', see class below
    """

    # catch wrongly parsed input data, we want all numeric!
    for dtype in raw_df.dtypes:
        if dtype == object:
            return None

    # map all columns to strings for the PandasModel and SignalBox
    str_col = map(str, raw_df.columns)
    raw_df.columns = str_col

    if debug:
        print("raw columns:", raw_df.columns)

    if not isinstance(raw_df.index, pd.core.indexes.range.RangeIndex):
        msgBox = QMessageBox()
        msgBox.setWindowTitle("Data Import Warning")
        msgBox.setIcon(QMessageBox.Icon.Warning)
        msg = ("Found one additional column which will be ignored.\n"
               "pyBOAT creates its own time axis on the fly\n"
               "by setting the `Sampling Interval`!")
        msgBox.setText(msg)
        msgBox.exec()

        # revert to default integer range index
        raw_df.reset_index(drop=True, inplace=True)

    # return data
    return raw_df


def interpol_NaNs(df):

    """
    Calls the NaN interpolator for each column
    of the DataFrame, only interpolates through
    non-contiguous (non-trailing) NaNs!
    """

    for signal_id in df:
        a = df[signal_id]
        # exclude leading and trailing NaNs
        raw_signal = np.array(a.loc[a.first_valid_index():a.last_valid_index()])

        NaNswitches = np.sum(np.diff(np.isnan(raw_signal)))
        if NaNswitches > 0:
            interp_signal = interpolate_NaNs(raw_signal)
            # inject into DataFrame on the fly, re-attaching trailing NaNs
            df[signal_id] = pd.Series(interp_signal)

    return df


class PandasModel(QAbstractTableModel):
    """
    Class to populate a table view with a pandas dataframe
    """

    def __init__(self, data, parent=None):
        QAbstractTableModel.__init__(self, parent)
        self._data = data

    def rowCount(self, parent=None):
        return len(self._data.values)

    def columnCount(self, parent=None):
        return self._data.columns.size

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if index.isValid():
            if role == Qt.ItemDataRole.DisplayRole:
                return str(self._data.values[index.row()][index.column()])
        return None

    def headerData(self, col, orientation, role):
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            return self._data.columns[col]
        return None


def set_max_width(qwidget, width):

    size_pol = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
    qwidget.setSizePolicy(size_pol)
    qwidget.setMaximumWidth(width)
    # qwidget.resize( 10,10 )


def is_dark_color_scheme() -> bool:
    """Qt6 styles itself depending on the system style"""

    return  QGuiApplication.styleHints().colorScheme() != Qt.ColorScheme.Light


def write_df(df: DataFrame, file_name: str) -> None:

    # the write out calls
    settings = QSettings()
    float_format = settings.value("default-settings/float_format", "%.3f")

    file_ext = file_name.split(".")[-1]
    if file_ext not in ["txt", "csv", "xlsx"]:

        msgBox = QMessageBox()
        msgBox.setWindowTitle("Unknown File Format")
        msgBox.setText("Please append .txt, .csv or .xlsx to the file name!")
        msgBox.exec()

    if file_ext == "txt":
        df.to_csv(
            file_name, index=False, sep="\t", float_format=float_format
        )

    elif file_ext == "csv":
        df.to_csv(
            file_name, index=False, sep=",", float_format=float_format
        )

    elif file_ext == "xlsx":
        df.to_excel(file_name, index=False, float_format=float_format)


class StoreGeometry:
    """Mixin in to store and restore window geometry"""

    def __init__(self, pos: tuple[int], size: tuple[int]):
        self._default_geometry: tuple[QPoint, QSize] = QPoint(*pos), QSize(*size)

    def closeEvent(self, event):
        settings = QSettings()
        settings.setValue(f'{self.__class__.__name__}/geometry', (self.pos(), self.size()))
        super().closeEvent(event)
        event.accept()

    def restore_geometry(self, pos_offset: int = 0) -> None:
        settings = QSettings()
        pos, size = (settings.value(f'{self.__class__.__name__}/geometry')
                     or self._default_geometry)
        self.move(pos + QPoint(pos_offset, pos_offset))
        self.resize(size)


def create_spinbox(start_value: int | float,
                   minimum: int | float,
                   maximum: int | float,
                   step: int | float = 1,
                   unit: str = '',
                   status_tip: str = '',
                   double: bool = False) -> QSpinBox | QDoubleSpinBox:

    if double:
        sb = QDoubleSpinBox()
        sb.setDecimals(1)
    else:
        sb = QSpinBox()

    sb.setMinimum(minimum)
    sb.setMaximum(maximum)
    sb.setSingleStep(step)
    if unit:
        sb.setSuffix(' ' + unit)
    if status_tip:
        sb.setStatusTip(status_tip)
    sb.setValue(start_value)
    return sb


def mk_spinbox_unit_slot(sb: QAbstractSpinBox) -> Callable:
    def unit_slot(unit: str):
        sb.setSuffix(' ' + unit)
    return unit_slot
