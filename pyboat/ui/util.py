import os
import pandas as pd

from PyQt5.QtWidgets import (
    QCheckBox,
    QTableView,
    QComboBox,
    QFileDialog,
    QAction,
    QMainWindow,
    QApplication,
    QLabel,
    QLineEdit,
    QPushButton,
    QMessageBox,
    QSizePolicy,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QDialog,
    QGroupBox,
    QFormLayout,
    QGridLayout,
    QTabWidget,
    QTableWidget,
)

from PyQt5.QtCore import Qt, QAbstractTableModel

from PyQt5.QtGui import QDoubleValidator, QIntValidator

# some Qt Validators, they accept floats with ','!
posfloatV = QDoubleValidator(bottom=1e-16, top=1e16)
posintV = QIntValidator(bottom=1, top=9999999999)


class MessageWindow(QWidget):
    def __init__(self, message, title):
        super().__init__()
        self.message = message
        self.title = title
        self.initUI()

    def initUI(self):
        error = QLabel(self.message)
        self.setGeometry(300, 300, 220, 100)
        self.setWindowTitle(self.title)
        okButton = QPushButton("OK", self)
        okButton.clicked.connect(self.close)
        main_layout_v = QVBoxLayout()
        main_layout_v.addWidget(error)
        main_layout_v.addWidget(okButton)
        self.setLayout(main_layout_v)
        self.show()


def load_data(no_header, debug=False):

    """
    Spawns a Qt FileDialog to point to the input file,
    then uses pandas read_* routines to read in the data
    and returns a DataFrame.
    """

    if debug:
        # file_names = ['../../data_examples/synth_signals.csv']
        print("no_header?", no_header)

    # returns a list, stand alone File Dialog
    file_names = QFileDialog.getOpenFileName(
        parent=None, caption="Import Data", directory="./"
    )

    if debug:
        print(file_names)

    file_name = file_names[0]
    file_ext = file_name.split(".")[-1]

    # check for valid path
    if not os.path.isfile(file_name):
        return None, "No valid file path supplied!"

    try:
        print("Loading", file_ext, file_name)
        # open file according to extension
        if file_ext == "csv":
            if debug:
                print("CSV")

            if no_header:
                raw_df = pd.read_table(file_name, sep=",", header=None)
            else:
                raw_df = pd.read_table(file_name, sep=",")

        elif file_ext == "tsv":
            if debug:
                print("TSV")

            if no_header:
                raw_df = pd.read_table(file_name, sep="\t", header=None)
            else:
                raw_df = pd.read_table(file_name, sep="\t")

        elif file_ext in ["xls", "xlsx"]:
            if debug:
                print("EXCEL")
            if no_header:
                raw_df = pd.read_excel(file_name, header=None)
            else:
                raw_df = pd.read_excel(file_name)
        # try white space separation as a fallback
        # (internal Python parsing engine)
        else:
            if debug:
                print("WHITESPACE")
            if no_header:
                raw_df = pd.read_table(file_name, sep="\s+", header=None)
            else:
                raw_df = pd.read_table(file_name, sep="\s+")

        if debug:
            print("Raw Columns:", raw_df.columns)

    except pd.errors.ParserError:
        raw_df = pd.DataFrame()
        err_msg = f"Parsing errors encountered in\n\n{file_name}\n\nCheck input file!"
        return None, err_msg

    if debug:
        print("Data Types:", raw_df.dtypes)

    # catch wrongly parsed input data
    for dtype in raw_df.dtypes:
        if dtype == object:
            err_msg = f"Non-numeric values encountered in\n\n{file_name}\n\nCheck input file/header?!"
            print("Error loading data..")
            return None, err_msg

    # map all columns to strings for the PandasModel and SignalBox
    str_col = map(str, raw_df.columns)
    raw_df.columns = str_col

    if debug:
        print("raw columns:", raw_df.columns)

    ## TODO drop NaNs
    ## later TODO deal with 'holes'

    # attach a name for later reference in the DataViewer
    raw_df.name = os.path.basename(file_name)

    # return data and empty error message
    return raw_df, ""


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

    def data(self, index, role=Qt.DisplayRole):
        if index.isValid():
            if role == Qt.DisplayRole:
                return str(self._data.values[index.row()][index.column()])
        return None

    def headerData(self, col, orientation, role):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self._data.columns[col]
        return None
