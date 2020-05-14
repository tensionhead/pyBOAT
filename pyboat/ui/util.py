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

from pyboat.core import interpolate_NaNs


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


def get_file_path(debug=False):

    """
    Spawns a Qt FileDialog to point to the input file
    """

    if debug:        
        # file_names = ['../../data_examples/synth_signals.csv']
        pass

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

    return file_name, file_ext

def load_data(debug = False, **kwargs):

    '''
    This is the entry point to import the data

    **kwargs: keyword arguments for pandas.read_... functions
    to control things like *header*, *sep* and so on.

    If no 'sep' is present in kwargs, infer column separator from
    default extensions: 
    
    'csv' : ',' 
    'tsv' : '\t' 
    ['xlsx', 'xls'] : use pandas read_excel 

    any other extension (like .txt) calls Python's csv.Sniffer

    '''
    
    err_msg1 = f"Non-numeric values encountered in\n\n"
    err_msg2 = f"Parsing errors encountered in\n\n"
    err_suffix = "\n\ncheck input..!"
    
    file_name, file_ext = get_file_path(debug)
    print("Loading", file_ext, file_name)

    # check if it's an excel file:
    if file_ext in ["xls", "xlsx"]:
        
        # not needed for reading excel sheets
        if 'sep' in kwargs:
            del kwargs['sep']
            
        try:            
            raw_df = pd.read_excel(file_name, **kwargs)
            san_df = sanitize_df(raw_df, debug)
            if san_df is None:
                print("Error loading data..")
                return None, f'{err_msg1}{file_name}{err_suffix}'
            else:
                # attach a name for later reference in the DataViewer
                san_df.name = os.path.basename(file_name)                
                return san_df, '' # empty error msg
        
        except pd.errors.ParserError:
            return None, f'{err_msg2}{file_name}{err_suffix}'
        
    # infer table type from extension?
    if 'sep' not in kwargs:

        if file_ext == "csv":
            delimiter = ','
        elif file_ext == "tsv":
            delimiter = '\t'
        else:
            # calls Python's inbuilt csv.Sniffer, works good for white spaces        
            delimiter = None
            
        kwargs['sep'] = delimiter
        
        if debug:
            print(f"Infered separator from extension: {delimiter}'")
    else:
        if debug:
            print(f"Separator was given as: '{kwargs['sep']}'")

        
    try:
        raw_df = pd.read_table(file_name, **kwargs)
        san_df = sanitize_df(raw_df, debug)
        
        if san_df is None:
                print("Error loading data..")
                return None, f'{err_msg1}{file_name}{err_suffix}'
        else:
            # attach a name for later reference in the DataViewer
            san_df.name = os.path.basename(file_name)                            
            return san_df, '' # empty error msg
        
    except pd.errors.ParserError:
        return None, f'{err_msg2}{file_name}{err_suffix}'
        
            
def sanitize_df(raw_df, debug = False):

    '''
    Makes sure the DataFrame is in a form to
    work as a 'PandasModel', see class below
    '''

    # catch wrongly parsed input data, we want all numeric!
    for dtype in raw_df.dtypes:
        if dtype == object:
            return None    
    
    # map all columns to strings for the PandasModel and SignalBox
    str_col = map(str, raw_df.columns)
    raw_df.columns = str_col

    if debug:
        print("raw columns:", raw_df.columns)

    # return data and empty error message
    return raw_df

def interpol_NaNs(df):

    '''
    Calls the NaN interpolator for each column
    of the DataFrame 
    '''

    df = df.apply(interpolate_NaNs, axis = 0)

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

    def data(self, index, role=Qt.DisplayRole):
        if index.isValid():
            if role == Qt.DisplayRole:
                return str(self._data.values[index.row()][index.column()])
        return None

    def headerData(self, col, orientation, role):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self._data.columns[col]
        return None
