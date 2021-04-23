import os
import pandas as pd
import matplotlib.pyplot as plt
import csv

from PyQt5.QtWidgets import (
    QFileDialog,
    QLabel,
    QPushButton,
    QSizePolicy,
    QWidget,
    QGridLayout,
)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtCore import Qt, QAbstractTableModel
from PyQt5.QtGui import QDoubleValidator, QIntValidator

from pyboat.core import interpolate_NaNs

# some Qt Validators, they accept floats with ','!
floatV = QDoubleValidator(bottom=-1e16, top=1e16)
posfloatV = QDoubleValidator(bottom=1e-16, top=1e16)
posintV = QIntValidator(bottom=1, top=9999999999)

# --- the analysis parameter dictionary with defaults ---

default_par_dict = {
    'dt' : 1,
    'time_unit' : 'min',
    'cut_off' : None,
    'window_size' : None,
    'Tmin' : None,
    'Tmax' : None,
    'nT' : 200,
    'pow_max' : None,
    'float_format' : '%.3f',
    'graphics_format' : 'png'
}
    

class MessageWindow(QWidget):

    ''' 
    A generic window do display a message
    and an Ok buttong to close
    '''
    
    def __init__(self, message, title):
        super().__init__()
        self.message = message
        self.title = title
        self.initUI()

    def initUI(self):
        error = QLabel(self.message)
        self.setGeometry(300, 450, 220, 100) # x,y, xlen, ylen
        self.setWindowTitle(self.title)
        okButton = QPushButton("OK", self)
        okButton.clicked.connect(self.close)
        main_layout_v = QGridLayout()
        main_layout_v.addWidget(error,0,0,1,3)
        main_layout_v.addWidget(okButton,1,1)
        self.setLayout(main_layout_v)
        self.show()


class mkGenericCanvas(FigureCanvas):
    def __init__(self):
        self.fig, self.ax = plt.subplots(1,1)

        FigureCanvas.__init__(self, self.fig)

        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

        
def load_data(dir_path='./', debug = False, **kwargs):

    '''
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

    '''
    
    err_msg1 = "Non-numeric values encountered in\n\n"
    err_msg2 = "Parsing errors encountered in\n\n"
    err_suffix = "\n\ncheck input..!"
    
    # returns a list with 1 element, stand alone File Dialog
    file_names = QFileDialog.getOpenFileName(
        parent=None, caption="Open Tabular Data", directory=dir_path
    )

    if debug:
        print(file_names, type(file_names))

    file_name = file_names[0]
    file_ext = file_name.split(".")[-1]

    # check if cancelled -> Null string
    if file_name == '':
        return None, "cancelled", None
    
    print("Loading", file_ext, file_name)
    new_dir_path = os.path.dirname(file_name)
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
                return None, f'{err_msg1}{file_name}{err_suffix}', new_dir_path
            else:
                # attach a name for later reference in the DataViewer
                # strip off extension
                bname = os.path.basename(file_name)                
                san_df.name = ''.join(bname.split('.')[:-1])
                return san_df, '', os.path.dirname(file_name) # empty error msg
        
        except (pd.errors.ParserError, csv.Error):
            return None, f'{err_msg2}{file_name}{err_suffix}', new_dir_path
        
    # infer table type from extension
    if 'sep' not in kwargs:

        if file_ext == "csv":
            delimiter = ','
        elif file_ext == "tsv":
            delimiter = '\t'
        elif file_ext == "txt":
            delimiter = '\s+'
        else:
            # calls Python's inbuilt csv.Sniffer
            delimiter = None
            kwargs['engine'] = 'python'
            
        kwargs['sep'] = delimiter
                
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
            return None, f'{err_msg1}{file_name}{err_suffix}', new_dir_path
        
        san_df = sanitize_df(raw_df, debug)
        if san_df is None:
            print("Error sanitizing data..")
            return None, f'{err_msg1}{file_name}{err_suffix}', new_dir_path
        
        else:
            # attach a name for later reference in the DataViewer
            bname = os.path.basename(file_name)                
            san_df.name = ''.join(bname.split('.')[:-1])
            return san_df, '', new_dir_path # empty error msg
        
    except pd.errors.ParserError:
        return None, f'{err_msg2}{file_name}{err_suffix}', new_dir_path
        
            
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


def set_max_width(qwidget, width):

    size_pol = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
    qwidget.setSizePolicy(size_pol)
    qwidget.setMaximumWidth(width)
    # qwidget.resize( 10,10 )

    
def retrieve_double_edit(edit):
    
    text = edit.text()
    text = text.replace(',','.')
    try:
        value = float(text)
    except ValueError:
        value = None

    return value
