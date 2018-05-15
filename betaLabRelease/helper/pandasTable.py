#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys, time, os

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

import numpy as np
import pandas as pd
import pylab as pl

import random

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



if __name__ == '__main__':
    app = QApplication(sys.argv)
    test= QTableView()
    df = pd.read_excel('C:/Users/USER/ownCloud/Shared/TFAnalyzer/src/synth_signals2c.xlsx', header=0)
    model= PandasModel(df)
    test.setModel(model)
    test.show()
    sys.exit(app.exec_())
        
