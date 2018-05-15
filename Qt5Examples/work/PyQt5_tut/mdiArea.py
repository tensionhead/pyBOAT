#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
from PyQt5.QtWidgets import QMainWindow, QApplication, QLabel, QLineEdit, QPushButton, QMessageBox, QSizePolicy, QWidget, QVBoxLayout, QHBoxLayout, QMdiArea


from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from PyQt5.QtCore import pyqtSlot

import random
import numpy as np

DEBUG = True

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.centralW = QMdiArea()
        self.setCentralWidget(self.centralW)
        sub = QMdiSubWindow(SubWindow())
        self.centralW.addSubWindow(sub)
        
        
class SubWindow(QWidget):
    def __init__(self):
        self.setWindowTitle('Test' + str(1))
        self.setGeometry(300,300,300,300)
        
        

        
app = QApplication(sys.argv)
mainW =MainWindow()
mainW.show()
sys.exit(app.exec_())