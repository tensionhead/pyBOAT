#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys, time, os

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

import numpy as np
import pylab as pl

import random

class Example(QMainWindow):

    def __init__(self):
        super().__init__()

        self.initUI()


    def initUI(self):

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)


        self.folderLayout = QWidget();

        self.pathRoot = QDir.rootPath()

        self.dirmodel = QFileSystemModel(self)
        self.dirmodel.setRootPath(QDir.currentPath())

        self.indexRoot = self.dirmodel.index(self.dirmodel.rootPath())

        self.folder_view = QTreeView();
        self.folder_view.setDragEnabled(True)
        self.folder_view.setModel(self.dirmodel)
        self.folder_view.setRootIndex(self.indexRoot)

        self.selectionModel = self.folder_view.selectionModel()

        self.left_layout = QVBoxLayout()
        self.left_layout.addWidget(self.folder_view)

        self.folderLayout.setLayout(self.left_layout)        

        splitter_filebrowser = QSplitter(Qt.Horizontal)
        splitter_filebrowser.addWidget(self.folderLayout)
        splitter_filebrowser.addWidget(Figure_Canvas(self))
        splitter_filebrowser.setStretchFactor(1, 1)

        hbox = QHBoxLayout(self)
        hbox.addWidget(splitter_filebrowser)

        self.centralWidget().setLayout(hbox)


        self.setWindowTitle('Simple drag & drop')
        self.setGeometry(750, 100, 600, 500)



class Figure_Canvas(QWidget):

    def __init__(self, parent):
        super().__init__(parent)

        self.setAcceptDrops(True)

        blabla = QLineEdit()

        self.right_layout = QVBoxLayout()
        self.right_layout.addWidget(blabla)


        self.buttonLayout = QWidget()
        self.buttonLayout.setLayout(self.right_layout)

    def dragEnterEvent(self, e):

        if e.mimeData().hasFormat('text/uri-list'):
            e.accept()
        else:
            e.ignore() 

    def dropEvent(self, e):

        print("something")
        data = np.loadtxt(e.mimeData())
        print(data)


if __name__ == '__main__':

    app = QApplication(sys.argv)
    ex = Example()
    ex.show()
    app.exec_()          