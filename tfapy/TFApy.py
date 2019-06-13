#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys,os
from PyQt5.QtWidgets import QCheckBox, QTableView, QComboBox, QFileDialog, QAction, QMainWindow, QApplication, QLabel, QLineEdit, QPushButton, QMessageBox, QSizePolicy, QWidget, QVBoxLayout, QHBoxLayout, QDialog, QGroupBox, QFormLayout, QGridLayout, QTabWidget, QTableWidget

from ui.util import ErrorWindow
from ui.data_viewer import DataViewer

# matplotlib settings
from matplotlib import rc
rc('text', usetex=False) # better for the UI


# -------------
DEBUG = True
# -------------


class MainWindow(QMainWindow):
    
    def __init__(self):
        super().__init__()
        self.nViewers = 0
        self.DataViewers = {} # no Viewers yet
        self.detr = {}
        self.initUI()
        
    def initUI(self):
        
        self.setGeometry(1400,100,400,100)
        self.setWindowTitle('TFAnalyzer')

        # Actions for the menu - status bar
        main_widget = QWidget()
        
        self.quitAction = QAction("&Quit", self)
        self.quitAction.setShortcut("Ctrl+Q")
        self.quitAction.setStatusTip('Quit pyTFA')
        self.quitAction.triggered.connect(self.close_application)

        openFile = QAction("&Load data", self)
        openFile.setShortcut("Ctrl+L")
        openFile.setStatusTip('Load data')
        openFile.triggered.connect(self.Load_init_Viewer)


        # ?
        self.statusBar()

        # the menu bar
        
        mainMenu = self.menuBar()
        mainMenu.setNativeMenuBar(False)
        
        fileMenu = mainMenu.addMenu('&File')
        fileMenu.addAction(self.quitAction)
        fileMenu.addAction(openFile)
        
        # analyzerMenu = mainMenu.addMenu('&Analyzer')
        # analyzerMenu.addAction(plotSynSig)

        # main groups
        
        main_layout = QGridLayout()

        load_box = QGroupBox('Import Data')
        load_box_layout = QVBoxLayout()

        synsig_box = QGroupBox('Synthetic Signals')
        synsig_box_layout = QVBoxLayout()
        
        synsigButton = QPushButton("Start Generator",self)
        synsigButton.clicked.connect(self.init_synsig_generator)

        synsig_box_layout.addWidget(synsigButton)
        synsig_box_layout.addStretch(0)
        synsig_box.setLayout(synsig_box_layout)

        
        openFileButton = QPushButton("Open",self)
        openFileButton.clicked.connect(self.Load_init_Viewer)
        # quitButton.resize(quitButton.minimumSizeHint())
        load_box_layout.addWidget(openFileButton)
                
        self.cb_header = QCheckBox('With table header', self)
        self.cb_header.setChecked(True) # detrend by default
                
        load_box_layout.addWidget(self.cb_header)

        load_box.setLayout(load_box_layout)

        # not used right now
        quitButton = QPushButton("Quit", self)
        quitButton.clicked.connect(self.close_application)
 
        #quitButton.setMinimumSize(40,20)
        #quitButton.resize(quitButton.minimumSizeHint())

        # fill grid main layout
        main_layout.addWidget(load_box,0,0,3,2)
        main_layout.addWidget(synsig_box,0,2,2,2)
        main_layout.addWidget(quitButton,2,3,1,1)

        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        self.show()


    def init_synsig_generator(self):
        self.e = ErrorWindow('Sorry, not implemented yet!',"Error")
        return

        
    def close_application(self):

        # no confirmation window
        if DEBUG:
            appc = QApplication.instance()
            appc.closeAllWindows()
            return

        choice = QMessageBox.question(self, 'Quitting',
                                            'Do you want to exit TFApy?',
                                            QMessageBox.Yes | QMessageBox.No)
        if choice == QMessageBox.Yes:
            print("Quitting ...")
            #sys.exit()
            appc = QApplication.instance()
            appc.closeAllWindows()
        else:
            pass
        
    def Load_init_Viewer(self, cb_status):

        cb_status = self.cb_header.isChecked()
        
        if DEBUG:
            print ('function Viewer_Ini called')
            
        self.nViewers += 1

        # make new DataViewer and get the data
        self.DataViewers[self.nViewers] = DataViewer(cb_status, DEBUG)

        

#if __name__ == '__main__':
app = QApplication(sys.argv)

if DEBUG:
    print(
        '''
        ----------------
        DEBUG enabled!!
        ---------------
        ''')

    screen = app.primaryScreen()
    print('Screen: %s' % screen.name())
    size = screen.size()
    print('Size: %d x %d' % (size.width(), size.height()))
    rect = screen.availableGeometry()
    print('Available: %d x %d' % (rect.width(), rect.height()))

window = MainWindow()

sys.exit(app.exec_())
        
