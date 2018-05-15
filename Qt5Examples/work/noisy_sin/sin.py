import sys
 
from PyQt5.QtWidgets import QApplication, QMainWindow, QMenu, QVBoxLayout, QSizePolicy, QMessageBox, QWidget, QPushButton, QWidget, QPushButton, QAction, QLineEdit, QMessageBox, QLabel

from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QIcon
 
 
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

import random
import numpy as np

class para_control(QMainWindow):
 
    def __init__(self):
        super().__init__()
        self.title = 'Parameters for noisy sin'
        self.left = 100
        self.top = 100
        self.width = 320
        self.height = 180
        #self.amp = ()
        self.initUI()
 
    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

 	#label for textbox
        label = QLabel('Amplitude:', self)
        label.move(50,25)

        label2 = QLabel('Periode:', self)
        label2.move(50,55)

        label3 = QLabel('Sigma:', self)
        label3.move(50,85)
 
        # Create textbox
        self.textbox = QLineEdit(self)
        self.textbox.move(220, 30)
        self.textbox.resize(60,20)
	
	# Create textbox2
        self.textbox2 = QLineEdit(self)
        self.textbox2.move(220, 60)
        self.textbox2.resize(60,20)

	# Create textbox3
        self.textbox3 = QLineEdit(self)
        self.textbox3.move(220, 90)
        self.textbox3.resize(60,20)
 
 
        # Create a button in the window
        self.button = QPushButton('Save', self)
        self.button.resize(60,20)
        self.button.move(220,140)
	
 
        # connect button to function save_on_click
        self.button.clicked.connect(self.save_on_click)
        self.show()
 
    @pyqtSlot()
    def save_on_click(self):
        textboxValue = self.textbox.text()
        #self.amp = textboxValue
        textbox2Value = self.textbox2.text()
        textbox3Value = self.textbox3.text()
        
        QMessageBox.question(self, 'Message - pythonspot.com', "You typed: " + textboxValue + textbox2Value + textbox3Value, QMessageBox.Ok, QMessageBox.Ok)
        #self.textbox.setText("")


class ApplicationWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        super().__init__()

        self.left = 10
        self.top = 10
        self.title = 'Sin - Plot'
        self.width = 640
        self.height = 400
        self.initUI()
 
    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
 
        m = PlotCanvas(self)
        m.move(0,0)
        self.show()

class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)

 
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)
 
        FigureCanvas.setSizePolicy(self,
                QSizePolicy.Expanding,
                QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        # Adding update button
        self.button = QPushButton('Update', self)
        self.button.resize(60,20)
        self.button.move(430,380)
        self.button.clicked.connect(self.update_on_click)
        self.plot()
    @pyqtSlot()
    def update_on_click(self):
        action = para_control()
        action.show()
         
 
 
    def plot(self):
        t=np.array(range(0,256))

        sigma = 1
        amp = 3
        per = 40

       
       
        noise=np.random.normal(0, sigma, len(t))
        s=amp*np.sin(2*np.pi/per*t)+noise
        self.axes.plot(t,s)


        





App = QApplication(sys.argv)

w=ApplicationWindow()
sys.exit(App.exec_())
