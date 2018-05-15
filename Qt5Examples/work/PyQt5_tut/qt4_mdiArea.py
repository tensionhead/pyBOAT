#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
from PyQt5.QtWidgets import QMainWindow, QApplication, QMdiSubWindow
#from PyQt4.QtCore import *
#from PyQt4.QtGui import *

class MainWindow(QMainWindow):
   count = 0
	
   def __init__(self):
      super().__init__()
      self.mdi = QMdiArea()
      self.setCentralWidget(self.mdi)
      bar = self.menuBar()
      q = 'Tilted'
      
      file = bar.addMenu("File")
      file.addAction("New")
      file.addAction("cascade")
      file.addAction("Tiled")
      file.triggered[QAction].connect(self.windowaction)
      self.setWindowTitle("MDI demo")
		
   def windowaction(self):
      print ("triggered")
		
   if 1:
      #MainWindow.count = MainWindow.count+1
      sub = QMdiSubWindow()
      sub.setWidget(QTextEdit())
      sub.setWindowTitle("subwindow"+str(1))
      self.mdi.addSubWindow(sub)
      sub.show()
		
   if q.text() == "cascade":
      self.mdi.cascadeSubWindows()
		
   if q.text() == "Tiled":
      self.mdi.tileSubWindows()
		
   def main():
      app = QApplication(sys.argv)
      ex = MainWindow()
      ex.show()
      sys.exit(app.exec_())
	
   if __name__ == '__main__':
      main()