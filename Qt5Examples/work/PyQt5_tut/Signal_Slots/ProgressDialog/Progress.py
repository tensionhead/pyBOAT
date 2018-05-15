from PyQt5.QtWidgets import QDialog, QProgressBar, QLabel, QHBoxLayout
from PyQt5.QtCore import pyqtSlot

class ProgressBar_Dialog(QDialog):
    def __init__(self):
        super(ProgressBar_Dialog ,self).__init__()
        self.init_ui()

    def init_ui(self):
        # Creating a label
        self.progressLabel = QLabel('Progress Bar:', self)

        # Creating a progress bar and setting the value limits
        self.progressBar = QProgressBar(self)
        self.progressBar.setMaximum(100)
        self.progressBar.setMinimum(0)

        # Creating a Horizontal Layout to add all the widgets
        self.hboxLayout = QHBoxLayout(self)

        # Adding the widgets
        self.hboxLayout.addWidget(self.progressLabel)
        self.hboxLayout.addWidget(self.progressBar)

        # Setting the hBoxLayout as the main layout
        self.setLayout(self.hboxLayout)
        self.setWindowTitle('Dialog with Progressbar')

        self.show()

    def make_connection(self, slider_object):
        slider_object.changedValue.connect(self.get_slider_value)

    @pyqtSlot(int)
    def get_slider_value(self, val):
        self.progressBar.setValue(val)