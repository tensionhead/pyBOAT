from PyQt5.QtWidgets import QSlider, QDialog, QLabel, QHBoxLayout
from PyQt5.QtCore import Qt, pyqtSignal


class Slider_Dialog(QDialog):

    # Added a signal
    changedValue = pyqtSignal(int)

    def __init__(self):
        super(Slider_Dialog, self).__init__()
        self.init_ui()

    def init_ui(self):
        # Creating a label
        self.sliderLabel = QLabel('Slider:', self)

        # Creating a slider and setting its maximum and minimum value
        self.slider = QSlider(self)
        self.slider.setMinimum(0)
        self.slider.setMaximum(100)
        self.slider.setOrientation(Qt.Horizontal)

        # Creating a horizontalBoxLayout
        self.hboxLayout = QHBoxLayout(self)

        # Adding the widgets
        self.hboxLayout.addWidget(self.sliderLabel)
        self.hboxLayout.addWidget(self.slider)

        # Setting main layout
        self.setLayout(self.hboxLayout)

        # Setting a connection between slider position change and on_changed_value function
        self.slider.valueChanged.connect(self.on_changed_value)

        self.setWindowTitle("Dialog with a Slider")
        self.show()

    def on_changed_value(self, value):
        self.changedValue.emit(value)