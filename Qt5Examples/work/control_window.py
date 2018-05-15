import sys
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QPushButton, QAction, QLineEdit, QMessageBox, QLabel
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot
 
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
        self.amp = textboxValue
        textbox2Value = self.textbox2.text()
        textbox3Value = self.textbox3.text()
        
        QMessageBox.question(self, 'Message - pythonspot.com', "You typed: " + textboxValue + textbox2Value + textbox3Value, QMessageBox.Ok, QMessageBox.Ok)
        #self.textbox.setText("")
        return [self.amp]
 
 
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = para_control()
    sys.exit(app.exec_())
