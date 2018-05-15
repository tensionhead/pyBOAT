import sys
from control_window import para_control
from PyQt5.QtWidgets import QApplication
from noisy_sin_embeded_qt5 import ApplicationWindow

if 0:# __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = para_control()
    sys.exit(app.exec_())


qApp = QApplication(sys.argv)

aw = ApplicationWindow()
aw.setWindowTitle("%s" % progname)
aw.show()
sys.exit(qApp.exec_())



