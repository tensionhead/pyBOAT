## Not sure if that is needed here?!
## #!/usr/bin/python3
# -*- coding: utf-8 -*-

# ------------------------------
# --- entry point for the UI ---
# ------------------------------

import sys,os
from PyQt5.QtWidgets import QApplication
from pyboat.ui import start_menu
# -------------
DEBUG = False
# -------------

if __name__ == '__main__':
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

    window = start_menu.MainWindow(DEBUG)

    sys.exit(app.exec_())
        
