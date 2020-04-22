""" pyBOAT - The Biological Oscillations Analysis Toolkit """

import sys,os

__version__ = '0.7'

# the object oriented API
from .api import WAnalyzer

# the core functions
from .core import sinc_smooth
from .core import sliding_window_amplitude
from .core import normalize_with_envelope
from .core import compute_spectrum
from .core import get_maxRidge
from .core import eval_ridge

# -- to start the ui --

def main(DEBUG = False):
    
    # import PyQt only here
    from PyQt5.QtWidgets import QApplication
    from pyboat.ui import start_menu
    
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
