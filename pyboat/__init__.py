""" pyBOAT - A Biological Oscillations Analysis Toolkit """

import sys,os
import argparse

__version__ = '0.8'

# the object oriented API
from .api import WAnalyzer

# the core functions
from .core import sinc_smooth
from .core import sliding_window_amplitude
from .core import normalize_with_envelope
from .core import compute_spectrum
from .core import get_maxRidge_ys
from .core import eval_ridge
from .core import interpolate_NaNs

# ------------------------------
# --- entry point for the UI ---
# ------------------------------

def main(argv=None):

    # import PyQt only here, no need to
    # generally import if only
    # scripting is needed..
    
    from PyQt5.QtWidgets import QApplication
    from pyboat.ui import start_menu
    # args get not parsed inside Qt app    
    app = QApplication(sys.argv) 

    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--version', action='version', version='pyBOAT '+__version__)
    args = parser.parse_args(argv)
    
    debug = args.debug        

    if debug:
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

    window = start_menu.MainWindow(debug)

    sys.exit(app.exec_())
