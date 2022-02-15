""" pyBOAT - A Biological Oscillations Analysis Toolkit """

import sys
import os
import argparse

__version__ = '0.9.4'

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


# --------------
# UI Entry Point
# --------------

def main(argv=None):

    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtGui import QIcon
    from PyQt5.QtCore import QSize

    from pyboat.ui import start_menu

    # --- initialize the Qt App ---

    # args get not parsed inside Qt app
    app = QApplication(sys.argv)

    # add an application icon
    abs_path = os.path.dirname(os.path.realpath(__file__))
    icon_path = os.path.join(abs_path, 'logo_circ128x128.png')
    icon = QIcon()
    icon.addFile(icon_path, QSize(128, 128))
    app.setWindowIcon(icon)

    # needed for QSettings
    app.setOrganizationName("tensionhead")
    app.setOrganizationDomain("https://github.com/tensionhead")
    app.setApplicationName("pyBOAT")

    # -- parse command line arguments ---

    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--version', action='version', version='pyBOAT '+__version__)
    args = parser.parse_args(argv)
    
    debug = args.debug

    if debug:
        print(
            '''
            ---------------
            DEBUG enabled!!
            ---------------
            ''')

        screen = app.primaryScreen()
        print('Screen: %s' % screen.name())
        size = screen.size()
        print('Size: %d x %d' % (size.width(), size.height()))
        rect = screen.availableGeometry()
        print('Available: %d x %d' % (rect.width(), rect.height()))

    # this starts up the Program
    window = start_menu.MainWindow(debug)

    return app.exec()
