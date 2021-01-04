# ------------------------------
# --- entry point for the UI ---
# ------------------------------
import sys, os
import argparse

from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import QSize
    
from pyboat.ui import start_menu
from . import __version__


def main(argv=None):
    
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

    app.exec()
                        
        
if __name__ == '__main__':
    main()


