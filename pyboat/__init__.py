"""pyBOAT - A Biological Oscillations Analysis Toolkit"""

import sys
import os
import argparse
import logging

__version__ = "1.0.1"

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

__all__ = [
    "WAnalyzer",
    "sinc_smooth",
    "sliding_window_amplitude",
    "normalize_with_envelope",
    "compute_spectrum",
    "get_maxRidge_ys",
    "eval_ridge",
    "interpolate_NaNs",
]

# --------------
# UI Entry Point
# --------------


def main(argv=None):
    """Entry point into the UI"""

    # import Qt modules only here when needed for the UI
    from PySide6.QtWidgets import QApplication
    from PySide6.QtGui import QIcon, QGuiApplication
    from PySide6.QtCore import QSize

    from pyboat.ui import start_menu

    # -- parse command line arguments and setup logging ---

    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--version", action="version", version="pyBOAT " + __version__)
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if not args.debug else logging.DEBUG,
        format="%(levelname)s [%(name)s] %(message)s",
        force=True
        )

    # silence matplotlib
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    # TODO: since Qt6 this is apparently at least for the moment
    # needed for plotting within the Qt UI
    # otherwise pyplot unsuccessfully tries to use TkAgg
    import matplotlib
    matplotlib.use("qtagg")

    # --- initialize the Qt App ---

    # args get not parsed inside Qt app
    app = QApplication(sys.argv)
    # add an application icon
    abs_path = os.path.dirname(os.path.realpath(__file__))
    icon_path = os.path.join(abs_path, "logo_circ128x128.png")
    icon = QIcon()
    icon.addFile(icon_path, QSize(128, 128))
    app.setWindowIcon(icon)

    # needed for QSettings
    app.setOrganizationName("tensionhead")
    app.setOrganizationDomain("https://github.com/tensionhead")
    app.setApplicationName("pyBOAT")

    if args.debug:
        print(
            """
            ---------------
            DEBUG enabled!!
            ---------------
            """
        )

        screen = app.primaryScreen()
        print("Screen: %s" % screen.name())
        size = screen.size()
        print("Size: %d x %d" % (size.width(), size.height()))
        rect = screen.availableGeometry()
        print("Available: %d x %d" % (rect.width(), rect.height()))
        print("Color scheme: ", QGuiApplication.styleHints().colorScheme())

    # this starts up the Program
    window = start_menu.MainWindow()

    return app.exec()
