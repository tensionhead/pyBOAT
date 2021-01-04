""" pyBOAT - A Biological Oscillations Analysis Toolkit """

import sys,os
import argparse

__version__ = '0.9.0'

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

