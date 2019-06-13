import matplotlib.pyplot as ppl
from numpy.random import randn
import numpy as np
from tfa_lib import analyzer_api as ana

ppl.ion()

periods = np.linspace(2,50,150)
dt = 2

signal = randn(500)
wAn = ana.TFAnalyzer(periods = periods, dt = dt, T_cut_off = 40)

wAn.compute_spectrum(signal, Plot = True)

