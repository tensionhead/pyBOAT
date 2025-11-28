"""Abstraction layer to support both PyQt6 and PySide6"""
try:
    import PyQt6 as QtLib
    from PyQt6 import QtCore, QtGui, QtWidgets
    Signal = QtCore.pyqtSignal
except ImportError:
    import PySide6 as QtLib
    from PySide6 import QtCore, QtGui, QtWidgets
    Signal = QtCore.Signal

__all__ = ["QtLib", "QtCore", "QtGui", "QtWidgets", "Signal"]
