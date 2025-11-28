"""Abstraction layer to support both PySide6 and PySide6"""
try:
    import PySide6 as QtLib
    from PySide6 import QtCore, QtGui, QtWidgets
    Signal = QtCore.pyqtSignal
except ImportError:
    import PySide6 as QtLib
    from PySide6 import QtCore, QtGui, QtWidgets
    Signal = QtCore.Signal

__all__ = ["QtLib", "QtCore", "QtGui", "QtWidgets", "Signal"]
