"""Very basic test that the pyqt imports work, mostly interesting for cross-platform tests (apple silicon, windows)"""

def test_qt_import():
    from PySide6.QtWidgets import QWidget
