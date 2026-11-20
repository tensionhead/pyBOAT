from PySide6.QtGui import QGuiApplication, Qt


def is_dark_color_scheme() -> bool:
    """Qt6 styles itself depending on the system style"""

    return  QGuiApplication.styleHints().colorScheme() != Qt.ColorScheme.Light
