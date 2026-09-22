"""Dark theme tokens (UX-1) and the two hand accents (UX-2.5)."""

from __future__ import annotations

from PySide6.QtGui import QColor

BACKGROUND = QColor("#14161a")
SURFACE = QColor("#1c2027")
SURFACE_RAISED = QColor("#242933")
BORDER = QColor("#333a45")
TEXT = QColor("#e6e9ef")
TEXT_DIM = QColor("#8f98a8")
WARNING = QColor("#f0a648")
ERROR = QColor("#ef6461")
OK = QColor("#6bcf8a")

#: Right hand plays degrees 1-5, left hand 6-10.
RIGHT_ACCENT = QColor("#4fc3f7")
LEFT_ACCENT = QColor("#ffb74d")

STYLESHEET = f"""
QWidget {{
    background-color: {BACKGROUND.name()};
    color: {TEXT.name()};
    font-size: 13px;
}}
QLabel#sectionTitle {{
    color: {TEXT_DIM.name()};
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1px;
}}
QLabel#statusValue {{
    font-family: "Cascadia Mono", "Consolas", monospace;
}}
QComboBox, QSpinBox, QDoubleSpinBox {{
    background-color: {SURFACE_RAISED.name()};
    border: 1px solid {BORDER.name()};
    border-radius: 6px;
    padding: 6px 8px;
    min-height: 20px;
}}
QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover {{
    border-color: {RIGHT_ACCENT.name()};
}}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox QAbstractItemView {{
    background-color: {SURFACE_RAISED.name()};
    border: 1px solid {BORDER.name()};
    selection-background-color: {RIGHT_ACCENT.name()};
    selection-color: {BACKGROUND.name()};
    outline: none;
}}
QPushButton {{
    background-color: {SURFACE_RAISED.name()};
    border: 1px solid {BORDER.name()};
    border-radius: 6px;
    padding: 8px 12px;
}}
QPushButton:hover {{ border-color: {TEXT_DIM.name()}; }}
QPushButton:pressed {{ background-color: {SURFACE.name()}; }}
QPushButton#panicButton {{
    border-color: {ERROR.name()};
    color: {ERROR.name()};
    font-weight: 600;
}}
QPushButton#panicButton:hover {{
    background-color: {ERROR.name()};
    color: {BACKGROUND.name()};
}}
QCheckBox {{ spacing: 8px; }}
QSlider::groove:horizontal {{
    height: 4px;
    background: {BORDER.name()};
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: {RIGHT_ACCENT.name()};
    width: 14px;
    height: 14px;
    margin: -6px 0;
    border-radius: 7px;
}}
QSlider::sub-page:horizontal {{
    background: {RIGHT_ACCENT.name()};
    border-radius: 2px;
}}
QFrame#panel {{
    background-color: {SURFACE.name()};
    border-left: 1px solid {BORDER.name()};
}}
QFrame#separator {{ background-color: {BORDER.name()}; max-height: 1px; }}
QFrame#wavetablePanel {{
    background-color: {SURFACE.name()};
    border-top: 1px solid {BORDER.name()};
}}
QSplitter#stage::handle:vertical {{
    background-color: {BACKGROUND.name()};
    height: 6px;
}}
QSplitter#stage::handle:vertical:hover {{ background-color: {BORDER.name()}; }}
"""
