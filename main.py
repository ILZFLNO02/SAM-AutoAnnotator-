#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
应用主程序入口
"""

import sys
from PySide6.QtWidgets import QApplication
from ui_components import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()