#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт запуска приложения PCDViewer
"""

import sys
import os

# Добавляем путь к ui в sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
ui_dir = os.path.join(current_dir, 'ui')
sys.path.insert(0, ui_dir)

from PyQt5.QtWidgets import QApplication
from PCDViewer import PCDViewer


def main():
    app = QApplication(sys.argv)
    
    # Создаем и показываем основное окно
    window = PCDViewer()
    window.setWindowTitle("SPUTNIK.U - Semantic Segmentation Tool")
    window.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
