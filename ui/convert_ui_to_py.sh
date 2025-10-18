#!/bin/bash

# Конвертирование ui/MainWindow.ui в ui/MainWindow.py
pyuic5 -o ui/MainWindow.py ui/MainWindow.ui

# Конвертирование ui/icon/icon.qrc в ui/icon_rc.py
pyrcc5 ui/icon/icon.qrc -o ui/icon_rc.py

echo "Конвертация завершена."
