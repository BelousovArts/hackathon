#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для проверки корректности установки всех зависимостей
"""

import sys


def check_module(module_name, import_statement=None):
    """Проверка наличия модуля"""
    try:
        if import_statement:
            exec(import_statement)
        else:
            __import__(module_name)
        print(f"✓ {module_name:20} - OK")
        return True
    except ImportError as e:
        print(f"✗ {module_name:20} - FAILED: {e}")
        return False
    except Exception as e:
        print(f"⚠ {module_name:20} - WARNING: {e}")
        return True


def main():
    print("=" * 60)
    print("Проверка установки зависимостей PCD Viewer")
    print("=" * 60)
    print()
    
    checks = {
        "PyQt5": "from PyQt5.QtWidgets import QApplication",
        "pyvista": "import pyvista",
        "pyvistaqt": "import pyvistaqt",
        "open3d": "import open3d",
        "open3d.ml": "import open3d.ml",
        "open3d.ml (models)": "from open3d.ml.torch.models import RandLANet",
        "numpy": "import numpy",
        "scipy": "import scipy",
        "torch": "import torch",
        "matplotlib": "import matplotlib",
        "sklearn": "import sklearn",
        "PIL": "from PIL import Image",
        "yaml": "import yaml",
        "pandas": "import pandas",
        "tqdm": "import tqdm",
    }
    
    print("Проверка Python модулей:")
    print("-" * 60)
    
    all_ok = True
    for module, import_stmt in checks.items():
        if not check_module(module, import_stmt):
            all_ok = False
    
    print()
    print("=" * 60)
    
    # Проверка CUDA
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print(f"✓ CUDA доступна: {torch.version.cuda}")
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠ CUDA недоступна - инференс будет работать на CPU (медленнее)")
    except:
        print("⚠ Не удалось проверить CUDA")
    
    print()
    print("=" * 60)
    
    if all_ok:
        print("✓ Все зависимости установлены корректно!")
        print("  Можно запускать приложение: python3 run.py")
        return 0
    else:
        print("✗ Некоторые зависимости отсутствуют!")
        print("  Установите их: pip install -r requirements.txt")
        return 1


if __name__ == "__main__":
    sys.exit(main())
