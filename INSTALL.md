# Краткая инструкция по установке

## Быстрый старт (Ubuntu/Linux)

### 1. Установка зависимостей системы

```bash
# Обновление системы
sudo apt update

# Установка Python 3.10 (если еще не установлен)
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install -y python3.10 python3.10-venv python3.10-dev python3-pip

# Проверка версии Python
python3.10 --version
# Должно быть: Python 3.10.8 или 3.10.x

# Установка OpenGL библиотек (для визуализации)
sudo apt install -y libgl1-mesa-glx libglu1-mesa libxrender1
```

### 2. Создание виртуального окружения

```bash
cd hackathon

# ВАЖНО: Используйте python3.10 (требуется версия 3.10.8)
python3.10 -m venv venv
source venv/bin/activate

# Проверьте версию Python в виртуальном окружении
python --version
# Должно быть: Python 3.10.8 или 3.10.x
```

### 3. Установка Python пакетов

**Если есть NVIDIA GPU с CUDA:**

```bash
# Установка PyTorch с CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Установка остальных зависимостей
pip install -r requirements.txt
```

**Если нет GPU (только CPU):**

```bash
# Установка PyTorch 2.2.2 CPU (ВАЖНО: именно эта версия!)
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cpu

# Установка остальных зависимостей
pip install -r requirements.txt
```

### 4. Проверка установки

```bash
python3 check_installation.py
```

Если все модули установлены корректно, вы увидите:
```
✓ Все зависимости установлены корректно!
  Можно запускать приложение: python3 run.py
```

### 5. Запуск приложения

```bash
python3 run.py
```

## Возможные проблемы

### Ошибка "Cannot mix incompatible Qt library"

```bash
pip uninstall PyQt5 PyQt5-sip -y
pip install PyQt5 PyQt5-sip
```

### Ошибка при импорте pyvistaqt

```bash
pip install pyvistaqt --force-reinstall
```

### CUDA не определяется

Проверьте установку CUDA:
```bash
nvidia-smi
nvcc --version
```

Если CUDA не установлена, используйте CPU версию PyTorch.

## Альтернативная установка через conda

Если вы используете Anaconda/Miniconda:

```bash
# ВАЖНО: Используйте python=3.10.8
conda create -n pcdviewer python=3.10.8
conda activate pcdviewer

# Проверка версии
python --version

# С GPU
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Остальные пакеты
pip install -r requirements.txt
```

## Контакты

При возникновении проблем с установкой, обратитесь к полному README.md или к разработчикам проекта.
