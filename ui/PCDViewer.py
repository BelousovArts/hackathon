# -*- coding: utf-8 -*-
import sys
import os
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import pyvista as pv
import pyvistaqt as pvqt
import open3d as o3d
from MainWindow import Ui_MainWindow


class PCDViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Настройка UI
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        
        # Инициализация PyVista
        self.setup_pyvista_widget()
        
        # Настройка treeWidget
        self.setup_tree_widget()
        
        # Подключение сигналов
        self.ui.pushButton_import.clicked.connect(self.import_pcd)
        self.ui.pushButton_analys_net.clicked.connect(self.on_click_analyze_net)
        self.ui.pushButton_remove.clicked.connect(self.on_click_remove_by_class)
        self.ui.pushButton_export.clicked.connect(self.on_click_export)
        self.ui.pushButton_interpolate.clicked.connect(self.on_click_interpolate)
        self.ui.pushButton_reset_point.clicked.connect(self.on_click_reset_point)
        self.ui.pushButton_remove_point.clicked.connect(self.on_click_remove_point)
        self.ui.pushButton_crop_tile.clicked.connect(self.on_click_crop_tile)
        self.ui.pushButton_compare.clicked.connect(self.on_click_compare)
        self.ui.treeWidget.itemChanged.connect(self.on_tree_item_changed)
        self.ui.treeWidget.currentItemChanged.connect(self.on_tree_current_item_changed)
        
        # Размер точек (spinbox)
        self.ui.spinBox_point_size.setRange(1, 20)
        self.ui.spinBox_point_size.setValue(3)
        self.ui.spinBox_point_size.valueChanged.connect(self.on_point_size_changed)
        
        # Переключение Eye Dome Lighting (EDL)
        self.ui.comboBox_eye_dome.currentIndexChanged.connect(self.on_eye_dome_changed)
        self.ui.comboBox_eye_dome.setCurrentIndex(0)

        # Переключение режима проекции (ортографический/перспективный)
        self.ui.comboBox_display_mode.currentIndexChanged.connect(self.on_display_mode_changed)
        self.ui.comboBox_display_mode.setCurrentIndex(0)

        # Смена цвета фона
        self.background_palette = [
            ("Белый", 'white'),
            ("Светло-серый", (200, 200, 200)),
            ("Серый", (150, 150, 150)),
            ("Тёмно-серый", (80, 80, 80)),
            ("Угольный", (30, 30, 30)),
            ("Сланцевый серый", 'slategray'),
            ("Тёмный сланцевый", 'darkslategray'),
            ("Ночной синий", 'midnightblue'),
            ("Тёмно-синий", 'navy'),
            ("Стальной синий", 'steelblue'),
        ]
        self.ui.comboBox_background_color.blockSignals(True)
        self.ui.comboBox_background_color.clear()
        self.ui.comboBox_background_color.addItems([name for name, _ in self.background_palette])
        self.ui.comboBox_background_color.blockSignals(False)
        self.ui.comboBox_background_color.setCurrentIndex(0)
        self.ui.comboBox_background_color.currentIndexChanged.connect(self.on_background_color_changed)

        # Переключение цвета точек
        # Проверяем, нужно ли добавить пункт "Distance" (для облаков сравнения)
        if self.ui.comboBox_point_color.count() < 10:
            # Если пунктов меньше 10, значит "Distance" еще не добавлен
            # Добавляем его на позицию 5 (после Predict)
            self.ui.comboBox_point_color.insertItem(5, "Distance")
        self.ui.comboBox_point_color.currentIndexChanged.connect(self.on_point_color_changed)

        # Кнопка преобразования классов выделенных точек
        if hasattr(self.ui, 'pushButton_transform'):
            self.ui.pushButton_transform.clicked.connect(self.on_click_transform)

        # Кнопки масштабирования и сброса вида
        self.ui.pushButton_zoom_up.clicked.connect(self.on_zoom_up)
        self.ui.pushButton_zoom_down.clicked.connect(self.on_zoom_down)
        self.ui.pushButton_reset_view.clicked.connect(self.on_reset_view)

        # Кнопки стандартных видов
        self.ui.pushButton_front.clicked.connect(self.on_view_front)
        self.ui.pushButton_back.clicked.connect(self.on_view_back)
        self.ui.pushButton_left.clicked.connect(self.on_view_left)
        self.ui.pushButton_right.clicked.connect(self.on_view_right)
        self.ui.pushButton_top.clicked.connect(self.on_view_top)
        self.ui.pushButton_bottom.clicked.connect(self.on_view_bottom)

        # Режим покликового выделения точек
        self.selection_mode = False
        if hasattr(self.ui, 'pushButton_point'):
            self.ui.pushButton_point.setCheckable(True)
            self.ui.pushButton_point.clicked.connect(self.on_toggle_point_select)
        
        # Режим выделения кистью
        self.brush_mode = False
        self.brush_radius = 1.0  # радиус кисти в единицах сцены
        self.brush_size = 1      # размер кисти 1..20 (шаг 1); 1 => радиус 1.0
        if hasattr(self.ui, 'pushButton_brush'):
            self.ui.pushButton_brush.setCheckable(True)
            self.ui.pushButton_brush.clicked.connect(self.on_toggle_brush_select)
        # Синхронизация радиуса с размером кисти (size -> radius)
        self._sync_brush_radius()
        # Кнопки изменения размера кисти
        if hasattr(self.ui, 'pushButton_brush_1'):
            self.ui.pushButton_brush_1.clicked.connect(self.on_brush_size_inc)
        if hasattr(self.ui, 'pushButton_brush_2'):
            self.ui.pushButton_brush_2.clicked.connect(self.on_brush_size_dec)

        # Тултипы для всех кнопок UI
        self._setup_tooltips()

        # Сопоставление индексов комбобокса режимам окраски и обратно
        self.mode_by_index = {
            0: 'height',        # Градиент по высоте
            1: 'rgb',           # RGB
            2: 'intensity',     # Интенсивность
            3: 'label',         # Label
            4: 'predict',       # Predict
            5: 'distance',      # Distance (для сравнения облаков)
            6: 'white',         # Белый
            7: 'black',         # Черный
            8: 'blue',          # Синий
            9: 'gray',          # Серый
        }
        self.index_by_mode = {v: k for k, v in self.mode_by_index.items()}
        
        # Настройка контекстного меню для treeWidget
        self.ui.treeWidget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.ui.treeWidget.customContextMenuRequested.connect(self.show_context_menu)
        
        # Хранение данных об облаках
        self.point_clouds = {}
        # Счетчик для уникальных групп тайлов
        self._tile_group_counter = 0
        # Поля для ML-инференса
        self._ml_pipeline = None
        self._ml_pipelines = {}
        # Определяем базовую директорию (где находится скрипт)
        self._base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self._ml_ckpt_path = os.path.join(self._base_dir, "models", "randlanet_hack2c.pth")
        # Карта конфигов по именам (нормализованным)
        self._ml_cfg_map = {
            'randlanet': os.path.join(self._base_dir, "configs", "randlanet_hack2c.yml"),
            'randlanet_toronto3d': os.path.join(self._base_dir, "configs", "randlanet_toronto3d.yml"),
            'randlanet_semantickitti': os.path.join(self._base_dir, "configs", "randlanet_semantickitti.yml"),
            'randlanet_lidar_scenes': os.path.join(self._base_dir, "configs", "randlanet_lidar_scenes.yml"),
        }
        # Карта чекпоинтов по конфигам
        self._ml_ckpt_map = {
            'randlanet': os.path.join(self._base_dir, "models", "randlanet_hack2c.pth"),
            'randlanet_toronto3d': os.path.join(self._base_dir, "models", "randlanet_toronto3d.pth"),
            'randlanet_semantickitti': os.path.join(self._base_dir, "models", "randlanet_semantickitti.pth"),
            'randlanet_lidar_scenes': os.path.join(self._base_dir, "models", "randlanet_lidar_scenes.pth"),
        }
        
        # Инициализация статусбара
        self.update_status_bar()
        
        # Устанавливаем фокус на главное окно для обработки клавиш
        self.setFocusPolicy(Qt.StrongFocus)

    def _setup_tooltips(self):
        """Установить подсказки (tooltips) для всех кнопок UI."""
        try:
            # Импорт/Экспорт
            if hasattr(self.ui, 'pushButton_import'):
                self.ui.pushButton_import.setToolTip('Импортировать облако(а) точек из файла')
            if hasattr(self.ui, 'pushButton_export'):
                self.ui.pushButton_export.setToolTip('Экспортировать текущее облако точек в .pcd')

            # Анализ и классы
            if hasattr(self.ui, 'pushButton_analys_net'):
                self.ui.pushButton_analys_net.setToolTip('Запустить сегментацию: выбрать модель и выполнить анализ')
            if hasattr(self.ui, 'pushButton_transform'):
                self.ui.pushButton_transform.setToolTip('Сменить класс у выделенных точек')
            if hasattr(self.ui, 'pushButton_remove'):
                self.ui.pushButton_remove.setToolTip('Удалить все точки выбранного класса')

            # Выделение и правки
            if hasattr(self.ui, 'pushButton_interpolate'):
                self.ui.pushButton_interpolate.setToolTip('Интерполяция области по выделенным граничным точкам')
            if hasattr(self.ui, 'pushButton_reset_point'):
                self.ui.pushButton_reset_point.setToolTip('Сбросить текущее выделение точек')
            if hasattr(self.ui, 'pushButton_remove_point'):
                self.ui.pushButton_remove_point.setToolTip('Удалить выделенные точки')

            # Тайлы и сравнение
            if hasattr(self.ui, 'pushButton_crop_tile'):
                self.ui.pushButton_crop_tile.setToolTip('Разбить облако на тайлы (размер и перекрытие)')
            if hasattr(self.ui, 'pushButton_compare'):
                self.ui.pushButton_compare.setToolTip('Сравнить два облака точек (тепловая карта/различия/статистика)')

            # Режимы выделения
            if hasattr(self.ui, 'pushButton_point'):
                self.ui.pushButton_point.setToolTip('Режим: выбор точек по щелчку (ПКМ)')
            if hasattr(self.ui, 'pushButton_brush'):
                self.ui.pushButton_brush.setToolTip('Режим: выделение кистью (ПКМ), Alt+ПКМ — снятие')
            if hasattr(self.ui, 'pushButton_brush_1'):
                self.ui.pushButton_brush_1.setToolTip('Увеличить размер кисти')
            if hasattr(self.ui, 'pushButton_brush_2'):
                self.ui.pushButton_brush_2.setToolTip('Уменьшить размер кисти')

            # Виды и масштаб
            if hasattr(self.ui, 'pushButton_zoom_up'):
                self.ui.pushButton_zoom_up.setToolTip('Увеличить масштаб')
            if hasattr(self.ui, 'pushButton_zoom_down'):
                self.ui.pushButton_zoom_down.setToolTip('Уменьшить масштаб')
            if hasattr(self.ui, 'pushButton_reset_view'):
                self.ui.pushButton_reset_view.setToolTip('Показать всю сцену (сброс вида)')
            if hasattr(self.ui, 'pushButton_front'):
                self.ui.pushButton_front.setToolTip('Вид спереди')
            if hasattr(self.ui, 'pushButton_back'):
                self.ui.pushButton_back.setToolTip('Вид сзади')
            if hasattr(self.ui, 'pushButton_left'):
                self.ui.pushButton_left.setToolTip('Вид слева')
            if hasattr(self.ui, 'pushButton_right'):
                self.ui.pushButton_right.setToolTip('Вид справа')
            if hasattr(self.ui, 'pushButton_top'):
                self.ui.pushButton_top.setToolTip('Вид сверху')
            if hasattr(self.ui, 'pushButton_bottom'):
                self.ui.pushButton_bottom.setToolTip('Вид снизу')
        except Exception:
            pass
        
    def _get_selected_cfg_name(self, cfg_name=None):
        """Вернуть нормализованное имя конфига."""
        if cfg_name is None:
            try:
                text = self.ui.comboBox_analys_net.currentText()
            except Exception:
                text = ''
            name = str(text).strip().lower()
            if 'toronto' in name:
                return 'randlanet_toronto3d'
            if 'kitti' in name or 'semantik' in name or 'semantic' in name:
                return 'randlanet_semantickitti'
            if 'lidar' in name or 'scenes' in name:
                return 'randlanet_lidar_scenes'
            if name.startswith('randla-net') or name.startswith('randlanet'):
                return 'randlanet'
            return 'randlanet'
        return cfg_name

    def _ensure_ml_pipeline(self, cfg_name=None):
        """Ленивая инициализация пайплайна по указанному конфигу."""
        if cfg_name is None:
            cfg_name = self._get_selected_cfg_name()
        if cfg_name in self._ml_pipelines:
            self._ml_pipeline = self._ml_pipelines[cfg_name]
            return
        # Подготовим импорты/конфиг
        # Финальная инициализация (без нестандартных оберток)
        from open3d.ml.utils import Config
        from open3d.ml.torch.models import RandLANet
        from open3d.ml.torch.pipelines import SemanticSegmentation
        import tempfile
        cfg_path = self._ml_cfg_map.get(cfg_name)
        if not cfg_path:
            raise RuntimeError(f"Неизвестный конфиг: {cfg_name}")
        cfg = Config.load_from_file(cfg_path)

        # Переопределяем параметры pipeline для inference-режима (без создания логов)
        pipeline_cfg = dict(cfg.pipeline)
        # Используем временную директорию для логов, чтобы не создавать папки в проекте
        pipeline_cfg['main_log_dir'] = tempfile.gettempdir()
        pipeline_cfg['save_ckpt_freq'] = -1  # Отключаем сохранение чекпоинтов
        
        # Создаём модель и пайплайн без датасета (desktop: только веса и конфиг модели)
        model = RandLANet(**cfg.model)
        pipeline = SemanticSegmentation(model, dataset=None, **pipeline_cfg)
        # Загрузка чекпойнта
        try:
            ckpt_path = self._ml_ckpt_map.get(cfg_name, self._ml_ckpt_path)
            pipeline.load_ckpt(ckpt_path)
        except Exception as e:
            QMessageBox.critical(self, 'Ошибка', f'Не удалось загрузить чекпойнт:\n{str(e)}')
            raise
        self._ml_pipeline = pipeline
        self._ml_pipelines[cfg_name] = pipeline

    def _run_inference_on_points(self, points_xyz, intensity=None):
        """Выполнить инференс на numpy массиве Nx3, вернуть массив меток int32.
        
        Args:
            points_xyz: Координаты точек Nx3
            intensity: Интенсивность точек Nx1 (опционально)
        """
        import numpy as np
        import open3d as o3d
        points = np.asarray(points_xyz, dtype=np.float32)
        labels_dummy = np.zeros(points.shape[0], dtype=np.int32)
        
        # Определяем количество дополнительных признаков
        try:
            in_ch = int(getattr(self._ml_pipeline.model.cfg, 'in_channels', 3))
        except Exception:
            in_ch = 3
        extra_feat = max(0, in_ch - 3)
        
        # Создаём матрицу признаков
        if extra_feat == 0:
            feat = None
        elif extra_feat == 1:
            # Один дополнительный канал - используем intensity если есть
            if intensity is not None and intensity.shape[0] == points.shape[0]:
                feat = np.asarray(intensity, dtype=np.float32).reshape(-1, 1)
            else:
                feat = np.zeros((points.shape[0], 1), dtype=np.float32)
        else:
            # Несколько каналов - intensity в первый, остальные нули
            feat = np.zeros((points.shape[0], extra_feat), dtype=np.float32)
            if intensity is not None and intensity.shape[0] == points.shape[0]:
                feat[:, 0:1] = np.asarray(intensity, dtype=np.float32).reshape(-1, 1)
        
        data = { 'point': points, 'feat': feat, 'label': labels_dummy }
        results = self._ml_pipeline.run_inference(data)
        pred = results.get('predict_labels')
        if pred is None:
            raise RuntimeError('Pipeline не вернул predict_labels')
        return pred.astype(np.int32)

    def _get_class_names_and_colors(self, cfg_name):
        """Вернуть фиксированные имена и цвета классов для выбранного конфига."""
        if cfg_name == 'randlanet_toronto3d':
            names = [
                'Road', 'Road marking', 'Natural', 'Building',
                'Utility line', 'Pole', 'Car', 'Fence'
            ]
            colors = [
                (128, 128, 128),  # Road
                (255, 255, 255),  # Road marking
                (0, 255, 0),      # Natural
                (255, 0, 0),      # Building
                (255, 255, 0),    # Utility line
                (0, 0, 255),      # Pole
                (255, 0, 255),    # Car
                (0, 255, 255),    # Fence
            ]
        elif cfg_name == 'randlanet_semantickitti':
            # SemanticKITTI: 20 классов (включая 'unlabeled')
            names = [
                'Unlabeled', 'Car', 'Bicycle', 'Motorcycle', 'Truck', 'Other-vehicle',
                'Person', 'Bicyclist', 'Motorcyclist', 'Road', 'Parking', 'Sidewalk',
                'Other-ground', 'Building', 'Fence', 'Vegetation', 'Trunk', 'Terrain',
                'Pole', 'Traffic-sign'
            ]
            # Палитра (интуитивные, различимые цвета)
            colors = [
                (0, 0, 0),        # Unlabeled
                (245, 150, 100),  # Car
                (245, 230, 100),  # Bicycle
                (150, 60, 30),    # Motorcycle
                (180, 30, 80),    # Truck
                (255, 0, 0),      # Other-vehicle
                (30, 30, 255),    # Person
                (200, 40, 255),   # Bicyclist
                (90, 30, 150),    # Motorcyclist
                (255, 0, 255),    # Road
                (255, 150, 255),  # Parking
                (75, 0, 75),      # Sidewalk
                (75, 0, 175),     # Other-ground
                (0, 200, 255),    # Building
                (50, 120, 255),   # Fence
                (0, 175, 0),      # Vegetation
                (0, 60, 135),     # Trunk
                (80, 240, 150),   # Terrain
                (150, 240, 255),  # Pole
                (0, 0, 255),      # Traffic-sign
            ]
        elif cfg_name == 'randlanet_lidar_scenes':
            # Lidar Scenes: 4 класса
            names = [
                'Background', 'Car', 'Person', 'Tram'
            ]
            colors = [
                (128, 128, 128),  # Background (серый)
                (255, 0, 0),      # Car (красный)
                (0, 0, 255),      # Person (синий)
                (255, 255, 0),    # Tram (желтый)
            ]
        else:
            # Hack2C / RandLA-Net (2 класса)
            names = ['Background', 'Car']
            colors = [
                (128, 128, 128),  # Background
                (255, 0, 0),      # Car
            ]
        return names, colors

    def _populate_class_combos(self, cloud_name):
        """Заполнить comboBox_transform и comboBox_remove классами из данных облака.

        Предпочитается поле 'predict'; если его нет — используем 'label'. Комбобоксы
        остаются пустыми до выполнения инференса.
        """
        try:
            if not hasattr(self.ui, 'comboBox_transform') or not hasattr(self.ui, 'comboBox_remove'):
                return
            cloud = self.point_clouds.get(cloud_name)
            if not cloud:
                return
            poly = cloud.get('poly_data')
            if poly is None:
                self.ui.comboBox_transform.clear()
                self.ui.comboBox_remove.clear()
                return

            # Выбираем источник классов
            source_field = None
            if 'predict' in poly.array_names:
                source_field = 'predict'
            elif 'label' in poly.array_names:
                source_field = 'label'
            if source_field is None:
                self.ui.comboBox_transform.clear()
                self.ui.comboBox_remove.clear()
                return

            import numpy as np
            preds = np.asarray(poly[source_field]).reshape(-1)
            uniq = np.unique(preds)

            cfg_name = cloud.get('predict_cfg') if isinstance(cloud.get('predict_cfg'), str) else self._get_selected_cfg_name()
            names, colors = self._get_class_names_and_colors(cfg_name)

            self.ui.comboBox_transform.blockSignals(True)
            self.ui.comboBox_remove.blockSignals(True)
            self.ui.comboBox_transform.clear()
            self.ui.comboBox_remove.clear()

            for cls_id in uniq.tolist():
                cls_int = int(cls_id)
                title = names[cls_int] if 0 <= cls_int < len(names) else f"Class {cls_int}"
                color = colors[cls_int % len(colors)]
                qcol = QColor(color[0], color[1], color[2])

                self.ui.comboBox_transform.addItem(title, cls_int)
                idx_s = self.ui.comboBox_transform.count() - 1
                self.ui.comboBox_transform.setItemData(idx_s, QBrush(qcol), Qt.BackgroundRole)

                self.ui.comboBox_remove.addItem(title, cls_int)
                idx_r = self.ui.comboBox_remove.count() - 1
                self.ui.comboBox_remove.setItemData(idx_r, QBrush(qcol), Qt.BackgroundRole)

            self.ui.comboBox_transform.blockSignals(False)
            self.ui.comboBox_remove.blockSignals(False)
        except Exception:
            pass

    def on_click_analyze_net(self):
        """Обработчик кнопки анализа: открывает диалог выбора нейросети."""
        current = self.ui.treeWidget.currentItem()
        if current is None:
            QMessageBox.information(self, 'Инфо', 'Не выбрано облако точек')
            return
        cloud_name = current.data(0, Qt.UserRole)
        if not cloud_name or cloud_name not in self.point_clouds:
            QMessageBox.information(self, 'Инфо', 'Некорректный выбор облака')
            return

        # Диалог выбора нейросети
        dialog = QDialog(self)
        dialog.setWindowTitle('Выбор нейросети для анализа')
        dialog_layout = QVBoxLayout(dialog)
        
        # Метка
        label = QLabel('Выберите нейросеть для сегментации:')
        dialog_layout.addWidget(label)
        
        # Список нейросетей
        networks_list = QListWidget()
        networks = [
            ('RandLA-Net (Hack2C)', 'randlanet'),
            ('RandLA-Net (Toronto3D)', 'randlanet_toronto3d'),
            ('RandLA-Net (SemanticKITTI)', 'randlanet_semantickitti'),
            ('RandLA-Net (Lidar Scenes)', 'randlanet_lidar_scenes')
        ]
        for name, cfg in networks:
            item = QListWidgetItem(name)
            item.setData(Qt.UserRole, cfg)
            networks_list.addItem(item)
        networks_list.setCurrentRow(0)
        dialog_layout.addWidget(networks_list)
        
        # Кнопки OK/Cancel
        button_layout = QHBoxLayout()
        ok_button = QPushButton('Запустить анализ')
        cancel_button = QPushButton('Отмена')
        ok_button.clicked.connect(dialog.accept)
        cancel_button.clicked.connect(dialog.reject)
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        dialog_layout.addLayout(button_layout)
        
        if dialog.exec_() != QDialog.Accepted:
            return
        
        # Получаем выбранную конфигурацию
        selected_item = networks_list.currentItem()
        if selected_item is None:
            return
        cfg_name = selected_item.data(Qt.UserRole)
        
        # Запускаем инференс с выбранной конфигурацией
        self._run_inference(cloud_name, cfg_name)

    def _run_inference(self, cloud_name, cfg_name):
        """Запустить инференс на облаке с указанной конфигурацией."""
        btn = self.ui.pushButton_analys_net if hasattr(self.ui, 'pushButton_analys_net') else None
        if btn is not None:
            btn.setEnabled(False)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            # 1) Инициализируем/берем pipeline
            self._ensure_ml_pipeline(cfg_name)
            # 2) Достаём точки текущего облака
            cloud = self.point_clouds[cloud_name]
            poly = cloud['poly_data']
            pts = np.asarray(poly.points)
            if pts.size == 0:
                QMessageBox.warning(self, 'Предупреждение', 'Облако пустое')
                return
            
            # 3) Извлекаем intensity если есть
            intensity = None
            if 'intensity' in poly.array_names:
                intensity = np.asarray(poly['intensity']).reshape(-1, 1)
            
            # 4) Запускаем инференс с intensity
            pred = self._run_inference_on_points(pts[:, :3], intensity)
            if pred.shape[0] != pts.shape[0]:
                QMessageBox.warning(self, 'Предупреждение', 'Размер предсказаний не совпадает с числом точек')
                return
            # 5) Кладём предсказания в poly_data и применяем режим
            cloud['poly_data']['predict'] = pred
            cloud['predict_cfg'] = cfg_name
            cloud['color_mode'] = 'predict'
            self.apply_point_color_mode(cloud_name, 'predict')
            # После инференса заполняем трансформ и remove классами выбранной нейросети
            self._populate_class_combos(cloud_name)
            self._populate_transform_combo(cloud_name)
        except Exception as e:
            QMessageBox.critical(self, 'Ошибка инференса', str(e))
        finally:
            QApplication.restoreOverrideCursor()
            if btn is not None:
                btn.setEnabled(True)

    def on_click_export(self):
        """Экспорт текущего облака точек в выбранный формат (пока .pcd)."""
        current = self.ui.treeWidget.currentItem()
        if current is None:
            QMessageBox.information(self, 'Инфо', 'Не выбрано облако точек')
            return
        cloud_name = current.data(0, Qt.UserRole)
        cloud = self.point_clouds.get(cloud_name)
        if not cloud:
            QMessageBox.information(self, 'Инфо', 'Некорректный выбор облака')
            return

        fmt = None
        try:
            fmt = self.ui.comboBox_export.currentText().strip().lower()
        except Exception:
            fmt = '.pcd'
        if fmt not in ('.pcd', 'pcd'):
            QMessageBox.information(self, 'Инфо', f'Формат {fmt} пока не поддерживается')
            return

        # Диалог выбора файла
        default_name = os.path.splitext(cloud_name)[0] + '.pcd'
        out_path, _ = QFileDialog.getSaveFileName(
            self,
            'Сохранить облако точек как PCD',
            default_name,
            'PCD files (*.pcd)'
        )
        if not out_path:
            return
        if not out_path.lower().endswith('.pcd'):
            out_path += '.pcd'

        try:
            import open3d.core as o3c
            poly = cloud['poly_data']
            pts = np.asarray(poly.points).astype(np.float32)

            tpc = o3d.t.geometry.PointCloud()
            tpc.point['positions'] = o3c.Tensor(pts)

            # Сохраняем цвета, если есть
            if 'colors' in poly.array_names:
                cols = np.asarray(poly['colors']).astype(np.uint8)
                tpc.point['colors'] = o3c.Tensor(cols)

            # intensity
            if 'intensity' in poly.array_names:
                intens = np.asarray(poly['intensity']).reshape(-1, 1).astype(np.float32)
                tpc.point['intensity'] = o3c.Tensor(intens)

            # label
            if 'label' in poly.array_names:
                labels = np.asarray(poly['label']).reshape(-1, 1).astype(np.int32)
                tpc.point['label'] = o3c.Tensor(labels)

            # predict
            if 'predict' in poly.array_names:
                preds = np.asarray(poly['predict']).reshape(-1, 1).astype(np.int32)
                tpc.point['predict'] = o3c.Tensor(preds)

            # height как вспомогательный скаляр (опционально)
            if 'height' in poly.array_names:
                h = np.asarray(poly['height']).reshape(-1, 1).astype(np.float32)
                tpc.point['height'] = o3c.Tensor(h)

            # Запись PCD (binary)
            o3d.t.io.write_point_cloud(out_path, tpc, write_ascii=False)
            QMessageBox.information(self, 'Экспорт', f'Сохранено: {out_path}')
        except Exception as e:
            QMessageBox.critical(self, 'Ошибка экспорта', f'Не удалось сохранить PCD: {str(e)}')

    def on_click_interpolate(self):
        """Интерполировать область по выделенным точкам текущего облака"""
        current = self.ui.treeWidget.currentItem()
        if current is None:
            QMessageBox.information(self, 'Инфо', 'Не выбрано облако точек')
            return
        cloud_name = current.data(0, Qt.UserRole)
        if cloud_name in self.point_clouds:
            self.fill_hole(cloud_name)

    def on_click_reset_point(self):
        """Сбросить выделение точек текущего облака"""
        current = self.ui.treeWidget.currentItem()
        if current is None:
            return
        cloud_name = current.data(0, Qt.UserRole)
        if cloud_name in self.point_clouds:
            self.clear_selection(cloud_name)

    def on_click_remove_point(self):
        """Удалить выделенные точки текущего облака"""
        current = self.ui.treeWidget.currentItem()
        if current is None:
            return
        cloud_name = current.data(0, Qt.UserRole)
        if cloud_name in self.point_clouds:
            self.delete_selected_points(cloud_name)

    def on_click_compare(self):
        """Открыть диалог сравнения двух облаков точек."""
        # Получаем список всех облаков (не групп)
        available_clouds = []
        for name, data in self.point_clouds.items():
            if data.get('poly_data') is not None:
                available_clouds.append(name)
        
        if len(available_clouds) < 2:
            QMessageBox.information(
                self,
                'Инфо',
                'Для сравнения необходимо минимум 2 облака точек'
            )
            return
        
        # Диалог выбора облаков и режима сравнения
        dialog = QDialog(self)
        dialog.setWindowTitle('Сравнение облаков точек')
        dialog.setMinimumWidth(400)
        dialog_layout = QVBoxLayout(dialog)
        
        # Выбор исходного облака
        source_layout = QHBoxLayout()
        source_label = QLabel('Исходное облако:')
        source_combo = QComboBox()
        source_combo.addItems(available_clouds)
        source_layout.addWidget(source_label)
        source_layout.addWidget(source_combo)
        dialog_layout.addLayout(source_layout)
        
        # Выбор сравниваемого облака
        target_layout = QHBoxLayout()
        target_label = QLabel('Сравнить с:')
        target_combo = QComboBox()
        target_combo.addItems(available_clouds)
        if len(available_clouds) > 1:
            target_combo.setCurrentIndex(1)
        target_layout.addWidget(target_label)
        target_layout.addWidget(target_combo)
        dialog_layout.addLayout(target_layout)
        
        dialog_layout.addWidget(QLabel(''))  # Разделитель
        
        # Выбор режима сравнения
        mode_group = QGroupBox('Режим сравнения')
        mode_layout = QVBoxLayout()
        
        mode_heatmap = QRadioButton('Distance Heatmap (тепловая карта расстояний)')
        mode_heatmap.setChecked(True)
        mode_heatmap.setToolTip('Показать расстояния цветом: синий=совпадают, красный=удалены')
        mode_layout.addWidget(mode_heatmap)
        
        mode_diff = QRadioButton('Show Differences (только различия)')
        mode_diff.setToolTip('Показать только точки с расстоянием больше порога')
        mode_layout.addWidget(mode_diff)
        
        mode_stats = QRadioButton('Statistics (статистика + карта)')
        mode_stats.setToolTip('Показать метрики и тепловую карту')
        mode_layout.addWidget(mode_stats)
        
        mode_group.setLayout(mode_layout)
        dialog_layout.addWidget(mode_group)
        
        # Порог для режима различий
        threshold_layout = QHBoxLayout()
        threshold_label = QLabel('Порог различий (м):')
        threshold_spinbox = QDoubleSpinBox()
        threshold_spinbox.setRange(0.01, 10.0)
        threshold_spinbox.setSingleStep(0.1)
        threshold_spinbox.setValue(0.5)
        threshold_spinbox.setDecimals(2)
        threshold_layout.addWidget(threshold_label)
        threshold_layout.addWidget(threshold_spinbox)
        dialog_layout.addLayout(threshold_layout)
        
        # Тип colorbar
        colorbar_group = QGroupBox('Тип colorbar')
        colorbar_layout = QVBoxLayout()
        
        colorbar_gradient = QRadioButton('Gradient (плавный градиент)')
        colorbar_gradient.setChecked(True)
        colorbar_gradient.setToolTip('Синий → зеленый → желтый → красный (плавный переход)')
        colorbar_layout.addWidget(colorbar_gradient)
        
        colorbar_binary = QRadioButton('Binary (два цвета по порогу)')
        colorbar_binary.setToolTip('Синий (≤ порог) / Красный (> порог)')
        colorbar_layout.addWidget(colorbar_binary)
        
        colorbar_group.setLayout(colorbar_layout)
        dialog_layout.addWidget(colorbar_group)
        
        # Кнопки OK/Cancel
        button_layout = QHBoxLayout()
        ok_button = QPushButton('Сравнить')
        cancel_button = QPushButton('Отмена')
        ok_button.clicked.connect(dialog.accept)
        cancel_button.clicked.connect(dialog.reject)
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        dialog_layout.addLayout(button_layout)
        
        if dialog.exec_() != QDialog.Accepted:
            return
        
        # Получаем выбранные параметры
        source_name = source_combo.currentText()
        target_name = target_combo.currentText()
        threshold = threshold_spinbox.value()
        
        if source_name == target_name:
            QMessageBox.warning(self, 'Предупреждение', 'Выберите разные облака для сравнения')
            return
        
        # Определяем режим
        if mode_heatmap.isChecked():
            mode = 'heatmap'
        elif mode_diff.isChecked():
            mode = 'differences'
        else:
            mode = 'statistics'
        
        # Определяем тип colorbar
        colorbar_type = 'binary' if colorbar_binary.isChecked() else 'gradient'
        
        # Выполняем сравнение
        self._compare_clouds(source_name, target_name, mode, threshold, colorbar_type)

    def _compare_clouds(self, source_name, target_name, mode, threshold, colorbar_type='gradient'):
        """Выполнить сравнение двух облаков и визуализировать результат."""
        try:
            source_cloud = self.point_clouds.get(source_name)
            target_cloud = self.point_clouds.get(target_name)
            
            if not source_cloud or not target_cloud:
                QMessageBox.warning(self, 'Ошибка', 'Облака не найдены')
                return
            
            QApplication.setOverrideCursor(Qt.WaitCursor)
            
            # Получаем Open3D объекты
            source_pcd = source_cloud['pcd']
            target_pcd = target_cloud['pcd']
            
            # Вычисляем расстояния от исходного до целевого
            distances = np.asarray(source_pcd.compute_point_cloud_distance(target_pcd))
            
            # Статистика
            mean_dist = np.mean(distances)
            max_dist = np.max(distances)
            min_dist = np.min(distances)
            median_dist = np.median(distances)
            points_above_threshold = np.sum(distances > threshold)
            
            if mode == 'statistics':
                # Показываем статистику
                stats_text = (
                    f'Сравнение: {source_name} → {target_name}\n\n'
                    f'Всего точек: {len(distances)}\n'
                    f'Среднее расстояние: {mean_dist:.4f} м\n'
                    f'Медианное расстояние: {median_dist:.4f} м\n'
                    f'Минимальное расстояние: {min_dist:.4f} м\n'
                    f'Максимальное расстояние: {max_dist:.4f} м\n'
                    f'Точек с расстоянием > {threshold} м: {points_above_threshold} '
                    f'({100.0 * points_above_threshold / len(distances):.2f}%)'
                )
                QApplication.restoreOverrideCursor()
                QMessageBox.information(self, 'Статистика сравнения', stats_text)
                QApplication.setOverrideCursor(Qt.WaitCursor)
            
            # Создаем визуализацию
            source_points = np.asarray(source_cloud['poly_data'].points)
            
            if mode == 'differences':
                # Показываем только точки с расстоянием > порога
                mask = distances > threshold
                if not np.any(mask):
                    QApplication.restoreOverrideCursor()
                    QMessageBox.information(
                        self,
                        'Результат',
                        f'Нет точек с расстоянием больше {threshold} м.\n'
                        f'Облака практически идентичны.'
                    )
                    return
                
                diff_points = source_points[mask]
                diff_distances = distances[mask]
                
                # Создаем облако различий
                diff_poly = pv.PolyData(diff_points)
                diff_poly['distance'] = diff_distances
                diff_poly['height'] = diff_points[:, 2]  # Добавляем height для возможности переключения режимов
                
                comparison_name = f"{source_name}_vs_{target_name}_diff"
            else:
                # Режим heatmap - показываем все точки с расстояниями
                diff_poly = pv.PolyData(source_points)
                diff_poly['distance'] = distances
                diff_poly['height'] = source_points[:, 2]  # Добавляем height для возможности переключения режимов
                
                comparison_name = f"{source_name}_vs_{target_name}_heatmap"
            
            # Проверяем уникальность имени
            counter = 1
            base_name = comparison_name
            while comparison_name in self.point_clouds:
                comparison_name = f"{base_name}_{counter}"
                counter += 1
            
            # Создаем Open3D объект
            comparison_pcd = o3d.geometry.PointCloud()
            comparison_pcd.points = o3d.utility.Vector3dVector(np.asarray(diff_poly.points))
            
            # Настройка визуализации в зависимости от типа colorbar
            if colorbar_type == 'binary':
                # Бинарная раскраска: синий (≤ порог) / красный (> порог)
                import matplotlib.colors as mcolors
                
                # Создаем дискретный colormap с двумя цветами
                colors_list = ['blue', 'red']
                n_bins = 2
                cmap_binary = mcolors.ListedColormap(colors_list)
                
                # Создаем границы: все что ≤ threshold -> 0 (синий), > threshold -> 1 (красный)
                binary_values = (diff_poly['distance'] > threshold).astype(int)
                diff_poly['binary'] = binary_values
                
                actor = self.plotter.add_mesh(
                    diff_poly,
                    scalars='binary',
                    cmap=cmap_binary,
                    clim=[0, 1],  # Фиксированный диапазон 0-1
                    point_size=5,
                    show_scalar_bar=False  # Отключаем, создадим отдельно
                )
                
                # Создаем scalar bar отдельно для управления видимостью
                scalar_bar_actor = self.plotter.add_scalar_bar(
                    title=f'Distance\n≤{threshold}m / >{threshold}m',
                    vertical=True,
                    height=0.4,
                    width=0.05,
                    position_x=0.9,
                    position_y=0.3,
                    n_labels=2,
                    mapper=actor.GetMapper()
                )
            else:
                # Градиентная раскраска (по умолчанию)
                actor = self.plotter.add_mesh(
                    diff_poly,
                    scalars='distance',
                    cmap='jet',  # Синий -> зеленый -> желтый -> красный
                    point_size=5,
                    show_scalar_bar=False  # Отключаем, создадим отдельно
                )
                
                # Создаем scalar bar отдельно для управления видимостью
                scalar_bar_actor = self.plotter.add_scalar_bar(
                    title='Distance (m)',
                    vertical=True,
                    height=0.6,
                    width=0.05,
                    position_x=0.9,
                    position_y=0.2,
                    mapper=actor.GetMapper()
                )
            
            # Сохраняем данные
            self.point_clouds[comparison_name] = {
                'pcd': comparison_pcd,
                'poly_data': diff_poly,
                'actor': actor,
                'scalar_bar_actor': scalar_bar_actor,  # Сохраняем scalar bar для управления видимостью
                'visible': True,
                'file_path': '',
                'color_mode': 'distance',
                'selected_indices': set(),
                'selection_actor': None,
                'comparison_info': {
                    'source': source_name,
                    'target': target_name,
                    'mode': mode,
                    'threshold': threshold,
                    'colorbar_type': colorbar_type,
                    'mean_dist': mean_dist,
                    'max_dist': max_dist
                }
            }
            
            # Добавляем в дерево
            self.add_cloud_to_tree(comparison_name)
            
            # Скрываем исходные облака для лучшего просмотра
            self.set_cloud_visibility(source_name, False)
            self.set_cloud_visibility(target_name, False)
            
            # Обновляем чекбоксы в дереве
            root = self.ui.treeWidget.invisibleRootItem()
            for i in range(root.childCount()):
                item = root.child(i)
                cloud_name = item.data(0, Qt.UserRole)
                if cloud_name == source_name or cloud_name == target_name:
                    self.ui.treeWidget.blockSignals(True)
                    item.setCheckState(0, Qt.Unchecked)
                    self.ui.treeWidget.blockSignals(False)
            
            self.plotter.reset_camera()
            
            QApplication.restoreOverrideCursor()
            
            result_text = f'Создано облако сравнения: {comparison_name}\n\n'
            if mode == 'differences':
                result_text += f'Показано {len(diff_points)} точек с различиями > {threshold} м'
            else:
                result_text += (
                    f'Среднее расстояние: {mean_dist:.4f} м\n'
                    f'Максимальное расстояние: {max_dist:.4f} м'
                )
            
            QMessageBox.information(self, 'Сравнение завершено', result_text)
            
        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, 'Ошибка', f'Не удалось сравнить облака: {str(e)}')

    def on_click_crop_tile(self):
        """Открыть диалог параметров тайлинга и разделить облако на тайлы."""
        current = self.ui.treeWidget.currentItem()
        if current is None:
            QMessageBox.information(self, 'Инфо', 'Не выбрано облако точек')
            return
        
        # Получаем имя облака (может быть группа или отдельное облако)
        cloud_name = current.data(0, Qt.UserRole)
        if not cloud_name or cloud_name not in self.point_clouds:
            QMessageBox.information(self, 'Инфо', 'Некорректный выбор облака')
            return
        
        cloud = self.point_clouds[cloud_name]
        poly = cloud.get('poly_data')
        if poly is None or len(poly.points) == 0:
            QMessageBox.information(self, 'Инфо', 'Облако пустое')
            return
        
        # Диалог параметров тайлинга
        dialog = QDialog(self)
        dialog.setWindowTitle('Параметры разделения на тайлы')
        dialog_layout = QVBoxLayout(dialog)
        
        # Размер тайла
        tile_size_layout = QHBoxLayout()
        tile_size_label = QLabel('Размер тайла (метры):')
        tile_size_spinbox = QDoubleSpinBox()
        tile_size_spinbox.setRange(1.0, 1000.0)
        tile_size_spinbox.setSingleStep(1.0)
        tile_size_spinbox.setValue(50.0)
        tile_size_spinbox.setDecimals(1)
        tile_size_layout.addWidget(tile_size_label)
        tile_size_layout.addWidget(tile_size_spinbox)
        dialog_layout.addLayout(tile_size_layout)
        
        # Процент перекрытия
        overlap_layout = QHBoxLayout()
        overlap_label = QLabel('Перекрытие (%):')
        overlap_spinbox = QDoubleSpinBox()
        overlap_spinbox.setRange(0.0, 50.0)
        overlap_spinbox.setSingleStep(5.0)
        overlap_spinbox.setValue(10.0)
        overlap_spinbox.setDecimals(1)
        overlap_layout.addWidget(overlap_label)
        overlap_layout.addWidget(overlap_spinbox)
        dialog_layout.addLayout(overlap_layout)
        
        # Информация об облаке
        points = np.asarray(poly.points)
        bbox_min = points.min(axis=0)
        bbox_max = points.max(axis=0)
        bbox_size = bbox_max - bbox_min
        
        info_label = QLabel(
            f'Размер облака:\n'
            f'X: {bbox_size[0]:.2f} м\n'
            f'Y: {bbox_size[1]:.2f} м\n'
            f'Z: {bbox_size[2]:.2f} м\n'
            f'Точек: {len(points)}'
        )
        dialog_layout.addWidget(info_label)
        
        # Кнопки OK/Cancel
        button_layout = QHBoxLayout()
        ok_button = QPushButton('Создать тайлы')
        cancel_button = QPushButton('Отмена')
        ok_button.clicked.connect(dialog.accept)
        cancel_button.clicked.connect(dialog.reject)
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        dialog_layout.addLayout(button_layout)
        
        if dialog.exec_() != QDialog.Accepted:
            return
        
        tile_size = tile_size_spinbox.value()
        overlap_percent = overlap_spinbox.value()
        
        # Выполняем тайлинг
        self._create_tiles(cloud_name, tile_size, overlap_percent)

    def _create_tiles(self, cloud_name, tile_size, overlap_percent):
        """Разделить облако на тайлы и создать группу в дереве."""
        try:
            cloud = self.point_clouds[cloud_name]
            poly = cloud['poly_data']
            points = np.asarray(poly.points)
            
            # Уникальный ID для группы тайлов
            self._tile_group_counter += 1
            group_id = self._tile_group_counter
            
            # Вычисляем границы облака
            bbox_min = points.min(axis=0)
            bbox_max = points.max(axis=0)
            
            # Вычисляем шаг с учетом перекрытия
            overlap = tile_size * overlap_percent / 100.0
            step = tile_size - overlap
            
            # Генерируем сетку тайлов
            x_coords = np.arange(bbox_min[0], bbox_max[0], step)
            y_coords = np.arange(bbox_min[1], bbox_max[1], step)
            
            if len(x_coords) == 0:
                x_coords = np.array([bbox_min[0]])
            if len(y_coords) == 0:
                y_coords = np.array([bbox_min[1]])
            
            # Создаем тайлы
            tiles = []
            tile_count = 0
            
            QApplication.setOverrideCursor(Qt.WaitCursor)
            
            for i, x in enumerate(x_coords):
                for j, y in enumerate(y_coords):
                    # Границы тайла
                    x_min, x_max = x, x + tile_size
                    y_min, y_max = y, y + tile_size
                    
                    # Фильтруем точки внутри тайла
                    mask = (
                        (points[:, 0] >= x_min) & (points[:, 0] < x_max) &
                        (points[:, 1] >= y_min) & (points[:, 1] < y_max)
                    )
                    
                    if not np.any(mask):
                        continue
                    
                    # Создаем тайл
                    tile_points = points[mask]
                    tile_poly_data = pv.PolyData(tile_points)
                    
                    # Копируем атрибуты
                    for attr_name in poly.array_names:
                        try:
                            attr_data = np.asarray(poly[attr_name])
                            if len(attr_data) == len(points):
                                tile_poly_data[attr_name] = attr_data[mask]
                        except Exception:
                            pass
                    
                    # Имя тайла с уникальным group_id
                    tile_name = f"{cloud_name}_tiles{group_id:03d}_i{i:04d}_j{j:04d}"
                    
                    # Создаем Open3D объект
                    tile_pcd = o3d.geometry.PointCloud()
                    tile_pcd.points = o3d.utility.Vector3dVector(tile_points)
                    
                    # Добавляем в плоттер (изначально скрытым)
                    mode = cloud.get('color_mode', 'height')
                    kwargs = {'point_size': 3, 'show_scalar_bar': False}
                    
                    if mode == 'rgb' and 'colors' in tile_poly_data.array_names:
                        kwargs.update({'scalars': 'colors', 'rgb': True})
                    elif mode == 'height':
                        kwargs.update({'scalars': 'height', 'cmap': 'coolwarm'})
                    else:
                        kwargs.update({'color': 'white'})
                    
                    actor = self.plotter.add_mesh(tile_poly_data, **kwargs)
                    actor.SetVisibility(False)  # Скрываем по умолчанию
                    
                    # Сохраняем данные тайла
                    self.point_clouds[tile_name] = {
                        'pcd': tile_pcd,
                        'poly_data': tile_poly_data,
                        'actor': actor,
                        'visible': False,
                        'file_path': cloud.get('file_path', ''),
                        'color_mode': mode,
                        'selected_indices': set(),
                        'selection_actor': None,
                        'parent_group': f"{cloud_name}_tiles{group_id:03d}",
                        'tile_info': {
                            'i': i,
                            'j': j,
                            'x_min': x_min,
                            'x_max': x_max,
                            'y_min': y_min,
                            'y_max': y_max
                        }
                    }
                    
                    tiles.append(tile_name)
                    tile_count += 1
            
            QApplication.restoreOverrideCursor()
            
            if tile_count == 0:
                QMessageBox.information(self, 'Инфо', 'Не удалось создать тайлы')
                return
            
            # Создаем группу в дереве
            group_name = f"{cloud_name}_tiles{group_id:03d}"
            self._add_tile_group_to_tree(group_name, tiles, cloud_name)
            
            QMessageBox.information(
                self,
                'Успех',
                f'Создано {tile_count} тайлов\n'
                f'Размер: {tile_size}м\n'
                f'Перекрытие: {overlap_percent}%'
            )
            
        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, 'Ошибка', f'Не удалось создать тайлы: {str(e)}')

    def _add_tile_group_to_tree(self, group_name, tiles, source_cloud_name):
        """Добавить группу тайлов в дерево."""
        # Создаем элемент группы
        group_item = QTreeWidgetItem(self.ui.treeWidget)
        group_item.setText(0, f"{source_cloud_name} - tiles ({len(tiles)})")
        group_item.setFlags(group_item.flags() | Qt.ItemIsUserCheckable)
        group_item.setCheckState(0, Qt.Unchecked)
        group_item.setData(0, Qt.UserRole, group_name)
        group_item.setData(0, Qt.UserRole + 1, 'group')  # Метка группы
        group_item.setData(0, Qt.UserRole + 2, tiles)  # Список тайлов в группе
        
        # Добавляем дочерние элементы (тайлы)
        for tile_name in tiles:
            tile_item = QTreeWidgetItem(group_item)
            # Отображаем только координаты тайла
            tile_suffix = tile_name.split('_i')[-1] if '_i' in tile_name else tile_name
            tile_item.setText(0, f"i{tile_suffix}")
            tile_item.setFlags(tile_item.flags() | Qt.ItemIsUserCheckable)
            tile_item.setCheckState(0, Qt.Unchecked)
            tile_item.setData(0, Qt.UserRole, tile_name)
            tile_item.setData(0, Qt.UserRole + 1, 'tile')  # Метка тайла
        
        # Раскрываем группу
        group_item.setExpanded(True)
    
    def setup_pyvista_widget(self):
        """Настройка PyVista виджета в frame_view"""
        layout = QVBoxLayout(self.ui.frame_view)
        
        self.plotter = pvqt.QtInteractor(self.ui.frame_view)
        layout.addWidget(self.plotter.interactor)

        self.plotter.enable_trackball_style()
        self.plotter.enable_eye_dome_lighting()
        self.plotter.enable_parallel_projection()
        
        self.plotter.set_background('white')
        self.plotter.show_axes()
        
    def setup_tree_widget(self):
        """Настройка treeWidget"""
        self.ui.treeWidget.clear()
        self.ui.treeWidget.setHeaderLabels(['Имя'])
        self.ui.treeWidget.setColumnCount(1)
        
    def import_pcd(self):
        """Импорт PCD файлов"""
        file_dialog = QFileDialog()
        file_paths, _ = file_dialog.getOpenFileNames(
            self,
            'Выберите файлы облаков точек',
            '',
            'Point clouds (*.pcd *.ply *.bin);;PCD files (*.pcd);;PLY files (*.ply);;BIN files (*.bin);;All files (*.*)'
        )
        
        for file_path in file_paths:
            self.load_pcd_file(file_path)
            
    def load_pcd_file(self, file_path):
        """Загрузка одного PCD файла"""
        try:
            # Определяем расширение
            ext = os.path.splitext(file_path)[1].lower()
            intensity_from_bin = None
            # Загрузка через open3d или парсинг .bin (KITTI-формат)
            if ext == '.bin':
                points_arr, intensity_from_bin = self._read_bin_points(file_path)
                if points_arr.size == 0:
                    QMessageBox.warning(self, 'Ошибка', f'Файл {file_path} пуст или поврежден')
                    return
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points_arr.astype(np.float32))
            else:
                pcd = o3d.io.read_point_cloud(file_path)
            
            if len(pcd.points) == 0:
                QMessageBox.warning(self, 'Ошибка', f'Файл {file_path} пуст или поврежден')
                return
                
            # Получение имени файла
            file_name = os.path.basename(file_path)
            
            # Проверка на дубликаты
            original_name = file_name
            counter = 1
            while file_name in self.point_clouds:
                file_name = f"{original_name}_{counter}"
                counter += 1
            
            # Конвертация в numpy массивы
            points = np.asarray(pcd.points)
            
            # Проверка наличия цветов
            if pcd.has_colors():
                colors = np.asarray(pcd.colors) * 255  # open3d использует 0-1, pyvista 0-255
            else:
                colors = None
            
            # Создание PyVista PolyData
            poly_data = pv.PolyData(points)
            if colors is not None:
                poly_data['colors'] = colors.astype(np.uint8)
            # Скаляр для градиента по высоте
            poly_data['height'] = points[:, 2]

            # Пытаемся прочитать дополнительные атрибуты через Open3D Tensor API (интенсивность/лейблы)
            if ext != '.bin':
                try:
                    pcd_t = o3d.t.io.read_point_cloud(file_path)
                    # intensity
                    if 'intensity' in pcd_t.point:
                        intensity = pcd_t.point['intensity'].numpy().reshape(-1)
                        if intensity.shape[0] == points.shape[0]:
                            poly_data['intensity'] = intensity.astype(float)
                    # label
                    if 'label' in pcd_t.point:
                        labels = pcd_t.point['label'].numpy().reshape(-1)
                        if labels.shape[0] == points.shape[0]:
                            poly_data['label'] = labels.astype(int)
                    # predict
                    if 'predict' in pcd_t.point:
                        preds = pcd_t.point['predict'].numpy().reshape(-1)
                        if preds.shape[0] == points.shape[0]:
                            poly_data['predict'] = preds.astype(int)
                except Exception:
                    pass
            else:
                # Для .bin поддержим intensity, если прочитали
                if intensity_from_bin is not None and intensity_from_bin.shape[0] == points.shape[0]:
                    poly_data['intensity'] = intensity_from_bin.astype(float)
            
            # Добавление в плоттер (временно белым, без colorbar)
            actor = self.plotter.add_mesh(poly_data, color='white', point_size=3, show_scalar_bar=False)
            
            # Сохранение данных
            self.point_clouds[file_name] = {
                'pcd': pcd,
                'poly_data': poly_data,
                'actor': actor,
                'visible': True,
                'file_path': file_path
            }
            # Инициализация структуры выделения точек
            self.point_clouds[file_name]['selected_indices'] = set()
            self.point_clouds[file_name]['selection_actor'] = None
            
            # Добавление в treeWidget
            self.add_cloud_to_tree(file_name)

            # Немедленно применяем выбранный режим цвета из комбобокса
            index = self.ui.comboBox_point_color.currentIndex()
            mode = self.mode_by_index.get(index, 'height')
            # Сохраняем режим на облаке и применяем
            self.point_clouds[file_name]['color_mode'] = mode
            self.apply_point_color_mode(file_name, mode)
            
            # Автоматическая подгонка камеры
            self.plotter.reset_camera()
            
        except Exception as e:
            QMessageBox.critical(self, 'Ошибка загрузки', f'Не удалось загрузить файл {file_path}:\n{str(e)}')

    def _read_bin_points(self, file_path):
        """Прочитать .bin (KITTI) файл и вернуть (Nx3 float32 points, N intensity or None)."""
        try:
            data = np.fromfile(file_path, dtype=np.float32)
            if data.size == 0:
                return np.empty((0, 3), dtype=np.float32), None
            if data.size % 4 == 0:
                pts = data.reshape(-1, 4)
                return pts[:, :3], pts[:, 3]
            elif data.size % 3 == 0:
                pts = data.reshape(-1, 3)
                return pts[:, :3], None
            else:
                # Нестандартный формат — попытаемся взять первые 3 колонки как XYZ
                n = data.size // 3
                if n > 0:
                    pts = data[:n*3].reshape(-1, 3)
                    return pts[:, :3], None
                return np.empty((0, 3), dtype=np.float32), None
        except Exception:
            return np.empty((0, 3), dtype=np.float32), None
    
    def add_cloud_to_tree(self, cloud_name):
        """Добавление облака в дерево"""
        item = QTreeWidgetItem(self.ui.treeWidget)
        item.setText(0, cloud_name)
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(0, Qt.Checked)
        
        # Связываем элемент дерева с именем облака
        item.setData(0, Qt.UserRole, cloud_name)
        
    def on_tree_item_changed(self, item, column):
        """Обработка изменения состояния чекбокса"""
        if column == 0:
            item_type = item.data(0, Qt.UserRole + 1)
            
            if item_type == 'group':
                # Это группа тайлов - управляем видимостью всех дочерних тайлов
                is_checked = item.checkState(0) == Qt.Checked
                # Блокируем сигналы чтобы избежать рекурсии
                self.ui.treeWidget.blockSignals(True)
                for i in range(item.childCount()):
                    child = item.child(i)
                    child.setCheckState(0, Qt.Checked if is_checked else Qt.Unchecked)
                    tile_name = child.data(0, Qt.UserRole)
                    if tile_name in self.point_clouds:
                        self.set_cloud_visibility(tile_name, is_checked)
                self.ui.treeWidget.blockSignals(False)
            elif item_type == 'tile':
                # Это отдельный тайл - обновляем его видимость и состояние группы
                cloud_name = item.data(0, Qt.UserRole)
                if cloud_name and cloud_name in self.point_clouds:
                    is_checked = item.checkState(0) == Qt.Checked
                    self.set_cloud_visibility(cloud_name, is_checked)
                    
                    # Обновляем состояние родительской группы
                    parent = item.parent()
                    if parent:
                        self._update_group_check_state(parent)
            else:
                # Это обычное облако
                cloud_name = item.data(0, Qt.UserRole)
                if cloud_name and cloud_name in self.point_clouds:
                    is_checked = item.checkState(0) == Qt.Checked
                    self.set_cloud_visibility(cloud_name, is_checked)

    def _update_group_check_state(self, group_item):
        """Обновить состояние чекбокса группы на основе состояния дочерних элементов."""
        if group_item.childCount() == 0:
            return
        
        checked_count = 0
        total_count = group_item.childCount()
        
        for i in range(total_count):
            child = group_item.child(i)
            if child.checkState(0) == Qt.Checked:
                checked_count += 1
        
        # Блокируем сигналы чтобы не вызвать рекурсию
        self.ui.treeWidget.blockSignals(True)
        if checked_count == 0:
            group_item.setCheckState(0, Qt.Unchecked)
        elif checked_count == total_count:
            group_item.setCheckState(0, Qt.Checked)
        else:
            # Частично выбрано - оставляем unchecked (или можно добавить tristate)
            group_item.setCheckState(0, Qt.Unchecked)
        self.ui.treeWidget.blockSignals(False)

    def on_tree_current_item_changed(self, current, previous):
        """Обновление информации при смене выбранного элемента"""
        if current is None:
            self.ui.label_count_value.setText("0")
            self.ui.label_center_value_x.setText("X:0")
            self.ui.label_center_value_y.setText("Y:0")
            self.ui.label_center_value_z.setText("Z:0")
            return
        cloud_name = current.data(0, Qt.UserRole)
        if cloud_name in self.point_clouds:
            pcd = self.point_clouds[cloud_name]['pcd']
            self.ui.label_count_value.setText(str(len(pcd.points)))
            # Обновляем центр
            cx, cy, cz = pcd.get_center()
            self.ui.label_center_value_x.setText(f"X:{cx:.3f}")
            self.ui.label_center_value_y.setText(f"Y:{cy:.3f}")
            self.ui.label_center_value_z.setText(f"Z:{cz:.3f}")
            # Синхронизируем размер точек со спинбоксом
            actor = self.point_clouds[cloud_name]['actor']
            self.ui.spinBox_point_size.setValue(int(actor.GetProperty().GetPointSize()))
            # Синхронизируем режим цвета точек согласно сохраненному состоянию облака
            cloud_data = self.point_clouds[cloud_name]
            mode = cloud_data.get('color_mode', 'height')
            idx = self.index_by_mode.get(mode, 0)
            self.ui.comboBox_point_color.blockSignals(True)
            self.ui.comboBox_point_color.setCurrentIndex(idx)
            self.ui.comboBox_point_color.blockSignals(False)
            # Больше не заполняем комбобоксы при смене облака — они пустые до инференса
        else:
            self.ui.label_count_value.setText("0")
            self.ui.label_center_value_x.setText("X:0")
            self.ui.label_center_value_y.setText("Y:0")
            self.ui.label_center_value_z.setText("Z:0")

    def on_point_size_changed(self, value):
        """Изменение размера точек у выбранного облака"""
        current = self.ui.treeWidget.currentItem()
        if current is None:
            return
        cloud_name = current.data(0, Qt.UserRole)
        if cloud_name in self.point_clouds:
            actor = self.point_clouds[cloud_name]['actor']
            actor.GetProperty().SetPointSize(int(value))
            self.plotter.render()

    def on_eye_dome_changed(self, index):
        """Вкл/Выкл Eye Dome Lighting из комбобокса"""
        try:
            if index == 0:  # Вкл
                self.plotter.enable_eye_dome_lighting()
            else:  # Выкл
                self.plotter.disable_eye_dome_lighting()
            self.plotter.render()
        except Exception:
            pass

    def _populate_transform_combo(self, cloud_name):
        """Заполнить comboBox_transform доступными классами для назначения."""
        try:
            if not hasattr(self.ui, 'comboBox_transform'):
                return
            cloud = self.point_clouds.get(cloud_name)
            if not cloud:
                return
            poly = cloud.get('poly_data')

            # Определяем доступные классы: сначала по данным облака, затем по конфигу
            class_ids = []
            if poly is not None:
                if 'predict' in poly.array_names:
                    class_ids = np.unique(np.asarray(poly['predict']).reshape(-1)).tolist()
                elif 'label' in poly.array_names:
                    class_ids = np.unique(np.asarray(poly['label']).reshape(-1)).tolist()

            cfg_name = cloud.get('predict_cfg') if isinstance(cloud.get('predict_cfg'), str) else self._get_selected_cfg_name()
            names, colors = self._get_class_names_and_colors(cfg_name)
            if not class_ids:
                class_ids = list(range(len(names)))

            self.ui.comboBox_transform.blockSignals(True)
            self.ui.comboBox_transform.clear()
            for cls_id in class_ids:
                cls_int = int(cls_id)
                title = names[cls_int] if 0 <= cls_int < len(names) else f"Class {cls_int}"
                color = colors[cls_int % len(colors)] if colors else (200, 200, 200)
                qcol = QColor(color[0], color[1], color[2])
                self.ui.comboBox_transform.addItem(title, cls_int)
                idx = self.ui.comboBox_transform.count() - 1
                self.ui.comboBox_transform.setItemData(idx, QBrush(qcol), Qt.BackgroundRole)
            self.ui.comboBox_transform.blockSignals(False)
        except Exception:
            pass

    def on_click_transform(self):
        """Открыть диалог выбора класса для преобразования выделенных точек."""
        current = self.ui.treeWidget.currentItem()
        if current is None:
            QMessageBox.information(self, 'Инфо', 'Не выбрано облако точек')
            return
        cloud_name = current.data(0, Qt.UserRole)
        cloud = self.point_clouds.get(cloud_name)
        if not cloud:
            QMessageBox.information(self, 'Инфо', 'Некорректный выбор облака')
            return

        poly = cloud.get('poly_data')
        if poly is None or len(poly.points) == 0:
            QMessageBox.information(self, 'Инфо', 'Нет данных точек для преобразования')
            return

        sel = cloud.get('selected_indices', set())
        if not sel:
            QMessageBox.information(self, 'Инфо', 'Нет выделенных точек для преобразования')
            return

        # Определяем доступные классы
        class_ids = []
        if 'predict' in poly.array_names:
            class_ids = np.unique(np.asarray(poly['predict']).reshape(-1)).tolist()
        elif 'label' in poly.array_names:
            class_ids = np.unique(np.asarray(poly['label']).reshape(-1)).tolist()
        else:
            QMessageBox.information(self, 'Инфо', 'Нет данных классов для преобразования')
            return

        # Получаем конфигурацию и имена классов
        cfg_name = cloud.get('predict_cfg') if isinstance(cloud.get('predict_cfg'), str) else 'randlanet'
        names, colors = self._get_class_names_and_colors(cfg_name)

        # Диалог выбора класса
        dialog = QDialog(self)
        dialog.setWindowTitle('Выбор класса для преобразования')
        dialog_layout = QVBoxLayout(dialog)

        # Метка
        label = QLabel(f'Выберите класс для {len(sel)} выделенных точек:')
        dialog_layout.addWidget(label)

        # Список классов
        classes_list = QListWidget()
        for cls_id in class_ids:
            cls_int = int(cls_id)
            title = names[cls_int] if 0 <= cls_int < len(names) else f"Class {cls_int}"
            color = colors[cls_int % len(colors)] if colors else (200, 200, 200)
            
            item = QListWidgetItem(title)
            item.setData(Qt.UserRole, cls_int)
            # Устанавливаем цвет фона для визуализации
            qcolor = QColor(color[0], color[1], color[2])
            item.setBackground(QBrush(qcolor))
            classes_list.addItem(item)
        
        classes_list.setCurrentRow(0)
        dialog_layout.addWidget(classes_list)

        # Кнопки OK/Cancel
        button_layout = QHBoxLayout()
        ok_button = QPushButton('Преобразовать')
        cancel_button = QPushButton('Отмена')
        ok_button.clicked.connect(dialog.accept)
        cancel_button.clicked.connect(dialog.reject)
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        dialog_layout.addLayout(button_layout)

        if dialog.exec_() != QDialog.Accepted:
            return

        # Получаем выбранный класс
        selected_item = classes_list.currentItem()
        if selected_item is None:
            return
        target_cls = selected_item.data(Qt.UserRole)

        # Выполняем преобразование
        self._transform_points_to_class(cloud_name, target_cls)

    def _transform_points_to_class(self, cloud_name, target_cls):
        """Преобразовать выделенные точки облака в указанный класс."""
        try:
            cloud = self.point_clouds.get(cloud_name)
            if not cloud:
                return

            poly = cloud.get('poly_data')
            sel = cloud.get('selected_indices', set())
            if not sel or poly is None:
                return

            indices = np.array(sorted(list(sel)), dtype=int)

            updated_field = None
            if 'predict' in poly.array_names:
                arr = np.asarray(poly['predict']).copy()
                arr[indices] = int(target_cls)
                poly['predict'] = arr
                updated_field = 'predict'
            elif 'label' in poly.array_names:
                arr = np.asarray(poly['label']).copy()
                arr[indices] = int(target_cls)
                poly['label'] = arr
                updated_field = 'label'
            else:
                arr = np.zeros(len(poly.points), dtype=int)
                arr[indices] = int(target_cls)
                poly['predict'] = arr
                updated_field = 'predict'

            # Обновляем окраску, если активен соответствующий режим
            mode = cloud.get('color_mode', 'height')
            if mode in ('predict', 'label') and updated_field in ('predict', 'label'):
                self.apply_point_color_mode(cloud_name, mode)

            # Обновим списки классов
            self._populate_class_combos(cloud_name)
            self._populate_transform_combo(cloud_name)
            
            # Сбрасываем выделение после преобразования
            self.clear_selection(cloud_name)
        except Exception as e:
            QMessageBox.critical(self, 'Ошибка', f'Не удалось преобразовать точки: {str(e)}')

    def on_display_mode_changed(self, index):
        """Переключение ортографического/перспективного режима"""
        try:
            if index == 0:  # Ортографический
                self.plotter.enable_parallel_projection()
            else:  # Перспективный
                self.plotter.disable_parallel_projection()
            self.plotter.render()
        except Exception:
            pass

    def on_background_color_changed(self, index):
        """Смена цвета фона сцены из комбобокса"""
        try:
            if 0 <= index < len(self.background_palette):
                color = self.background_palette[index][1]
            else:
                color = 'white'
            self.plotter.set_background(color)
            self.plotter.render()
        except Exception:
            pass

    def on_zoom_up(self):
        """Увеличить масштаб камеры"""
        try:
            self.plotter.camera.Zoom(1.2)
            self.plotter.render()
        except Exception:
            pass

    def on_zoom_down(self):
        """Уменьшить масштаб камеры"""
        try:
            self.plotter.camera.Zoom(1.0/1.2)
            self.plotter.render()
        except Exception:
            pass

    def on_reset_view(self):
        """Сбросить вид так, чтобы было видно все"""
        try:
            self.plotter.reset_camera()
            self.plotter.render()
        except Exception:
            pass

    def on_view_front(self):
        """Вид спереди (с +Y к центру)"""
        try:
            self.plotter.view_xz(negative=False)
            self.plotter.reset_camera()
            self.plotter.render()
        except Exception:
            pass

    def on_view_back(self):
        """Вид сзади (с -Y к центру)"""
        try:
            self.plotter.view_xz(negative=True)
            self.plotter.reset_camera()
            self.plotter.render()
        except Exception:
            pass

    def on_view_left(self):
        """Вид слева (с -X к центру)"""
        try:
            self.plotter.view_yz(negative=True)
            self.plotter.reset_camera()
            self.plotter.render()
        except Exception:
            pass

    def on_view_right(self):
        """Вид справа (с +X к центру)"""
        try:
            self.plotter.view_yz(negative=False)
            self.plotter.reset_camera()
            self.plotter.render()
        except Exception:
            pass

    def on_view_top(self):
        """Вид сверху (с +Z к центру)"""
        try:
            self.plotter.view_xy(negative=False)
            self.plotter.reset_camera()
            self.plotter.render()
        except Exception:
            pass

    def on_view_bottom(self):
        """Вид снизу (с -Z к центру)"""
        try:
            self.plotter.view_xy(negative=True)
            self.plotter.reset_camera()
            self.plotter.render()
        except Exception:
            pass

    def on_point_color_changed(self, index):
        """Смена цвета точек у выбранного облака"""
        current = self.ui.treeWidget.currentItem()
        if current is None:
            return
        cloud_name = current.data(0, Qt.UserRole)
        if cloud_name not in self.point_clouds:
            return
        mode = self.mode_by_index.get(index, 'height')
        # Применяем режим (color_mode обновится внутри функции)
        self.apply_point_color_mode(cloud_name, mode)

    def apply_point_color_mode(self, cloud_name, mode):
        """Применить режим окраски точек к облаку"""
        cloud_data = self.point_clouds[cloud_name]
        poly_data = cloud_data['poly_data']
        old_actor = cloud_data['actor']
        point_size = int(old_actor.GetProperty().GetPointSize())
        was_visible = cloud_data.get('visible', True)

        # Удаляем старый актор
        try:
            self.plotter.remove_actor(old_actor)
        except Exception:
            pass
        
        # Управляем scalar bar при смене режима
        old_mode = cloud_data.get('color_mode', 'height')
        has_scalar_bar = cloud_data.get('scalar_bar_actor') is not None
        
        # Скрываем старый scalar bar если был
        if has_scalar_bar and old_mode == 'distance':
            try:
                cloud_data['scalar_bar_actor'].SetVisibility(False)
            except Exception:
                pass

        # Выбор параметров отображения
        kwargs = { 'point_size': point_size, 'show_scalar_bar': False }

        if mode == 'rgb':
            if 'colors' in poly_data.array_names:
                kwargs.update({ 'scalars': 'colors', 'rgb': True })
            else:
                kwargs.update({ 'color': 'white' })
        elif mode == 'intensity':
            # Ищем массив интенсивностей
            intensity_name = None
            for name in poly_data.array_names:
                lname = name.lower()
                if 'intens' in lname:
                    intensity_name = name
                    break
            if intensity_name is not None:
                kwargs.update({ 'scalars': intensity_name, 'cmap': 'viridis' })
            else:
                kwargs.update({ 'color': 'white' })
        elif mode == 'label':
            if 'label' in poly_data.array_names:
                kwargs.update({ 'scalars': 'label', 'cmap': 'tab20' })
            else:
                kwargs.update({ 'color': 'white' })
        elif mode == 'predict' and 'predict' in poly_data.array_names:
            # Переопределим цвета для predict на фиксированные цвета классов
            try:
                # Построим кастомный colormap на основе фиксированных RGB
                import matplotlib.colors as mcolors
                cfg_name = cloud_data.get('predict_cfg') if isinstance(cloud_data.get('predict_cfg'), str) else 'randlanet'
                names, colors = self._get_class_names_and_colors(cfg_name)
                # colors в 0..255 -> 0..1
                colors01 = [(r/255.0, g/255.0, b/255.0) for (r, g, b) in colors]
                # Сгенерируем дискретный ListedColormap
                cmap = mcolors.ListedColormap(colors01, name='fixed_pred')
                kwargs.update({ 'scalars': 'predict', 'cmap': cmap })
            except Exception:
                kwargs.update({ 'scalars': 'predict', 'cmap': 'tab20' })
        elif mode == 'predict':
            kwargs.update({ 'color': 'white' })
        elif mode == 'height':
            # Градиент по высоте: снизу синий, сверху красный
            if 'height' in poly_data.array_names:
                kwargs.update({ 'scalars': 'height', 'cmap': 'coolwarm' })
            else:
                # Если нет height, вычисляем из координат Z
                points = np.asarray(poly_data.points)
                poly_data['height'] = points[:, 2]
                kwargs.update({ 'scalars': 'height', 'cmap': 'coolwarm' })
        elif mode == 'distance':
            # Режим сравнения облаков
            if has_scalar_bar:
                # Восстанавливаем визуализацию distance
                comparison_info = cloud_data.get('comparison_info', {})
                colorbar_type = comparison_info.get('colorbar_type', 'gradient')
                
                if colorbar_type == 'binary' and 'binary' in poly_data.array_names:
                    # Бинарный режим
                    import matplotlib.colors as mcolors
                    colors_list = ['blue', 'red']
                    cmap_binary = mcolors.ListedColormap(colors_list)
                    kwargs.update({ 'scalars': 'binary', 'cmap': cmap_binary, 'clim': [0, 1] })
                elif 'distance' in poly_data.array_names:
                    # Градиентный режим
                    kwargs.update({ 'scalars': 'distance', 'cmap': 'jet' })
                else:
                    kwargs.update({ 'color': 'white' })
            elif 'distance' in poly_data.array_names:
                # Нет scalar bar, но есть данные distance
                kwargs.update({ 'scalars': 'distance', 'cmap': 'jet' })
            else:
                kwargs.update({ 'color': 'white' })
        elif mode in ('white', 'black', 'blue', 'gray'):
            kwargs.update({ 'color': mode })
        else:
            kwargs.update({ 'color': 'white' })

        # Добавляем новый актор
        new_actor = self.plotter.add_mesh(poly_data, **kwargs)
        # Сохраняем текущую видимость
        if not was_visible:
            try:
                new_actor.SetVisibility(False)
            except Exception:
                pass
        cloud_data['actor'] = new_actor
        
        # Показываем scalar bar если переключились на distance и облако видимо
        if mode == 'distance' and has_scalar_bar and was_visible:
            try:
                cloud_data['scalar_bar_actor'].SetVisibility(True)
            except Exception:
                pass
        
        # Обновляем режим окраски в данных облака
        cloud_data['color_mode'] = mode
        
        self.plotter.render()
    
    def set_cloud_visibility(self, cloud_name, visible):
        """Установка видимости облака"""
        if cloud_name in self.point_clouds:
            cloud_data = self.point_clouds[cloud_name]
            actor = cloud_data['actor']
            
            if visible:
                if not cloud_data['visible']:
                    # Показать облако
                    actor.SetVisibility(True)
                    cloud_data['visible'] = True
                    # Показать оверлей выделения, если есть
                    if cloud_data.get('selection_actor') is not None:
                        try:
                            cloud_data['selection_actor'].SetVisibility(True)
                        except Exception:
                            pass
                    # Показать scalar bar, если это облако сравнения с режимом distance
                    if cloud_data.get('scalar_bar_actor') is not None and cloud_data.get('color_mode') == 'distance':
                        try:
                            cloud_data['scalar_bar_actor'].SetVisibility(True)
                        except Exception:
                            pass
            else:
                if cloud_data['visible']:
                    # Скрыть облако
                    actor.SetVisibility(False)
                    cloud_data['visible'] = False
                    # Скрыть оверлей выделения, если есть
                    if cloud_data.get('selection_actor') is not None:
                        try:
                            cloud_data['selection_actor'].SetVisibility(False)
                        except Exception:
                            pass
                    # Скрыть scalar bar, если есть
                    if cloud_data.get('scalar_bar_actor') is not None:
                        try:
                            cloud_data['scalar_bar_actor'].SetVisibility(False)
                        except Exception:
                            pass
                    
                    # Очистка при скрытии облака
                try:
                    if hasattr(self.plotter, 'picker'):
                        self.plotter.picker = None
                except Exception:
                    pass
            
            # Обновить рендер
            self.plotter.render()
    
    def show_context_menu(self, position):
        """Показать контекстное меню для элемента дерева"""
        item = self.ui.treeWidget.itemAt(position)
        if item is None:
            return
        
        item_type = item.data(0, Qt.UserRole + 1)
        
        # Создаем контекстное меню
        context_menu = QMenu(self)
        
        if item_type == 'group':
            # Это группа тайлов
            group_name = item.data(0, Qt.UserRole)
            tiles = item.data(0, Qt.UserRole + 2)
            
            # Объединение тайлов в облако
            merge_action = context_menu.addAction("Объединить тайлы в облако")
            merge_action.triggered.connect(lambda: self.merge_tiles_to_cloud(group_name, tiles))
            
            context_menu.addSeparator()
            
            # Удаление группы тайлов
            delete_group_action = context_menu.addAction("Удалить группу тайлов")
            delete_group_action.triggered.connect(lambda: self.delete_tile_group(group_name, tiles, item))
        else:
            # Это обычное облако или тайл
            cloud_name = item.data(0, Qt.UserRole)
            if not cloud_name or cloud_name not in self.point_clouds:
                return
            
            # Очистка выделения точек
            clear_sel_action = context_menu.addAction("Очистить выделение")
            clear_sel_action.triggered.connect(lambda: self.clear_selection(cloud_name))
            
            # Удаление выделенных точек
            del_sel_action = context_menu.addAction("Удалить выделенные точки")
            del_sel_action.triggered.connect(lambda: self.delete_selected_points(cloud_name))
            
            # Интерполяция области
            fill_hole_action = context_menu.addAction("Интерполировать область (beta)")
            fill_hole_action.triggered.connect(lambda: self.fill_hole(cloud_name))
            
            context_menu.addSeparator()
            
            # Добавляем действие удаления
            delete_action = context_menu.addAction("Удалить")
            delete_action.triggered.connect(lambda: self.delete_cloud(cloud_name))
        
        # Показываем меню в глобальных координатах
        context_menu.exec_(self.ui.treeWidget.mapToGlobal(position))
    
    def merge_tiles_to_cloud(self, group_name, tiles):
        """Объединить тайлы группы в единое облако точек"""
        if not tiles:
            QMessageBox.information(self, 'Инфо', 'Нет тайлов для объединения')
            return
        
        try:
            # Собираем все точки и атрибуты из тайлов
            all_points = []
            all_attributes = {}
            color_mode = 'height'
            
            QApplication.setOverrideCursor(Qt.WaitCursor)
            
            for tile_name in tiles:
                if tile_name not in self.point_clouds:
                    continue
                
                tile_data = self.point_clouds[tile_name]
                poly = tile_data['poly_data']
                
                # Собираем точки
                points = np.asarray(poly.points)
                all_points.append(points)
                
                # Собираем атрибуты
                for attr_name in poly.array_names:
                    try:
                        attr_data = np.asarray(poly[attr_name])
                        if attr_name not in all_attributes:
                            all_attributes[attr_name] = []
                        all_attributes[attr_name].append(attr_data)
                    except Exception:
                        pass
                
                # Запоминаем режим окраски
                if 'color_mode' in tile_data:
                    color_mode = tile_data['color_mode']
            
            if not all_points:
                QApplication.restoreOverrideCursor()
                QMessageBox.warning(self, 'Предупреждение', 'Не удалось собрать точки из тайлов')
                return
            
            # Объединяем все точки
            merged_points = np.vstack(all_points)
            
            # Объединяем атрибуты
            merged_attributes = {}
            for attr_name, attr_list in all_attributes.items():
                try:
                    merged_attributes[attr_name] = np.concatenate(attr_list)
                except Exception:
                    pass
            
            # Создаем имя для объединенного облака
            base_name = group_name.replace('_tiles', '_merged')
            merged_name = base_name
            counter = 1
            while merged_name in self.point_clouds:
                merged_name = f"{base_name}_{counter}"
                counter += 1
            
            # Создаем PyVista PolyData
            merged_poly = pv.PolyData(merged_points)
            for attr_name, attr_data in merged_attributes.items():
                if len(attr_data) == len(merged_points):
                    merged_poly[attr_name] = attr_data
            
            # Создаем Open3D объект
            merged_pcd = o3d.geometry.PointCloud()
            merged_pcd.points = o3d.utility.Vector3dVector(merged_points)
            
            # Добавляем в плоттер
            kwargs = {'point_size': 3, 'show_scalar_bar': False}
            
            if color_mode == 'rgb' and 'colors' in merged_poly.array_names:
                kwargs.update({'scalars': 'colors', 'rgb': True})
            elif color_mode == 'height':
                kwargs.update({'scalars': 'height', 'cmap': 'coolwarm'})
            elif color_mode == 'predict' and 'predict' in merged_poly.array_names:
                # Используем фиксированные цвета классов
                try:
                    import matplotlib.colors as mcolors
                    # Пытаемся получить конфигурацию из первого тайла
                    first_tile = self.point_clouds.get(tiles[0])
                    cfg_name = first_tile.get('predict_cfg', 'randlanet') if first_tile else 'randlanet'
                    names, colors = self._get_class_names_and_colors(cfg_name)
                    colors01 = [(r/255.0, g/255.0, b/255.0) for (r, g, b) in colors]
                    cmap = mcolors.ListedColormap(colors01, name='fixed_pred')
                    kwargs.update({'scalars': 'predict', 'cmap': cmap})
                except Exception:
                    kwargs.update({'scalars': 'predict', 'cmap': 'tab20'})
            else:
                kwargs.update({'color': 'white'})
            
            actor = self.plotter.add_mesh(merged_poly, **kwargs)
            
            # Сохраняем данные объединенного облака
            self.point_clouds[merged_name] = {
                'pcd': merged_pcd,
                'poly_data': merged_poly,
                'actor': actor,
                'visible': True,
                'file_path': '',
                'color_mode': color_mode,
                'selected_indices': set(),
                'selection_actor': None
            }
            
            # Копируем конфигурацию предсказаний если есть
            first_tile = self.point_clouds.get(tiles[0])
            if first_tile and 'predict_cfg' in first_tile:
                self.point_clouds[merged_name]['predict_cfg'] = first_tile['predict_cfg']
            
            # Добавляем в дерево
            self.add_cloud_to_tree(merged_name)
            
            # Подгоняем камеру
            self.plotter.reset_camera()
            
            QApplication.restoreOverrideCursor()
            
            # Спрашиваем, удалить ли исходные тайлы
            reply = QMessageBox.question(
                self,
                'Объединение завершено',
                f'Облако "{merged_name}" создано ({len(merged_points)} точек).\n\n'
                f'Удалить исходную группу тайлов?',
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                # Находим элемент группы в дереве и удаляем
                root = self.ui.treeWidget.invisibleRootItem()
                for i in range(root.childCount()):
                    item = root.child(i)
                    if item.data(0, Qt.UserRole) == group_name:
                        self.delete_tile_group(group_name, tiles, item)
                        break
            
        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, 'Ошибка', f'Не удалось объединить тайлы: {str(e)}')

    def delete_tile_group(self, group_name, tiles, group_item):
        """Удаление группы тайлов"""
        if not tiles:
            return
        
        # Подтверждение удаления только если вызвано напрямую (не из merge)
        import inspect
        caller_name = inspect.stack()[1].function
        if caller_name != 'merge_tiles_to_cloud':
            reply = QMessageBox.question(
                self,
                'Подтверждение удаления',
                f'Вы уверены, что хотите удалить группу с {len(tiles)} тайлами?',
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply != QMessageBox.Yes:
                return
        
        try:
            # Удаляем все тайлы из группы
            for tile_name in tiles:
                if tile_name in self.point_clouds:
                    cloud_data = self.point_clouds[tile_name]
                    
                    # Удаляем актор из плоттера
                    try:
                        self.plotter.remove_actor(cloud_data['actor'])
                    except Exception:
                        pass
                    
                    # Удаляем актор выделения если есть
                    if cloud_data.get('selection_actor') is not None:
                        try:
                            self.plotter.remove_actor(cloud_data['selection_actor'])
                        except Exception:
                            pass
                    
                    # Удаляем из словаря
                    del self.point_clouds[tile_name]
            
            # Удаляем группу из дерева
            root = self.ui.treeWidget.invisibleRootItem()
            root.removeChild(group_item)
            
            # Обновляем рендер
            self.plotter.render()
            
            # Обновляем счетчик
            current = self.ui.treeWidget.currentItem()
            if current is not None:
                self.on_tree_current_item_changed(current, None)
            else:
                self.ui.label_count_value.setText("0")
                
        except Exception as e:
            QMessageBox.critical(self, 'Ошибка удаления', f'Не удалось удалить группу тайлов:\n{str(e)}')

    def delete_cloud(self, cloud_name):
        """Удаление облака точек"""
        if cloud_name not in self.point_clouds:
            return
            
        # Подтверждение удаления
        reply = QMessageBox.question(
            self, 
            'Подтверждение удаления', 
            f'Вы уверены, что хотите удалить облако "{cloud_name}"?',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
            return
            
        try:
            # Получаем данные облака
            cloud_data = self.point_clouds[cloud_name]
            actor = cloud_data['actor']
            
            # Удаляем актор из плоттера
            self.plotter.remove_actor(actor)
            
            # Удаляем актор выделения если есть
            if cloud_data.get('selection_actor') is not None:
                try:
                    self.plotter.remove_actor(cloud_data['selection_actor'])
                except Exception:
                    pass
            
            # Удаляем scalar bar если есть (для облаков сравнения)
            if cloud_data.get('scalar_bar_actor') is not None:
                try:
                    self.plotter.remove_actor(cloud_data['scalar_bar_actor'])
                except Exception:
                    pass
            
            # Удаляем из словаря
            del self.point_clouds[cloud_name]
            
            # Удаляем из дерева
            self.remove_cloud_from_tree(cloud_name)
            
            # Обновляем рендер
            self.plotter.render()
            
            # Обновляем счетчик по текущему выделению
            current = self.ui.treeWidget.currentItem()
            if current is not None:
                self.on_tree_current_item_changed(current, None)
            else:
                self.ui.label_count_value.setText("0")
            
        except Exception as e:
            QMessageBox.critical(self, 'Ошибка удаления', f'Не удалось удалить облако {cloud_name}:\n{str(e)}')
    
    def remove_cloud_from_tree(self, cloud_name):
        """Удаление элемента из дерева"""
        root = self.ui.treeWidget.invisibleRootItem()
        for i in range(root.childCount()):
            item = root.child(i)
            if item.data(0, Qt.UserRole) == cloud_name:
                root.removeChild(item)
                break

    def on_toggle_point_select(self, checked):
        """Вкл/Выкл режима покликового выбора точек правой кнопкой"""
        self.selection_mode = bool(checked)
        
        # Взаимное исключение режимов
        if self.selection_mode and self.brush_mode:
            self.brush_mode = False
            self.ui.pushButton_brush.setChecked(False)
        
        try:
            if self.selection_mode:
                # Отключаем любой предыдущий picking перед включением нового
                try:
                    if hasattr(self.plotter, 'disable_picking'):
                        self.plotter.disable_picking()
                except Exception:
                    pass
                
                # Сохраняем текущую позицию камеры перед включением picking
                saved_camera_pos = self.plotter.camera_position
                
                # Включаем point picking
                try:
                    self.plotter.enable_point_picking(
                        callback=self.on_point_pick,
                        left_clicking=False,
                        show_message=False,
                        show_point=False,
                        color='white',
                        opacity=0.0
                    )
                except Exception:
                    pass
                
                # Восстанавливаем позицию камеры после включения picking
                self.plotter.camera_position = saved_camera_pos
                self.plotter.render()
                
                # Устанавливаем фокус на главное окно для обработки клавиш
                self.setFocus()
            else:
                # Отключаем picking
                try:
                    if hasattr(self.plotter, 'disable_picking'):
                        self.plotter.disable_picking()
                except Exception:
                    pass
                
                # Очистка внутреннего выделения PyVista
                try:
                    if hasattr(self.plotter, 'picker'):
                        self.plotter.picker = None
                    self.plotter.render()
                except Exception:
                    pass
        except Exception:
            pass
        
        self.update_status_bar()

    def on_point_pick(self, picked_point):
        """Правый клик: выбрать/снять выбор точки у текущего облака"""
        try:
            # Проверяем, что режим выделения включен
            if not self.selection_mode:
                return
                
            # Сохраняем текущее состояние камеры
            camera_pos = self.plotter.camera_position
            
            current = self.ui.treeWidget.currentItem()
            if current is None:
                return
            cloud_name = current.data(0, Qt.UserRole)
            if cloud_name not in self.point_clouds:
                return
            
            cloud = self.point_clouds[cloud_name]
            pts = np.asarray(cloud['poly_data'].points)
            if pts.size == 0:
                return
                
            # Найдем ближайшую точку к месту клика
            p = np.asarray(picked_point, dtype=float).reshape(1, 3)
            d2 = np.sum((pts - p) ** 2, axis=1)
            idx = int(np.argmin(d2))
            
            # Переключаем выделение
            sel = cloud.get('selected_indices', set())
            if idx in sel:
                sel.remove(idx)
            else:
                sel.add(idx)
            cloud['selected_indices'] = sel
            
            # Обновляем визуальный оверлей
            self.update_selection_overlay(cloud_name)
            
            # Восстанавливаем позицию камеры
            self.plotter.camera_position = camera_pos
            
        except Exception:
            pass

    def update_selection_overlay(self, cloud_name):
        """Обновить визуальный оверлей выделенных точек"""
        cloud = self.point_clouds.get(cloud_name)
        if not cloud:
            return
            
        try:
            # Удаляем старый оверлей
            if cloud.get('selection_actor') is not None:
                try:
                    self.plotter.remove_actor(cloud['selection_actor'])
                except Exception:
                    pass
                cloud['selection_actor'] = None

            sel = cloud.get('selected_indices', set())
            if not sel:
                self.plotter.render()
                return

            # Создаем оверлей из выделенных точек
            total = len(cloud['poly_data'].points)
            mask = np.zeros(total, dtype=bool)
            mask[list(sel)] = True
            sub = cloud['poly_data'].extract_points(mask, include_cells=False)

            # Размер точек для оверлея
            point_size = max(1, int(self.ui.spinBox_point_size.value()) + 2)
            actor = self.plotter.add_mesh(
                sub,
                color='yellow',
                point_size=point_size,
                show_scalar_bar=False,
                pickable=False,
            )
            
            # Учитываем видимость основного облака
            if not cloud.get('visible', True):
                try:
                    actor.SetVisibility(False)
                except Exception:
                    pass
                    
            cloud['selection_actor'] = actor
            self.plotter.render()
            
        except Exception:
            pass

    def clear_selection(self, cloud_name):
        """Очистить выделение точек у облака"""
        cloud = self.point_clouds.get(cloud_name)
        if not cloud:
            return
            
        try:
            cloud['selected_indices'] = set()
            if cloud.get('selection_actor') is not None:
                try:
                    self.plotter.remove_actor(cloud['selection_actor'])
                except Exception:
                    pass
                cloud['selection_actor'] = None
            self.plotter.render()
        except Exception:
            pass

    def delete_selected_points(self, cloud_name):
        """Удалить выделенные точки из облака"""
        cloud = self.point_clouds.get(cloud_name)
        if not cloud:
            return
            
        sel = cloud.get('selected_indices', set())
        if not sel:
            QMessageBox.information(self, 'Инфо', 'Нет выделенных точек для удаления')
            return
            
        # Подтверждение удаления
        reply = QMessageBox.question(
            self, 
            'Подтверждение удаления', 
            f'Удалить {len(sel)} выделенных точек из облака "{cloud_name}"?',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
            return
            
        try:
            # Создаем маску для сохранения невыделенных точек
            total = len(cloud['poly_data'].points)
            keep_mask = np.ones(total, dtype=bool)
            keep_mask[list(sel)] = False
            
            # Извлекаем невыделенные точки
            new_poly_data = cloud['poly_data'].extract_points(keep_mask, include_cells=False)
            
            if len(new_poly_data.points) == 0:
                QMessageBox.warning(self, 'Предупреждение', 'Удаление всех точек приведет к пустому облаку')
                return
                
            # Обновляем Open3D объект
            new_points = np.asarray(new_poly_data.points)
            cloud['pcd'].points = o3d.utility.Vector3dVector(new_points)
            
            # Обновляем цвета если есть
            if 'colors' in new_poly_data.array_names:
                colors = np.asarray(new_poly_data['colors']) / 255.0
                cloud['pcd'].colors = o3d.utility.Vector3dVector(colors)
            
            # Обновляем poly_data
            cloud['poly_data'] = new_poly_data
            
            # Очищаем выделение
            cloud['selected_indices'] = set()
            if cloud.get('selection_actor') is not None:
                try:
                    self.plotter.remove_actor(cloud['selection_actor'])
                except Exception:
                    pass
                cloud['selection_actor'] = None
            
            # Применяем текущий режим окраски
            mode = cloud.get('color_mode', 'height')
            self.apply_point_color_mode(cloud_name, mode)
            
            # Обновляем информацию в интерфейсе
            current = self.ui.treeWidget.currentItem()
            if current and current.data(0, Qt.UserRole) == cloud_name:
                self.on_tree_current_item_changed(current, None)
                
        except Exception as e:
            QMessageBox.critical(self, 'Ошибка', f'Не удалось удалить точки: {str(e)}')

    def fill_hole(self, cloud_name):
        """Заполнить дыру в облаке точек на основе выделенных граничных точек"""
        cloud = self.point_clouds.get(cloud_name)
        if not cloud:
            return
            
        sel = cloud.get('selected_indices', set())
        if not sel or len(sel) < 3:
            QMessageBox.information(
                self, 
                'Инфо', 
                'Выберите как минимум 3 граничные точки дыры для заполнения'
            )
            return
        
        # Диалог настройки параметров интерполяции
        dialog = QDialog(self)
        dialog.setWindowTitle('Параметры интерполяции области')
        dialog_layout = QVBoxLayout(dialog)
        
        # Параметр плотности
        density_layout = QHBoxLayout()
        density_label = QLabel('Коэффициент разреженности:')
        density_spinbox = QDoubleSpinBox()
        density_spinbox.setRange(1.0, 10.0)
        density_spinbox.setSingleStep(0.5)
        density_spinbox.setValue(2.0)
        density_spinbox.setToolTip('Чем больше значение, тем реже точки (1.0 = исходная плотность)')
        density_layout.addWidget(density_label)
        density_layout.addWidget(density_spinbox)
        dialog_layout.addLayout(density_layout)
        
        # Параметр шума
        noise_layout = QHBoxLayout()
        noise_label = QLabel('Уровень шума:')
        noise_spinbox = QDoubleSpinBox()
        noise_spinbox.setRange(0.0, 1.0)
        noise_spinbox.setSingleStep(0.05)
        noise_spinbox.setValue(0.1)
        noise_spinbox.setToolTip('0.0 = без шума, 1.0 = максимальный шум (доля от среднего расстояния)')
        noise_layout.addWidget(noise_label)
        noise_layout.addWidget(noise_spinbox)
        dialog_layout.addLayout(noise_layout)
        
        # Кнопки OK/Cancel
        button_layout = QHBoxLayout()
        ok_button = QPushButton('OK')
        cancel_button = QPushButton('Отмена')
        ok_button.clicked.connect(dialog.accept)
        cancel_button.clicked.connect(dialog.reject)
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        dialog_layout.addLayout(button_layout)
        
        if dialog.exec_() != QDialog.Accepted:
            return
        
        density_factor = density_spinbox.value()
        noise_level = noise_spinbox.value()
            
        try:
            from scipy.spatial import Delaunay
            from scipy.interpolate import griddata
            
            # Получаем выделенные точки (границы дыры)
            poly_data = cloud['poly_data']
            all_points = np.asarray(poly_data.points)
            boundary_indices = np.array(sorted(list(sel)))
            boundary_points = all_points[boundary_indices]
            
            # Вычисляем среднюю плотность выделенных точек
            # (среднее расстояние до ближайшего соседа)
            from scipy.spatial import KDTree
            tree = KDTree(boundary_points)
            distances, _ = tree.query(boundary_points, k=2)  # k=2: сама точка и ближайший сосед
            avg_density = np.mean(distances[:, 1])  # берём расстояние до ближайшего соседа
            
            if avg_density <= 0:
                avg_density = 0.1  # fallback значение
            
            # Применяем коэффициент разреженности
            avg_density *= density_factor
            
            # Проецируем граничные точки на плоскость (используем локальную систему координат)
            # Находим главные компоненты для определения плоскости
            centroid = np.mean(boundary_points, axis=0)
            centered = boundary_points - centroid
            
            # SVD для нахождения базиса плоскости
            U, S, Vt = np.linalg.svd(centered)
            # Первые два вектора из Vt — базис плоскости
            basis_u = Vt[0]
            basis_v = Vt[1]
            normal = Vt[2]
            
            # Проецируем граничные точки на 2D координаты в плоскости
            boundary_2d = np.column_stack([
                np.dot(centered, basis_u),
                np.dot(centered, basis_v)
            ])
            
            # Создаём триангуляцию Делоне на 2D точках
            tri = Delaunay(boundary_2d)
            
            # Создаём сетку точек внутри выпуклой оболочки граничных точек
            min_x, min_y = boundary_2d.min(axis=0)
            max_x, max_y = boundary_2d.max(axis=0)
            
            # Количество точек по каждой оси на основе плотности
            n_x = int((max_x - min_x) / avg_density) + 1
            n_y = int((max_y - min_y) / avg_density) + 1
            
            # Ограничиваем количество точек чтобы не было слишком много
            n_x = min(max(n_x, 5), 500)
            n_y = min(max(n_y, 5), 500)
            
            x_grid = np.linspace(min_x, max_x, n_x)
            y_grid = np.linspace(min_y, max_y, n_y)
            xx, yy = np.meshgrid(x_grid, y_grid)
            grid_2d = np.column_stack([xx.ravel(), yy.ravel()])
            
            # Фильтруем точки: оставляем только те, что внутри выпуклой оболочки
            simplex_indices = tri.find_simplex(grid_2d)
            inside_mask = simplex_indices >= 0
            inside_2d = grid_2d[inside_mask]
            
            if len(inside_2d) == 0:
                QMessageBox.warning(
                    self,
                    'Предупреждение',
                    'Не удалось создать точки внутри выделенной области'
                )
                return
            
            # Интерполируем Z-координату (высоту вдоль нормали к плоскости)
            # Используем значения высоты граничных точек
            boundary_heights = np.dot(centered, normal)
            interpolated_heights = griddata(
                boundary_2d,
                boundary_heights,
                inside_2d,
                method='linear',
                fill_value=np.mean(boundary_heights)
            )
            
            # Преобразуем 2D координаты обратно в 3D
            new_points_3d = centroid + \
                inside_2d[:, 0:1] * basis_u + \
                inside_2d[:, 1:2] * basis_v + \
                interpolated_heights.reshape(-1, 1) * normal
            
            # Добавляем шум к новым точкам
            if noise_level > 0:
                # Генерируем случайный шум в 3D
                noise_magnitude = avg_density * noise_level
                noise = np.random.normal(0, noise_magnitude, new_points_3d.shape)
                new_points_3d += noise
            
            # Добавляем новые точки к облаку
            updated_points = np.vstack([all_points, new_points_3d])
            
            # Создаём новый poly_data с обновлёнными точками
            new_poly_data = pv.PolyData(updated_points)
            
            # Копируем существующие атрибуты и добавляем значения для новых точек
            n_old = len(all_points)
            n_new = len(new_points_3d)
            
            # Цвета: для новых точек используем средний цвет граничных точек
            if 'colors' in poly_data.array_names:
                old_colors = np.asarray(poly_data['colors'])
                boundary_colors = old_colors[boundary_indices]
                avg_color = np.mean(boundary_colors, axis=0).astype(np.uint8)
                new_colors = np.vstack([
                    old_colors,
                    np.tile(avg_color, (n_new, 1))
                ])
                new_poly_data['colors'] = new_colors
            
            # Height
            new_poly_data['height'] = updated_points[:, 2]
            
            # Intensity: для новых точек используем среднюю интенсивность граничных
            if 'intensity' in poly_data.array_names:
                old_intensity = np.asarray(poly_data['intensity'])
                boundary_intensity = old_intensity[boundary_indices]
                avg_intensity = np.mean(boundary_intensity)
                new_intensity = np.concatenate([
                    old_intensity,
                    np.full(n_new, avg_intensity)
                ])
                new_poly_data['intensity'] = new_intensity
            
            # Label: для новых точек используем наиболее частый лейбл границы
            if 'label' in poly_data.array_names:
                old_labels = np.asarray(poly_data['label'])
                boundary_labels = old_labels[boundary_indices]
                # Наиболее частый лейбл
                unique, counts = np.unique(boundary_labels, return_counts=True)
                most_common_label = unique[np.argmax(counts)]
                new_labels = np.concatenate([
                    old_labels,
                    np.full(n_new, most_common_label, dtype=old_labels.dtype)
                ])
                new_poly_data['label'] = new_labels
            
            # Predict: аналогично label
            if 'predict' in poly_data.array_names:
                old_predict = np.asarray(poly_data['predict'])
                boundary_predict = old_predict[boundary_indices]
                unique, counts = np.unique(boundary_predict, return_counts=True)
                most_common_predict = unique[np.argmax(counts)]
                new_predict = np.concatenate([
                    old_predict,
                    np.full(n_new, most_common_predict, dtype=old_predict.dtype)
                ])
                new_poly_data['predict'] = new_predict
            
            # Обновляем Open3D объект
            cloud['pcd'].points = o3d.utility.Vector3dVector(updated_points)
            
            # Обновляем poly_data
            cloud['poly_data'] = new_poly_data
            
            # Очищаем выделение
            cloud['selected_indices'] = set()
            if cloud.get('selection_actor') is not None:
                try:
                    self.plotter.remove_actor(cloud['selection_actor'])
                except Exception:
                    pass
                cloud['selection_actor'] = None
            
            # Применяем текущий режим окраски
            mode = cloud.get('color_mode', 'height')
            self.apply_point_color_mode(cloud_name, mode)
            
            # Обновляем информацию в интерфейсе
            current = self.ui.treeWidget.currentItem()
            if current and current.data(0, Qt.UserRole) == cloud_name:
                self.on_tree_current_item_changed(current, None)
            
            QMessageBox.information(
                self,
                'Успех',
                f'Область интерполирована: добавлено {n_new} точек'
            )
                
        except ImportError:
            QMessageBox.critical(
                self, 
                'Ошибка', 
                'Для интерполяции областей требуется библиотека scipy.\nУстановите её: pip install scipy'
            )
        except Exception as e:
            QMessageBox.critical(
                self, 
                'Ошибка', 
                f'Не удалось интерполировать область: {str(e)}'
            )

    def on_click_remove_by_class(self):
        """Открыть диалог выбора класса для удаления всех точек этого класса."""
        current = self.ui.treeWidget.currentItem()
        if current is None:
            QMessageBox.information(self, 'Инфо', 'Не выбрано облако точек')
            return
        cloud_name = current.data(0, Qt.UserRole)
        cloud = self.point_clouds.get(cloud_name)
        if not cloud:
            QMessageBox.information(self, 'Инфо', 'Некорректный выбор облака')
            return

        poly = cloud.get('poly_data')
        if poly is None:
            QMessageBox.information(self, 'Инфо', 'Нет данных облака')
            return

        # Определяем поле классов: predict предпочтительно, иначе label
        field = 'predict' if 'predict' in poly.array_names else ('label' if 'label' in poly.array_names else None)
        if field is None:
            QMessageBox.information(self, 'Инфо', 'Нет данных классов (predict/label) для удаления')
            return

        # Получаем уникальные классы и их количество точек
        preds = np.asarray(poly[field]).reshape(-1)
        if preds.size == 0:
            QMessageBox.information(self, 'Инфо', 'Нет точек для удаления')
            return

        unique_classes, counts = np.unique(preds, return_counts=True)
        
        # Получаем конфигурацию и имена классов
        cfg_name = cloud.get('predict_cfg') if isinstance(cloud.get('predict_cfg'), str) else 'randlanet'
        names, colors = self._get_class_names_and_colors(cfg_name)

        # Диалог выбора класса
        dialog = QDialog(self)
        dialog.setWindowTitle('Удаление точек по классу')
        dialog_layout = QVBoxLayout(dialog)

        # Метка
        label = QLabel('Выберите класс для удаления всех его точек:')
        dialog_layout.addWidget(label)

        # Список классов с количеством точек
        classes_list = QListWidget()
        for cls_id, count in zip(unique_classes, counts):
            cls_int = int(cls_id)
            title = names[cls_int] if 0 <= cls_int < len(names) else f"Class {cls_int}"
            color = colors[cls_int % len(colors)] if colors else (200, 200, 200)
            
            item = QListWidgetItem(f"{title} ({count} точек)")
            item.setData(Qt.UserRole, cls_int)
            # Устанавливаем цвет фона для визуализации
            qcolor = QColor(color[0], color[1], color[2])
            item.setBackground(QBrush(qcolor))
            classes_list.addItem(item)
        
        classes_list.setCurrentRow(0)
        dialog_layout.addWidget(classes_list)

        # Кнопки OK/Cancel
        button_layout = QHBoxLayout()
        ok_button = QPushButton('Удалить класс')
        cancel_button = QPushButton('Отмена')
        ok_button.clicked.connect(dialog.accept)
        cancel_button.clicked.connect(dialog.reject)
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        dialog_layout.addLayout(button_layout)

        if dialog.exec_() != QDialog.Accepted:
            return

        # Получаем выбранный класс
        selected_item = classes_list.currentItem()
        if selected_item is None:
            return
        cls_id = selected_item.data(Qt.UserRole)

        # Выполняем удаление
        self._remove_points_by_class(cloud_name, cls_id, field)

    def _remove_points_by_class(self, cloud_name, cls_id, field):
        """Удалить все точки облака, принадлежащие указанному классу."""
        try:
            cloud = self.point_clouds.get(cloud_name)
            if not cloud:
                return

            poly = cloud.get('poly_data')
            if poly is None:
                return

            preds = np.asarray(poly[field]).reshape(-1)
            if preds.size == 0:
                return

            # Маска точек, которые нужно оставить (все, кроме выбранного класса)
            keep_mask = preds != int(cls_id)
            if not np.any(keep_mask):
                QMessageBox.warning(self, 'Предупреждение', 'Удаление всех точек приведет к пустому облаку')
                return

            new_poly_data = poly.extract_points(keep_mask, include_cells=False)
            if len(new_poly_data.points) == 0:
                QMessageBox.warning(self, 'Предупреждение', 'После удаления облако пусто')
                return

            # Обновляем Open3D объект
            new_points = np.asarray(new_poly_data.points)
            cloud['pcd'].points = o3d.utility.Vector3dVector(new_points)

            # Перенесём при наличии — цвета/скаляры (polydata уже содержит нужные массивы)
            cloud['poly_data'] = new_poly_data

            # Очистим выделение
            cloud['selected_indices'] = set()
            if cloud.get('selection_actor') is not None:
                try:
                    self.plotter.remove_actor(cloud['selection_actor'])
                except Exception:
                    pass
                cloud['selection_actor'] = None

            # Применим текущий режим окраски и обновим комбобоксы классов
            mode = cloud.get('color_mode', 'height')
            self.apply_point_color_mode(cloud_name, mode)
            self._populate_class_combos(cloud_name)

            # Обновим панель информации
            current_item = self.ui.treeWidget.currentItem()
            if current_item and current_item.data(0, Qt.UserRole) == cloud_name:
                self.on_tree_current_item_changed(current_item, None)

        except Exception as e:
            QMessageBox.critical(self, 'Ошибка', f'Не удалось удалить точки по классу: {str(e)}')

    def update_status_bar(self):
        """Обновить статусбар с информацией о текущем режиме"""
        if self.brush_mode:
            self.statusBar().showMessage(
                f"Режим кисти: радиус {self.brush_radius:.3f} (размер {self.brush_size}) | Кнопки [+]/[−] для изменения размера | Alt+ПКМ для снятия выделения"
            )
        elif self.selection_mode:
            self.statusBar().showMessage("Режим точечного выделения | ПКМ для выделения/снятия выделения точек")
        else:
            self.statusBar().clearMessage()

    def _sync_brush_radius(self):
        """Синхронизировать внутренний радиус кисти с целочисленным размером 1..20.
        """
        try:
            size = int(self.brush_size)
        except Exception:
            size = 10
        size = max(1, min(20, size))
        self.brush_size = size
        self.brush_radius = float(size)

    def on_toggle_brush_select(self, checked):
        """Вкл/Выкл режима выделения кистью"""
        self.brush_mode = bool(checked)
        
        # Взаимное исключение режимов
        if self.brush_mode and self.selection_mode:
            self.selection_mode = False
            self.ui.pushButton_point.setChecked(False)
        
        try:
            if self.brush_mode:
                # Отключаем любой предыдущий picking перед включением нового
                try:
                    if hasattr(self.plotter, 'disable_picking'):
                        self.plotter.disable_picking()
                except Exception:
                    pass
                
                # Сохраняем текущую позицию камеры
                saved_camera_pos = self.plotter.camera_position
                
                # Включаем point picking для кисти
                try:
                    self.plotter.enable_point_picking(
                        callback=self.on_brush_pick,
                        left_clicking=False,
                        show_message=False,
                        show_point=False,
                        color='white',
                        opacity=0.0
                    )
                except Exception:
                    pass
                
                # Восстанавливаем позицию камеры
                self.plotter.camera_position = saved_camera_pos
                self.plotter.render()
                
                # Устанавливаем фокус на главное окно для обработки клавиш +/-
                self.setFocus()
            else:
                # Отключаем picking
                try:
                    if hasattr(self.plotter, 'disable_picking'):
                        self.plotter.disable_picking()
                except Exception:
                    pass
                
                # Очистка внутреннего выделения PyVista
                try:
                    if hasattr(self.plotter, 'picker'):
                        self.plotter.picker = None
                    self.plotter.render()
                except Exception:
                    pass
        except Exception:
            pass
        
        self.update_status_bar()

    def on_brush_pick(self, picked_point):
        """Правый клик: выделение кистью с радиусом"""
        try:
            # Проверяем, что режим кисти включен
            if not self.brush_mode:
                return
            
            # Сохраняем текущее состояние камеры
            camera_pos = self.plotter.camera_position
            
            current = self.ui.treeWidget.currentItem()
            if current is None:
                return
            cloud_name = current.data(0, Qt.UserRole)
            if cloud_name not in self.point_clouds:
                return
            
            cloud = self.point_clouds[cloud_name]
            pts = np.asarray(cloud['poly_data'].points)
            if pts.size == 0:
                return
            
            # Определяем режим: выделение или снятие выделения
            from PyQt5.QtWidgets import QApplication
            modifiers = QApplication.keyboardModifiers()
            is_removing = bool(modifiers & Qt.AltModifier)
            
            # Найдем все точки в радиусе от места клика
            p = np.asarray(picked_point, dtype=float).reshape(1, 3)
            distances = np.linalg.norm(pts - p, axis=1)
            indices_in_radius = np.where(distances <= self.brush_radius)[0]
            
            # Получаем текущее выделение
            sel = cloud.get('selected_indices', set())
            
            if is_removing:
                # Alt + ПКМ: снимаем выделение с точек в радиусе
                for idx in indices_in_radius:
                    sel.discard(int(idx))
            else:
                # ПКМ: выделяем точки в радиусе
                for idx in indices_in_radius:
                    sel.add(int(idx))
            
            cloud['selected_indices'] = sel
            
            # Обновляем визуальный оверлей
            self.update_selection_overlay(cloud_name)
            
            # Восстанавливаем позицию камеры
            self.plotter.camera_position = camera_pos
            
        except Exception:
            pass

    def keyPressEvent(self, event):
        """Обработка нажатий клавиш"""
        # Больше не изменяем размер кисти по клавишам +/- — используем кнопки UI
        super().keyPressEvent(event)

    def on_brush_size_inc(self):
        """Увеличить размер кисти кнопкой [+] (1..20)."""
        try:
            self.brush_size = min(int(self.brush_size) + 1, 20)
            self._sync_brush_radius()
            self.update_status_bar()
        except Exception:
            pass

    def on_brush_size_dec(self):
        """Уменьшить размер кисти кнопкой [−] (1..20)."""
        try:
            self.brush_size = max(int(self.brush_size) - 1, 1)
            self._sync_brush_radius()
            self.update_status_bar()
        except Exception:
            pass


if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    # Создаем и показываем основное окно
    window = PCDViewer()
    window.show()
    
    sys.exit(app.exec_())