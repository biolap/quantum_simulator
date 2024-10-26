import sys
import numpy as np
import pyqtgraph.opengl as gl
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QLabel,
                            QLineEdit, QComboBox, QPushButton, QCheckBox,
                            QFileDialog, QSpinBox, QTabWidget, QHBoxLayout, QStackedWidget)
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QDoubleValidator
from pathlib import Path
from schrodinger.simulation import Simulate
from schrodinger.wavefunctions import make_wavelines, surf_smoothing
import json

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Квантовый симулятор волн")

        layout = QVBoxLayout()

        # --- Размер сетки ---
        size_label = QLabel("Размер сетки:")
        layout.addWidget(size_label)
        self.size_edit = QLineEdit("125")
        layout.addWidget(self.size_edit)
        
        # --- Выбор цветовой карты ---
        layout.addWidget(QLabel("Цветовая карта:"))
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(plt.colormaps())  # Добавляем все доступные цветовые карты matplotlib
        layout.addWidget(self.colormap_combo)
        self.colormap_combo.currentIndexChanged.connect(self.update_colormap) # Подключаем к слоту
        
        # --- Кнопки управления камерой ---
        camera_controls_layout = QHBoxLayout()  # Layout для кнопок управления камерой
        self.rotate_left_button = QPushButton("Вращать влево")
        camera_controls_layout.addWidget(self.rotate_left_button)
        self.rotate_right_button = QPushButton("Вращать вправо")
        camera_controls_layout.addWidget(self.rotate_right_button)
        self.zoom_in_button = QPushButton("Приблизить")
        camera_controls_layout.addWidget(self.zoom_in_button)
        self.zoom_out_button = QPushButton("Отдалить")
        camera_controls_layout.addWidget(self.zoom_out_button)
        self.reset_camera_button = QPushButton("Сброс камеры")
        camera_controls_layout.addWidget(self.reset_camera_button)
        layout.addLayout(camera_controls_layout) # Добавляем layout в основной layout

        # --- Слой ---
        layer_label = QLabel("Слой:")
        layout.addWidget(layer_label)
        self.layer_combo = QComboBox()
        self.layer_combo.addItems(["", "real", "imag", "surf"])
        layout.addWidget(self.layer_combo)

        # --- Опции симуляции ---
        self.collapse_checkbox = QCheckBox("Collapse")
        layout.addWidget(self.collapse_checkbox)
        self.smoothing_checkbox = QCheckBox("Smoothing")
        layout.addWidget(self.smoothing_checkbox)
        self.record_checkbox = QCheckBox("Record")
        layout.addWidget(self.record_checkbox)

        # --- Сценарий ---
        scenario_label = QLabel("Сценарий:")
        layout.addWidget(scenario_label)
        self.scenario_combo = QComboBox()
        self.scenario_combo.addItems(["collision", "collision1", "movement", "collapse_init", "collapse3", "entanglement", "Пользовательский"])
        layout.addWidget(self.scenario_combo)

        # --- Путь сохранения ---
        save_path_label = QLabel("Путь сохранения:")
        layout.addWidget(save_path_label)
        self.save_path_edit = QLineEdit("./frames")
        layout.addWidget(self.save_path_edit)

        # --- Потенциал ---
        potential_label = QLabel("Потенциал:")
        layout.addWidget(potential_label)
        self.potential_edit = QLineEdit("0")
        layout.addWidget(self.potential_edit)
        self.time_dependent_checkbox = QCheckBox("Зависящий от времени потенциал")
        layout.addWidget(self.time_dependent_checkbox)

        # --- Начальные условия (Stacked Widget) ---
        initial_conditions_label = QLabel("Начальные условия:")
        layout.addWidget(initial_conditions_label)

        self.initial_conditions_stack = QStackedWidget()

        # --- Виджет для выбора типа начальных условий ---
        self.initial_conditions_widget = QWidget()
        initial_conditions_layout = QVBoxLayout()
        initial_conditions_type_label = QLabel("Тип начальных условий:")
        initial_conditions_layout.addWidget(initial_conditions_type_label)
        self.initial_conditions_combo = QComboBox()
        self.initial_conditions_combo.addItems(["Гауссиан", "Загрузить из файла"])
        initial_conditions_layout.addWidget(self.initial_conditions_combo)
        self.initial_conditions_widget.setLayout(initial_conditions_layout)
        self.initial_conditions_stack.addWidget(self.initial_conditions_widget)

        # --- Параметры гауссовых пакетов (в закладках) ---
        self.gaussian_params_tab_widget = QTabWidget()
        self.initial_conditions_stack.addWidget(self.gaussian_params_tab_widget)

        # --- Количество волновых пакетов ---
        self.num_packets_widget = QWidget()
        num_packets_layout = QHBoxLayout()
        num_packets_label = QLabel("Количество волновых пакетов:")
        num_packets_layout.addWidget(num_packets_label)
        self.num_packets_spinbox = QSpinBox()
        self.num_packets_spinbox.setMinimum(1)
        self.num_packets_spinbox.setMaximum(5)
        self.num_packets_spinbox.setValue(1)
        num_packets_layout.addWidget(self.num_packets_spinbox)
        self.num_packets_widget.setLayout(num_packets_layout)
        layout.addWidget(self.num_packets_widget)
        
        # --- Загрузка из файла ---
        self.file_input_widget = QWidget()
        file_input_layout = QVBoxLayout()
        self.load_file_button = QPushButton("Загрузить файл")
        file_input_layout.addWidget(self.load_file_button)
        self.file_input_widget.setLayout(file_input_layout)
        self.initial_conditions_stack.addWidget(self.file_input_widget)

        layout.addWidget(self.initial_conditions_stack)
        self.initial_conditions_stack.hide()

        # --- Кнопки управления ---
        self.start_button = QPushButton("Start")
        layout.addWidget(self.start_button)
        self.stop_button = QPushButton("Stop")
        layout.addWidget(self.stop_button)
        self.stop_button.setEnabled(False)

        self.setLayout(layout)

        # --- Подключения ---
        self.start_button.clicked.connect(self.start_simulation)
        self.stop_button.clicked.connect(self.stop_simulation)
        self.initial_conditions_combo.currentIndexChanged.connect(self.update_initial_condition_options)
        self.load_file_button.clicked.connect(self.load_initial_conditions_from_file)
        self.scenario_combo.currentIndexChanged.connect(self.update_scenario_options) # <--- Добавляем подключение
        self.num_packets_spinbox.valueChanged.connect(self.update_gaussian_params_widgets)
        
        self.rotate_left_button.clicked.connect(self.rotate_camera_left)  # <---  Подключаем кнопки
        self.rotate_right_button.clicked.connect(self.rotate_camera_right)
        self.zoom_in_button.clicked.connect(self.zoom_camera_in)
        self.zoom_out_button.clicked.connect(self.zoom_camera_out)
        self.reset_camera_button.clicked.connect(self.reset_camera)
        
        # --- Другие атрибуты ---
        self.gl_widget_window = QWidget()
        self.gl_widget_window.setWindowTitle("Симуляция")
        self.camera_distance = 80 # <---  Начальное расстояние камеры
        self.cmap = plt.get_cmap('viridis') # Начальная цветовая карта
        self.view = gl.GLViewWidget()
        gl_layout = QVBoxLayout()
        gl_layout.addWidget(self.view)
        self.gl_widget_window.setLayout(gl_layout)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update)
        self.sim = None
        self.frame_index = 0
        self.frames = None
        self.record = False
        self.folder = None
        self.cmap = plt.get_cmap('viridis')
        self.rescale = 1000
        self.surf_smooth = 8
        self.zscale = 10
        self.rcol_bias = 0.7
        self.icol_bias = 0.3
        self.sec = 30
        self.fps = 30
        self.do_parallel = False
        self.do_collapse = False
        self.do_smoothing = True
        self.layer = ""
        self.initial_wavefunction = None
        self.gaussian_params_widgets = []  # Инициализируем пустой список
        self.update_gaussian_params_widgets(self.num_packets_spinbox.value())  # Создаем виджеты для начального значения spinbox
        self.update_initial_condition_options(self.initial_conditions_combo.currentIndex()) # Инициализация начального состояния виджетов
    
    def update_scenario_options(self, index):
        """Обновляет видимость виджетов начальных условий в зависимости от выбранного сценария."""
        if self.scenario_combo.itemText(index) == "Пользовательский":
            self.initial_conditions_stack.show()
            self.update_initial_condition_options(self.initial_conditions_combo.currentIndex())  # Обновляем видимость гауссовых параметров или загрузки из файла
        else:
            self.initial_conditions_stack.hide()
    
    def update_gaussian_params_widgets(self, num_packets):
        """Обновляет виджеты параметров гауссовых пакетов."""

        self.gaussian_params_tab_widget.clear()
        self.gaussian_params_widgets.clear()

        for i in range(num_packets):
            packet_widget = QWidget()
            packet_layout = QVBoxLayout(packet_widget)

            x0_edit = QLineEdit("0")
            y0_edit = QLineEdit("0")
            kx_edit = QLineEdit("5000")
            ky_edit = QLineEdit("2500")
            ax_edit = QLineEdit(".2")
            ay_edit = QLineEdit(".2")

            # Создаем валидаторы для числовых полей
            double_validator = QDoubleValidator()
            x0_edit.setValidator(double_validator)
            y0_edit.setValidator(double_validator)
            kx_edit.setValidator(double_validator)
            ky_edit.setValidator(double_validator)
            ax_edit.setValidator(double_validator)
            ay_edit.setValidator(double_validator)

            packet_layout.addWidget(QLabel("x0:"))
            packet_layout.addWidget(x0_edit)
            packet_layout.addWidget(QLabel("y0:"))
            packet_layout.addWidget(y0_edit)
            packet_layout.addWidget(QLabel("kx:"))
            packet_layout.addWidget(kx_edit)
            packet_layout.addWidget(QLabel("ky:"))
            packet_layout.addWidget(ky_edit)
            packet_layout.addWidget(QLabel("ax:"))
            packet_layout.addWidget(ax_edit)
            packet_layout.addWidget(QLabel("ay:"))
            packet_layout.addWidget(ay_edit)


            self.gaussian_params_tab_widget.addTab(packet_widget, f"Пакет {i+1}")
            self.gaussian_params_widgets.append((x0_edit, y0_edit, kx_edit, ky_edit, ax_edit, ay_edit))
        
    def update_initial_condition_options(self, index):
        self.initial_conditions_stack.setCurrentIndex(index + 1)  # Переключаем stacked widget
        if index == 0:  # Гауссиан
            self.num_packets_widget.show() # Показываем виджет выбора количества пакетов
        else: # Загрузка из файла
            self.num_packets_widget.hide() # Скрываем виджет выбора количества пакетов
            
    def update_colormap(self, index):
        """Обновляет цветовую карту."""
        cmap_name = self.colormap_combo.itemText(index)
        self.cmap = get_cmap(cmap_name) # Получаем цветовую карту по имени

        # Перерисовываем сцену (если симуляция уже запущена)
        if self.sim:
            self.update()  # Вызываем update для обновления графиков
            
    def rotate_camera_left(self):
        self.view.orbit(-10, 0)  # Вращаем на -10 градусов по горизонтали

    def rotate_camera_right(self):
        self.view.orbit(10, 0)   # Вращаем на 10 градусов по горизонтали

    def zoom_camera_in(self):
        self.camera_distance -= 5
        self.view.setCameraPosition(distance=self.camera_distance)

    def zoom_camera_out(self):
        self.camera_distance += 5
        self.view.setCameraPosition(distance=self.camera_distance)

    def reset_camera(self):
        self.camera_distance = 80  # Сбрасываем расстояние камеры
        self.view.setCameraPosition(distance=self.camera_distance, elevation=30, azimuth=45)  # <--- Добавляем elevation и azimuth
            
    def load_initial_conditions_from_file(self):
        # Открываем диалог выбора файла
        options = QFileDialog.Option.ReadOnly
        file_path, _ = QFileDialog.getOpenFileName(self, "Загрузить начальные условия", "", "JSON Files (*.json);;All Files (*)", options=options)
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    # Загружаем волновую функцию как комплексный numpy массив
                    self.initial_wavefunction = np.array(data['wavefunction'], dtype=complex)
            except Exception as e:
                print(f"Ошибка при загрузке начальных условий из файла: {e}")
                
    def start_simulation(self):
        size = int(self.size_edit.text())

        try:
            potential_expr = self.potential_edit.text()
            test_x, test_y, test_t = 0, 0, 0
            test_potential = eval(potential_expr, {}, {'x': test_x, 'y': test_y, 't': test_t})
        except Exception as e:
            print(f"Ошибка в выражении потенциала: {e}")
            return

        self.sim = Simulate(size,
                            collapse=self.collapse_checkbox.isChecked(),
                            potential=potential_expr,
                            obstacles="False",
                            delta_t=0.005,
                            verbose=True,
                            time_dependent=self.time_dependent_checkbox.isChecked())

        if self.sim is None:
            print("Ошибка: объект Simulate не инициализирован.")
            return

        scenario = self.scenario_combo.currentText()

        # --- Пользовательские начальные условия ---
        if scenario == "Пользовательский":
            if self.initial_conditions_combo.currentIndex() == 0:  # Гауссиан
                try:
                    gaussian_params = []
                    for x0_edit, y0_edit, kx_edit, ky_edit, ax_edit, ay_edit in self.gaussian_params_widgets:
                        x0 = float(x0_edit.text())
                        y0 = float(y0_edit.text())
                        kx = float(kx_edit.text())
                        ky = float(ky_edit.text())
                        ax = float(ax_edit.text())
                        ay = float(ay_edit.text())
                        gaussian_params.append((x0, y0, kx, ky, ax, ay))
                    self.sim.simulation_initialize_multiple(*gaussian_params)
                except ValueError as e:
                    print(f"Неверный ввод для параметров гауссиана: {e}")
                    return
            elif self.initial_conditions_combo.currentIndex() == 1:  # Загрузка из файла
                if self.initial_wavefunction is None:
                    print("Начальная волновая функция не загружена из файла.")
                    return
                else:
                    if self.initial_wavefunction.shape != (size, size):
                        print("Размерности загруженной волновой функции не соответствуют размеру симуляции.")
                        return
                    self.sim.wave_function = self.initial_wavefunction
                    self.sim.x_axis = np.linspace(-self.sim.size / 2, self.sim.size / 2, size)
                    self.sim.y_axis = np.linspace(-self.sim.size / 2, self.sim.size / 2, size)
                    self.sim.simulation_initialize()
        elif scenario == "collision":
            self.sim.collision()
        elif scenario == "collision1":
            self.sim.collision1()
        elif scenario == "movement":
            self.sim.movement()
        elif scenario == "collapse_init":
            self.sim.collapse_init()
        elif scenario == "collapse3":
            self.sim.collapse3()
        elif scenario == "entanglement":
            self.sim.entanglement()
        elif gaussian_params:  # Гауссиан, если не выбран базовый сценарий
            self.sim.simulation_initialize(*gaussian_params) # Распаковываем параметры
        elif self.initial_conditions_combo.currentIndex() == 1: # Загрузка из файла, если не выбран базовый сценарий
            self.sim.simulation_initialize() # Инициализируем с загруженной волновой функцией

        self.frames = self.sim.frames

        self.layer = self.layer_combo.currentText()
        self.record = self.record_checkbox.isChecked()
        self.do_smoothing = self.smoothing_checkbox.isChecked()
        self.folder = Path(self.save_path_edit.text())

        if self.record and not self.folder.exists():
            self.folder.mkdir(parents=True, exist_ok=True)

        self.frame_index = 0
        self.stop_button.setEnabled(True)
        self.start_button.setEnabled(False)

        self.create_graphics_items() # Создаем графические элементы после инициализации sim

        self.gl_widget_window.show()
        self.gl_widget_window.move(self.geometry().right(), self.geometry().top())

        self.timer.start(0) # Запускаем таймер после отображения окна
        
    def stop_simulation(self):
        """Останавливает симуляцию и очищает ресурсы."""
        if self.sim is not None:
            self.timer.stop()
            self.sim = None
            self.view.clear()
            self.frame_index = 0
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
        
    def create_graphics_items(self):
        ds = 100
        md = gl.MeshData.sphere(rows=ds, cols=ds)
        sphere_colors = np.zeros((md.faceCount(), 4), dtype=float)
        sphere_colors[:] = self.cmap(0)
        md.setFaceColors(sphere_colors)
        self.sphere = gl.GLMeshItem(meshdata=md, smooth=True, glOptions='translucent')
        self.sphere.translate(5, -5, 0)
        self.sphere.scale(1000, 1000, 1000)

        self.real = gl.GLLinePlotItem(antialias=True)
        self.imag = gl.GLLinePlotItem(antialias=True)
        self.rhzn = gl.GLLinePlotItem(antialias=True)
        self.ihzn = gl.GLLinePlotItem(antialias=True)
        self.surf = gl.GLSurfacePlotItem(computeNormals=False, smooth=True)
        self.surf.setGLOptions('translucent')
        self.surf.setDepthValue(10)

        # Добавляем элементы в self.view
        self.view.addItem(self.sphere)
        self.view.addItem(self.real)
        self.view.addItem(self.imag)
        self.view.addItem(self.rhzn)
        self.view.addItem(self.ihzn)
        self.view.addItem(self.surf)

        # Настройка графических элементов
        for elem in [self.real, self.imag, self.rhzn, self.ihzn]:
            elem.scale(1 / self.sim.n, 1 / self.sim.n, 1 / self.sim.n)
            self._extracted_from_create_graphics_items_30(elem)
            elem.scale(1, 1, self.zscale)

        for elem in [self.surf]:
            elem.scale(1 / (self.sim.n * self.surf_smooth), 1 / (self.sim.n * self.surf_smooth), 1 / (self.sim.n * self.surf_smooth))
            self._extracted_from_create_graphics_items_30(elem)
            elem.scale(1, 1, self.zscale * self.surf_smooth)

        self.imag.rotate(90, 0, 0, 1)
        self.imag.translate(-2, 0, 0)

        self.view.setCameraPosition(distance=80)
        self.view.resize(1200, 800)
        self.view.setFixedSize(1200, 800)

    # TODO Rename this here and in `create_graphics_items`
    def _extracted_from_create_graphics_items_30(self, elem):
        elem.translate(-.5, -.5, 0)
        elem.scale(*(self.rescale,) * 3)
        elem.translate(-self.rescale / 2, -self.rescale / 2, 0)
        
    def update(self):
        if self.sim is None:
            return  # Просто выходим, если self.sim is None

        if self.frame_index >= self.frames and self.record:
            self.timer.stop()
            return

        d = self.sim.simulate_frame(debug=False)
        zdata = np.abs(d)**2

        if self.do_smoothing:
            zdata = surf_smoothing(zdata, smoothing=self.surf_smooth)

        zcol = self.cmap(np.interp(zdata, [0, 4], [0, 1]))
        zcol[:, :, 3] = zdata + .1
        zcol[:, :, 3] *= 0.75

        dreal, dimag = d.real, np.flipud(d.imag)
        rpoints, realcolors = make_wavelines(dreal, axis=0, smoothing=self.do_smoothing)
        ipoints, imagcolors = make_wavelines(dimag, axis=1, smoothing=self.do_smoothing)

        realcolors[:, 0:3:2] *= self.rcol_bias
        imagcolors[:, 0] *= self.icol_bias

        self.surf.setData(z=zdata, colors=zcol)
        self.real.setData(pos=rpoints, color=realcolors)
        self.imag.setData(pos=ipoints, color=imagcolors)

        self.view.update()
        self.frame_index += 1

        if self.record and self.folder.exists():
            img = self.view.grabFramebuffer()
            img.save(str(self.folder / f"frame_{self.frame_index:04d}.png"))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())