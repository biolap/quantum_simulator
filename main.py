import sys
import numpy as np
import pyqtgraph.opengl as gl
import matplotlib.pyplot as plt
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QLabel,
                            QLineEdit, QComboBox, QPushButton, QCheckBox,
                            QFileDialog)
from PyQt6.QtCore import QTimer
from pathlib import Path
from schrodinger.simulation import Simulate
from schrodinger.wavefunctions import make_wavelines, surf_smoothing
import json

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Квантовый симулятор волн")

        # Виджеты
        self.size_label = QLabel("Size:")
        self.size_edit = QLineEdit("125")
        self.layer_label = QLabel("Layer:")
        self.layer_combo = QComboBox()
        self.layer_combo.addItems(["", "real", "imag", "surf"])
        self.collapse_checkbox = QCheckBox("Collapse")
        self.smoothing_checkbox = QCheckBox("Smoothing")
        self.record_checkbox = QCheckBox("Record")
        self.start_button = QPushButton("Start")
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_simulation)
        self.stop_button.setEnabled(False)
        self.scenario_combo = QComboBox()
        self.scenario_combo.addItems(["collision", "collision1", "movement", "collapse_init", "collapse3", "entanglement"])
        self.save_path_edit = QLineEdit("./frames")

        self.potential_label = QLabel("Потенциал:")
        self.potential_edit = QLineEdit("0")
        self.time_dependent_checkbox = QCheckBox("Зависящий от времени потенциал")

        self.initial_conditions_label = QLabel("Начальные условия:")
        self.initial_conditions_combo = QComboBox()
        self.initial_conditions_combo.addItems(["Гауссиан", "Загрузить из файла"])
        self.initial_conditions_combo.currentIndexChanged.connect(self.update_initial_condition_options)

        self.gaussian_params_widget = QWidget()
        gaussian_params_layout = QVBoxLayout()
        self.x0_edit = QLineEdit("0")
        self.y0_edit = QLineEdit("0")
        self.kx_edit = QLineEdit("5000")
        self.ky_edit = QLineEdit("2500")
        self.ax_edit = QLineEdit(".2")
        self.ay_edit = QLineEdit(".2")
        gaussian_params_layout.addWidget(QLabel("x0:"))
        gaussian_params_layout.addWidget(self.x0_edit)
        gaussian_params_layout.addWidget(QLabel("y0:"))
        gaussian_params_layout.addWidget(self.y0_edit)
        gaussian_params_layout.addWidget(QLabel("kx:"))
        gaussian_params_layout.addWidget(self.kx_edit)
        gaussian_params_layout.addWidget(QLabel("ky:"))
        gaussian_params_layout.addWidget(self.ky_edit)
        gaussian_params_layout.addWidget(QLabel("ax:"))
        gaussian_params_layout.addWidget(self.ax_edit)
        gaussian_params_layout.addWidget(QLabel("ay:"))
        gaussian_params_layout.addWidget(self.ay_edit)
        self.gaussian_params_widget.setLayout(gaussian_params_layout)

        self.file_input_widget = QWidget()
        file_input_layout = QVBoxLayout()
        self.load_file_button = QPushButton("Загрузить файл")
        self.load_file_button.clicked.connect(self.load_initial_conditions_from_file)
        file_input_layout.addWidget(self.load_file_button)
        self.file_input_widget.setLayout(file_input_layout)

        self.initial_conditions_stack = QWidget()
        self.initial_conditions_stack_layout = QVBoxLayout()
        self.initial_conditions_stack_layout.addWidget(self.gaussian_params_widget)
        self.initial_conditions_stack_layout.addWidget(self.file_input_widget)
        self.initial_conditions_stack.setLayout(self.initial_conditions_stack_layout)
        self.file_input_widget.hide()

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.size_label)
        layout.addWidget(self.size_edit)
        layout.addWidget(self.layer_label)
        layout.addWidget(self.layer_combo)
        layout.addWidget(self.collapse_checkbox)
        layout.addWidget(self.smoothing_checkbox)
        layout.addWidget(self.record_checkbox)
        layout.addWidget(QLabel("Scenario:"))
        layout.addWidget(self.scenario_combo)
        layout.addWidget(QLabel("Save Path:"))
        layout.addWidget(self.save_path_edit)
        layout.addWidget(self.start_button)
        layout.addWidget(self.stop_button)
        layout.addWidget(self.potential_label)
        layout.addWidget(self.potential_edit)
        layout.addWidget(self.time_dependent_checkbox)
        layout.addWidget(self.initial_conditions_label)
        layout.addWidget(self.initial_conditions_combo)
        layout.addWidget(self.initial_conditions_stack)
        self.setLayout(layout)
        
        # Подключения  <---  Этот блок должен быть здесь
        self.start_button.clicked.connect(self.start_simulation)
        self.stop_button.clicked.connect(self.stop_simulation)

        # Другие атрибуты
        self.gl_widget_window = QWidget() # Окно для GLViewWidget
        self.gl_widget_window.setWindowTitle("Симуляция")
        self.view = gl.GLViewWidget()
        gl_layout = QVBoxLayout()
        gl_layout.addWidget(self.view)
        self.gl_widget_window.setLayout(gl_layout)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update)
        self.sim = None  # Объект симуляции
        self.frame_index = 0 # Индекс текущего кадра
        self.frames = None # Количество кадров
        self.record = False # Запись видео
        self.folder = None # Папка для сохранения
        self.cmap = plt.get_cmap('viridis') # Цветовая карта
        self.rescale = 1000 # Масштабирование
        self.surf_smooth = 8 # Сглаживание поверхности
        self.zscale = 10 # Масштабирование по Z
        self.rcol_bias = 0.7  # Настройка цвета (real)
        self.icol_bias = 0.3 # Настройка цвета (imag)
        self.sec = 30 # Длительность видео (секунды)
        self.fps = 30 # FPS видео
        self.do_parallel = False # Параллельные вычисления (не используется)
        self.do_collapse = False # Коллапс волновой функции (не используется)
        self.do_smoothing = True # Сглаживание
        self.layer = ""  # Слой для отображения
        self.initial_wavefunction = None # Загруженная волновая функция


        # НЕ создаем графические элементы здесь.
        # Они будут создаваться в start_simulation после инициализации self.sim
        
    def update_initial_condition_options(self, index):
        # Переключаем видимость виджетов в зависимости от выбранного варианта
        if index == 0:  # Гауссиан
            self.gaussian_params_widget.show()
            self.file_input_widget.hide()
        elif index == 1:  # Загрузка из файла
            self.gaussian_params_widget.hide()
            self.file_input_widget.show()
            
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

        try:  # Проверяем корректность ввода потенциала
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

        if self.sim is None:  # Проверяем, что self.sim успешно создан
            print("Ошибка: объект Simulate не инициализирован.")
            return

        if self.initial_conditions_combo.currentIndex() == 0:  # Гауссиан
            try:
                x0 = [float(self.x0_edit.text())]
                y0 = [float(self.y0_edit.text())]
                kx = [float(self.kx_edit.text())]
                ky = [float(self.ky_edit.text())]
                ax = [float(self.ax_edit.text())]
                ay = [float(self.ay_edit.text())]
                self.sim.simulation_initialize(x0=x0, y0=y0, k_x=kx, k_y=ky, a_x=ax, a_y=ay)  # Инициализируем здесь для гауссиана
            except ValueError as e:
                print(f"Неверный ввод для параметров гауссиана: {e}") # Выводим конкретную ошибку
                return

        elif self.initial_conditions_combo.currentIndex() == 1:  # Загрузка из файла
            if self.initial_wavefunction is not None:
                if self.initial_wavefunction.shape != (size, size):
                    print("Размерности загруженной волновой функции не соответствуют размеру симуляции.")
                    return
                self.sim.wave_function = self.initial_wavefunction
                self.sim.x_axis = np.linspace(-self.sim.size / 2, self.sim.size / 2, size)
                self.sim.y_axis = np.linspace(-self.sim.size / 2, self.sim.size / 2, size)
                self.sim.simulation_initialize() # Инициализируем здесь для загрузки из файла
            else:
                print("Начальная волновая функция не загружена из файла.")
                return

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
            elem.translate(-.5, -.5, 0)
            elem.scale(*(self.rescale,) * 3)
            elem.translate(-self.rescale / 2, -self.rescale / 2, 0)
            elem.scale(1, 1, self.zscale)

        for elem in [self.surf]:
            elem.scale(1 / (self.sim.n * self.surf_smooth), 1 / (self.sim.n * self.surf_smooth), 1 / (self.sim.n * self.surf_smooth))
            elem.translate(-.5, -.5, 0)
            elem.scale(*(self.rescale,) * 3)
            elem.translate(-self.rescale / 2, -self.rescale / 2, 0)
            elem.scale(1, 1, self.zscale * self.surf_smooth)

        self.imag.rotate(90, 0, 0, 1)
        self.imag.translate(-2, 0, 0)

        self.view.setCameraPosition(distance=80)
        self.view.resize(1200, 800)
        self.view.setFixedSize(1200, 800)
        
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