import sys
import numpy as np
import pyqtgraph.opengl as gl
import matplotlib.pyplot as plt
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QLineEdit, QComboBox, QPushButton, QCheckBox
from PyQt6.QtCore import QTimer
from pathlib import Path
from schrodinger.simulation import Simulate
from schrodinger.wavefunctions import make_wavelines, surf_smoothing

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Quantum Wave Simulator")

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
        
        self.scenario_combo = QComboBox()
        self.scenario_combo.addItems(["collision", "collision1", "movement", "collapse_init", "collapse3", "entanglement"])
        self.save_path_edit = QLineEdit("./frames")

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
        self.setLayout(layout)

        # Подключения
        self.start_button.clicked.connect(self.start_simulation)

        # Другие атрибуты
        self.view = gl.GLViewWidget()  # Виджет для 3D-графики
        layout.addWidget(self.view)  # Добавляем виджет в layout
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

        # Graphics Items - создаём, но не добавляем в self.view
        self.sphere = gl.GLMeshItem(meshdata=gl.MeshData.sphere(rows=100, cols=100), smooth=True, glOptions='translucent')
        self.sphere.translate(5, -5, 0)
        self.sphere.scale(1000, 1000, 1000)

        self.real = gl.GLLinePlotItem(antialias=True)
        self.imag = gl.GLLinePlotItem(antialias=True)
        self.rhzn = gl.GLLinePlotItem(antialias=True)
        self.ihzn = gl.GLLinePlotItem(antialias=True)
        self.surf = gl.GLSurfacePlotItem(computeNormals=False, smooth=True)
        self.surf.setGLOptions('translucent')
        self.surf.setDepthValue(10)

    def start_simulation(self):
        self.sim = Simulate(int(self.size_edit.text()),
                                        collapse=self.collapse_checkbox.isChecked(),
                                        potential="0",
                                        obstacles="False",
                                        delta_t=0.005,
                                        verbose=True)
        self.layer = self.layer_combo.currentText()
        self.record = self.record_checkbox.isChecked()
        self.do_smoothing = self.smoothing_checkbox.isChecked()
        self.folder = Path(self.save_path_edit.text())
        scenario = self.scenario_combo.currentText()
        getattr(self.sim, scenario)()
        self.frames = self.sim.frames
        self.frame_index = 0
        self.stop_button.setEnabled(True) # Включаем кнопку Stop
        self.start_button.setEnabled(False) # Выключаем кнопку Start
        self.create_graphics_items()
        self.timer.start(0)
        
    def stop_simulation(self):
        """Останавливает симуляцию и очищает ресурсы."""
        if self.sim is not None:
            self.timer.stop()
            self.sim = None
            self.view.clear() # Очищаем содержимое GLViewWidget
            self.frame_index = 0 #Сброс счётчика кадров
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
        self.view.resize(1200, 800)  # Устанавливаем размер виджета
        self.view.setFixedSize(1200, 800)  # Фиксируем размер
    
    def update(self):
        print("update called")
        if self.sim is None:
            print("sim is None")
            return

        if self.frame_index >= self.frames and self.record:
            self.timer.stop()
            return

        d = self.sim.simulate_frame(debug=False)
        zdata = np.abs(d)**2
        print(f"zdata shape: {zdata.shape}, dtype: {zdata.dtype}, min: {zdata.min()}, max: {zdata.max()}")

        if self.do_smoothing:
            zdata = surf_smoothing(zdata, smoothing=self.surf_smooth)
            print(f"zdata shape after smoothing: {zdata.shape}, dtype: {zdata.dtype}, min: {zdata.min()}, max: {zdata.max()}")

        zcol = self.cmap(np.interp(zdata, [0, 4], [0, 1]))
        zcol[:, :, 3] = zdata + .1
        zcol[:, :, 3] *= 0.75
        print(f"zcol shape: {zcol.shape}, dtype: {zcol.dtype}")

        dreal, dimag = d.real, np.flipud(d.imag)
        rpoints, realcolors = make_wavelines(dreal, axis=0, smoothing=self.do_smoothing)
        ipoints, imagcolors = make_wavelines(dimag, axis=1, smoothing=self.do_smoothing)
        print(f"rpoints shape: {rpoints.shape}, dtype: {rpoints.dtype}")
        print(f"realcolors shape: {realcolors.shape}, dtype: {realcolors.dtype}")
        print(f"ipoints shape: {ipoints.shape}, dtype: {ipoints.dtype}")
        print(f"imagcolors shape: {imagcolors.shape}, dtype: {imagcolors.dtype}")


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