import numpy as np
from scipy import interpolate
from time import perf_counter as pc
from time import time
from .field import Field
from .wavefunctions import make_wavelines, surf_smoothing
from quantumwaves.interpolation import cubic_interp1d
import scipy.sparse as sp
import scipy.sparse.linalg
from pathlib import Path
from schrodinger.solve import solve
from schrodinger import util

class Simulate:
    """
    Класс для симуляции квантовых волн.
    """
    FPS = 60
    DURATION = 5

    def __init__(self, n, size=10, collapse=False, potential="0", obstacles="False", delta_t=0.005, verbose=False):
        """Инициализирует симуляцию."""
        self.n = n
        self.size = size
        self.collapse = collapse
        self.frames = self.FPS * self.DURATION
        self.dt = delta_t / self.FPS if delta_t else 0.005
        self.step = self.size / self.n

        self.field = Field()
        self.field.set_potential(potential)
        self.field.set_obstacle(obstacles)

        self.wave_function = None
        self.x_axis = None
        self.y_axis = None
        self.V_x = None
        self.V_y = None
        self.HX = None
        self.HY = None
        self.counter = 0
        self.start_time = None
        self.i_time = None

        if verbose:
            memory = 16 * self.n * self.n * 1e-9
            print(f"Estimated memory usage: {memory:.3f} GB")

    def simulation_initialize(self, x0=[0], y0=[0], k_x=[5000], k_y=[2500], a_x=[.2], a_y=[.2], wall_potential=1e10):
        """Инициализирует волновую функцию."""
        try:
            n = self.n
            step = self.step
            dt = self.dt
            self.counter = 0
            self.x_axis = np.linspace(-self.size / 2, self.size / 2, n)
            self.y_axis = np.linspace(-self.size / 2, self.size / 2, n)
            X, Y = np.meshgrid(self.x_axis, self.y_axis)

            self.wave_function = np.zeros((n, n), dtype=complex)

            for i in range(len(x0)):
                print(f"--- Initializing wave packet {i+1}/{len(x0)} ---")
                phase = np.exp(1j * (X * k_x[i] + Y * k_y[i]))
                px = np.exp(-((x0[i] - X)**2) / (4 * a_x[i]**2))
                py = np.exp(-((y0[i] - Y)**2) / (4 * a_y[i]**2))
                wave_packet = phase * px * py
                norm = np.sqrt(util.integrate(np.abs(wave_packet)**2, n, step))
                print(f"  Norm for packet {i+1}: {norm}")
                self.wave_function += wave_packet / norm

            print(f"Wavefunction initialized: shape={self.wave_function.shape}, dtype={self.wave_function.dtype}")
            print(f"Wavefunction max: {np.max(self.wave_function)}, min: {np.min(self.wave_function)}")
            print(f"Wavefunction sum: {np.sum(self.wave_function)}")

        except (ValueError, TypeError) as e:
            print(f"Error in simulation_initialize: {e}")
            return

        self.start_time = pc()
        self.i_time = pc()


    def simulate_frame(self, debug=True):
        """Выполняет один шаг симуляции."""
        if self.wave_function is None:
            raise RuntimeError("Simulation not initialized. Call simulation_initialize() first.")

        n = self.n
        step = self.step
        dt = self.dt

        self.V_x = np.zeros(n * n, dtype='c16')
        for j in range(n):
            for i in range(n):
                xx = i
                yy = n * j
                self.V_x[xx + yy] = self.field.get_potential(self.x_axis[j], self.y_axis[i]) if not self.field.is_obstacle(self.x_axis[j], self.y_axis[i]) else 1e10

        self.V_y = np.zeros(n * n, dtype='c16')
        for j in range(n):
            for i in range(n):
                xx = j * n
                yy = i
                self.V_y[xx + yy] = self.field.get_potential(self.x_axis[i], self.y_axis[j]) if not self.field.is_obstacle(self.x_axis[i], self.y_axis[j]) else 1e10

        LAPLACE_MATRIX = sp.lil_matrix((-2 * sp.eye(n * n)))
        for i in range(n):
            for j in range(n - 1):
                k = i * n + j
                LAPLACE_MATRIX[k, k + 1] = 1
                LAPLACE_MATRIX[k + 1, k] = 1

        LAPLACE_MATRIX = LAPLACE_MATRIX.tocsc()
        LAPLACE_MATRIX = LAPLACE_MATRIX / (step**2)

        self.HX = (sp.eye(n * n) - 1j * (dt / 2) * (LAPLACE_MATRIX - sp.diags(self.V_x, 0))).tocsc()
        self.HY = (sp.eye(n * n) - 1j * (dt / 2) * (LAPLACE_MATRIX - sp.diags(self.V_y, 0))).tocsc()

        self.wave_function = solve(self.wave_function, self.V_x, self.V_y, self.HX, self.HY, self.n, self.step, self.dt)

        print(f"Simulate frame: wave_function shape={self.wave_function.shape}")

        if debug:
            self.print_update()
        self.counter += 1
        return self.wave_function

    # ... (остальные методы)

    def collapse_wavefunction(self, num_samples=1, collapse_width=10):
        """
        Коллапсирует волновую функцию и возвращает координаты коллапса.

        Args:
            num_samples (int, optional): Количество выборок для коллапса. Defaults to 1.
            collapse_width (float, optional): Ширина коллапса волновой функции. Defaults to 10.

        Returns:
            tuple: Кортеж координат коллапса (x, y).
        """
        if self.wave_function is None:
            raise RuntimeError("Wavefunction not initialized. Call simulation_initialize() first.")

        probability_distribution = np.abs(self.wave_function)**2
        probability_distribution /= probability_distribution.sum()

        indices = np.random.choice(np.arange(self.n**2), size=num_samples, replace=False, p=probability_distribution.flatten())
        x, y = np.unravel_index(indices, (self.n, self.n))

        # Преобразование индексов в координаты
        x = (x / self.n - 0.5) * self.size
        y = (y / self.n - 0.5) * self.size

        # Проверка границ
        x = np.clip(x, -self.size / 2, self.size / 2)
        y = np.clip(y, -self.size / 2, self.size / 2)

        cw = collapse_width / self.n  # Ширина коллапса
        print(f">>> Collapsed to: x = {x}, y = {y}, collapse_width = {cw}")
        self.simulation_initialize(x0=x, y0=y, k_x=[0] * num_samples, k_y=[0] * num_samples, a_x=[cw] * num_samples, a_y=[cw] * num_samples)

        return x, y

    def dual_collapse_wavefunction(self, num_samples=10, min_distance=2, collapse_width=10):
        """
        Выполняет двойной коллапс волновой функции.

        Args:
            num_samples (int, optional): Максимальное количество выборок. Defaults to 10.
            min_distance (float, optional): Минимальное расстояние между точками коллапса. Defaults to 2.
            collapse_width (float, optional): Ширина коллапса. Defaults to 10.

        Returns:
            tuple: Кортеж координат коллапса ((x1, y1), (x2, y2)).
        """
        if self.wave_function is None:
            raise RuntimeError("Wavefunction not initialized.")

        probability_distribution = np.abs(self.wave_function)**2
        probability_distribution /= probability_distribution.sum()

        indices = np.random.choice(np.arange(self.n**2), size=num_samples, replace=False, p=probability_distribution.flatten())
        
        # Находим две точки коллапса на достаточном расстоянии друг от друга
        first_index = indices[0]
        x1, y1 = np.unravel_index(first_index, (self.n, self.n))
        x1 = (x1 / self.n - 0.5) * self.size
        y1 = (y1 / self.n - 0.5) * self.size

        second_index = None
        for index in indices[1:]:
            x, y = np.unravel_index(index, (self.n, self.n))
            x = (x / self.n - 0.5) * self.size
            y = (y / self.n - 0.5) * self.size

            if np.linalg.norm((x, y) - (x1, y1)) >= min_distance:
                second_index = index
                break

        if second_index is None:
            print("Warning: Could not find a second collapse point with sufficient distance.")
            return (x1, y1), (x1, y1)  # Возвращаем одну и ту же точку, если не нашли вторую


        x2, y2 = np.unravel_index(second_index, (self.n, self.n))
        x2 = (x2 / self.n - 0.5) * self.size
        y2 = (y2 / self.n - 0.5) * self.size


        # Проверка границ
        x1 = np.clip(x1, -self.size / 2, self.size / 2)
        y1 = np.clip(y1, -self.size / 2, self.size / 2)
        x2 = np.clip(x2, -self.size / 2, self.size / 2)
        y2 = np.clip(y2, -self.size / 2, self.size / 2)


        cw = collapse_width / self.n
        print(f">>> Collapsed to: (x1, y1) = ({x1:.2f}, {y1:.2f}), (x2, y2) = ({x2:.2f}, {y2:.2f}), collapse_width = {cw:.2f}")
        self.simulation_initialize(x0=[x1, x2], y0=[y1, y2], k_x=[0, 0], k_y=[0, 0], a_x=[cw, cw], a_y=[cw, cw])

        return (x1, y1), (x2, y2)

    def print_update(self, progress_bar_length=20):
        """Выводит информацию о ходе симуляции."""
        norm = np.sqrt(util.integrate(np.abs(self.wave_function)**2, self.n, self.step))
        progress = self.counter / self.frames
        elapsed_time = time() - self.start_time
        remaining_time = (time() - self.i_time) * (self.frames - self.counter)
        remaining_time_min = int(remaining_time // 60)
        remaining_time_sec = int(round(remaining_time % 60))

        progress_bar = "[" + "#" * int(progress * progress_bar_length) + "-" * (progress_bar_length - int(progress * progress_bar_length)) + "]"
        print(f"--- Simulation in progress: {''} ---")
        print(f"{progress_bar}   {progress*100:.2f}%")
        print(f"Elapsed time: {elapsed_time:.1f} s")
        print(f"Estimated time remaining: {remaining_time_min}:{remaining_time_sec:02d}")
        print(f"Function standard: {norm:.3f}")
        self.i_time = time()

    def save_wave(self, wave_function, save_path, frame_index):
        """Сохраняет волновую функцию в файл."""

        filename = Path(save_path) / f"wavefunction_{frame_index:04d}.npy"  # Используем Path
        np.save(filename, wave_function)
        
    def collision(self):
        self.simulation_initialize(x0=[0, 0], y0=[0, 1], k_x=[0, 0], k_y=[0, 90000], a_x=[10.2, 10.2], a_y=[10.2, 10.2]) # a_x и a_y увеличены

    def collision1(self):
        self.simulation_initialize(x0=[0, 0], y0=[0, 1.5], k_x=[10, 0], k_y=[0, 90000], a_x=[.15, .15], a_y=[.15, .15])

    def movement(self):
        self.simulation_initialize(x0=[0], y0=[0], k_x=[5000], k_y=[2500], a_x=[.2], a_y=[.2])

    def collapse_init(self):
        self.simulation_initialize(x0=[0], y0=[0], k_x=[50], k_y=[25], a_x=[.25], a_y=[.25])

    def collapse3(self):
        self.simulation_initialize(x0=[0], y0=[0], k_x=[50], k_y=[25], a_x=[.28], a_y=[.28])

    def entanglement(self):
        self.simulation_initialize(x0=[0, 0], y0=[1, -1], k_x=[0, 0], k_y=[-3000, 3000], a_x=[.15, .15], a_y=[.15, .15])