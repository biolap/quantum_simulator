import numpy as np
from numpy import angle, abs
from colorsys import hls_to_rgb
from numba import njit, prange

def colorize(z):
    """Преобразует комплексную волновую функцию в цветовой массив."""
    r = abs(z)
    arg = angle(z)
    h = (arg + np.pi) / (2 * np.pi) + 0.5
    lightness = 1.0 - 1.0 / (1.0 + 2 * r**1.2)
    s = 0.8
    c = np.apply_along_axis(lambda x: hls_to_rgb(*x), 2, np.stack((h, lightness, s)))
    return c

@njit(cache=True)
def to_vector(array_2d, n: int):
    """Преобразует 2D массив в 1D вектор."""
    return array_2d.reshape((n * n,))

@njit(cache=True)
def to_array(vector_1d, n: int):
    """Преобразует 1D вектор в 2D массив."""
    return vector_1d.reshape((n, n))

@njit(cache=True, parallel=True)
def dx_square(array_2d, n: int, step: float):
    """Вычисляет вторую производную по x."""
    result = np.zeros_like(array_2d)
    for j in prange(n):
        result[0, j] = array_2d[1, j] - 2 * array_2d[0, j]
        for i in prange(1, n - 1):
            result[i, j] = array_2d[i + 1, j] + array_2d[i - 1, j] - 2 * array_2d[i, j]
        result[n - 1, j] = array_2d[n - 2, j] - 2 * array_2d[n - 1, j]
    return result / (step**2)

@njit(cache=True, parallel=True)
def dy_square(array_2d, n: int, step: float):
    """Вычисляет вторую производную по y."""
    result = np.zeros_like(array_2d)
    for i in prange(n):
        result[i, 0] = array_2d[i, 1] - 2 * array_2d[i, 0]
        for j in prange(1, n - 1):
            result[i, j] = array_2d[i, j + 1] + array_2d[i, j - 1] - 2 * array_2d[i, j]
        result[i, n - 1] = array_2d[i, n - 2] - 2 * array_2d[i, n - 1]
    return result / (step**2)

@njit(cache=True)
def integrate(array_2d, n: int, step: float):
    """Вычисляет интеграл от функции двух переменных методом трапеций."""
    area = step**2
    return area * np.sum(array_2d)