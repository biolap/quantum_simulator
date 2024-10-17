import numpy as np
from numba import njit
from math import sqrt

@njit(cache=True)
def cubic_interp1d(x0, x, y):
    """
    Интерполирует одномерную функцию кубическими сплайнами.

    Args:
        x0 (float): Значение, для которого нужно вычислить интерполяцию.
        x (numpy.ndarray): Массив узлов интерполяции.
        y (numpy.ndarray): Массив значений функции в узлах.

    Returns:
        float: Интерполированное значение функции в точке x0.
    """
    x = np.ascontiguousarray(x)
    y = np.ascontiguousarray(y)

    if np.any(np.diff(x) < 0):
        indexes = np.argsort(x)
        x = x[indexes]
        y = y[indexes]

    size = len(x)
    xdiff = np.diff(x)
    ydiff = np.diff(y)
    Li = np.empty(size)
    Li_1 = np.empty(size - 1)
    z = np.empty(size)

    Li[0] = sqrt(2 * xdiff[0])
    Li_1[0] = 0.0
    z[0] = 0.0  # natural boundary

    for i in range(1, size - 1):
        Li_1[i] = xdiff[i - 1] / Li[i - 1]
        Li[i] = sqrt(2 * (xdiff[i - 1] + xdiff[i]) - Li_1[i - 1] * Li_1[i - 1])
        Bi = 6 * (ydiff[i] / xdiff[i] - ydiff[i - 1] / xdiff[i - 1])
        z[i] = (Bi - Li_1[i - 1] * z[i - 1]) / Li[i]

    Li_1[size - 2] = xdiff[-1] / Li[size - 2]
    Li[size - 1] = sqrt(2 * xdiff[-1] - Li_1[size - 2] * Li_1[size - 2])
    z[size - 1] = 0.0  # natural boundary


    z[size - 1] = z[size - 1] / Li[size - 1]
    for i in range(size - 2, -1, -1):
        z[i] = (z[i] - Li_1[i] * z[i + 1]) / Li[i]

    index = np.searchsorted(x, x0)
    xi1, xi0 = x[index], x[index - 1]
    yi1, yi0 = y[index], y[index - 1]
    zi1, zi0 = z[index], z[index - 1]
    hi1 = xi1 - xi0

    return zi0 / (6 * hi1) * (xi1 - x0)**3 + \
           zi1 / (6 * hi1) * (x0 - xi0)**3 + \
           (yi1 / hi1 - zi1 * hi1 / 6) * (x0 - xi0) + \
           (yi0 / hi1 - zi0 * hi1 / 6) * (xi1 - x0)