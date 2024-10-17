import numpy as np
from numba import njit, prange
from quantumwaves.interpolation import cubic_interp1d
from scipy import interpolate

@njit(cache=True, parallel=True)
def make_wavelines(wavedata, P=5000, axis=0, stride=4, smoothing=True):
    """
    Генерирует координаты для волновых линий.

    Args:
        wavedata (numpy.ndarray): Данные волновой функции.
        P (int, optional): Количество точек на каждой линии. Defaults to 5000.
        axis (int, optional): Ось, вдоль которой генерируются линии (0 или 1). Defaults to 0.
        stride (int, optional): Шаг между линиями. Defaults to 4.
        smoothing (bool, optional): Применять ли сглаживание. Defaults to True.

    Returns:
        tuple: Кортеж массивов координат (points) и цветов (colors).
    """
    X = wavedata.shape[0]
    Y = wavedata.shape[1]
    N = X // stride

    if not smoothing:
        P = int(Y)

    M = N * (P + 2)
    points = np.zeros((M, 3))
    colors = np.ones((M, 4))
    y0 = np.arange(0, X)
    y = np.linspace(0, X - 1, P) if smoothing else np.linspace(0, X - 1, X)

    for x in prange(N):
        w = x * stride
        i0 = x * (P + 2)
        i1 = i0 + (P + 2) - 1
        points[i0:i1 + 1, 0] = w

        points[i0, 1] = -Y
        points[i0, 2] = 0
        colors[i0, 3] = 0

        z0 = wavedata[w, :] if axis == 0 else wavedata[:, w]
        z = cubic_interp1d(y, y0, z0) if smoothing else z0

        points[i0 + 1:i1, 1] = y
        points[i0 + 1:i1, 2] = z
        colors[i0 + 1:i1, 3] = 0.33

        points[i1, 1] = 2 * Y
        points[i1, 2] = 0
        colors[i1, 3] = 0

    return points, colors


def surf_smoothing(surf_data, smoothing=2):
    """
    Сглаживает данные поверхности с помощью RectBivariateSpline.

    Args:
        surf_data (numpy.ndarray): Входные данные поверхности.
        smoothing (int): Коэффициент сглаживания.

    Returns:
        numpy.ndarray: Сглаженные данные поверхности.
    """
    X = surf_data.shape[0]
    Y = surf_data.shape[1]
    x, y = np.arange(X), np.arange(Y)
    f = interpolate.RectBivariateSpline(x, y, surf_data, kx=3, ky=3)
    xnew = np.linspace(0, X - 1, X * smoothing)
    ynew = np.linspace(0, Y - 1, Y * smoothing)
    return f(xnew, ynew)