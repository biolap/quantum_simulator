import numpy as np
from numba import njit

@njit(cache=True)
def make_gridlines(X, Y, axis=0, stride=4, extend=3):
    """
    Создает координаты для сетки линий.

    Args:
        X (int): Максимальное значение по оси X.
        Y (int): Максимальное значение по оси Y.
        axis (int, optional): Ось, вдоль которой генерируются линии (0 или 1). Defaults to 0.
        stride (int, optional): Шаг между линиями. Defaults to 4.
        extend (int, optional):  Дополнительное расстояние за пределами области. Defaults to 3.

    Returns:
        tuple: Кортеж массивов координат (points) и цветов (colors).
    """
    if X != Y:
        raise ValueError('X and Y must be equal')

    P = 4  # Точек на линии
    N = (2 * X) // stride  # Количество линий
    M = N * P  # Общее количество точек

    points = np.zeros((M, 3))
    colors = np.ones((M, 4))

    for n in range(N):
        i0 = n * P
        i1 = i0 + P - 1

        a = int(axis)
        b = int(1 - axis)  # Инвертированная ось

        if n < N // 2:
            points[i0:i1 + 1, a] = -n * stride
            fading = 1 - n / (N // 2)
        else:
            points[i0:i1 + 1, a] = n * stride
            fading = 1 - (n - N // 2) / (N // 2)

        points[i0, b] = -Y
        points[i0 + 1, b] = 0
        points[i0 + 2, b] = Y
        points[i0 + 3, b] = 2 * Y

        colors[i0, 3] = 0
        colors[i0 + 1, 3] = 0.25 * fading
        colors[i0 + 2, 3] = 0.25 * fading
        colors[i0 + 3, 3] = 0

        if n in (0, N - 1):
            colors[i0 + 1, 3] = 0
            colors[i0 + 2, 3] = 0

    return points, colors