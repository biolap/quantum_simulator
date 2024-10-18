import scipy.sparse.linalg
from schrodinger.util import to_vector, to_array, dx_square, dy_square

def solve(wavefunction, potential_x, potential_y, hamiltonian_x, hamiltonian_y, n, step, dt):
    """
    Решает уравнение Шредингера одним шагом методом ADI.

    Args:
        wavefunction (numpy.ndarray): Входная волновая функция.
        potential_x (numpy.ndarray): Потенциал вдоль оси x.
        potential_y (numpy.ndarray): Потенциал вдоль оси y.
        hamiltonian_x (scipy.sparse.csc_matrix): Разреженная матрица Гамильтона для оси x.
        hamiltonian_y (scipy.sparse.csc_matrix): Разреженная матрица Гамильтона для оси y.
        n (int): Размер сетки.
        step (float): Шаг сетки.
        dt (float): Шаг по времени.

    Returns:
        numpy.ndarray: Обновленная волновая функция.
    """
    wf_vec_x = to_vector(wavefunction, n)
    wf_deriv_y = to_vector(dy_square(wavefunction, n, step), n)
    updated_wf_x = wf_vec_x + (1j * dt / 2) * (wf_deriv_y - potential_x * wf_vec_x)
    updated_wf_x = scipy.sparse.linalg.spsolve(hamiltonian_x, updated_wf_x)
    wavefunction = to_array(updated_wf_x, n)

    wf_vec_y = to_vector(wavefunction, n)
    wf_deriv_x = to_vector(dx_square(wavefunction, n, step), n)
    updated_wf_y = wf_vec_y + (1j * dt / 2) * (wf_deriv_x - potential_y * wf_vec_y)
    updated_wf_y = scipy.sparse.linalg.spsolve(hamiltonian_y, updated_wf_y)
    wavefunction = to_array(updated_wf_y, n)

    return wavefunction