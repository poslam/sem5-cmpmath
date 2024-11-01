# тема: метод окаймления

import numpy as np
from tabulate import tabulate


def print_matrix(matrix):
    if len(matrix.shape) == 1:
        matrix = matrix.reshape((1, matrix.shape[0]))
    str_matrix = [[str(cell) for cell in row] for row in matrix]
    print(f"{tabulate(str_matrix, tablefmt='fancy_grid')}\n")


import numpy as np


def method_okaimleniy(M: np.ndarray) -> np.ndarray:
    A = M[:, :-1]
    b = M[:, -1]

    n = A.shape[0]
    A_inv = np.array([[1 / A[0, 0]]])  # Начинаем с матрицы 1x1
    x = np.zeros((n, 1))

    for k in range(1, n):
        A_inv_k = A_inv  # Обратная матрица на предыдущем шаге

        # Новые элементы
        a_k_k = A[k, k]  # Диагональный элемент
        a_k_row = A[k, :k]  # Новая строка
        a_k_col = A[:k, k]  # Новый столбец

        # Вычисляем компоненту обратной матрицы
        d_k = a_k_k - np.dot(a_k_row, np.dot(A_inv_k, a_k_col))

        if np.abs(d_k) < 1e-9:
            raise ValueError("Матрица вырождена или близка к вырожденной.")

        A_inv_new = np.zeros((k + 1, k + 1))

        A_inv_new[:k, :k] = (
            A_inv_k
            + np.outer(
                np.dot(A_inv_k, a_k_col),
                np.dot(a_k_row, A_inv_k),
            )
            / d_k
        )
        A_inv_new[:k, k] = -np.dot(A_inv_k, a_k_col) / d_k
        A_inv_new[k, :k] = -np.dot(a_k_row, A_inv_k) / d_k
        A_inv_new[k, k] = 1 / d_k

        A_inv = A_inv_new

    x = A_inv @ b
    return x


# Пример использования:
M = np.array(
    [
        [1, 3, -2, 0, -2, 0.5],
        [3, 4, -5, 1, -3, 5.4],
        [-2, -5, 3, -2, 2, 5],
        [0, 1, -2, 5, 3, 7.5],
        [-2, -3, 2, 3, 4, 3.3],
    ]
)

size = (6, 7)
M = np.random.uniform(-1000, 1000, size=(size[0], size[1]))

ans = method_okaimleniy(M)
np_ans = np.linalg.solve(M[:, :-1], M[:, -1])

# print("matrix:")
# print_matrix(M)

print(
    f"""
delta:
{''.join(f"{i[0]}: {abs(i[1] - ans[i[0]])}\n" for i in enumerate(np_ans))}
"""
)
