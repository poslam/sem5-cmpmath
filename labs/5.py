# тема: метод квадратного корня (1.1.4)

from cmath import sqrt as csqrt

import numpy as np
from tabulate import tabulate


def print_matrix(matrix):
    if len(matrix.shape) == 1:
        matrix = matrix.reshape((1, matrix.shape[0]))
    str_matrix = [[str(cell) for cell in row] for row in matrix]
    print(f"{tabulate(str_matrix, tablefmt='fancy_grid')}\n")


def make_triangular(M: np.ndarray) -> np.ndarray:
    n = M.shape[0]
    S = np.zeros((n, n), dtype=np.complex128)

    S[0, 0] = np.sqrt(M[0, 0])
    S[0, 1:] = M[0, 1:] / S[0, 0]

    for i in range(1, n):
        S[i, i] = csqrt(M[i, i] - sum(S[:i, i] ** 2))

        for j in range(i + 1, n):
            S[i, j] = (M[i, j] - sum(S[:i, i] * S[:i, j])) / S[i, i]

    return S


def square_solve(M: np.ndarray) -> np.ndarray:
    n = M.shape[0]
    x = np.zeros(n, dtype=np.complex128)
    y = np.zeros(n, dtype=np.complex128)

    S = make_triangular(M[:, :-1])
    S_T = S.T

    # print("S:")
    # print_matrix(S)

    # recovered_m = S.T @ S
    # print(np.allclose(M[:, :-1], recovered_m))

    for i in range(n):
        y[i] = (M[i, -1] - sum(S_T[i, j] * y[j] for j in range(i))) / S_T[i, i]

    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - sum(S[i, j] * x[j] for j in range(i + 1, n))) / S[i, i]

    return np.array([val.real for val in x])


M = np.array(
    [
        [1, 3, -2, 0, -2, 0.5],
        [3, 4, -5, 1, -3, 5.4],
        [-2, -5, 3, -2, 2, 5],
        [0, 1, -2, 5, 3, 7.5],
        [-2, -3, 2, 3, 4, 3.3],
    ]
)

# генерирует симметричные матрицы
size = (6, 7)
M = np.random.uniform(-1000, 1000, size=(size[0], size[1]))
M = np.hstack((M[:, :-1] @ M[:, :-1].T, M[:, -1].reshape(-1, 1)))

ans = square_solve(M)
np_ans = np.linalg.solve(M[:, :-1], M[:, -1])

print("matrix:")
print_matrix(M)

print(
    f"""
alg solve:
{''.join(f"{i}: {ans[i]}\n" for i in range(len(ans)))}

np solve:
{''.join(f"{i[0]}: {i[1]}\n" for i in enumerate(np_ans))}

delta:
{''.join(f"{i[0]}: {abs(i[1] - ans[i[0]])}\n" for i in enumerate(np_ans))}
"""
)
