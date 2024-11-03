# тема: метод окаймления

import numpy as np
from tabulate import tabulate


def print_matrix(matrix):
    if len(matrix.shape) == 1:
        matrix = matrix.reshape((1, matrix.shape[0]))
    str_matrix = [[str(cell) for cell in row] for row in matrix]
    print(f"{tabulate(str_matrix, tablefmt='fancy_grid')}\n")


def check_ans(M: np.ndarray, ans: np.ndarray) -> bool:
    print_matrix(M[:, :-1] @ ans - M[:, -1])


def method_okaimleniy(M: np.ndarray) -> np.ndarray:
    A = M[:, :-1]
    b = M[:, -1]

    A_inv = np.array([[1 / A[0, 0]]])

    for k in range(1, A.shape[0]):
        print_matrix(A_inv)

        A_inv_k = A_inv

        a_k_col = A[:k, k]
        a_k_row = A[k, :k]
        a_k_k = A[k, k]

        d_k = a_k_k - np.dot(a_k_row, np.dot(A_inv_k, a_k_col))

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

    print_matrix(A_inv)

    x = A_inv @ b
    return x


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

print("matrix:")
print_matrix(M)

print(
    f"""
delta:
{''.join(f"{i[0]}: {abs(i[1] - ans[i[0]])}\n" for i in enumerate(np_ans))}
"""
)

print("check (M @ x - b):")
check_ans(M, ans)
