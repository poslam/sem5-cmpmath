# тема: метод отражений (1.1.6)

import numpy as np
from tabulate import tabulate


def print_matrix(matrix):
    if len(matrix.shape) == 1:
        matrix = matrix.reshape((1, matrix.shape[0]))
    str_matrix = [[str(cell) for cell in row] for row in matrix]
    print(f"{tabulate(str_matrix, tablefmt='fancy_grid')}\n")


def check_ans(M: np.ndarray, ans: np.ndarray) -> bool:
    print_matrix(M[:, :-1] @ ans - M[:, -1])


def QR(M: np.ndarray) -> np.ndarray:
    A = M[:, :-1]
    b = M[:, -1]
    n = M.shape[0]

    x = np.zeros(n)

    P_s = []

    for k in range(n - 1):
        p = np.zeros(n)
        P = np.zeros((n, n))

        p[k] = A[k, k] + (1 if A[k, k] >= 0 else -1) * np.sqrt(sum(A[k:, k] ** 2))
        p[k + 1 :] = A[k + 1 :, k]

        for i in range(n):
            for j in range(n):
                P[i, j] = (1 if i == j else 0) - 2 * p[i] * p[j] / float(
                    np.sum(p[k:n] ** 2)
                )

        P_s.append(P)
        A = P @ A

        print_matrix(A)

    Q = np.linalg.multi_dot(P_s)
    R = A

    Q1, R1 = np.linalg.qr(M[:, :-1])

    print("Q, R vs np Q, R:")
    print_matrix(np.abs(Q1 - Q))
    print_matrix(np.abs(R1 - R))

    g = np.dot(Q.T, b)

    for i in range(n - 1, -1, -1):
        x[i] = (g[i] - sum(R[i, j] * x[j] for j in range(i + 1, n))) / R[i, i]

    return x


size = (6, 7)
M = np.random.uniform(-1000, 1000, size=(size[0], size[1])).astype(np.double)

print("matrix:")
print_matrix(M)

ans = QR(M)
np_ans = np.linalg.solve(M[:, :-1], M[:, -1])

print(
    f"""
delta:
{''.join(f"{i[0]}: {abs(i[1] - ans[i[0]])}\n" for i in enumerate(np_ans))}
"""
)

print("check (M @ x - b):")
check_ans(M, ans)
