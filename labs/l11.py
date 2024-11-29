# тема: Ричардсон

import sys

import numpy as np

from labs.funcs import *
from labs.l10 import rotation_with_barriers

sys.stdout = open("./labs/output.txt", "w")


def richardson(
    A: np.ndarray,
    k: int,
    a: float,
    b: float,
    eps: float = 1e-10,
    max_iter: int = 1000,
) -> tuple:
    """
    - A - кваратная матрица
    - k - степень полинома
    - a - нижнаяя граница собственного значения
    - b - верхняя граница собственного значения

    returns: (eigval, eigvec, iter_count)
    """
    n = A.shape[0]
    x = np.ones(n)
    x = x / np.linalg.norm(x)

    c = (b + a) / 2
    e = (b - a) / 2
    tau = np.zeros(k)

    for j in range(k):
        theta = np.pi * (2 * j + 1) / (2 * k)
        tau[j] = 1 / (c + e * np.cos(theta))

    lambda_old = 0

    for iter in range(max_iter):
        y = x.copy()

        for j in range(k):
            y = tau[j] * (A @ y)
            y = y / np.linalg.norm(y)

        lambda_new = (x.T @ A @ x) / (x.T @ x)

        if np.abs(lambda_new - lambda_old) < eps:
            print(f"{iter}\t{np.abs(lambda_new - lambda_old)}")
            return lambda_new, y, iter * k + 1

        x = y
        lambda_old = lambda_new

    return lambda_new, y, iter * k + 1


# size = (6, 6)
# M = generate_symmetric_matrix(*size).astype(np.double)
# M /= np.max(M)

# print_matrix(M, "matrix")

matrices = [
    np.array(
        [
            [-0.168700, 0.353699, 0.008540, 0.733624],
            [0.353699, 0.056519, -0.723182, -0.076440],
            [0.008540, -0.723182, 0.015938, 0.342333],
            [0.733624, -0.076440, 0.342333, -0.045744],
        ]
    ),
    np.array(
        [
            [2.2, 1.0, 0.5, 2.0],
            [1.0, 1.3, 2, 1],
            [0.5, 2, 0.5, 1.6],
            [2, 1, 1.6, 2],
        ]
    ),
    np.array(
        [
            [2, 0.42, 0.54, 0.66],
            [0.42, 2, 0.32, 0.44],
            [0.54, 0.32, 2, 0.22],
            [0.66, 0.44, 0.22, 2],
        ]
    ),
]

for M in matrices:
    eigval, eigvec, iters = richardson(
        M,
        k=5,
        a=-1000,
        b=1000,
        eps=1e-20,
        max_iter=10**5,
    )

    mx_np_eigval = np.max(np.abs(rotation_with_barriers(M, p=8)))

    print(f"\niters: {iters}")
    print(f"eignval delta: {np.abs(mx_np_eigval - np.abs(eigval))}\n")
    check_eigvec(M, eigvec, eigval)
