# тема: Ричардсон

import sys

import numpy as np

from labs.funcs import *

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
    x = np.random.rand(n)

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

        lambda_new = (x.T @ A @ x) / (x.T @ x)

        print(f"{iter}\t{lambda_new}")

        if abs(lambda_new - lambda_old) < eps:
            return lambda_new, iter * k + 1

        x = y
        lambda_old = lambda_new

    return lambda_new, iter * k + 1


size = (6, 6)
M = generate_symmetric_matrix(*size).astype(np.double)
M /= np.max(M)

print("matrix:")
print_matrix(M)

eigenval, iters = richardson(
    M,
    k=5,
    a=-1000,
    b=1000,
    eps=1e-20,
    max_iter=10**5,
)

mx_np_eigval = np.max(np.linalg.eigvals(M))

print(f"\niters: {iters}")
print(f"eignval delta: {np.abs(mx_np_eigval - eigenval)}")
