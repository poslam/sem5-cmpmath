# тема: простая итерация
# src: методичка

import sys

from labs.funcs import *

sys.stdout = open("./labs/output.txt", "w")


def simple_iter(
    A: np.ndarray,
    x0: np.ndarray = None,
    eps: float = 1e-10,
    max_iter: int = 10**5,
) -> tuple:
    n = A.shape[0]
    x = x0.copy() if x0 is not None else np.random.rand(n)
    x = x / np.linalg.norm(x)

    l = 0
    for iter in range(max_iter):
        y = A @ x
        l_new = y @ x

        y_norm = y / np.linalg.norm(y)

        # print(f"{iter}\t{l_new}")

        if np.abs(l - l_new) < eps:
            break

        x = y_norm
        l = l_new

    return x, l, iter


M = np.array(
    [
        [2, 1, 1],
        [1, 2.5, 1],
        [1, 1, 3],
    ]
)

size = (6, 6)
M = generate_symmetric_matrix(*size).astype(np.double)
M /= np.max(M)

print_matrix(M, header="matrix")

eigvec, eigval, iters = simple_iter(M, eps=1e-16, max_iter=10**5)
np_mx_eigval = np.max(np.linalg.eigvals(M))

print(f"\niters: {iters}")
print(f"eigval delta: {np.abs(np_mx_eigval - eigval)}\n")
check_eigvec(M, eigvec, eigval)
