# тема: метод прямых итераций
# src: методичка

import sys

from funcs import *

sys.stdout = open("./labs/output.txt", "w", encoding="utf-8")


def straight_iter(
    A: np.ndarray,
    x0: np.ndarray = None,
    eps: float = 1e-10,
    max_iter: int = 10**5,
) -> tuple:
    n = A.shape[0]
    x = x0.copy() if x0 is not None else np.random.rand(n)
    x = x / np.linalg.norm(x)

    l = 0
    iters = 0
    a = x.max()

    while iters < max_iter:
        y: np.ndarray = A @ (x / a)
        a_new = np.abs(y).max()

        iters += 1

        if np.abs(a - a_new) < eps:
            return y, a_new, iters

        x = y
        a = a_new

    return x, a, iters


matrices = [
    np.array(
        [
            [-0.168700, 0.353699, 0.008540, 0.733624],
            [0.353699, 0.056519, -0.723182, -0.076440],
            [0.008540, -0.723182, 0.015938, 0.342333],
            [0.733624, -0.076440, 0.342333, -0.045744],
        ]
    )
]

size = (6, 6)
M = generate_symmetric_matrix(*size).astype(np.double)
M /= np.max(M)
matrices.append(M)


for M in matrices:
    # print_matrix(M, header="matrix")

    eigvec, eigval, iters = straight_iter(M, eps=1e-16, max_iter=10**5)
    np_mx_eigval = np.max(np.abs(np.linalg.eigvals(M)))

    print(f"\niters: {iters}")
    print(eigval)
    print(f"eigval delta: {np.abs(np_mx_eigval - np.abs(eigval))}\n")
    check_eigvec(M, eigvec, eigval)
