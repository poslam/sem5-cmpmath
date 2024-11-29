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

        if np.abs(l - l_new) < eps:
            print(f"{iter}\t{np.abs(l - l_new)}")
            return x, l, iter

        x = y_norm
        l = l_new

    return x, l, iter


M = np.array(
    [
        [-0.168700, 0.353699, 0.008540, 0.733624],
        [0.353699, 0.056519, -0.723182, -0.076440],
        [0.008540, -0.723182, 0.015938, 0.342333],
        [0.733624, -0.076440, 0.342333, -0.045744],
    ]
)

# size = (6, 6)
# M = generate_symmetric_matrix(*size).astype(np.double)
# M /= np.max(M)

print_matrix(M, header="matrix")

eigvec, eigval, iters = simple_iter(M, eps=1e-16, max_iter=10**5)
np_mx_eigval = np.max(np.abs(np.linalg.eigvals(M)))

print(f"\niters: {iters}")
print(f"eigval delta: {np.abs(np_mx_eigval - eigval)}\n")
check_eigvec(M, eigvec, eigval)
