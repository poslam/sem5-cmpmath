# тема: Ричардсон
# src: https://www-users.cse.umn.edu/~saad/IterMethBook_2ndEd.pdf
# pages: 43 (Rayleigh quotient), 136, 137, 448 (Richardson)

import sys

import numpy as np
from funcs import *
from l10 import rotation_with_barriers

sys.stdout = open("./labs/output.txt", "w", encoding="utf-8")


def richardson(
    A: np.ndarray,
    a: float,
    b: float,
    eps: float = 1e-10,
    max_iter: int = 10**5,
) -> tuple:
    n = A.shape[0]
    x = np.ones(n)
    x = x / np.linalg.norm(x)

    if a + b == 0:
        a += 1

    alpha = 2 / np.abs(a + b)

    l = 0
    iters = 0

    while iters < max_iter:
        y = x.copy()

        y = y - alpha * (A @ y)
        y = y / np.linalg.norm(y)

        l_new = np.dot(A @ y, y) / np.dot(y, y)

        if np.abs(l_new - l) < eps:
            print(f"{iters}\t{np.abs(l_new - l)}")
            break

        x = y
        l = l_new
        iters += 1

    return l_new, y, iters


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
            [1, 0.42, 0.54, 0.66],
            [0.42, 1, 0.32, 0.44],
            [0.54, 0.32, 1, 0.22],
            [0.66, 0.44, 0.22, 1],
        ]
    ),
]

for M in matrices:
    eigval, eigvec, iters = richardson(
        M,
        a=-100,
        b=100,
        eps=1e-20,
        max_iter=10**5,
    )

    mx_np_eigval = np.max(np.abs(rotation_with_barriers(M, p=8)))

    print(f"\niters: {iters}")
    print(f"eignval delta: {np.abs(mx_np_eigval - np.abs(eigval))}\n")
    check_eigvec(M, eigvec, eigval)
