# тема: Ричардсон

import sys

import numpy as np
from funcs import *
from l10 import rotation_with_barriers

sys.stdout = open("./labs/output.txt", "w", encoding="utf-8")


def richardson(
    M: np.ndarray,
    k: int,
    mn: float,
    mx: float,
    eps: float = 1e-10,
    max_iter: int = 1000,
) -> tuple:
    A = M[:, :-1]
    b = M[:, -1]

    n = A.shape[0]
    x = np.ones(n)
    x = x / np.linalg.norm(x)

    gm = mn / mx
    tau0 = 2 / (mx + mn)
    ro0 = (1 - gm) / (1 + gm)

    tau = np.zeros(k)

    for j in range(k):
        nu0 = np.pi * (2 * j - 1) / k
        tau[j] = tau0 / (1 + ro0 * np.cos(nu0))

    iters = 0

    while iters * k < max_iter:
        y = x.copy()

        for j in range(k):
            x = x - tau[j] * (A @ x - b)

        iters += 1

        if np.linalg.norm(x - y) < eps:
            break

    print(f"iters: {iters * k}")

    return x


size = (6, 7)
M = generate_symmetric_pos_def_matrix(*size).astype(np.double)
M /= np.max(M)


rb_eigvals = rotation_with_barriers(M[:, :-1], p=8)

ans = richardson(
    M,
    k=10,
    mn=rb_eigvals.min(),
    mx=rb_eigvals.max(),
    eps=1e-16,
    max_iter=10**5,
)

np_ans = np.linalg.solve(M[:, :-1], M[:, -1])

print(
    f"""
delta:
{''.join(f"{i[0]}: {abs(i[1] - ans[i[0]])}\n" for i in enumerate(np_ans))}
"""
)

print("check (M @ x - b):")
check_ans(M, ans)
