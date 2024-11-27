# тема: вращение с преградами
# src: методичка

import sys

import numpy as np

from labs.funcs import *

sys.stdout = open("./labs/output.txt", "w")


def rotation_with_barriers(
    A: np.ndarray,
    p: int = 4,
) -> np.ndarray:
    D = A.copy()
    n = D.shape[0]

    counter = 0

    for K in range(1, p + 1):
        sigma = np.sqrt(np.max(np.abs(np.diag(np.diag(M))))) * 10 ** (-K)
        # sigma = 10 ** (-K)

        while True:
            if counter > 1e5:
                raise ValueError("inf cycle")

            mx_val = -np.inf
            idx = ()

            for i in range(n):
                for j in range(n):
                    if D[i, j] > mx_val and np.abs(D[i, j]) >= sigma and i != j:
                        mx_val = D[i, j]
                        idx = (i, j)

            if mx_val == -np.inf:
                break

            i, j = idx[0], idx[1]

            d = np.sqrt((D[i, i] - D[j, j]) ** 2 + 4 * D[i, j] ** 2)
            s = np.sign(D[i, j] * (D[i, i] - D[j, j])) * np.sqrt(
                1 / 2 * (1 - np.linalg.norm(D[i, i] - D[j, j]) / d)
            )
            c = np.sqrt(1 / 2 * (1 + np.linalg.norm(D[i, i] - D[j, j]) / d))

            # print(f"K: {K} \nsigma: {sigma} \ni,j: {i+1,j+1} \nmx_val: {mx_val}")
            # print(f"c:\t{c}\ts:\t{s}")
            # print_matrix(D)

            T = np.eye(n)
            T[i, i] = T[j, j] = c
            T[i, j] = -s
            T[j, i] = s

            D = T.T @ D @ T

            counter += 1

    print(f"steps: {counter}")

    return np.diag(D)


size = (6, 6)
M = generate_symmetric_matrix(*size).astype(np.double)
M /= np.max(M)

print("matrix:")
print_matrix(M)

ans = rotation_with_barriers(M, p=8)  # max(p)=8
np_ans = np.linalg.eigvals(M)

ans = np.array(sorted(ans))
np_ans = np.array(sorted(np_ans))

print(
    f"""
delta:
{''.join(f"{i[0]}: {abs(i[1] - ans[i[0]])}\n" for i in enumerate(np_ans))}
"""
)
