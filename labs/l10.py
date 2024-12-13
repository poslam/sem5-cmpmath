# тема: вращение с преградами
# src: методичка

import sys

import numpy as np
from funcs import *

sys.stdout = open("./labs/output.txt", "w", encoding="utf-8")


def sign(x: float):
    if x >= 0:
        return 1
    else:
        return -1


def rotation_with_barriers(
    A: np.ndarray,
    p: int = 4,
) -> np.ndarray:
    D = A.copy()
    n = D.shape[0]

    if np.linalg.det(D) == 0:
        raise ValueError("matrix is singular")

    counter = 0

    for K in range(1, p + 1):
        sigma = np.sqrt(np.max(np.abs(np.diag(np.diag(D))))) * 10 ** (-K)
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
            s = sign(D[i, j] * (D[i, i] - D[j, j])) * np.sqrt(
                1 / 2 * (1 - abs(D[i, i] - D[j, j]) / d)
            )
            c = np.sqrt(1 / 2 * (1 + abs(D[i, i] - D[j, j]) / d))

            C = D.copy()

            for k in range(n):
                if k in [i, j]:
                    continue
                C[k, i] = C[i, k] = c * D[k, i] + s * D[k, j]
                C[k, j] = C[j, k] = -s * D[k, i] + c * D[k, j]

            C[i, j] = C[j, i] = 0
            C[i, i] = c**2 * D[i, i] + 2 * c * s * D[i, j] + s**2 * D[j, j]
            C[j, j] = s**2 * D[i, i] - 2 * c * s * D[i, j] + c**2 * D[j, j]

            # print(f"K: {K} \nsigma: {sigma} \ni,j: {i+1,j+1} \nmx_val: {mx_val}")
            # print(f"c:\t{c}\ts:\t{s}")
            # print_matrix(D)
            # print_matrix(C)

            D = C

            counter += 1

    # print(f"steps: {counter}")

    return np.diag(D)


if __name__ == "__main__":
    # size = (6, 6)
    # M = generate_symmetric_matrix(*size).astype(np.double)
    # M /= np.max(M)

    # print_matrix(M, "matrix")

    # ans = rotation_with_barriers(M, p=8)  # max(p)=8
    # np_ans = np.linalg.eigvals(M)

    # ans = np.array(sorted(ans))
    # np_ans = np.array(sorted(np_ans))

    # print(
    #     f"""
    # delta:
    # {''.join(f"{i[0]}: {abs(i[1] - ans[i[0]])}\n" for i in enumerate(np_ans))}
    # """
    # )

    # check_eigvals(M, ans)

    M1 = np.array(
        [
            [-0.168700, 0.353699, 0.008540, 0.733624],
            [0.353699, 0.056519, -0.723182, -0.076440],
            [0.008540, -0.723182, 0.015938, 0.342333],
            [0.733624, -0.076440, 0.342333, -0.045744],
        ]
    )

    eigenvalues1 = np.array([-0.943568, -0.744036, 0.687843, 0.857774])

    M2 = np.array(
        [
            [2.2, 1.0, 0.5, 2.0],
            [1.0, 1.3, 2, 1],
            [0.5, 2, 0.5, 1.6],
            [2, 1, 1.6, 2],
        ]
    )

    eigenvalues2 = np.array([5.652, 1.545, -1.420, 0.2226])

    M3 = np.array(
        [
            [1, 0.42, 0.54, 0.66],
            [0.42, 1, 0.32, 0.44],
            [0.54, 0.32, 1, 0.22],
            [0.66, 0.44, 0.22, 1],
        ]
    )

    eigenvalues3 = np.array([2.3227, 0.7967, 0.6383, 0.2423])

    matrices = [M1, M2, M3]
    vals = [eigenvalues1, eigenvalues2, eigenvalues3]

    for i in range(len(matrices)):
        print(
            np.array(sorted(np.linalg.eigvals(matrices[i])))
            - np.array(sorted(rotation_with_barriers(matrices[i], p=8))),
            "\n",
        )

    print("given: ", np.array(sorted(vals[i])))
    print("numpy: ", np.array(sorted(np.linalg.eigvals(matrices[i]))))
    print(
        "delta (given - algo): ",
        np.array(sorted(vals[i]))
        - np.array(sorted(rotation_with_barriers(matrices[i], p=8))),
        "\n",
    )
