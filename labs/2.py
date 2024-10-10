# тема: выбор главного элемента

import numpy as np


def find_max(M):
    mx = None
    index_i = None
    index_j = None

    counter = -1

    for i in range(len(M)):
        counter += 1
        for j in range(len(M[i]) - 1):
            if mx is None or abs(M[i][j]) > abs(mx):
                mx = M[i][j]
                index_i = counter
                index_j = j

    return (mx, index_i, index_j)


def cut_matrix(M):
    mx, i_mx, j_mx = find_max(M)
    m = [-M[i][j_mx] / M[i_mx][j_mx] for i in range(len(M))]
    mx_row = M[i_mx]
    alphas = [mx_row[i] / mx for i in range(len(mx_row))]

    M1 = np.zeros(M.shape)

    for i in range(len(M1)):
        if i == i_mx:
            continue

        for j in range(len(M1[i])):
            if j == j_mx:
                continue

            M1[i][j] = M[i][j] + mx_row[j] * m[i]

    return {
        "matrix": M1,
        "alphas": alphas[:-1],
        "beta": alphas[-1],
    }


def cut_matric_to_min(M):
    alphas = []
    betas = []

    for i in range(min(M.shape)):
        ans = cut_matrix(M)

        M = ans["matrix"]

        betas.append(ans["beta"])
        alphas.append(ans["alphas"])

    return {
        "alphas": alphas,
        "betas": betas,
    }


M = np.array(
    [
        [2.1, -4.5, -2, 19.07],
        [3, 2.5, 4.3, 3.21],
        [-6, 3.5, 2.5, -18.25],
    ]
)

min_matrix = cut_matric_to_min(M)

alphas = min_matrix["alphas"]
betas = min_matrix["betas"]

[print(alphas[i], betas[i]) for i in range(len(alphas))]
