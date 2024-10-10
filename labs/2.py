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


def cut_matric_to_triangular(M):
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


def solve_system(M):
    min_matrix = cut_matric_to_triangular(M)

    alphas = min_matrix["alphas"]
    betas = min_matrix["betas"]

    ans = {}

    for i in range(len(alphas) - 1, -1, -1):
        eq = alphas[i]

        for index, coef in ans.items():
            betas[i] = betas[i] - eq[index] * coef
            eq[index] = 0

        counter = 0

        for k in range(len(eq)):
            if eq[k] != 0 and ans.get(k) is None:
                ans[k] = betas[i] / eq[k]
                counter += 1

        if counter > 1:
            raise Exception("system has many solutions")

    return {k: ans[k] for k in sorted(ans)}


M = np.array(
    [
        [2.1, -4.5, -2, 19.07],
        [3, 2.5, 4.3, 3.21],
        [-6, 3.5, 2.5, -18.25],
    ]
)

ans = solve_system(M)


np_ans = np.linalg.solve(M[:, :-1], M[:, -1])

print(
    f"""

alg solve: 
{[(f"{i[0]}: {i[1]}") for i in ans.items()]}

np solve:
{[(f"{i[0]}: {i[1]}") for i in enumerate(np_ans)]}

delta:
{[(f"{i[0]}: {abs(i[1] - ans[i[0]])}") for i in enumerate(np_ans)]}

"""
)
