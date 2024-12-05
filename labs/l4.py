# тема: LU (1.1.5)

import sys

from funcs import *

sys.stdout = open("./labs/output.txt", "w", encoding="utf-8")


def get_LU(M: np.ndarray) -> np.ndarray:
    n = M.shape[0]
    L = np.eye(n)
    U = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            U[i, j] = M[i, j] - sum(L[i, k] * U[k, j] for k in range(i))
        for j in range(i, n):
            L[j, i] = (M[j, i] - sum(L[j, k] * U[k, i] for k in range(i))) / U[i, i]

    return L, U


def LU(M: np.ndarray) -> np.ndarray:
    L, U = get_LU(M)
    n = M.shape[0]
    y = np.zeros(n)
    x = np.zeros(n)

    for i in range(n):
        y[i] = M[i, -1] - sum(L[i, j] * y[j] for j in range(i))

    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - sum(U[i, j] * x[j] for j in range(i + 1, n))) / U[i, i]

    return (L, U, x)


M = np.array(
    [
        [3, 1, -1, 2, 6],
        [-5, 1, 3, -4, -12],
        [2, 0, 1, -1, 1],
        [1, -5, 3, -3, 3],
    ]
)

size = (9, 10)
M = np.random.uniform(-1000, 1000, size=(size[0], size[1]))

ans = LU(M)
np_ans = np.linalg.solve(M[:, :-1], M[:, -1])

# print("matrix:")
# print_matrix(M)

# print("L:")
# print_matrix(ans[0])

# print("U:")
# print_matrix(ans[1])

# print(
#     f"""
# check:
# {np.allclose(ans[0] @ ans[1], M[:, :-1])}

# alg solve:
# {''.join(f"{i}: {ans[2][i]}\n" for i in range(len(ans[2])))}

# np solve:
# {''.join(f"{i[0]}: {i[1]}\n" for i in enumerate(np_ans))}

# delta:
# {''.join(f"{i[0]}: {abs(i[1] - ans[2][i[0]])}\n" for i in enumerate(np_ans))}
# """
# )
