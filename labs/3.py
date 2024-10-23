# тема: метод оптимального исключения (1.1.3)

import numpy as np


def find_maxes(M: np.ndarray) -> np.ndarray:
    max_cols = []

    for i in range(M.shape[0]):
        max_col = max(
            [(k, abs(M[i, k])) for k in range(M.shape[0]) if k not in max_cols],
            key=lambda x: x[1],
        )[0]
        max_cols.append(max_col)

    return max_cols


def rearrange_matrix(M: np.ndarray, max_cols: np.ndarray) -> np.ndarray:
    done = []

    for i in range(len(max_cols)):
        if (max_cols[i], i) in done or (i, max_cols[i]) in done:
            continue

        done.append((max_cols[i], i))
        M[[i, max_cols[i]]] = M[[max_cols[i], i]]

    return M


def basic_elimination(M: np.ndarray) -> np.ndarray:
    for i in range(M.shape[0]):
        for j in range(i):
            M[i, :] -= M[i, j] * M[j, :]

        M[i, :] /= M[i, i]

        for j in range(i):
            M[j, :] -= M[j, i] * M[i, :]

        # print(M, "\n")

    return M


def solve_easy_system(M: np.ndarray) -> np.ndarray:
    ans = []
    for i in range(M.shape[0]):
        index = np.argmax(np.abs(M[:, i]))
        ans.append((index, M[index, -1]))

    return list(map(lambda x: float(x[1]), sorted(ans, key=lambda x: x[0])))


def optimal_elimination(M: np.ndarray) -> np.ndarray:
    M = M.astype(float)

    M = rearrange_matrix(M, find_maxes(M))
    M = basic_elimination(M)

    # print(M, "\n")

    return solve_easy_system(M)


M = np.array(
    [
        [5, 2, 3, 3],
        [1, 6, 1, 5],
        [3, -4, -2, 8],
    ]
)

# size = (9, 10)
# M = np.random.uniform(-1000, 1000, size=(size[0], size[1]))

ans = optimal_elimination(M)
np_ans = np.linalg.solve(M[:, :-1], M[:, -1])

print(
    f"""
matrix:
{M}

alg solve:
{''.join(f"{i}: {ans[i]}\n" for i in range(len(ans)))}

np solve:
{''.join(f"{i[0]}: {i[1]}\n" for i in enumerate(np_ans))}

delta:
{''.join(f"{i[0]}: {abs(i[1] - ans[i[0]])}\n" for i in enumerate(np_ans))}
"""
)
