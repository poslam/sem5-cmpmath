# тема: метод оптимального исключения

import numpy as np


def optimal_elimination(M: np.ndarray) -> np.ndarray:
    M = M.astype(float)
    n = M.shape[0]
    max_cols = []

    for i in range(n):
        max_col = max(
            [(k, abs(M[i, k])) for k in range(n) if k not in max_cols],
            key=lambda x: x[1],
        )[0]
        max_cols.append(max_col)

    done = []

    for i in range(len(max_cols)):
        if (max_cols[i], i) in done or (i, max_cols[i]) in done:
            continue

        done.append((max_cols[i], i))
        M[[i, max_cols[i]]] = M[[max_cols[i], i]]

    # print(M, "\n")

    for i in range(n):
        for j in range(i):
            M[i, :] -= M[i, j] * M[j, :]

        M[i, :] /= M[i, i]

        for j in range(i):
            M[j, :] -= M[j, i] * M[i, :]

        # print(M, "\n")

    ans = []
    for i in range(n):
        index = np.argmax(np.abs(M[:, i]))
        ans.append((index, M[index, -1]))

    return list(map(lambda x: float(x[1]), sorted(ans, key=lambda x: x[0])))


M = np.array(
    [
        [0, 2, 3, 3],
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
{''.join(f"{i}: {ans}\n" for i in range(len(ans)))}

np solve:
{''.join(f"{i[0]}: {i[1]}\n" for i in enumerate(np_ans))}

delta:
{''.join(f"{i[0]}: {abs(i[1] - ans[i[0]])}\n" for i in enumerate(np_ans))}
"""
)
