# тема: метод квадратного корня (1.1.4)

import numpy as np

M = np.array(
    [
        [3, 1, -1, 2, 6],
        [-5, 1, 3, -4, -12],
        [2, 0, 1, -1, 1],
        [1, -5, 3, -3, 3],
    ]
)

np_ans = np.linalg.solve(M[:, :-1], M[:, -1])

print(np_ans)
