# тема: простая итерация. релаксация (1.2.1, 1.2.3)

from labs.funcs import *


def simple_iteration(M: np.ndarray, eps=1e-10, max_iter=None) -> np.ndarray:
    A = M[:, :-1]
    b = M[:, -1]
    n = M.shape[0]

    alpha = np.zeros((n, n))
    beta = np.zeros(n)

    for i in range(n):

        if np.abs(A[i, i]) <= np.abs(sum(A[i, 0:i])) + np.abs(sum(A[i, i + 1 :])):
            raise ValueError("Matrix is not diagonally dominant")

        beta[i] = b[i] / A[i, i]

        for j in range(n):
            if i != j:
                alpha[i, j] = -A[i, j] / A[i, i]

    counter = 0

    xk = beta
    xk_prev = beta + alpha @ xk

    while np.linalg.norm(xk_prev - xk) > eps:
        print(xk, np.linalg.norm(xk_prev - xk))
        xk_prev = xk
        xk = beta + alpha @ xk_prev

        counter += 1

        if max_iter is not None and counter >= max_iter:
            break

    print(counter)
    print_matrix(xk)

    return (xk, counter)


M = np.array(
    [
        [10, 2, 1, 10],
        [1, 10, 2, 12],
        [1, 1, 10, 8],
    ]
)

# size = (6, 7)
# M = np.random.uniform(-1000, 1000, size=(size[0], size[1])).astype(np.double)
# M /= np.max(M)

print("matrix:")
print_matrix(M)

ans = simple_iteration(M, eps=1e-30)
np_ans = np.linalg.solve(M[:, :-1], M[:, -1])

print(f"iteration: {ans[1]}")
ans = ans[0]

print(
    f"""
delta:
{''.join(f"{i[0]}: {abs(i[1] - ans[i[0]])}\n" for i in enumerate(np_ans))}
"""
)

print("check (M @ x - b):")
check_ans(M, ans)
