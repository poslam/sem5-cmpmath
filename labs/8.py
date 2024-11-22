# тема: простая итерация. релаксация (1.2.1, 1.2.3)
# src: https://kpfu.ru/portal/docs/F_939319029/drz.tmr.ChM_2..pdf

from typing import Literal

from labs.funcs import *


def generate_diag_dominant_matrix(
    n: int,
    m: int,
    min_border: int = -100,
    max_border: int = 100,
) -> np.ndarray:
    if n >= m:
        raise ValueError("n must be less than m")

    A = np.random.uniform(min_border, max_border, size=(n, n))

    for i in range(n):
        row_sum = np.sum(np.abs(A[i, :])) - np.abs(A[i, i])

        if A[i, i] >= 0:
            A[i, i] = row_sum + np.random.uniform(1, 10)
        else:
            A[i, i] = -(row_sum + np.random.uniform(1, 10))

    b = np.random.uniform(min_border, max_border, size=(n, np.abs(n - m)))

    M = np.hstack((A, b)).astype(np.double)

    return M


def iteration(
    M: np.ndarray,
    eps=1e-10,
    max_iter=1e5,
    method: Literal["simple", "relax"] = "simple",
) -> np.ndarray:

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
        print(f"{counter}\t{np.linalg.norm(xk_prev - xk)}")
        xk_prev = xk
        xk = beta + alpha @ xk_prev

        counter += 1

        if counter >= max_iter:
            break

    return (xk, counter)


def sor_method(
    M: np.ndarray,
    omega: float,
    x0: np.ndarray = None,
    eps: float = 1e-15,
    max_iter=1e5,
) -> tuple:
    A = M[:, :-1]
    b = M[:, -1]
    n = A.shape[0]
    x = x0.copy() if x0 is not None else np.zeros(n)

    if not isinstance(max_iter, int):
        max_iter = int(max_iter)

    for iter_count in range(max_iter):
        x_old = x.copy()

        for i in range(n):
            if np.abs(A[i, i]) <= np.abs(sum(A[i, 0:i])) + np.abs(sum(A[i, i + 1 :])):
                raise ValueError("Matrix is not diagonally dominant")

            sum1 = np.dot(A[i, :i], x[:i])
            sum2 = np.dot(A[i, i + 1 :], x_old[i + 1 :])
            x[i] = (1 - omega) * x_old[i] + (omega / A[i, i]) * (b[i] - sum1 - sum2)

        print(f"{iter_count}\t{np.linalg.norm(x_old - x)}")

        if np.linalg.norm(x_old - x) <= eps:
            iter_count += 1
            break

    return (x, iter_count)


M = np.array(
    [
        [10, 2, 1, 10],
        [1, 10, 2, 12],
        [1, 1, 10, 8],
    ]
)

size = (6, 7)
M = generate_diag_dominant_matrix(*size).astype(np.double)
# M = np.random.uniform(-10, 10, size=size)
M /= np.max(M)

print("matrix:")
print_matrix(M)

print("steps:")

# ans = iteration(M, eps=1e-20, max_iter=1e5)
ans = sor_method(M, omega=1, eps=1e-20, max_iter=1e5)

np_ans = np.linalg.solve(M[:, :-1], M[:, -1])

print(f"\niterations: {ans[1]}")
ans = ans[0]

print(
    f"""
delta:
{''.join(f"{i[0]}: {abs(i[1] - ans[i[0]])}\n" for i in enumerate(np_ans))}
"""
)

print("check (M @ x - b):")
check_ans(M, ans)
