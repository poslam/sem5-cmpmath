# тема: градиентный спуск

import sys

from labs.funcs import *

sys.stdout = open("./labs/output.txt", "w")


def grad_descent_method(
    M: np.ndarray,
    x0: np.ndarray = None,
    eps: float = 1e-10,
    max_iter=1e5,
) -> tuple:
    A = M[:, :-1]
    b = M[:, -1]
    n = A.shape[0]
    x = x0.copy() if x0 is not None else np.zeros(n)

    if not isinstance(max_iter, int):
        max_iter = int(max_iter)

    for iter_count in range(max_iter):
        r = b - A @ x

        alpha = (r.T @ r) / (r.T @ A @ r)
        x_new = x + alpha * r

        if np.linalg.norm(r) < eps:
            iter_count += 1
            break

        x = x_new

    print(f"{iter_count}\t{np.linalg.norm(r)}")

    return x_new, iter_count


size = (6, 7)
M = generate_symmetric_pos_def_matrix(*size).astype(np.double)
M /= np.max(M)

print("matrix:")
print_matrix(M)

print("steps:")
ans = grad_descent_method(M, eps=1e-16, max_iter=1e5)

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
