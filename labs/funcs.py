import numpy as np
from tabulate import tabulate


def print_matrix(matrix):
    if len(matrix.shape) == 1:
        matrix = matrix.reshape((1, matrix.shape[0]))
    str_matrix = [[str(cell) for cell in row] for row in matrix]
    print(f"{tabulate(str_matrix, tablefmt='fancy_grid')}\n")


def check_ans(M: np.ndarray, ans: np.ndarray) -> bool:
    print_matrix(M[:, :-1] @ ans - M[:, -1])