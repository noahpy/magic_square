
from typing import List
import numpy as np
import random


A = np.array([[0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0]])
B = np.array([[0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [1, 0, 0, 0]])
C = np.array([[0, 0, 0, 1], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0]])
D = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1]])
E = np.array([[0, 0, 0, 1], [1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0]])
F = np.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
G = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 1, 0, 0]])

base_vectors = [A, B, C, D, E, F, G]
indexes = [i for i in range(7)]
vector_steps = [[a, b] for a in indexes for b in indexes if a != b]


def check_coefficients(coefficients: List, magic_number: int):
    sums = [[2], [6], [1], [5], [1, 3], [0, 5], [2, 4], [4, 5], [
        0, 2], [3, 6], [2, 3], [1, 4], [0, 6], [0, 1], [4, 6], [3, 5]]
    for s in sums:
        x = 0
        for i in s:
            x += coefficients[i]
        if x < 0:
            return False
    if check_magic_square(calc_magic_square(coefficients)) != magic_number:
        return False
    return True


def calc_magic_square(coefficients: List):
    return coefficients[0] * A + coefficients[1] * B + coefficients[2] * C + coefficients[3] * D + coefficients[4] * E + coefficients[5] * F + coefficients[6] * G


def check_magic_square(square: np.array):
    s = sum(square[0])
    for i in range(4):
        if sum(square[i]) != s:
            return -1
    for i in range(4):
        if sum(square[:, i]) != s:
            return -1
    # check diagonals
    sum_diag1 = 0
    for i in range(4):
        sum_diag1 += square[i][i]
    sum_diag2 = 0
    for i in range(4):
        sum_diag2 += square[i][3-i]
    if sum_diag1 != s or sum_diag2 != s:
        return -1
    return s


def get_random_coefficients(magic_number: int):
    coefficients = [magic_number // 7 for i in range(7)]
    coefficients[0] += magic_number % 7

    attempted_steps = random.randint(0, 500)
    count = 0

    while count < attempted_steps:
        step = random.choice(vector_steps)
        adding_coefficients = [0 for i in range(7)]
        adding_coefficients[step[0]] = 1
        adding_coefficients[step[1]] = -1
        new_coefficients = np.array(coefficients) + np.array(adding_coefficients)
        if check_coefficients(new_coefficients, magic_number):
            coefficients = new_coefficients
            count += 1

    return coefficients


durer = [-4, 8, 14, -5, -1, 6, 16]

sq = calc_magic_square(durer)

# print(sq)

c1 = get_random_coefficients(90)

sq2 = calc_magic_square(c1)

print(sq2)
