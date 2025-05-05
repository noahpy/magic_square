# magic_square_utils.py
from typing import List
import numpy as np
import random

# --- Base Matrices for Magic Square Construction ---
A = np.array([[0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0]])
B = np.array([[0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [1, 0, 0, 0]])
C = np.array([[0, 0, 0, 1], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0]])
D = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1]])
E = np.array([[0, 0, 0, 1], [1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0]])
F = np.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
G = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 1, 0, 0]])

base_matrices = [A, B, C, D, E, F, G]
n_dimensions = 7  # The search space is 7-dimensional coefficient vectors
indexes = [i for i in range(n_dimensions)]
# vector_steps_indices are pairs (a, b) indicating change +1 in coeff_a, -1 in coeff_b
vector_steps_indices = [[a, b] for a in indexes for b in indexes if a != b]

# Pre-calculate the actual vector step arrays [0,..,+1,..,-1,..,0]
# These are the valid discrete movements in the coefficient space.
vector_steps_vectors = []
for a, b in vector_steps_indices:
    step_vec = np.zeros(n_dimensions, dtype=int)
    step_vec[a] = 1
    step_vec[b] = -1
    vector_steps_vectors.append(step_vec)

# --- Validity Check and Magic Square Calculation Functions ---


def check_coefficients(coefficients: np.ndarray, magic_number: int) -> bool:
    """
    Checks if a given coefficient vector is valid.
    Validity requires certain sums of coefficients to be non-negative
    AND the resulting square to be a magic square with the target magic number.
    Assumes coefficients is a numpy array.
    """
    # Check non-negative sum constraints based on the structure derived from base matrices
    # These sums relate to specific entries or combinations of entries being non-negative.
    sums_to_check = [[2], [6], [1], [5], [1, 3], [0, 5], [2, 4], [4, 5], [
        0, 2], [3, 6], [2, 3], [1, 4], [0, 6], [0, 1], [4, 6], [3, 5]]
    for s_indices in sums_to_check:
        coeff_sum = 0
        for idx in s_indices:
            coeff_sum += coefficients[idx]
        if coeff_sum < 0:
            return False

    # Calculate the square to check its magic property
    if len(coefficients) != n_dimensions:
        return False  # Invalid dimensions

    square = calc_magic_square(coefficients)

    # Check if it forms a magic square with the correct magic number
    if check_magic_square(square) != magic_number:
        return False

    return True  # All checks passed


def calc_magic_square(coefficients: np.ndarray) -> np.ndarray:
    """
    Calculates the 4x4 magic square from the 7 coefficients.
    Assumes coefficients is a numpy array.
    """
    if len(coefficients) != n_dimensions:
        raise ValueError(f"Coefficient vector must have {
                         n_dimensions} dimensions.")

    # Ensure coefficients are treated as integers for matrix multiplication
    coeffs_int = coefficients.astype(int)

    return (coeffs_int[0] * base_matrices[0] +
            coeffs_int[1] * base_matrices[1] +
            coeffs_int[2] * base_matrices[2] +
            coeffs_int[3] * base_matrices[3] +
            coeffs_int[4] * base_matrices[4] +
            coeffs_int[5] * base_matrices[5] +
            coeffs_int[6] * base_matrices[6])


def check_magic_square(square: np.array) -> int:
    """
    Checks if a 4x4 square is a magic square and returns the magic sum,
    or -1 if it's not a magic square or not a 4x4 array.
    """
    if square.shape != (4, 4):
        return -1

    # Ensure square entries are integers for summing
    square_int = square.astype(int)

    s = np.sum(square_int[0])  # Sum of the first row

    # Check row sums
    for i in range(4):
        if np.sum(square_int[i]) != s:
            return -1
    # Check column sums
    for i in range(4):
        if np.sum(square_int[:, i]) != s:
            return -1
    # Check diagonals
    sum_diag1 = np.trace(square_int)
    sum_diag2 = np.trace(np.fliplr(square_int))
    if sum_diag1 != s or sum_diag2 != s:
        return -1

    return int(s)


def get_random_coefficients(magic_number: int, random_state: np.random.RandomState = None) -> np.ndarray:
    """
    Generates a random *valid* coefficient vector for a given magic number
    by starting with a base configuration and applying random valid steps.
    Uses the provided random_state for reproducibility.
    """
    rnd = random_state if random_state is not None else np.random.RandomState()

    coefficients = np.array([magic_number // n_dimensions for _ in range(n_dimensions)], dtype=int)
    remainder = magic_number % n_dimensions
    if n_dimensions > 0:  # Avoid division by zero
        random_index = rnd.choice(n_dimensions)
        coefficients[random_index] += remainder
    # Handle negative magic numbers if necessary (though magic squares usually have positive entries)
    # This initial distribution might need adjustment depending on expected coefficient ranges.

    # Attempt to apply random valid steps to reach a valid state or improve it.
    # We will apply a fixed number of steps *if* they result in a valid configuration.
    # This helps explore the valid state space from a starting point.
    num_initial_steps_to_attempt = 1000

    current_valid_coeffs = np.array(coefficients, dtype=int)

    for _ in range(num_initial_steps_to_attempt):
        # Pick a random valid step vector (e.g., [1, -1, 0, 0, 0, 0, 0])
        # Use random_state to pick
        chosen_step_index = rnd.choice(len(vector_steps_vectors))
        step_vector = vector_steps_vectors[chosen_step_index]

        # Calculate the candidate position by applying the chosen step
        candidate_coefficients = current_valid_coeffs + step_vector

        # Check if the candidate position is valid
        # We must check against the TARGET magic number, as this function's goal is to find
        # valid coeffs *for that specific magic number*.
        if check_coefficients(candidate_coefficients, magic_number):
            # Update to the new valid state
            current_valid_coeffs = candidate_coefficients.copy()

    return current_valid_coeffs.copy()


if __name__ == "__main__":
    # durer = np.array([-4, 8, 14, -5, -1, 6, 16], dtype=int)
    # sq = calc_magic_square(durer)
    # print(sq)

    while True:
        number = int(input("Enter a magic number: "))
        c = get_random_coefficients(number)
        sq = calc_magic_square(c)
        print(sq)

