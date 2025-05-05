
from typing import List
import numpy as np
import random
import math

from magic_square_utils import (
    check_coefficients,       # Function to check validity of a coefficient vector
    calc_magic_square,        # Function to calculate the square from coefficients
)

# --- Cost Function for DPSO ---


def magic_square_wish_cost(coefficients: np.ndarray, wish_numbers: List[int], magic_number: int) -> float:
    """
    Cost function for the magic square problem.
    Calculates the sum of squared distances of each square entry
    from the closest 'wish' number. Returns infinity for invalid coefficients.

    Args:
        coefficients (np.ndarray): The 7D integer coefficient vector.
        wish_numbers (List[int]): A list of target numbers for square entries.
        magic_number (int): The target magic sum. Required by check_coefficients.

    Returns:
        float: The total cost (sum of squared minimum distances),
               or float('inf') if coefficients are invalid.
    """
    # First, check if the coefficients produce a valid magic square with the target magic number
    # Pass the target magic number to the validity check.
    if not check_coefficients(coefficients, magic_number):
        # Return a very high cost for invalid coefficients
        return float('inf')

    # Calculate the magic square (validity confirmed by check_coefficients)
    square = calc_magic_square(coefficients)

    total_cost = 0
    flat_square = square.flatten()

    for entry in flat_square:
        # Find the squared distance to the closest wish number
        min_dist_sq = float('inf')
        # Handle case with no wish numbers (cost is 0 for all entries)
        if not wish_numbers:
            min_dist_sq = 0
        else:
            # Calculate distance to all wish numbers and find the minimum squared distance
            min_dist_sq = min([(entry - wish)**2 for wish in wish_numbers])

        total_cost += min_dist_sq

    return total_cost


def magic_square_wish_coverage_cost(coefficients: np.ndarray, wish_numbers: List[int],
                                    magic_number: int) -> float:
    """
    Cost function for the magic square problem.
    Calculates the sum of squared distances of each square entry
    from the closest 'wish' number. Returns infinity for invalid coefficients.
    Additionally, checks how much of the wish numbers are covered by the magic square.
    For every wish number that is not covered, the squared distance to the closest 
    existing entry + square root of the magic number is added to the cost.

    Args:
        coefficients (np.ndarray): The 7D integer coefficient vector.
        wish_numbers (List[int]): A list of target numbers for square entries.
        magic_number (int): The target magic sum. Required by check_coefficients.

    Returns:
        float: The total cost (sum of squared minimum distances),
               or float('inf') if coefficients are invalid.
    """
    # First, check if the coefficients produce a valid magic square with the target magic number
    # Pass the target magic number to the validity check.
    if not check_coefficients(coefficients, magic_number):
        # Return a very high cost for invalid coefficients
        return float('inf')

    # Calculate the magic square (validity confirmed by check_coefficients)
    square = calc_magic_square(coefficients)

    total_cost = 0
    flat_square = square.flatten()

    for entry in flat_square:
        # Find the squared distance to the closest wish number
        min_dist_sq = float('inf')
        # Handle case with no wish numbers (cost is 0 for all entries)
        if not wish_numbers:
            min_dist_sq = 0
        else:
            # Calculate distance to all wish numbers and find the minimum squared distance
            min_dist_sq = min([(entry - wish)**2 for wish in wish_numbers])

        total_cost += min_dist_sq

    for wish in wish_numbers:
        if wish not in flat_square:
            min_dist_sq = float('inf')
            for entry in flat_square:
                min_dist_sq = min(min_dist_sq, (entry - wish)**2)
            total_cost += min_dist_sq + math.sqrt(magic_number)

    return total_cost


def magic_coverage_cost(coefficients: np.ndarray, wish_numbers: List[int],
                                    magic_number: int) -> float:
    """
    Cost function for the magic square problem.
    Checks how much of the wish numbers are covered by the magic square.
    For every wish number that is not covered, the squared distance to the closest 
    existing entry + square root of the magic number is added to the cost.

    Args:
        coefficients (np.ndarray): The 7D integer coefficient vector.
        wish_numbers (List[int]): A list of target numbers for square entries.
        magic_number (int): The target magic sum. Required by check_coefficients.

    Returns:
        float: The total cost (sum of squared minimum distances),
               or float('inf') if coefficients are invalid.
    """
    # First, check if the coefficients produce a valid magic square with the target magic number
    # Pass the target magic number to the validity check.
    if not check_coefficients(coefficients, magic_number):
        # Return a very high cost for invalid coefficients
        return float('inf')

    # Calculate the magic square (validity confirmed by check_coefficients)
    square = calc_magic_square(coefficients)

    total_cost = 0
    flat_square = square.flatten()

    for wish in wish_numbers:
        if wish not in flat_square:
            min_dist_sq = float('inf')
            for entry in flat_square:
                min_dist_sq = min(min_dist_sq, (entry - wish)**2)
            total_cost += min_dist_sq + math.sqrt(magic_number)

    return total_cost

