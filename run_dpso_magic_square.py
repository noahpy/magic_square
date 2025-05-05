# run_dpso_magic_square.py
import numpy as np
# Import the DPSO Swarm class
from dpso_solver import DPSO_Swarm
# Import utility functions for displaying results
from magic_square_utils import check_coefficients, calc_magic_square, check_magic_square

from magic_dpso_cost_functions import *

import matplotlib.pyplot as plt

# Optional: For plotting the history
# import matplotlib.pyplot as plt

if __name__ == "__main__":

    # --- 1. Define the problem specifics ---
    # Number of particles (increased for potentially better exploration)
    dpso_n_particles = 200
    # Maximum iterations (discrete problems often need more)
    dpso_max_iterations = 1000
    magic_number = 90            # Target magic sum for the square
    # Example: Target values for square entries
    wish_numbers = [11, 3, 54, 19, 6, 22, 16, 5, 30, 21]

    # --- 2. DPSO parameters (often require tuning) ---
    # These parameters affect the real-valued velocity update.
    dpso_w = 0.8    # Inertia weight
    dpso_c1 = 2.0   # Cognitive coefficient
    dpso_c2 = 1.0   # Social coefficient

    # Max velocity (optional): Clips the real-valued velocity components.
    # Can help prevent probabilities in update_position from becoming too extreme.
    dpso_max_velocity = 100.0  # Limit velocity components

    # dpso_random_seed = 42  # Set a seed for reproducibility (optional)

    print("--- Magic Square DPSO Solver ---")
    print(f"Magic Number Target: {magic_number}")
    print(f"Wish Numbers: {wish_numbers}")
    print(f"Number of Particles: {dpso_n_particles}")
    print(f"Maximum Iterations: {dpso_max_iterations}")
    print("-" * 30)

    # --- 3. Create the DPSO swarm ---
    dpso_swarm = DPSO_Swarm(
        n_particles=dpso_n_particles,
        magic_number=magic_number,
        wish_numbers=wish_numbers,
        w=dpso_w,
        c1=dpso_c1,
        c2=dpso_c2,
        max_velocity=dpso_max_velocity,
        # random_seed=dpso_random_seed,
        cost_function=magic_coverage_cost
    )

    print("-" * 30)

    # --- 4. Run the optimization ---
    dpso_gbest_position, dpso_gbest_cost, dpso_history = dpso_swarm.optimize(
        max_iterations=dpso_max_iterations)

    # --- 5. Print results ---
    print("\n--- Optimization Results ---")
    print(f"Global Best Coefficient Vector Found: {dpso_gbest_position}")
    print(f"Global Best Cost Found: {dpso_gbest_cost:.6f}")

    # Optional: Verify the found coefficients produce a valid magic square
    is_valid = check_coefficients(dpso_gbest_position, magic_number)
    print(f"Coefficient Vector is Valid: {is_valid}")

    if is_valid:
        found_magic_sum = check_magic_square(
            calc_magic_square(dpso_gbest_position))
        print(f"Magic Square Sum (if valid): {found_magic_sum}")
        print("\nResulting Magic Square:")
        square = calc_magic_square(dpso_gbest_position)
        print(square)


        flat_square = square.flatten()
        coverage = 0
        for entry in flat_square:
            if entry in wish_numbers:
                coverage += 1
        print(f"\nCoverage of wished numbers: {coverage}/{len(wish_numbers)}")

        # print("\nMagic Square Entries and their cost contribution:")
        # flat_square = square.flatten()
        # for entry in flat_square:
        #     min_dist_sq = min(
        #         [(entry - wish)**2 for wish in wish_numbers]) if wish_numbers else 0
        #     print(f"  Entry: {entry}, Cost: {min_dist_sq}")
    else:
        print("\nWarning: The final global best position found is not a valid coefficient vector.")
        # This might happen if the optimization converges to a point from which
        # no valid moves are possible, or if the initial state generation
        # or validity checks have edge cases. The cost should be infinity in this case.
        print(f"Cost reported for this invalid position: {dpso_gbest_cost}")

    # Optional: Plot history (uncomment to use)
    # plt.figure(figsize=(10, 6))
    # plt.plot(dpso_history)
    # plt.xlabel("Iteration")
    # plt.ylabel("Global Best Cost")
    # plt.title(f"DPSO Optimization History (Magic Number {magic_number})")
    # plt.grid(True)
    # # Use log scale for cost if it decreases over orders of magnitude and starts high
    # if dpso_history and max(dpso_history) / min(dpso_history) > 100: # Check if range is large
    #    plt.yscale('log')
    # plt.show()
