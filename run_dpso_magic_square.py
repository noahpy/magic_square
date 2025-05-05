# run_dpso_magic_square.py
import numpy as np
# Import the DPSO Swarm class
from dpso_solver import DPSO_Swarm
# Import utility functions for displaying results
from magic_square_utils import check_coefficients, calc_magic_square, check_magic_square
# Import the cost function(s)
from magic_dpso_cost_functions import (
    magic_square_wish_cost, magic_coverage_cost, magic_coverage_duplicate_cost)

# Optional: For plotting the history
# import matplotlib.pyplot as plt

if __name__ == "__main__":

    # --- 1. Define the problem specifics ---
    dpso_n_particles = 200       # Number of particles
    # Maximum iterations (can be high, convergence might stop it early)
    dpso_max_iterations = 1000
    magic_number = 90           # Target magic sum for the square
    wish_numbers = [32, 2, 6, 10, 30, 25, 9, 11, 5, 28, 3, 7, 69, 35, 34]

    # --- 2. Choose the Cost Function ---
    # You can define other cost functions in magic_dpso_cost_functions.py
    # and select them here.
    selected_cost_function = magic_coverage_duplicate_cost

    # --- 3. DPSO parameters (often require tuning) ---
    dpso_w = 0.8    # Inertia weight
    dpso_c1 = 2.0   # Cognitive coefficient
    dpso_c2 = 2.0   # Social coefficient

    # Max velocity (optional): Clips the real-valued velocity components.
    dpso_max_velocity = 10.0  # Limit velocity components

    # --- 4. Step Combination Parameters ---
    # Set to True to enable combining top individual steps
    dpso_consider_combined_steps = True
    # If considering combined steps, how many of the best individual steps to use for combinations
    # Note: The number of *pairs* grows quadratically with this number,
    # plus the original N steps. Be mindful of performance.
    dpso_num_top_steps_to_combine = 10  # Example: Combines 10 best steps pairwise

    # --- 5. Convergence Parameters ---
    # Number of consecutive iterations gbest position must not change significantly
    convergence_threshold_iterations = 200
    # How small the change in gbest position must be to count towards convergence_count
    convergence_pos_tolerance = 1e-6

    # dpso_random_seed = 42 # Set a seed for reproducibility (optional)

    print("--- Magic Square DPSO Solver ---")
    print(f"Magic Number Target: {magic_number}")
    print(f"Wish Numbers: {wish_numbers}")
    print(f"Number of Particles: {dpso_n_particles}")
    print(f"Maximum Iterations: {dpso_max_iterations}")
    # Print the function name
    print(f"Selected Cost Function: {selected_cost_function.__name__}")
    print(f"Step Combination Enabled: {dpso_consider_combined_steps}")
    if dpso_consider_combined_steps:
        print(f"  Number of Top Steps for Combination: {
              dpso_num_top_steps_to_combine}")
    print(f"Convergence Threshold (iterations): {
          convergence_threshold_iterations}")
    print(f"Convergence Tolerance (position change): {
          convergence_pos_tolerance}")
    print("-" * 30)

    # --- 6. Create the DPSO swarm ---
    dpso_swarm = DPSO_Swarm(
        n_particles=dpso_n_particles,
        magic_number=magic_number,
        wish_numbers=wish_numbers,
        w=dpso_w,
        c1=dpso_c1,
        c2=dpso_c2,
        max_velocity=dpso_max_velocity,
        # random_seed=dpso_random_seed,
        cost_function=selected_cost_function,  # Pass the selected cost function
        consider_combined_steps=dpso_consider_combined_steps,  # Pass the flag
        # Pass the number for combination
        num_top_steps_to_combine=dpso_num_top_steps_to_combine
    )

    print("-" * 30)

    # --- 7. Run the optimization ---
    dpso_gbest_position, dpso_gbest_cost, dpso_history = dpso_swarm.optimize(
        max_iterations=dpso_max_iterations,
        convergence_threshold_iterations=convergence_threshold_iterations,
        convergence_pos_tolerance=convergence_pos_tolerance
    )

    # --- 8. Print results ---
    print("\n--- Optimization Results ---")
    if dpso_gbest_position is not None:
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

            # calculate coverage
            not_included = []
            coverage_count = 0
            flat_square = square.flatten()
            for n in wish_numbers:
                if n in flat_square:
                    coverage_count += 1
                else:
                    not_included.append(n)
            coverage = coverage_count / len(wish_numbers) * 100
            print(f"\nCoverage of wish numbers: {coverage_count} out of {
                  len(wish_numbers)} or {coverage:.2f}%")
            if len(not_included) > 0:
                print(f"Missing wish numbers: {not_included}")

            # print("\nMagic Square Entries and their cost contribution (squared distance to closest wish number):")
            # flat_square = square.flatten()
            # for entry in flat_square:
            #     min_dist_sq = min([(entry - wish)**2 for wish in wish_numbers]) if wish_numbers else 0
            #     print(f"  Entry: {entry}, Cost: {min_dist_sq}")
        else:
            print(
                "\nWarning: The final global best position found is not a valid coefficient vector.")
            print(f"Cost reported for this invalid position: {
                  dpso_gbest_cost}")

    else:
        print("Optimization failed or aborted, no global best position found.")

    # Optional: Plot history (uncomment to use)
    # if dpso_history:
    #     plt.figure(figsize=(10, 6))
    #     plt.plot(dpso_history)
    #     plt.xlabel("Iteration")
    #     plt.ylabel("Global Best Cost")
    #     plt.title(f"DPSO Optimization History (Magic Number {magic_number})")
    #     plt.grid(True)
    #     # Use log scale for cost if it decreases over orders of magnitude and starts high
    #     if dpso_history and max(dpso_history) > 0 and min(dpso_history) > 0 and max(dpso_history) / min(dpso_history) > 100: # Check if range is large
    #        plt.yscale('log')
    #     plt.show()
