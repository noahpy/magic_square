# dpso_solver.py
from typing import List, Callable, Optional
import numpy as np
import random
import math  # Still needed for sqrt

# Import the specific cost function(s) from the new file
from magic_dpso_cost_functions import magic_square_wish_cost  # Default cost function

# Import necessary components from the magic square utility file
from magic_square_utils import (
    n_dimensions,             # The problem dimension (7)
    vector_steps_vectors,     # The list of valid discrete step vectors
    check_coefficients,       # Function to check validity of a coefficient vector
    calc_magic_square,        # Function to calculate the square from coefficients
    check_magic_square,       # Function to check if a square is magic and get sum
    get_random_coefficients   # Function to get a random valid initial coefficient vector
)

# Define the expected signature for the cost function
CostFunction = Callable[[np.ndarray, List[int], int], float]


class DPSO_Particle:
    """Represents a single particle in the DPSO swarm for problems with discrete steps and validity."""

    def __init__(self, magic_number: int, wish_numbers: List[int],
                 w: float, c1: float, c2: float, max_velocity: Optional[float],
                 random_state: np.random.RandomState = None,
                 cost_function: CostFunction = magic_square_wish_cost,  # Accept cost function here
                 consider_combined_steps: bool = True,
                 num_top_steps_to_combine: int = 10):
        """
        Initializes a DPSO particle with a random *valid* position.

        Args:
            magic_number (int): The target magic sum.
            wish_numbers (List[int]): List of target values for square entries.
            w (float): Inertia weight for velocity update.
            c1 (float): Cognitive coefficient for velocity update.
            c2 (float): Social coefficient for velocity update.
            max_velocity (np.ndarray or float or None): Max velocity for clipping.
            random_state (np.random.RandomState, optional): For reproducibility.
            cost_function (Callable): The function to evaluate the cost of a
                                      coefficient vector. Takes (coeffs, wish_numbers, magic_number).
                                      Defaults to magic_square_wish_cost.
            consider_combined_steps (bool): Whether to consider linear
                                                      combinations of top steps.
                                                      Defaults to True.
            num_top_steps_to_combine (int): The number of best individual steps to consider
                                      when generating combined steps. Defaults to 10.
        """
        self.cost_function = cost_function  # Store the provided cost function
        self.rnd = random_state if random_state is not None else np.random.RandomState()
        self.n_dimensions = n_dimensions  # Get dimension from utilities
        self.vector_steps = vector_steps_vectors  # Get step vectors from utilities

        self.magic_number = magic_number
        self.wish_numbers = wish_numbers
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.max_velocity = max_velocity  # Renamed from max_velocity to match arg name
        self.consider_combined_steps = consider_combined_steps
        self.num_top_steps_to_combine = num_top_steps_to_combine

        # Initialize position with a random *valid* coefficient vector
        # The get_random_coefficients function from utilities handles finding a valid start
        self.position = get_random_coefficients(
            magic_number=self.magic_number, random_state=self.rnd)
        # Ensure position is numpy array of integers
        self.position = np.array(self.position, dtype=int)

        # Initialize velocity - standard continuous initialization
        # Velocity is real-valued, its magnitude and direction influence step selection.
        # Initialize velocities to be small random values around 0.
        # Max sum of abs coeffs in a fundamental step vector (usually 2: +1, -1)
        step_magnitude_approx = 2
        self.velocity = self.rnd.rand(
            self.n_dimensions) * step_magnitude_approx * 0.2 - step_magnitude_approx * 0.1

        # Initialize personal best
        self.pbest_position = self.position.copy()
        # Calculate initial cost using the specific cost function
        # Pass the necessary arguments to the cost function
        self.pbest_cost = self.cost_function(
            self.position, self.wish_numbers, self.magic_number)

    def update_velocity(self, gbest_position: np.ndarray):
        """
        Updates the particle's *real-valued* velocity using the standard PSO formula.
        Differences (pbest-pos, gbest-pos) are integer vectors.
        """
        r1 = self.rnd.rand(self.n_dimensions)
        r2 = self.rnd.rand(self.n_dimensions)

        # Differences are between integer vectors, resulting in integer vectors
        cognitive_component = self.pbest_position - self.position
        social_component = gbest_position - self.position

        # Velocity is updated in real-valued space
        self.velocity = (self.w * self.velocity +
                         self.c1 * r1 * cognitive_component +
                         self.c2 * r2 * social_component)

        # Apply velocity clipping (optional but can help stabilize probabilities)
        if self.max_velocity is not None:
            # Make sure max_velocity is positive before using it for clipping range
            max_velocity_val = abs(self.max_velocity) if isinstance(
                self.max_velocity, (int, float)) else np.abs(self.max_velocity)

            # If max_velocity is a single value, apply it to all dimensions
            if isinstance(max_velocity_val, (int, float)):
                max_velocity_arr = np.full(self.n_dimensions, max_velocity_val)
            elif isinstance(max_velocity_val, np.ndarray) and max_velocity_val.shape == (self.n_dimensions,):
                max_velocity_arr = max_velocity_val
            else:
                # This case should ideally be caught in Swarm init, but adding here too
                raise TypeError(
                    "max_velocity must be float, int, or np.ndarray of shape (n_dimensions,)")

            self.velocity = np.clip(
                self.velocity, -max_velocity_arr, max_velocity_arr)

    def update_position(self):
        """
        Updates the particle's integer position by selecting one candidate step vector
        (individual or combined) probabilistically, guided by the real-valued velocity.
        The move is only applied if the resulting coefficient vector is valid.
        """
        if self.consider_combined_steps:
            # 1. Calculate scores for all original individual step vectors based on current velocity
            individual_scores = np.array(
                [np.dot(self.velocity, step_vec) for step_vec in self.vector_steps])

            # 2. Identify the indices of the top N scoring individual steps
            # N is self.num_top_steps_to_combine
            num_steps_to_consider = min(self.num_top_steps_to_combine, len(
                self.vector_steps))  # Ensure N doesn't exceed total steps
            # Get indices that would sort the scores in descending order
            sorted_indices = np.argsort(individual_scores)[::-1]
            # Take the top N indices
            top_n_indices = sorted_indices[:num_steps_to_consider]
            top_n_steps = [self.vector_steps[i]
                           for i in top_n_indices]  # The actual top N vectors

            # 3. Generate candidate steps (top N individual steps + sums of pairs from top N)
            # Use a set of tuples for candidate steps to automatically handle duplicates
            # np.arrays are not hashable, convert to tuple
            # Add individual top N steps
            candidate_steps_set = {tuple(step) for step in top_n_steps}

            # Add sums of pairs from the top N steps
            for i in range(num_steps_to_consider):
                for j in range(num_steps_to_consider):
                    if i == j:
                        continue
                    sum_step = top_n_steps[i] + top_n_steps[j]
                    candidate_steps_set.add(
                        tuple(sum_step))  # Add sums of pairs

            # Convert back to a list of numpy arrays
            candidate_steps = [np.array(step_tuple, dtype=int)
                               for step_tuple in candidate_steps_set]

        else:  # If not considering combined steps, candidate steps are just the fundamental ones
            candidate_steps = self.vector_steps

        # 4. Calculate scores for ALL candidate steps based on current velocity
        # Use dot product: step vector s is desirable if v points in a similar direction (v . s > 0)
        candidate_scores = np.array(
            [np.dot(self.velocity, step_vec) for step_vec in candidate_steps])

        # 5. Convert candidate scores to probabilities using softmax-like scaling
        # Ensure numerical stability by subtracting the maximum score before exponentiation
        # Handle case where max(candidate_scores) might be very large or small
        max_score = np.max(candidate_scores)
        if np.isinf(max_score) or np.isnan(max_score):  # Handle potential inf or NaN values
            # If max is inf/nan, uniform prob is safest fallback.
            probabilities = np.full(
                len(candidate_steps), 1.0 / len(candidate_steps))
        else:
            # Use a small constant shift instead of max_score to avoid potential issues
            # if max_score is consistently large or small across iterations.
            # A common practice is to shift by some value like 1.0 or 0.
            # Let's stick to subtracting max_score for better numerical range,
            # but ensure the result is not too small or large before exp.
            shifted_scores = candidate_scores - max_score
            # Clip shifted scores to prevent overflow/underflow with exp
            # Arbitrary clipping range, can be tuned
            shifted_scores = np.clip(shifted_scores, -20, 20)

            exp_scores = np.exp(shifted_scores)
            sum_exp_scores = np.sum(exp_scores)

            if sum_exp_scores == 0:  # Sum is zero (all exp(shifted) were ~0)
                # Fallback to uniform
                probabilities = np.full(
                    len(candidate_steps), 1.0 / len(candidate_steps))
            else:
                probabilities = exp_scores / sum_exp_scores

        # Normalize probabilities just in case sum deviates slightly from 1 due to floating point
        sum_probabilities = np.sum(probabilities)
        if sum_probabilities > 0:
            probabilities /= sum_probabilities
        else:
            # Final fallback if probabilities are still zero or sum to zero
            probabilities = np.full(
                len(candidate_steps), 1.0 / len(candidate_steps))

        # 6. Select one candidate step vector based on probabilities
        # Use rnd.choice on the *indices* of the candidate steps list
        if not candidate_steps:  # Safety check if candidate_steps somehow became empty
            print("Warning: candidate_steps list is empty. Skipping position update.")
            return  # Skip update if no steps available

        chosen_candidate_index = self.rnd.choice(
            len(candidate_steps), p=probabilities)
        chosen_step_vector = candidate_steps[chosen_candidate_index]

        # --- Removed velocity normalization as it's not standard for DPSO ---
        # chosen_step_vector = chosen_step_vector / \
        #     np.linalg.norm(chosen_step_vector) * np.linalg.norm(self.velocity)
        # The probability selection based on the dot product already uses the velocity magnitude/direction.
        # The step vector itself should be the discrete integer step.

        # 7. Calculate candidate position and check validity
        candidate_position = self.position + chosen_step_vector

        # Check if the candidate position results in a valid coefficient vector
        # Pass the magic_number required for the check
        if check_coefficients(candidate_position, self.magic_number):
            # If valid, update the particle's position
            self.position = candidate_position.copy().astype(
                int)  # Ensure it's integer np array
        # Else: invalid move, particle stays at current position.

    def update_pbest(self):
        """Updates the particle's personal best position and cost."""
        # Calculate the cost of the current position using the specific cost function
        # Pass the necessary arguments to the cost function
        current_cost = self.cost_function(
            self.position, self.wish_numbers, self.magic_number)

        # Update personal best if the current position is better (lower cost)
        # The cost function returns inf for invalid positions, so this naturally
        # prevents updating pbest with an invalid position if one were reached
        # (though update_position tries to avoid this).
        if current_cost < self.pbest_cost:
            self.pbest_cost = current_cost
            self.pbest_position = self.position.copy()


class DPSO_Swarm:
    """Represents the swarm of particles for DPSO."""

    def __init__(self, n_particles: int, magic_number: int, wish_numbers: List[int],
                 w: float = 0.7, c1: float = 1.5, c2: float = 1.5, max_velocity: Optional[float] = None,
                 random_seed: int = None,
                 cost_function: CostFunction = magic_square_wish_cost,  # Accept cost function here
                 consider_combined_steps: bool = True,
                 num_top_steps_to_combine: int = 10):
        """
        Generically initialized DPSO swarm. Requires a specific cost function
        and validity check/step definition for the problem space.

        Args:
            n_particles (int): Number of particles in the swarm.
            magic_number (int): The target magic sum for the square (specific to this problem).
            wish_numbers (List[int]): List of target values for square entries (specific to this problem cost).
            w (float, optional): Inertia weight. Defaults to 0.7.
            c1 (float, optional): Cognitive coefficient. Defaults to 1.5.
            c2 (float, optional): Social coefficient. Defaults to 1.5.
            max_velocity (np.ndarray or float or None, optional): Max velocity for clipping.
                                Defaults to None (no clipping). Can be a single
                                float/int or an array matching n_dimensions (7).
            random_seed (int, optional): Seed for the random number generator
                                         for reproducibility. Defaults to None.
            cost_function (Callable): The function to evaluate the cost of a
                                      coefficient vector. Takes (coeffs, wish_numbers, magic_number).
                                      Defaults to magic_square_wish_cost.
            consider_combined_steps (bool, optional): Whether particles should
                                                      consider combined steps.
                                                      Defaults to True.
            num_top_steps_to_combine (int, optional): The number of best individual
                                      steps each particle considers for combination
                                      (if consider_combined_steps is True).
                                      Defaults to 10.
        """
        self.cost_function = cost_function  # Store the cost function
        self.rnd = np.random.RandomState(
            random_seed) if random_seed is not None else np.random.RandomState()
        self.n_dimensions = n_dimensions  # Get dimension from utilities

        self.n_particles = n_particles
        self.magic_number = magic_number  # Specific problem parameter
        self.wish_numbers = wish_numbers  # Specific problem parameter
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.max_velocity = max_velocity
        self.consider_combined_steps = consider_combined_steps
        self.num_top_steps_to_combine = num_top_steps_to_combine

        self.particles = []
        self.gbest_position = None  # Will be set after first particle init
        self.gbest_cost = float('inf')  # Initialize with a very large cost

        # Create particles and find initial global best
        initial_gbest_position = None
        initial_gbest_cost = float('inf')

        for i in range(n_particles):
            # Pass problem-specific parameters and DPSO coeffs to particle
            # Also pass cost_function and combined steps parameters
            particle = DPSO_Particle(
                magic_number=self.magic_number,
                wish_numbers=self.wish_numbers,
                w=self.w,
                c1=self.c1,
                c2=self.c2,
                max_velocity=self.max_velocity,
                random_state=self.rnd,  # Pass swarm's random state
                cost_function=self.cost_function,  # Pass the cost function
                consider_combined_steps=self.consider_combined_steps,
                num_top_steps_to_combine=self.num_top_steps_to_combine
            )
            self.particles.append(particle)

            # Update initial global best if this particle's pbest is better
            if particle.pbest_cost < initial_gbest_cost:
                initial_gbest_cost = particle.pbest_cost
                initial_gbest_position = particle.pbest_position.copy()

        self.gbest_cost = initial_gbest_cost
        self.gbest_position = initial_gbest_position

        # Handle the case where all initial particles had infinite cost
        if self.gbest_cost == float('inf'):
            print(f"Warning: Global best cost initialized as infinity ({
                  self.gbest_cost}). This likely means no initial valid positions were found for magic number {self.magic_number}.")
            # Set gbest_position to the first particle's initial pos as a fallback target,
            # even though its cost is inf.
            if self.particles and hasattr(self.particles[0], 'position'):
                self.gbest_position = self.particles[0].position.copy()
            else:
                # Critical failure if particles list is empty or position wasn't set
                print(
                    "Critical Error: Could not initialize gbest_position from particles.")
                self.gbest_position = None  # Cannot proceed if gbest is None

        print(f"Swarm initialized with {n_particles} particles.")
        print(f"Initial Global Best Cost: {self.gbest_cost:.6f}")

    def optimize(self, max_iterations: int, convergence_threshold_iterations: int = 100, convergence_pos_tolerance: float = 1e-5):
        """
        Runs the DPSO optimization process for a given number of iterations,
        with optional convergence detection.

        Args:
            max_iterations (int): The maximum number of iterations to run.
            convergence_threshold_iterations (int, optional): Number of iterations
                                                             with negligible change
                                                             in gbest position to
                                                             consider converged.
                                                             Defaults to 100.
            convergence_pos_tolerance (float, optional): The tolerance for the Euclidean
                                                         distance between gbest positions
                                                         in consecutive iterations to
                                                         be considered "no change".
                                                         Defaults to 1e-5.

        Returns:
            tuple: A tuple containing (gbest_position, gbest_cost, history).
                   history is a list of gbest_cost values per iteration.
        """
        history = []

        print(f"Starting DPSO optimization for up to {
              max_iterations} iterations...")

        # Check if gbest_position was successfully initialized
        if self.gbest_position is None:
            print("Optimization aborted: Global best position could not be initialized.")
            return None, float('inf'), history

        # Initialize convergence tracking
        convergence_count = 0
        # Store the global best position from the previous iteration
        past_gbest_position = self.gbest_position.copy()
        # Calculate squared tolerance for efficiency
        convergence_pos_tolerance_sq = convergence_pos_tolerance**2

        for iteration in range(max_iterations):
            for particle in self.particles:
                # 1. Update velocity (real-valued based on pbest and gbest)
                # Ensure gbest_position is not None (checked before the loop)
                particle.update_velocity(self.gbest_position)

                # 2. Update position (discrete step based on velocity and combined steps, checks validity)
                particle.update_position()

                # 3. Update personal best based on the new position's cost
                # The cost function internally uses magic_number
                particle.update_pbest()

                # 4. Update global best if the particle's pbest is better
                if particle.pbest_cost < self.gbest_cost:
                    self.gbest_cost = particle.pbest_cost
                    self.gbest_position = particle.pbest_position.copy()

            # Record gbest_cost for this iteration
            history.append(self.gbest_cost)

            # --- Check for Convergence ---
            # Calculate difference in global best position from last iteration
            # Ensure past_gbest_position is same type/shape as current gbest_position
            pos_diff = self.gbest_position - past_gbest_position
            # Calculate squared Euclidean distance
            dist_sq = np.dot(pos_diff, pos_diff)

            if dist_sq < convergence_pos_tolerance_sq:
                # Position changed by less than tolerance
                convergence_count += 1
            else:
                # Position changed significantly
                convergence_count = 0

            # Update past_gbest_position for the next iteration
            # Make sure to copy to avoid referencing the same array
            past_gbest_position = self.gbest_position.copy()

            # Check if converged based on consecutive iterations threshold
            if convergence_count >= convergence_threshold_iterations:
                print(f"\nConverged! Gbest position did not change significantly for {
                      convergence_threshold_iterations} iterations.")
                break  # Exit the main optimization loop

            # Check if optimal cost (0) is reached
            if self.gbest_cost == 0:
                print("\nOptimal cost (0) reached!")
                break  # Exit the main optimization loop
            # --- End Convergence Check ---

            # Optional: Print progress
            # Print first, every 10%, and last
            print_interval = max_iterations // 10 or 1
            if (iteration + 1) % print_interval == 0 or iteration == 0 or iteration == max_iterations - 1:
                print(f"Iteration {iteration+1}/{max_iterations}, Current Gbest Cost: {
                      self.gbest_cost:.6f}, Convergence Count: {convergence_count}")

        # If the loop finished because max_iterations was reached without convergence
        if iteration == max_iterations - 1 and self.gbest_cost != 0 and convergence_count < convergence_threshold_iterations:
            print("\nOptimization finished after reaching maximum iterations.")

        print("Optimization finished.")
        return self.gbest_position, self.gbest_cost, history
