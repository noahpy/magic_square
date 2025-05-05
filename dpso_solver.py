# dpso_solver.py
from typing import List
import numpy as np
import random
import math

from magic_dpso_cost_functions import magic_square_wish_cost

# Import necessary components from the magic square utility file
from magic_square_utils import (
    n_dimensions,             # The problem dimension (7)
    vector_steps_vectors,     # The list of valid discrete step vectors
    check_coefficients,       # Function to check validity of a coefficient vector
    calc_magic_square,        # Function to calculate the square from coefficients
    check_magic_square,       # Function to check if a square is magic and get sum
    get_random_coefficients   # Function to get a random valid initial coefficient vector
)


# --- Discrete PSO Implementation ---

class DPSO_Particle:
    """Represents a single particle in the DPSO swarm for problems with discrete steps and validity."""

    def __init__(self, magic_number: int, wish_numbers: List[int],
                 w: float, c1: float, c2: float, max_velocity,
                 random_state: np.random.RandomState = None, cost_function=magic_square_wish_cost):
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
        """
        self.cost_function = cost_function
        self.rnd = random_state if random_state is not None else np.random.RandomState()
        self.n_dimensions = n_dimensions  # Get dimension from utilities
        self.vector_steps = vector_steps_vectors  # Get step vectors from utilities

        self.magic_number = magic_number
        self.wish_numbers = wish_numbers
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.max_velocity = max_velocity

        # Initialize position with a random *valid* coefficient vector
        # The get_random_coefficients function from utilities handles finding a valid start
        self.position = get_random_coefficients(
            magic_number=self.magic_number, random_state=self.rnd)
        # Ensure position is numpy array of integers
        self.position = np.array(self.position, dtype=int)

        # Initialize velocity - standard continuous initialization
        # Velocity is real-valued, its magnitude and direction influence step selection.
        # Initialize velocities to be small random values around 0.
        # Small random values around 0
        self.velocity = self.rnd.rand(self.n_dimensions) * 0.2 - 0.1

        # Initialize personal best
        self.pbest_position = self.position.copy()
        # Calculate initial cost using the specific cost function
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
            if isinstance(self.max_velocity, (int, float)):
                max_velocity_arr = np.full(
                    self.n_dimensions, abs(self.max_velocity))
            elif isinstance(self.max_velocity, np.ndarray) and self.max_velocity.shape == (self.n_dimensions,):
                max_velocity_arr = np.abs(self.max_velocity)
            else:
                raise TypeError(
                    "max_velocity must be float, int, or np.ndarray of shape (n_dimensions,)")

            self.velocity = np.clip(
                self.velocity, -max_velocity_arr, max_velocity_arr)

    def update_position(self):
        """
        Updates the particle's integer position by selecting one valid vector step
        probabilistically, guided by the real-valued velocity.
        The move is only applied if the resulting coefficient vector is valid.
        """
        # Calculate a score/propensity for each possible vector step based on current velocity
        # Using dot product: a step vector s is desirable if v points in a similar direction (v . s > 0)
        # Ensure dot product is calculated between velocity (float) and step (int)
        scores = np.array([np.dot(self.velocity, step_vec)
                          for step_vec in self.vector_steps])

        # Convert scores to probabilities using softmax-like scaling for discrete choice
        # Subtract max score for numerical stability with large exponents
        # Add a small epsilon to prevent log(0) if any score leads to exp(0)
        e_scores = np.exp(scores - np.max(scores)) + \
            1e-9  # Add epsilon for robustness
        probabilities = e_scores / np.sum(e_scores)

        # Select one step vector based on these probabilities
        # np.random.choice needs the list of items to choose from or indices
        chosen_step_index = self.rnd.choice(
            len(self.vector_steps), p=probabilities)
        chosen_step_vector = self.vector_steps[chosen_step_index]

        # Calculate the candidate position by applying the chosen step
        candidate_position = self.position + chosen_step_vector

        # Check if the candidate position results in a valid coefficient vector
        # Pass the magic_number required for the check
        if check_coefficients(candidate_position, self.magic_number):
            # If valid, update the particle's position
            self.position = candidate_position.copy().astype(
                int)  # Ensure it's integer np array

    def update_pbest(self):
        """Updates the particle's personal best position and cost."""
        # Calculate the cost of the current position using the specific cost function
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
                 w: float = 0.7, c1: float = 1.5, c2: float = 1.5, max_velocity=None,
                 random_seed: int = None, cost_function=magic_square_wish_cost):
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
        """
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

        self.particles = []
        self.gbest_position = None
        self.gbest_cost = float('inf')  # Initialize with a very large cost

        # Create particles and find initial global best
        for _ in range(n_particles):
            # Pass problem-specific parameters and DPSO coeffs to particle
            particle = DPSO_Particle(
                magic_number=self.magic_number,
                wish_numbers=self.wish_numbers,
                w=self.w,
                c1=self.c1,
                c2=self.c2,
                max_velocity=self.max_velocity,
                random_state=self.rnd,
                cost_function=cost_function
            )
            self.particles.append(particle)

            # Update global best if this particle's pbest is better
            # Check for initial infinite cost cases
            if particle.pbest_cost < self.gbest_cost:
                self.gbest_cost = particle.pbest_cost
                self.gbest_position = particle.pbest_position.copy()

        # Handle the case where all initial particles had infinite cost
        # (e.g., get_random_coefficients couldn't find a valid start).
        # In this scenario, gbest_position would still be None.
        if self.gbest_position is None or self.gbest_cost == float('inf'):
            print("Warning: Global best could not be initialized with a finite cost. Setting gbest_position to the first particle's initial position.")
            self.gbest_position = self.particles[0].position.copy()
            # gbest_cost remains inf, which is correct if no valid position was found initially

        print(f"Swarm initialized with {n_particles} particles.")
        print(f"Initial Global Best Cost: {self.gbest_cost:.6f}")

    def optimize(self, max_iterations: int):
        """
        Runs the DPSO optimization process for a given number of iterations.

        Args:
            max_iterations (int): The maximum number of iterations to run.

        Returns:
            tuple: A tuple containing (gbest_position, gbest_cost, history).
                   history is a list of gbest_cost values per iteration.
        """
        history = []

        print(f"Starting DPSO optimization for {max_iterations} iterations...")

        convergence_count = 0
        CONVERGENCE_THRESHOLD = 100
        past_gbest_cost = self.gbest_cost

        for iteration in range(max_iterations):
            for particle in self.particles:
                # 1. Update velocity (real-valued based on pbest and gbest)
                particle.update_velocity(self.gbest_position)

                # 2. Update position (discrete step based on velocity, checks validity)
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

            if math.sqrt(np.dot(self.gbest_position - past_gbest_cost,
                                self.gbest_position - past_gbest_cost)) < 0.00001:
                convergence_count += 1
            else:
                convergence_count = 0
            past_gbest_cost = self.gbest_position

            if convergence_count >= CONVERGENCE_THRESHOLD:
                print("Converged!")
                break

            if self.gbest_cost == 0:
                break

            # Optional: Print progress
            if (iteration + 1) % (max_iterations // 10 or 1) == 0 or iteration == 0 or iteration == max_iterations - 1:
                print(f"Iteration {
                      iteration+1}/{max_iterations}, Current Gbest Cost: {self.gbest_cost:.6f}")

        print("Optimization finished.")
        return self.gbest_position, self.gbest_cost, history
