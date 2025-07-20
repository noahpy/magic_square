import customtkinter as ctk
import tkinter as tk
from threading import Thread
import queue
import sys

# Import necessary components from your existing files
from dpso_solver import DPSO_Swarm
from magic_square_utils import check_coefficients, calc_magic_square, check_magic_square
from magic_dpso_cost_functions import (
    magic_square_wish_cost, magic_coverage_cost, magic_coverage_duplicate_cost)

# --- A dictionary to map cost function names to the actual functions ---
COST_FUNCTIONS = {
    'magic_coverage_duplicate_cost': magic_coverage_duplicate_cost,
    'magic_square_wish_cost': magic_square_wish_cost,
    'magic_coverage_cost': magic_coverage_cost
}

# --- Helper class to redirect print statements to the GUI queue in real-time ---
class QueueLogger:
    def __init__(self, queue, widget_target):
        self.queue = queue
        self.widget_target = widget_target

    def write(self, text):
        """This method is called by print(). It sends the text to the GUI queue."""
        self.queue.put((text, self.widget_target))

    def flush(self):
        """Required for the file-like object interface."""
        pass

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        # --- Basic Window Configuration ---
        self.title("DPSO Magic Square Solver")
        self.geometry("900x600")
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        # --- Create a queue for thread-to-GUI communication ---
        self.log_queue = queue.Queue()

        # --- Create UI Frames ---
        self.create_input_frame()
        self.create_output_frame()

        # --- Start periodic check of the queue ---
        self.after(100, self.process_log_queue)

    def create_input_frame(self):
        """Creates the left frame for user inputs."""
        input_frame = ctk.CTkFrame(self, width=250, corner_radius=10)
        input_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        input_frame.grid_propagate(False)

        header = ctk.CTkLabel(input_frame, text="Solver Settings", font=ctk.CTkFont(size=20, weight="bold"))
        header.pack(pady=20)

        ctk.CTkLabel(input_frame, text="Wish Numbers (comma-separated)").pack(padx=20, pady=(10, 5), anchor="w")
        self.wish_numbers_entry = ctk.CTkEntry(input_frame, placeholder_text="e.g., 24, 30, 16, 8, 12")
        self.wish_numbers_entry.insert(0, "24, 30, 16, 8, 12")
        self.wish_numbers_entry.pack(padx=20, fill="x")

        ctk.CTkLabel(input_frame, text="Magic Number").pack(padx=20, pady=(10, 5), anchor="w")
        self.magic_number_entry = ctk.CTkEntry(input_frame, placeholder_text="e.g., 60")
        self.magic_number_entry.insert(0, "60")
        self.magic_number_entry.pack(padx=20, fill="x")

        ctk.CTkLabel(input_frame, text="Number of Particles").pack(padx=20, pady=(10, 5), anchor="w")
        self.particles_entry = ctk.CTkEntry(input_frame, placeholder_text="e.g., 200")
        self.particles_entry.insert(0, "200")
        self.particles_entry.pack(padx=20, fill="x")

        ctk.CTkLabel(input_frame, text="Cost Function").pack(padx=20, pady=(10, 5), anchor="w")
        self.cost_function_combo = ctk.CTkComboBox(input_frame, values=list(COST_FUNCTIONS.keys()))
        self.cost_function_combo.set('magic_coverage_duplicate_cost')
        self.cost_function_combo.pack(padx=20, fill="x")

        self.run_button = ctk.CTkButton(input_frame, text="Run Optimization", command=self.start_optimization_thread)
        self.run_button.pack(pady=30, padx=20)
        
        self.theme_switch = ctk.CTkSwitch(input_frame, text="Dark Mode", command=self.toggle_theme)
        self.theme_switch.pack(side="bottom", pady=20)
        self.theme_switch.select()

    def create_output_frame(self):
        """Creates the right frame for displaying progress and results."""
        output_frame = ctk.CTkFrame(self, corner_radius=10)
        output_frame.grid(row=0, column=1, padx=(0, 10), pady=10, sticky="nsew")
        output_frame.grid_rowconfigure(1, weight=1)
        output_frame.grid_rowconfigure(3, weight=1)
        output_frame.grid_columnconfigure(0, weight=1)

        progress_header = ctk.CTkLabel(output_frame, text="Progress & Output Log", font=ctk.CTkFont(size=16, weight="bold"))
        progress_header.grid(row=0, column=0, padx=15, pady=(15, 5), sticky="w")
        
        self.progress_textbox = ctk.CTkTextbox(output_frame, state="disabled", wrap="word") # wrap="word" is better for logs
        self.progress_textbox.grid(row=1, column=0, padx=15, pady=5, sticky="nsew")

        results_header = ctk.CTkLabel(output_frame, text="Final Result", font=ctk.CTkFont(size=16, weight="bold"))
        results_header.grid(row=2, column=0, padx=15, pady=(15, 5), sticky="w")
        
        self.results_textbox = ctk.CTkTextbox(output_frame, state="disabled", wrap="word", font=("Courier New", 12))
        self.results_textbox.grid(row=3, column=0, padx=15, pady=(5, 15), sticky="nsew")

    def toggle_theme(self):
        ctk.set_appearance_mode("dark" if self.theme_switch.get() == 1 else "light")

    def log_message(self, message, target_box):
        """Helper function to append text to a textbox."""
        target_box.configure(state="normal")
        if message == "CLEAR_BOX":
            target_box.delete("1.0", "end")
        else:
            target_box.insert("end", message)
        target_box.configure(state="disabled")
        target_box.see("end") # Autoscroll

    def process_log_queue(self):
        """Checks the queue for new messages and routes them to the correct textbox."""
        try:
            while True:
                message, target_widget = self.log_queue.get_nowait()
                self.log_message(message, target_widget)
        except queue.Empty:
            pass # No more messages
        finally:
            # Reschedule itself to run again
            self.after(100, self.process_log_queue)

    def start_optimization_thread(self):
        """Clears the GUI and starts the solver in a new thread."""
        self.log_queue.put(("CLEAR_BOX", self.progress_textbox))
        self.log_queue.put(("CLEAR_BOX", self.results_textbox))
        self.log_queue.put(("Starting optimization...\n", self.progress_textbox))
        self.log_queue.put(("Running...", self.results_textbox))
        
        self.run_button.configure(state="disabled", text="Running...")
        
        try:
            thread = Thread(target=self.optimization_worker, daemon=True)
            thread.start()
        except (ValueError, TypeError) as e:
            self.log_queue.put(("CLEAR_BOX", self.results_textbox))
            self.log_queue.put((f"Error: Invalid input values provided.\n{e}", self.results_textbox))
            self.run_button.configure(state="normal", text="Run Optimization")

    def optimization_worker(self):
        """The actual work of running the DPSO solver in the background."""
        
        # --- Redirect stdout to our live logger ---
        gui_logger = QueueLogger(self.log_queue, self.progress_textbox)
        original_stdout = sys.stdout
        sys.stdout = gui_logger

        results_summary = "Optimization failed to produce a valid result."
        try:
            # --- 1. Get parameters from GUI ---
            dpso_n_particles = int(self.particles_entry.get())
            magic_number = int(self.magic_number_entry.get())
            wish_numbers = [int(n.strip()) for n in self.wish_numbers_entry.get().split(',')]
            cost_func = COST_FUNCTIONS[self.cost_function_combo.get()]
            
            # --- 2. Hardcoded parameters ---
            dpso_max_iterations = 1000; dpso_w = 0.8; dpso_c1 = 2.0; dpso_c2 = 2.0
            dpso_max_velocity = 10.0; dpso_consider_combined_steps = True
            dpso_num_top_steps_to_combine = 10; convergence_threshold_iterations = 200
            convergence_pos_tolerance = 1e-6

            # --- 3. Create and Run the Swarm (all print()s are now redirected) ---
            swarm = DPSO_Swarm(n_particles=dpso_n_particles, magic_number=magic_number,
                               wish_numbers=wish_numbers, w=dpso_w, c1=dpso_c1, c2=dpso_c2,
                               max_velocity=dpso_max_velocity, cost_function=cost_func,
                               consider_combined_steps=dpso_consider_combined_steps,
                               num_top_steps_to_combine=dpso_num_top_steps_to_combine)
            
            gbest_pos, gbest_cost, _ = swarm.optimize(
                max_iterations=dpso_max_iterations,
                convergence_threshold_iterations=convergence_threshold_iterations,
                convergence_pos_tolerance=convergence_pos_tolerance
            )

            # --- 4. Process results ---
            if gbest_pos is not None:
                is_valid = check_coefficients(gbest_pos, magic_number)
                if is_valid:
                    square = calc_magic_square(gbest_pos)
                    magic_sum = check_magic_square(square)
                    flat_square = square.flatten()
                    coverage_count = sum(1 for n in wish_numbers if n in flat_square)
                    coverage_percent = (coverage_count / len(wish_numbers)) * 100
                    not_included = [n for n in wish_numbers if n not in flat_square]
                    new_numbers = [int(n) for n in flat_square if n not in wish_numbers]
                    results_summary = (
                        f"Magic Square (Sum: {magic_sum})\n-----------------------------\n{square}\n\n"
                        f"Coverage: {coverage_count}/{len(wish_numbers)} ({coverage_percent:.2f}%)\n"
                        f"  - Missing: {not_included}\n  - New Numbers: {new_numbers}\n\n"
                        f"Final Cost: {gbest_cost:.6f}"
                    )
                else:
                    results_summary = (
                        f"Warning: Solver returned an invalid coefficient vector.\n\n"
                        f"This can happen if the optimization stops without finding a perfect solution.\n\n"
                        f"Final Cost of Invalid Vector: {gbest_cost:.6f}\nVector: {gbest_pos}"
                    )
            else:
                results_summary = "Optimization failed or was aborted. No solution was found."

        except Exception as e:
            # Also print exceptions to the log
            print(f"\nAN UNEXPECTED ERROR OCCURRED: {e}")
            results_summary = f"An unexpected error occurred during optimization:\n\n{e}"
        finally:
            # --- Restore original stdout and update GUI ---
            sys.stdout = original_stdout
            self.log_queue.put(("CLEAR_BOX", self.results_textbox))
            self.log_queue.put((results_summary, self.results_textbox))
            self.run_button.configure(state="normal", text="Run Optimization")

if __name__ == "__main__":
    ctk.set_appearance_mode("dark")
    app = App()
    app.mainloop()
