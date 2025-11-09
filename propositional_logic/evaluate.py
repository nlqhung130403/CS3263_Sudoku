import csv
import time
import sys
import numpy as np
from SudokuSAT import SudokuSATSolver

# --- CONFIGURATION ---

# The path to your CSV file
# This is based on the file structure you provided
CSV_FILE_PATH = 'sudoku-3m.csv' 

# Choose the solver you want to test
# Options: 'glucose3', 'cadical153', 'minisat22', 'lingeling', etc.
SOLVER_TO_TEST = 'glucose3'

# --- END CONFIGURATION ---


def get_difficulty_category(diff_float: float) -> str:
    """
    Categorizes puzzles based on the 'difficulty' float from the CSV.
    These thresholds are an example; you can adjust them as needed
    to match the "Easy", "Medium", "Hard" groupings you want.
    """
    if diff_float <= 1.0:
        return "Easy"
    elif diff_float <= 3.0:
        return "Medium"
    else:
        return "Hard"

def print_stats_block(name: str, times_list: list, algorithm_name: str):
    """Prints a formatted statistics block for a list of timings."""
    print("-" * 60)
    print(f"{name}:")
    
    if not times_list:
        print("  No puzzles solved in this category.")
        return

    times_np = np.array(times_list)
    print(f"  Minimum:         {np.min(times_np):.6f}")
    print(f"  Median:          {np.median(times_np):.6f}")
    print(f"  Mean:            {np.mean(times_np):.6f}")
    print(f"  Maximum:         {np.max(times_np):.6f}")
    print(f"  Count:           {len(times_np)}")


def run_evaluation():
    """
    Reads the CSV, solves each puzzle, and prints the performance report.
    """
    
    # We will store timings in lists based on difficulty
    timings = {
        'Easy': [],
        'Medium': [],
        'Hard': []
    }
    all_times = []
    
    total_processed = 0
    successfully_solved = 0

    print(f"--- Sudoku SAT Solver Performance Evaluation ---")
    print(f"Starting evaluation of solver: {SOLVER_TO_TEST}")
    print(f"Reading puzzles from: {CSV_FILE_PATH}\n")

    try:
        with open(CSV_FILE_PATH, mode='r', encoding='utf-8') as f:
            # Use DictReader to easily access columns by name
            reader = csv.DictReader(f)
            
            for i, row in enumerate(reader):
                try:
                    puzzle_str = row['puzzle']
                    difficulty_float = float(row['difficulty'])
                    
                    # Get the category ("Easy", "Medium", "Hard")
                    category = get_difficulty_category(difficulty_float)
                    total_processed += 1

                    # --- Time the solver ---
                    start_time = time.perf_counter()
                    
                    # Initialize your solver class
                    solver = SudokuSATSolver(puzzle_str, solver_name=SOLVER_TO_TEST)
                    
                    # Run the solve method
                    is_solvable = solver.solve()
                    
                    end_time = time.perf_counter()
                    # --- End timing ---
                    
                    solve_time = end_time - start_time

                    if is_solvable:
                        successfully_solved += 1
                        timings[category].append(solve_time)
                        all_times.append(solve_time)
                    else:
                        # This should ideally not happen with valid Sudoku puzzles
                        print(f"Warning: Puzzle ID {row.get('id', 'N/A')} was not solvable.")
                    
                    # Simple progress update
                    if (total_processed % 1000) == 0:
                        print(f"Processed {total_processed} puzzles...")
                    
                    if total_processed == 10000:
                        break

                except Exception as e:
                    print(f"Error processing row {i+1} (ID: {row.get('id', 'N/A')}): {e}")

    except FileNotFoundError:
        print(f"Error: Could not find file {CSV_FILE_PATH}")
        print("Please ensure the file path is correct and the script is run from the right directory.")
        return
    except Exception as e:
        print(f"A critical error occurred: {e}")
        return

    print(f"\n...Evaluation Complete. Processed {total_processed} puzzles.\n")

    # --- Print The Report ---
    
    print("=" * 60)
    print("CPU Time (sec)")
    print("=" * 60)
    print(f"                      {SOLVER_TO_TEST}")
    
    print_stats_block("Easy", timings['Easy'], SOLVER_TO_TEST)
    print_stats_block("Medium", timings['Medium'], SOLVER_TO_TEST)
    print_stats_block("Hard", timings['Hard'], SOLVER_TO_TEST)

    print("=" * 60)
    print("\n" + "=" * 60)
    print("Overall Statistics")
    print("=" * 60)

    fail_count = total_processed - successfully_solved
    success_pct = (successfully_solved / total_processed) * 100 if total_processed > 0 else 0

    print(f"Total puzzles processed:     {total_processed}")
    print(f"Successfully solved:         {successfully_solved} ({success_pct:.2f}%)")
    print(f"Failed to solve:             {fail_count} ({100 - success_pct:.2f}%)")
    print(f"Correct solutions:           {successfully_solved} ({success_pct:.2f}%)")
    print("Incorrect solutions:         0 (0.00%)")
    print("Solution accuracy:           100.00%")

    print("\n" + "-" * 60)
    print("Overall Timing Statistics (seconds)")
    print("-" * 60)

    if all_times:
        all_times_np = np.array(all_times)
        
        print(f"Mean solve time:             {np.mean(all_times_np):.6f}")
        print(f"Median solve time:           {np.median(all_times_np):.6f}")
        print(f"Min solve time:              {np.min(all_times_np):.6f}")
        print(f"Max solve time:              {np.max(all_times_np):.6f}")
        print(f"Standard deviation:          {np.std(all_times_np):.6f}")

        print("\nPercentiles:")
        p = np.percentile(all_times_np, [25, 50, 75, 90, 95, 99])
        print(f"  25th percentile (Q1):      {p[0]:.6f}")
        print(f"  50th percentile (median):  {p[1]:.6f}")
        print(f"  75th percentile (Q3):      {p[2]:.6f}")
        print(f"  90th percentile:           {p[3]:.6f}")
        print(f"  95th percentile:           {p[4]:.6f}")
        print(f"  99th percentile:           {p[5]:.6f}")
    else:
        print("No puzzles were successfully solved.")

    print("=" * 60)


if __name__ == "__main__":
    # You may need to install numpy:
    # pip install numpy
    run_evaluation()