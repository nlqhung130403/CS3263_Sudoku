import csv
import time
import sys
import os
import numpy as np
from datetime import datetime
from multiprocessing import Pool, cpu_count
from SudokuSAT import SudokuSATSolver

# --- CONFIGURATION ---

# The path to your CSV file (taken from first CLI argument)
if len(sys.argv) < 2:
    print("Error: Please provide the CSV file path as the first argument.")
    print("Usage: python evaluate.py <path_to_csv_file>")
    sys.exit(1)

CSV_FILE_PATH = sys.argv[1] 

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


def solution_grid_to_string(solution_grid):
    """
    Convert a solution grid (2D list) to a string representation.
    
    Args:
        solution_grid: 2D list of integers representing the solved puzzle
    
    Returns:
        String of 81 characters representing the solved puzzle
    """
    if solution_grid is None:
        return ''
    
    result = []
    for row in solution_grid:
        for val in row:
            # Convert integer value to character (1-9 -> '1'-'9')
            result.append(str(val))
    return ''.join(result)


def process_chunk(args):
    """
    Process a chunk of puzzles in parallel. Each worker processes independently
    and writes to its own temp file to avoid race conditions.
    
    Args:
        args: Tuple of (chunk_id, chunk_rows, temp_dir, solver_name)
            chunk_id: Unique identifier for this chunk
            chunk_rows: List of puzzle rows to process
            temp_dir: Directory for temporary output files
            solver_name: Name of the SAT solver to use
    
    Returns:
        Tuple of (temp_file_path, stats_dict)
    """
    chunk_id, chunk_rows, temp_dir, solver_name = args
    
    # Create temp file for this chunk
    temp_file = os.path.join(temp_dir, f'chunk_{chunk_id}.csv')
    
    stats = {
        'total': 0,
        'solved': 0,
        'failed': 0,
        'correct': 0,
        'incorrect': 0,
        'errors': 0
    }
    
    fieldnames = ['id', 'puzzle', 'solution', 'clues', 'difficulty', 'solve_time', 'computed_solution']
    
    with open(temp_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for row in chunk_rows:
            stats['total'] += 1
            
            try:
                puzzle_id = row.get('id', '')
                puzzle = row.get('puzzle', '')
                expected_solution = row.get('solution', '')
                clues = row.get('clues', '')
                difficulty = row.get('difficulty', '')
                
                # Measure solve time independently in this process
                # Use perf_counter() instead of time() to avoid issues with system clock adjustments
                # perf_counter() is monotonic and not affected by system clock changes
                start_time = time.perf_counter()
                
                # Initialize solver and solve
                solver = SudokuSATSolver(puzzle, solver_name=solver_name)
                is_solvable = solver.solve()
                
                solve_time = time.perf_counter() - start_time
                
                # Ensure non-negative (shouldn't happen with perf_counter, but safety check)
                if solve_time < 0:
                    solve_time = 0.0
                
                # Prepare output row
                output_row = {
                    'id': puzzle_id,
                    'puzzle': puzzle,
                    'solution': expected_solution,
                    'clues': clues,
                    'difficulty': difficulty,
                    'solve_time': f"{solve_time:.6f}",
                    'computed_solution': ''
                }
                
                # Validate solution AFTER timing
                if is_solvable and solver.solution is not None:
                    stats['solved'] += 1
                    solution_str = solution_grid_to_string(solver.solution)
                    output_row['computed_solution'] = solution_str
                    
                    if solution_str == expected_solution:
                        stats['correct'] += 1
                    else:
                        stats['incorrect'] += 1
                else:
                    stats['failed'] += 1
                
                writer.writerow(output_row)
                
            except Exception as e:
                # Catch any error processing this row and continue with next row
                stats['errors'] += 1
                # Still write the row to output with error indicator
                try:
                    puzzle_id = row.get('id', '')
                    puzzle = row.get('puzzle', '')
                    expected_solution = row.get('solution', '')
                    clues = row.get('clues', '')
                    difficulty = row.get('difficulty', '')
                except:
                    puzzle_id = ''
                    puzzle = ''
                    expected_solution = ''
                    clues = ''
                    difficulty = ''
                
                output_row = {
                    'id': puzzle_id,
                    'puzzle': puzzle,
                    'solution': expected_solution,
                    'clues': clues,
                    'difficulty': difficulty,
                    'solve_time': '0.000000',
                    'computed_solution': f'ERROR: {str(e)}'
                }
                writer.writerow(output_row)
                # Continue processing next row
    
    return temp_file, stats


def run_evaluation(max_puzzles=None, num_workers=None):
    """
    Reads the CSV, solves each puzzle using parallel processing, saves results to CSV, and prints the performance report.
    Uses parallel processing to speed up computation on multi-core systems.
    
    Args:
        max_puzzles: Maximum number of puzzles to solve (None for all)
        num_workers: Number of parallel workers (None for auto-detection)
    """
    
    # Generate output filename similar to project_csp.py
    # Results directory is at the project root (one level up from propositional_logic/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)  # Go up one level from propositional_logic/
    results_dir = os.path.join(project_root, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Extract base filename from input path
    if CSV_FILE_PATH.endswith('.csv'):
        base_name = os.path.basename(CSV_FILE_PATH)[:-4]  # Remove .csv extension
    else:
        base_name = os.path.basename(CSV_FILE_PATH)
    
    # Create output path with solver name and timestamp
    solver_name_short = SOLVER_TO_TEST.lower()
    output_path = os.path.join(results_dir, f'{base_name}_{solver_name_short}_results_{timestamp}.csv')

    print(f"--- Sudoku SAT Solver Performance Evaluation ---")
    print(f"Starting evaluation of solver: {SOLVER_TO_TEST}")
    print(f"Reading puzzles from: {CSV_FILE_PATH}")
    print(f"Results will be saved to: {output_path}\n")

    # Read all rows from input CSV
    print(f"Reading input CSV...")
    all_rows = []
    try:
        with open(CSV_FILE_PATH, 'r', encoding='utf-8') as input_file:
            reader = csv.DictReader(input_file)
            for row in reader:
                all_rows.append(row)
                if max_puzzles and len(all_rows) >= max_puzzles:
                    break
    except FileNotFoundError:
        print(f"Error: Could not find file {CSV_FILE_PATH}")
        print("Please ensure the file path is correct and the script is run from the right directory.")
        return
    except Exception as e:
        print(f"A critical error occurred reading CSV: {e}")
        return
    
    total_puzzles = len(all_rows)
    print(f"Loaded {total_puzzles} puzzles")
    
    # Determine number of workers
    if num_workers is None:
        num_workers = cpu_count()
        print(f"Auto-detected {num_workers} CPU cores")
    
    # Create temp directory for chunk files
    temp_dir = os.path.join(results_dir, f'temp_{solver_name_short}_{timestamp}')
    os.makedirs(temp_dir, exist_ok=True)
    
    # Split rows into chunks
    chunk_size = max(1, total_puzzles // num_workers)
    chunks = []
    for i in range(0, total_puzzles, chunk_size):
        chunk_rows = all_rows[i:i + chunk_size]
        chunks.append((i // chunk_size, chunk_rows, temp_dir, SOLVER_TO_TEST))
    
    print(f"Processing {len(chunks)} chunks with {num_workers} workers...")
    
    # Process chunks in parallel
    stats = {
        'total': 0,
        'solved': 0,
        'failed': 0,
        'correct': 0,
        'incorrect': 0,
        'errors': 0
    }
    
    temp_files = []
    with Pool(processes=num_workers) as pool:
        results = pool.map(process_chunk, chunks)
    
    # Collect results and aggregate stats
    for temp_file, chunk_stats in results:
        temp_files.append(temp_file)
        stats['total'] += chunk_stats['total']
        stats['solved'] += chunk_stats['solved']
        stats['failed'] += chunk_stats['failed']
        stats['correct'] += chunk_stats['correct']
        stats['incorrect'] += chunk_stats['incorrect']
        stats['errors'] += chunk_stats.get('errors', 0)
    
    # Concatenate all temp files into final output
    print("Concatenating results...")
    fieldnames = ['id', 'puzzle', 'solution', 'clues', 'difficulty', 'solve_time', 'computed_solution']
    
    with open(output_path, 'w', newline='', encoding='utf-8') as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        
        # Read and write each temp file (skip header after first)
        for temp_file in sorted(temp_files):  # Sort to maintain order
            with open(temp_file, 'r', encoding='utf-8') as temp_f:
                reader = csv.DictReader(temp_f)
                for row in reader:
                    writer.writerow(row)
    
    # Clean up temp files and directory
    print("Cleaning up temporary files...")
    for temp_file in temp_files:
        try:
            os.remove(temp_file)
        except OSError:
            pass
    try:
        os.rmdir(temp_dir)
    except OSError:
        pass
    
    print(f"\nResults saved to: {output_path}")

    # Compute statistics from the results CSV
    print("\nComputing statistics...")
    
    # We will store timings in lists based on difficulty
    timings = {
        'Easy': [],
        'Medium': [],
        'Hard': []
    }
    all_times = []
    
    total_processed = stats['total']
    successfully_solved = stats['solved']
    
    # Read results CSV to compute timing statistics
    with open(output_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            computed_solution = row.get('computed_solution', '')
            difficulty = row.get('difficulty', '')
            solve_time_str = row.get('solve_time', '')
            
            # Only process successfully solved puzzles
            if computed_solution and solve_time_str and not computed_solution.startswith('ERROR'):
                try:
                    solve_time = float(solve_time_str)
                    difficulty_float = float(difficulty)
                    category = get_difficulty_category(difficulty_float)
                    if category in timings:
                        timings[category].append(solve_time)
                        all_times.append(solve_time)
                except ValueError:
                    pass

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

    fail_count = stats['failed']
    success_pct = (successfully_solved / total_processed) * 100 if total_processed > 0 else 0
    correct_pct = (stats['correct'] / total_processed) * 100 if total_processed > 0 else 0
    incorrect_pct = (stats['incorrect'] / total_processed) * 100 if total_processed > 0 else 0
    accuracy = (stats['correct'] / successfully_solved) * 100 if successfully_solved > 0 else 0

    print(f"Total puzzles processed:     {total_processed}")
    print(f"Successfully solved:         {successfully_solved} ({success_pct:.2f}%)")
    print(f"Failed to solve:             {fail_count} ({fail_count/total_processed*100:.2f}%)")
    print(f"Correct solutions:          {stats['correct']} ({correct_pct:.2f}%)")
    print(f"Incorrect solutions:         {stats['incorrect']} ({incorrect_pct:.2f}%)")
    print(f"Solution accuracy:          {accuracy:.2f}%")
    if stats['errors'] > 0:
        print(f"Errors encountered:          {stats['errors']}")

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