"""
Sudoku Simulated Annealing Solver
Processes CSV datasets with parallel processing
"""

import csv
import os
import time
import statistics
from collections import defaultdict
from datetime import datetime
from multiprocessing import Pool, cpu_count

from sudoku_simulated_annealing import solve_sudoku_simulated_annealing


def parse_sudoku(puzzle_string):
    """
    Parse a Sudoku puzzle string into a 9x9 grid.
    
    Args:
        puzzle_string: String of 81 characters where '.' represents empty cells
                      and digits 1-9 represent given values
    
    Returns:
        List of lists representing the 9x9 grid
    """
    if len(puzzle_string) != 81:
        raise ValueError("Puzzle string must be exactly 81 characters")
    
    grid = []
    for i in range(9):
        row = []
        for j in range(9):
            char = puzzle_string[i * 9 + j]
            row.append(char if char != '.' else None)
        grid.append(row)
    return grid


def solution_to_string(solution):
    """
    Convert a solution dictionary to a string representation.
    
    Args:
        solution: Dictionary mapping (row, col) to value
    
    Returns:
        String of 81 characters representing the solved puzzle
    """
    if solution is None:
        return None
    
    result = [''] * 81
    for (row, col), value in solution.items():
        result[row * 9 + col] = str(value)
    return ''.join(result)


def process_chunk(args):
    """
    Process a chunk of puzzles in parallel. Each worker processes independently
    and writes to its own temp file to avoid race conditions.
    
    Args:
        args: Tuple of (chunk_id, chunk_rows, temp_dir)
            chunk_id: Unique identifier for this chunk
            chunk_rows: List of puzzle rows to process
            temp_dir: Directory for temporary output files
    
    Returns:
        Tuple of (temp_file_path, stats_dict)
    """
    chunk_id, chunk_rows, temp_dir = args
    
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
    
    with open(temp_file, 'w', newline='') as f:
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
                start_time = time.perf_counter()
                solution = solve_sudoku_simulated_annealing(
                    puzzle,
                    initial_temp=10.0,
                    cooling_rate=0.999,
                    max_iterations=100000,
                    min_temp=0.01
                )
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
                if solution:
                    stats['solved'] += 1
                    solution_str = solution_to_string(solution)
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


def solve_dataset(csv_path, output_path=None, max_puzzles=None, num_workers=None):
    """
    Solve all puzzles from the CSV dataset using simulated annealing.
    Uses parallel processing to speed up computation on multi-core systems.
    
    Args:
        csv_path: Path to the CSV file containing puzzles
        output_path: Path to output CSV file (default: adds '_sa_results' to input filename)
        max_puzzles: Maximum number of puzzles to solve (None for all)
        num_workers: Number of parallel workers (None for auto-detection)
    
    Returns:
        Statistics dictionary with success count, failure count, etc.
    """
    # Generate output filename if not provided
    if output_path is None:
        # Create results subfolder
        results_dir = 'results'
        os.makedirs(results_dir, exist_ok=True)
        
        # Generate timestamp string
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Extract base filename from input path
        if csv_path.endswith('.csv'):
            base_name = os.path.basename(csv_path)[:-4]  # Remove .csv extension
        else:
            base_name = os.path.basename(csv_path)
        
        # Create output path with method and timestamp
        output_path = os.path.join(results_dir, f'{base_name}_sa_results_{timestamp}.csv')
    else:
        # Ensure output directory exists even for custom paths
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
    
    # Read all rows from input CSV
    print(f"Reading input CSV (using Simulated Annealing method)...")
    all_rows = []
    with open(csv_path, 'r') as input_file:
        reader = csv.DictReader(input_file)
        for row in reader:
            all_rows.append(row)
            if max_puzzles and len(all_rows) >= max_puzzles:
                break
    
    total_puzzles = len(all_rows)
    print(f"Loaded {total_puzzles} puzzles")
    
    # Determine number of workers
    if num_workers is None:
        num_workers = cpu_count()
        print(f"Auto-detected {num_workers} CPU cores")
    
    # Create temp directory for chunk files
    temp_dir = os.path.join(os.path.dirname(output_path) or '.', f'temp_sa_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    os.makedirs(temp_dir, exist_ok=True)
    
    # Split rows into chunks
    chunk_size = max(1, total_puzzles // num_workers)
    chunks = []
    for i in range(0, total_puzzles, chunk_size):
        chunk_rows = all_rows[i:i + chunk_size]
        chunks.append((i // chunk_size, chunk_rows, temp_dir))
    
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
    
    with open(output_path, 'w', newline='') as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        
        # Read and write each temp file (skip header after first)
        for temp_file in sorted(temp_files):  # Sort to maintain order
            with open(temp_file, 'r') as temp_f:
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
    return stats, output_path


def get_difficulty_category(difficulty_str):
    """
    Map difficulty value to category.
    Easy = 1, Medium = 2-3, Hard = 4-5
    
    Args:
        difficulty_str: String representation of difficulty value
    
    Returns:
        'Easy', 'Medium', 'Hard', or None if invalid
    """
    try:
        diff = float(difficulty_str)
        if 0.0 <= diff < 1.0:
            return 'Easy'
        elif 1.0 <= diff < 3.0:
            return 'Medium'
        elif 3.0 <= diff:
            return 'Hard'
    except (ValueError, TypeError):
        pass
    return None


def process_metrics(results_csv_path):
    """
    Process the results CSV and compute timing statistics by difficulty.
    
    Args:
        results_csv_path: Path to the results CSV file
    
    Returns:
        Dictionary with timing statistics grouped by difficulty category
    """
    # Group solve times by difficulty category
    difficulty_times = {
        'Easy': [],
        'Medium': [],
        'Hard': []
    }
    
    with open(results_csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            computed_solution = row.get('computed_solution', '')
            difficulty = row.get('difficulty', '')
            solve_time_str = row.get('solve_time', '')
            
            # Only process successfully solved puzzles
            if computed_solution and solve_time_str and not computed_solution.startswith('ERROR'):
                try:
                    solve_time = float(solve_time_str)
                    category = get_difficulty_category(difficulty)
                    if category and category in difficulty_times:
                        difficulty_times[category].append(solve_time)
                except ValueError:
                    pass
    
    # Compute statistics for each difficulty
    metrics = {}
    for category in ['Easy', 'Medium', 'Hard']:
        times = difficulty_times[category]
        if times:
            metrics[category] = {
                'Minimum': min(times),
                'Median': statistics.median(times),
                'Mean': statistics.mean(times),
                'Maximum': max(times),
                'count': len(times)
            }
        else:
            metrics[category] = {
                'Minimum': 0.0,
                'Median': 0.0,
                'Mean': 0.0,
                'Maximum': 0.0,
                'count': 0
            }
    
    return metrics


def print_metrics_table(metrics):
    """
    Print CPU Time statistics in table format matching the reference format.
    
    Args:
        metrics: Dictionary with timing statistics from process_metrics()
    """
    print("\n" + "="*60)
    print("CPU Time (sec)")
    print("="*60)
    print(f"{'':<20} {'Simulated Annealing':<15}")
    print("-"*60)
    
    # Print statistics for each difficulty level
    for category in ['Easy', 'Medium', 'Hard']:
        stats = metrics[category]
        
        # Print difficulty header
        print(f"{category}:")
        
        # Print each metric
        print(f"  {'Minimum:':<18} {stats['Minimum']:<15.6f}")
        print(f"  {'Median:':<18} {stats['Median']:<15.6f}")
        print(f"  {'Mean:':<18} {stats['Mean']:<15.6f}")
        print(f"  {'Maximum:':<18} {stats['Maximum']:<15.6f}")
        
        if category != 'Hard':  # Don't print separator after last category
            print("-"*60)
    
    print("="*60)
    print(f"\nNote: Statistics based on {sum(metrics[c]['count'] for c in metrics)} successfully solved puzzles")
    for category in ['Easy', 'Medium', 'Hard']:
        count = metrics[category]['count']
        if count > 0:
            print(f"  {category}: {count} puzzles")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Solve Sudoku puzzles from CSV dataset using Simulated Annealing')
    parser.add_argument('--csv', required=True,
                       help='Path to CSV file')
    parser.add_argument('--output', default=None,
                       help='Path to output CSV file (default: input_filename_sa_results_timestamp.csv)')
    parser.add_argument('--max', type=int, default=None,
                       help='Maximum number of puzzles to solve (default: all)')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of parallel workers (default: auto-detect CPU count)')
    
    args = parser.parse_args()
    
    # Process all puzzles from dataset
    print(f"Solving puzzles from {args.csv}...")
    print(f"Using Simulated Annealing method")
    if args.max:
        print(f"Limiting to first {args.max} puzzles")
    if args.workers:
        print(f"Using {args.workers} workers")
    print()
    
    stats, results_path = solve_dataset(
        args.csv, 
        output_path=args.output, 
        max_puzzles=args.max, 
        num_workers=args.workers
    )
    
    print("\n" + "="*50)
    print("Final Statistics:")
    print("="*50)
    print(f"Total puzzles processed: {stats['total']}")
    print(f"Successfully solved: {stats['solved']}")
    print(f"Failed to solve: {stats['failed']}")
    print(f"Correct solutions: {stats['correct']}")
    print(f"Incorrect solutions: {stats['incorrect']}")
    if stats['errors'] > 0:
        print(f"Errors encountered: {stats['errors']} (see computed_solution column for details)")
    if stats['total'] > 0:
        success_rate = (stats['solved'] / stats['total']) * 100
        print(f"Success rate: {success_rate:.2f}%")
    
    # Process metrics and print table
    print("\nProcessing metrics...")
    metrics = process_metrics(results_path)
    print_metrics_table(metrics)

