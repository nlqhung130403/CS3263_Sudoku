"""
Sudoku RL Solver using Q-Learning
Processes CSV datasets with parallel processing
"""

import csv
import os
import time
import statistics
from collections import defaultdict
from datetime import datetime
from multiprocessing import Pool, cpu_count, Manager, Lock

from sudoku_qlearning_shared import solve_sudoku_with_shared_qlearning
from sudoku_qlearning_inference import solve_sudoku_with_inference


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


def train_chunk(args):
    """
    Train on a chunk of puzzles in parallel. All workers share the same Q-table
    for knowledge transfer, using locks to prevent race conditions.
    
    Args:
        args: Tuple of (chunk_id, chunk_rows, shared_Q, shared_Nsa, Q_lock)
            chunk_id: Unique identifier for this chunk
            chunk_rows: List of puzzle rows to train on
            shared_Q: Shared dictionary for Q-values (from multiprocessing.Manager)
            shared_Nsa: Shared dictionary for visit counts
            Q_lock: Lock for synchronizing Q-table updates
    
    Returns:
        Tuple of (chunk_id, stats_dict)
    """
    chunk_id, chunk_rows, shared_Q, shared_Nsa, Q_lock = args
    
    stats = {
        'total': 0,
        'solved': 0,
        'failed': 0
    }
    
    for row in chunk_rows:
        stats['total'] += 1
        
        try:
            puzzle = row.get('puzzle', '')
            
            # Train on this puzzle (updates Q-table)
            solution = solve_sudoku_with_shared_qlearning(
                puzzle,
                shared_Q,      # Shared Q-table across all workers
                shared_Nsa,    # Shared visit counts
                Q_lock,        # Lock for synchronization
                max_episodes=500,  # Training episodes
                max_steps_per_episode=200
            )
            
            if solution:
                stats['solved'] += 1
            else:
                stats['failed'] += 1
                
        except Exception as e:
            stats['failed'] += 1
    
    return chunk_id, stats


def process_chunk_inference(args):
    """
    Process a chunk of puzzles for inference in parallel. Uses trained Q-table
    WITHOUT updating it. Each puzzle's solve time is measured independently.
    
    Args:
        args: Tuple of (chunk_id, chunk_rows, temp_dir, shared_Q, shared_Nsa, Q_lock)
            chunk_id: Unique identifier for this chunk
            chunk_rows: List of puzzle rows to process
            temp_dir: Directory for temporary output files
            shared_Q: Shared dictionary for Q-values (trained, read-only)
            shared_Nsa: Shared dictionary for visit counts (read-only)
            Q_lock: Lock for synchronizing Q-table reads
    
    Returns:
        Tuple of (temp_file_path, stats_dict)
    """
    chunk_id, chunk_rows, temp_dir, shared_Q, shared_Nsa, Q_lock = args
    
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
                
                # Measure solve time independently for each puzzle (inference only)
                start_time = time.perf_counter()
                solution = solve_sudoku_with_inference(
                    puzzle,
                    shared_Q,      # Trained Q-table (read-only)
                    shared_Nsa,    # Visit counts (read-only)
                    Q_lock,        # Lock for read synchronization
                    max_episodes=500,  # Inference episodes
                    max_steps_per_episode=200,
                    epsilon=0.0    # Pure exploitation (no exploration during inference)
                )
                solve_time = time.perf_counter() - start_time
                
                # Ensure non-negative
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


def train_dataset(train_csv_path, max_puzzles=None, num_workers=None):
    """
    Train Q-learning model on training dataset.
    
    Args:
        train_csv_path: Path to the training CSV file
        max_puzzles: Maximum number of puzzles to train on (None for all)
        num_workers: Number of parallel workers (None for auto-detection)
    
    Returns:
        Tuple of (shared_Q, shared_Nsa, Q_lock, stats)
    """
    print("="*60)
    print("PHASE 1: TRAINING")
    print("="*60)
    
    # Read training puzzles
    print(f"Reading training CSV: {train_csv_path}")
    all_rows = []
    with open(train_csv_path, 'r') as input_file:
        reader = csv.DictReader(input_file)
        for row in reader:
            all_rows.append(row)
            if max_puzzles and len(all_rows) >= max_puzzles:
                break
    
    total_puzzles = len(all_rows)
    print(f"Loaded {total_puzzles} training puzzles")
    
    # Determine number of workers
    if num_workers is None:
        num_workers = cpu_count()
        print(f"Auto-detected {num_workers} CPU cores")
    
    # Create shared Q-table and visit counts
    print("Creating shared Q-table for training...")
    manager = Manager()
    shared_Q = manager.dict()
    shared_Nsa = manager.dict()
    Q_lock = manager.Lock()
    
    # Split rows into chunks
    chunk_size = max(1, total_puzzles // num_workers)
    chunks = []
    for i in range(0, total_puzzles, chunk_size):
        chunk_rows = all_rows[i:i + chunk_size]
        chunks.append((i // chunk_size, chunk_rows, shared_Q, shared_Nsa, Q_lock))
    
    print(f"Training on {len(chunks)} chunks with {num_workers} workers...")
    
    # Train chunks in parallel
    stats = {
        'total': 0,
        'solved': 0,
        'failed': 0
    }
    
    with Pool(processes=num_workers) as pool:
        results = pool.map(train_chunk, chunks)
    
    # Aggregate stats
    for chunk_id, chunk_stats in results:
        stats['total'] += chunk_stats['total']
        stats['solved'] += chunk_stats['solved']
        stats['failed'] += chunk_stats['failed']
    
    print(f"\nTraining completed:")
    print(f"  Total puzzles trained: {stats['total']}")
    print(f"  Successfully solved during training: {stats['solved']}")
    print(f"  Failed during training: {stats['failed']}")
    print(f"  Q-table size: {len(shared_Q)} state-action pairs")
    print(f"  Total visits: {sum(shared_Nsa.values())}")
    
    return shared_Q, shared_Nsa, Q_lock, stats


def solve_dataset(test_csv_path, shared_Q, shared_Nsa, Q_lock, output_path=None, max_puzzles=None, num_workers=None):
    """
    Solve all puzzles from the test CSV dataset using trained Q-table (inference only).
    Each puzzle's solve time is measured independently.
    Uses parallel processing to speed up computation on multi-core systems.
    
    Args:
        test_csv_path: Path to the test CSV file containing puzzles
        shared_Q: Trained Q-table (read-only during inference)
        shared_Nsa: Visit counts (read-only during inference)
        Q_lock: Lock for synchronizing Q-table reads
        output_path: Path to output CSV file (default: adds '_rl_results' to input filename)
        max_puzzles: Maximum number of puzzles to solve (None for all)
        num_workers: Number of parallel workers (None for auto-detection)
    
    Returns:
        Statistics dictionary with success count, failure count, etc.
    """
    print("\n" + "="*60)
    print("PHASE 2: INFERENCE")
    print("="*60)
    
    # Generate output filename if not provided
    if output_path is None:
        # Create results subfolder
        results_dir = 'results'
        os.makedirs(results_dir, exist_ok=True)
        
        # Generate timestamp string
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Extract base filename from input path
        if test_csv_path.endswith('.csv'):
            base_name = os.path.basename(test_csv_path)[:-4]  # Remove .csv extension
        else:
            base_name = os.path.basename(test_csv_path)
        
        # Create output path with method and timestamp
        output_path = os.path.join(results_dir, f'{base_name}_rl_results_{timestamp}.csv')
    else:
        # Ensure output directory exists even for custom paths
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
    
    # Read all rows from test CSV
    print(f"Reading test CSV: {test_csv_path}")
    all_rows = []
    with open(test_csv_path, 'r') as input_file:
        reader = csv.DictReader(input_file)
        for row in reader:
            all_rows.append(row)
            if max_puzzles and len(all_rows) >= max_puzzles:
                break
    
    total_puzzles = len(all_rows)
    print(f"Loaded {total_puzzles} test puzzles")
    print(f"Using trained Q-table with {len(shared_Q)} state-action pairs")
    
    # Determine number of workers
    if num_workers is None:
        num_workers = cpu_count()
        print(f"Auto-detected {num_workers} CPU cores")
    
    # Create temp directory for chunk files
    temp_dir = os.path.join(os.path.dirname(output_path) or '.', f'temp_rl_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    os.makedirs(temp_dir, exist_ok=True)
    
    # Split rows into chunks
    chunk_size = max(1, total_puzzles // num_workers)
    chunks = []
    for i in range(0, total_puzzles, chunk_size):
        chunk_rows = all_rows[i:i + chunk_size]
        # Pass trained Q-table (read-only) to each worker
        chunks.append((i // chunk_size, chunk_rows, temp_dir, shared_Q, shared_Nsa, Q_lock))
    
    print(f"Processing {len(chunks)} chunks with {num_workers} workers (inference only)...")
    
    # Process chunks in parallel (inference only, no Q-table updates)
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
        results = pool.map(process_chunk_inference, chunks)
    
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
    
    print(f"\nInference results saved to: {output_path}")
    
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
    print(f"{'':<20} {'Q-Learning':<15}")
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
    
    parser = argparse.ArgumentParser(description='Train Q-Learning model on training set, then solve test set')
    parser.add_argument('--train', default='../test_set/sudoku_train_set_random_10k.csv',
                       help='Path to training CSV file (default: ../test_set/sudoku_train_set_random_10k.csv)')
    parser.add_argument('--test', default='../test_set/sudoku_test_set_random_10k.csv',
                       help='Path to test CSV file (default: ../test_set/sudoku_test_set_random_10k.csv)')
    parser.add_argument('--output', default=None,
                       help='Path to output CSV file (default: test_filename_rl_results_timestamp.csv)')
    parser.add_argument('--max-train', type=int, default=None,
                       help='Maximum number of puzzles to train on (default: all)')
    parser.add_argument('--max-test', type=int, default=None,
                       help='Maximum number of puzzles to test (default: all)')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of parallel workers (default: auto-detect CPU count)')
    
    args = parser.parse_args()
    
    # Phase 1: Train on training dataset
    shared_Q, shared_Nsa, Q_lock, train_stats = train_dataset(
        args.train,
        max_puzzles=args.max_train,
        num_workers=args.workers
    )
    
    # Phase 2: Inference on test dataset
    stats, results_path = solve_dataset(
        args.test,
        shared_Q,      # Trained Q-table
        shared_Nsa,    # Visit counts
        Q_lock,        # Lock for reads
        output_path=args.output,
        max_puzzles=args.max_test,
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

