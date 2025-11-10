"""
Sudoku CSP Solver using Backtracking Search with Generalized Arc Consistency (GAC)
Uses NaryCSP format with all-different constraints for rows, columns, and boxes

"""

import csv
import os
import time
import statistics
from collections import defaultdict
from datetime import datetime
from multiprocessing import Pool, cpu_count
from sortedcontainers import SortedSet

from utils import argmin_random_tie, count, first


# ============================================================================
# Classes and functions copied from CSP/csp.py for NaryCSP support
# ============================================================================

class Constraint:
    """
    A Constraint consists of:
    scope    : a tuple of variables
    condition: a function that can applied to a tuple of values
    for the variables.
    """

    def __init__(self, scope, condition):
        self.scope = scope
        self.condition = condition

    def __repr__(self):
        return self.condition.__name__ + str(self.scope)

    def holds(self, assignment):
        """Returns the value of Constraint con evaluated in assignment.

        precondition: all variables are assigned in assignment
        """
        return self.condition(*tuple(assignment[v] for v in self.scope))


class NaryCSP:
    """
    A nary-CSP consists of:
    domains     : a dictionary that maps each variable to its domain
    constraints : a list of constraints
    variables   : a set of variables
    var_to_const: a variable to set of constraints dictionary
    """

    def __init__(self, domains, constraints):
        """Domains is a variable:domain dictionary
        constraints is a list of constraints
        """
        self.variables = set(domains)
        self.domains = domains
        self.constraints = constraints
        self.var_to_const = {var: set() for var in self.variables}
        for con in constraints:
            for var in con.scope:
                self.var_to_const[var].add(con)

    def __str__(self):
        """String representation of CSP"""
        return str(self.domains)

    def display(self, assignment=None):
        """More detailed string representation of CSP"""
        if assignment is None:
            assignment = {}
        print(assignment)

    def consistent(self, assignment):
        """assignment is a variable:value dictionary
        returns True if all of the constraints that can be evaluated
                        evaluate to True given assignment.
        """
        return all(con.holds(assignment)
                   for con in self.constraints
                   if all(v in assignment for v in con.scope))


def all_diff_constraint(*values):
    """Returns True if all values are different, False otherwise"""
    return len(values) == len(set(values))


def sat_up(to_do):
    """Arc heuristic that prioritizes constraints with fewer variables"""
    return SortedSet(to_do, key=lambda t: 1 / len([var for var in t[1].scope]))


def no_heuristic(to_do):
    """No heuristic - return as-is"""
    return to_do


class ACSolver:
    """Solves a CSP with arc consistency and domain splitting"""

    def __init__(self, csp):
        """a CSP solver that uses arc consistency
        * csp is the CSP to be solved
        """
        self.csp = csp

    def GAC(self, orig_domains=None, to_do=None, arc_heuristic=sat_up):
        """
        Makes this CSP arc-consistent using Generalized Arc Consistency
        orig_domains: is the original domains
        to_do       : is a set of (variable,constraint) pairs
        returns the reduced domains (an arc-consistent variable:domain dictionary)
        """
        if orig_domains is None:
            orig_domains = self.csp.domains
        if to_do is None:
            to_do = {(var, const) for const in self.csp.constraints for var in const.scope}
        else:
            to_do = to_do.copy()
        domains = orig_domains.copy()
        to_do = arc_heuristic(to_do)
        checks = 0
        while to_do:
            var, const = to_do.pop()
            other_vars = [ov for ov in const.scope if ov != var]
            new_domain = set()
            if len(other_vars) == 0:
                for val in domains[var]:
                    if const.holds({var: val}):
                        new_domain.add(val)
                    checks += 1
            elif len(other_vars) == 1:
                other = other_vars[0]
                for val in domains[var]:
                    for other_val in domains[other]:
                        checks += 1
                        if const.holds({var: val, other: other_val}):
                            new_domain.add(val)
                            break
            else:  # general case
                for val in domains[var]:
                    holds, checks = self.any_holds(domains, const, {var: val}, other_vars, checks=checks)
                    if holds:
                        new_domain.add(val)
            if new_domain != domains[var]:
                domains[var] = new_domain
                if not new_domain:
                    return False, domains, checks
                add_to_do = self.new_to_do(var, const).difference(to_do)
                to_do |= add_to_do
        return True, domains, checks

    def new_to_do(self, var, const):
        """
        Returns new elements to be added to to_do after assigning
        variable var in constraint const.
        """
        return {(nvar, nconst) for nconst in self.csp.var_to_const[var]
                if nconst != const
                for nvar in nconst.scope
                if nvar != var}

    def any_holds(self, domains, const, env, other_vars, ind=0, checks=0):
        """
        Returns True if Constraint const holds for an assignment
        that extends env with the variables in other_vars[ind:]
        env is a dictionary
        Warning: this has side effects and changes the elements of env
        """
        if ind == len(other_vars):
            return const.holds(env), checks + 1
        else:
            var = other_vars[ind]
            for val in domains[var]:
                env[var] = val
                holds, checks = self.any_holds(domains, const, env, other_vars, ind + 1, checks)
                if holds:
                    return True, checks
            return False, checks


# ============================================================================
# Sudoku-specific functions (reused from project_csp.py)
# ============================================================================

def parse_sudoku(puzzle_string):
    """
    Parse a Sudoku puzzle string into an N×N grid.
    Grid size is determined dynamically from the puzzle string length.
    
    Args:
        puzzle_string: String of N×N characters where '.' represents empty cells
                      and digits/letters represent given values (1-9, A-P for 16x16, etc.)
    
    Returns:
        Tuple of (grid, grid_size, box_size) where:
        - grid: List of lists representing the N×N grid
        - grid_size: The size N of the grid
        - box_size: The size of each box (sqrt(N))
    """
    puzzle_len = len(puzzle_string)
    grid_size = int(puzzle_len ** 0.5)
    box_size = int(grid_size ** 0.5)
    
    # Validate that grid_size is a perfect square
    if box_size * box_size != grid_size:
        raise ValueError(f"Invalid puzzle length: {puzzle_len}. "
                        f"Length must be N×N where N is a perfect square (e.g., 16, 81, 256).")
    
    grid = []
    for i in range(grid_size):
        row = []
        for j in range(grid_size):
            char = puzzle_string[i * grid_size + j]
            row.append(char if char != '.' else None)
        grid.append(row)
    return grid, grid_size, box_size


def _char_to_val(char):
    """
    Convert a puzzle character to its integer value.
    
    Args:
        char: Character from puzzle string ('.', '1'-'9', 'A'-'Z')
    
    Returns:
        Integer value (1-N) or None if empty
    """
    if char in ('.', '0', '_'):
        return None
    
    val = 0
    if '1' <= char <= '9':
        val = int(char)
    elif 'A' <= char <= 'Z':
        val = ord(char) - ord('A') + 10
    elif 'a' <= char <= 'z':
        val = ord(char) - ord('a') + 10
    else:
        raise ValueError(f"Invalid character in puzzle string: '{char}'")
    
    return val


def _val_to_char(val, grid_size):
    """
    Convert an integer value to its character representation.
    
    Args:
        val: Integer value (1-N)
        grid_size: Size of the grid (N)
    
    Returns:
        Character representation ('1'-'9' for 1-9, 'A'-'P' for 10-25, etc.)
    """
    if val == 0:
        return '.'
    if 1 <= val <= 9:
        return str(val)
    if 10 <= val <= 35:
        return chr(ord('A') + val - 10)
    return '?'


def solution_to_string(solution, grid_size=None):
    """
    Convert a solution dictionary to a string representation.
    
    Args:
        solution: Dictionary mapping (row, col) to value
        grid_size: Size of the grid (N). If None, inferred from solution keys.
    
    Returns:
        String of N×N characters representing the solved puzzle
    """
    if solution is None:
        return None
    
    # Infer grid_size from solution if not provided
    if grid_size is None:
        if not solution:
            return None
        max_row = max(row for row, col in solution.keys())
        max_col = max(col for row, col in solution.keys())
        grid_size = max(max_row, max_col) + 1
    
    result = ['.'] * (grid_size * grid_size)
    for (row, col), value in solution.items():
        # Handle case where value might be a set (from domains)
        if isinstance(value, set):
            if len(value) == 1:
                value = next(iter(value))
            else:
                # Multiple values in set - this shouldn't happen in a complete solution
                raise ValueError(f"Variable ({row}, {col}) has multiple values: {value}")
        
        # Ensure value is an integer
        if not isinstance(value, int):
            raise TypeError(f"Variable ({row}, {col}) has non-integer value: {value} (type: {type(value)})")
        
        result[row * grid_size + col] = _val_to_char(value, grid_size)
    return ''.join(result)


def print_sudoku(puzzle_string):
    """
    Print a Sudoku puzzle in a readable N×N grid format.
    
    Args:
        puzzle_string: String of N×N characters representing the puzzle
    """
    grid, grid_size, box_size = parse_sudoku(puzzle_string)
    
    # Calculate separator width
    separator_width = (2 * grid_size) + (2 * box_size) - 3
    
    print("+" + "-" * separator_width + "+")
    for i in range(grid_size):
        row_str = "| "
        for j in range(grid_size):
            if grid[i][j] is not None:
                row_str += str(grid[i][j]) + " "
            else:
                row_str += ". "
            if (j + 1) % box_size == 0:
                row_str += "| "
        print(row_str)
        if (i + 1) % box_size == 0:
            print("+" + "-" * separator_width + "+")


# ============================================================================
# NaryCSP Sudoku creation and solving
# ============================================================================

def create_sudoku_narycsp(puzzle_string):
    """
    Create a NaryCSP instance for a Sudoku puzzle using all-different constraints.
    Grid size is determined dynamically from the puzzle string length.
    
    Args:
        puzzle_string: String of N×N characters representing the puzzle
    
    Returns:
        Tuple of (NaryCSP instance, grid_size)
    """
    grid, grid_size, box_size = parse_sudoku(puzzle_string)
    
    # Variables: each cell is represented as (row, col) tuple
    variables = [(i, j) for i in range(grid_size) for j in range(grid_size)]
    
    # Domains: each variable can be 1-N, but pre-filled cells have only one value
    domains = {}
    for i in range(grid_size):
        for j in range(grid_size):
            char = grid[i][j]
            if char is not None:
                # Pre-filled cell: domain is just the given value
                val = _char_to_val(char)
                if val is None:
                    raise ValueError(f"Invalid character '{char}' at position ({i}, {j})")
                domains[(i, j)] = {val}  # Use set for NaryCSP
            else:
                # Empty cell: domain is all values 1 to grid_size
                domains[(i, j)] = set(range(1, grid_size + 1))
    
    # Constraints: all-different constraints for rows, columns, and boxes
    constraints = []
    
    # Row constraints: each row must have all different values
    for i in range(grid_size):
        row_vars = [(i, j) for j in range(grid_size)]
        constraints.append(Constraint(tuple(row_vars), all_diff_constraint))
    
    # Column constraints: each column must have all different values
    for j in range(grid_size):
        col_vars = [(i, j) for i in range(grid_size)]
        constraints.append(Constraint(tuple(col_vars), all_diff_constraint))
    
    # Box constraints: each box must have all different values
    for box_row in range(0, grid_size, box_size):
        for box_col in range(0, grid_size, box_size):
            box_vars = []
            for i in range(box_row, box_row + box_size):
                for j in range(box_col, box_col + box_size):
                    box_vars.append((i, j))
            constraints.append(Constraint(tuple(box_vars), all_diff_constraint))
    
    return NaryCSP(domains, constraints), grid_size


def mrv_nary(assignment, csp, domains):
    """Minimum-remaining-values heuristic for NaryCSP"""
    unassigned = [v for v in csp.variables if v not in assignment]
    if not unassigned:
        return None
    return argmin_random_tie(unassigned, key=lambda var: len(domains[var]))


def lcv_nary(var, assignment, csp, domains):
    """Least-constraining-values heuristic for NaryCSP"""
    # For n-ary constraints, LCV is complex to compute accurately
    # Use a simplified version: count how many constraints involve this variable
    # and prefer values that appear less frequently in other variables' domains
    
    def count_constraints(val):
        # Count how many unassigned neighbors share this value in their domain
        constraint_count = 0
        for const in csp.var_to_const[var]:
            other_vars = [v for v in const.scope if v != var and v not in assignment]
            for other_var in other_vars:
                if val in domains[other_var]:
                    constraint_count += 1
        return constraint_count
    
    # Sort by constraint count (lower is better - least constraining)
    return sorted(domains[var], key=count_constraints)


def gac_inference(csp, var, value, assignment, domains, ac_solver):
    """
    GAC inference function for backtracking search.
    
    Args:
        csp: NaryCSP instance
        var: Variable being assigned
        value: Value being assigned
        assignment: Current assignment dictionary
        domains: Current domain dictionary
        ac_solver: ACSolver instance
    
    Returns:
        Tuple of (success: bool, new_domains: dict)
    """
    # Create new domains with var=value
    new_domains = {v: domains[v].copy() for v in csp.variables}
    new_domains[var] = {value}
    
    # Find constraints involving var
    to_do = {(var, const) for const in csp.var_to_const[var]}
    
    # Run GAC
    consistent, reduced_domains, _ = ac_solver.GAC(new_domains, to_do)
    
    if consistent:
        # Return new domains dictionary
        return True, reduced_domains
    else:
        return False, domains


def backtracking_search_nary(csp, select_unassigned_variable=None, order_domain_values=None):
    """
    Backtracking search for NaryCSP with GAC inference.
    
    Args:
        csp: NaryCSP instance
        select_unassigned_variable: Function to select next variable (default: MRV)
        order_domain_values: Function to order domain values (default: LCV)
    
    Returns:
        Dictionary mapping variables to values, or None if no solution exists
    """
    ac_solver = ACSolver(csp)
    
    # Initialize domains
    initial_domains = {v: csp.domains[v].copy() for v in csp.variables}
    
    # Set default heuristics if not provided
    if select_unassigned_variable is None:
        def select_var(assignment, domains):
            return mrv_nary(assignment, csp, domains)
        select_unassigned_variable = select_var
    if order_domain_values is None:
        def order_values(var, assignment, domains):
            return lcv_nary(var, assignment, csp, domains)
        order_domain_values = order_values
    
    def backtrack(assignment, domains):
        if len(assignment) == len(csp.variables):
            return assignment
        
        var = select_unassigned_variable(assignment, domains)
        if var is None:
            return None
        
        for value in order_domain_values(var, assignment, domains):
            if value not in domains[var]:
                continue
            
            # Check if value is consistent with current assignment
            consistent = True
            for const in csp.var_to_const[var]:
                # Check if all variables in constraint scope (except var) are assigned
                other_vars = [v for v in const.scope if v != var]
                if all(v in assignment for v in other_vars):
                    test_assignment = assignment.copy()
                    test_assignment[var] = value
                    if not const.holds(test_assignment):
                        consistent = False
                        break
            
            if not consistent:
                continue
            
            # Assign value
            assignment[var] = value
            
            # Save current domains
            old_domains = {v: domains[v].copy() for v in csp.variables}
            
            # Perform GAC inference
            success, new_domains = gac_inference(csp, var, value, assignment, domains, ac_solver)
            
            if success:
                result = backtrack(assignment, new_domains)
                if result is not None:
                    return result
            
            # Restore domains and unassign
            domains = old_domains
            del assignment[var]
        
        return None
    
    result = backtrack({}, initial_domains)
    return result


def solve_sudoku_narycsp(narycsp, grid_size):
    """
    Solve a Sudoku NaryCSP using backtracking search with GAC.
    
    Args:
        narycsp: NaryCSP instance for the Sudoku puzzle
        grid_size: Size of the grid
    
    Returns:
        Dictionary mapping (row, col) tuples to values, or None if no solution exists
    """
    solution = backtracking_search_nary(narycsp)
    return solution


def solve_sudoku_gac(puzzle_string):
    """
    Solve a Sudoku puzzle using backtracking search with GAC.
    This function creates the NaryCSP and then solves it.
    For timing purposes, use create_sudoku_narycsp() and solve_sudoku_narycsp() separately.
    
    Args:
        puzzle_string: String of N×N characters representing the puzzle
    
    Returns:
        Dictionary mapping (row, col) tuples to values, or None if no solution exists
    """
    narycsp, grid_size = create_sudoku_narycsp(puzzle_string)
    return solve_sudoku_narycsp(narycsp, grid_size)


# ============================================================================
# Dataset processing (same format as project_csp.py)
# ============================================================================

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
                
                # Preprocessing: Create NaryCSP (this is excluded from timing)
                narycsp, grid_size = create_sudoku_narycsp(puzzle)
                
                # Measure solve time independently in this process
                # Use perf_counter() instead of time() to avoid issues with system clock adjustments
                # perf_counter() is monotonic and not affected by system clock changes
                # Timing starts AFTER preprocessing
                start_time = time.perf_counter()
                solution = solve_sudoku_narycsp(narycsp, grid_size)
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
                    # Check if solution is complete (all variables assigned)
                    if len(solution) != grid_size * grid_size:
                        raise ValueError(f"Incomplete solution: {len(solution)}/{grid_size * grid_size} variables assigned")
                    
                    stats['solved'] += 1
                    solution_str = solution_to_string(solution, grid_size)
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
    Solve all puzzles from the CSV dataset using GAC and record results with timing.
    Uses parallel processing to speed up computation on multi-core systems.
    
    Args:
        csv_path: Path to the CSV file containing puzzles
        output_path: Path to output CSV file (default: adds '_gac_results' to input filename)
        max_puzzles: Maximum number of puzzles to solve (None for all)
        num_workers: Number of parallel workers (None for auto-detection)
    
    Returns:
        Tuple of (statistics dictionary, output_path)
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
        output_path = os.path.join(results_dir, f'{base_name}_gac_results_{timestamp}.csv')
    else:
        # Ensure output directory exists even for custom paths
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
    
    # Read all rows from input CSV
    print(f"Reading input CSV (using GAC method)...")
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
    temp_dir = os.path.join(os.path.dirname(output_path) or '.', f'temp_gac_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
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
            if computed_solution and solve_time_str:
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
    print(f"{'':<20} {'Backtracking':<15}")
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
    
    parser = argparse.ArgumentParser(description='Solve Sudoku puzzles from CSV dataset using GAC')
    parser.add_argument('--csv', default='sudoku-3m.csv', 
                       help='Path to CSV file (default: sudoku-3m.csv)')
    parser.add_argument('--output', default=None,
                       help='Path to output CSV file (default: input_filename_gac_results_timestamp.csv)')
    parser.add_argument('--max', type=int, default=None,
                       help='Maximum number of puzzles to solve (default: all)')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of parallel workers (default: auto-detect CPU count)')
    
    args = parser.parse_args()
    
    # Process all puzzles from dataset
    print(f"Solving puzzles from {args.csv}...")
    print(f"Using GAC method")
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

