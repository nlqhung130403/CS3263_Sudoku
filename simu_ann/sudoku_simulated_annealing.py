"""
Simulated Annealing Solver for Sudoku Puzzles

This module implements simulated annealing as described in the specification:
- State: 9x9 matrix where each digit 1-9 appears exactly 9 times
- Move: Swap two non-clue cells
- Cost: Number of constraint violations
- Acceptance: U ≤ min{exp([c(B_n) - c(B)] / τ_n), 1}
- Cooling: Geometric schedule
- Cell selection: Probability proportional to exp(violations)
"""

import random
import numpy as np
import copy
from utils import probability, weighted_sampler


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
            row.append(int(char) if char != '.' else None)
        grid.append(row)
    return grid


def get_box(row, col):
    """
    Get the 3x3 box number (0-8) for a given cell position.
    
    Args:
        row: Row index (0-8)
        col: Column index (0-8)
    
    Returns:
        Box number (0-8)
    """
    return (row // 3) * 3 + (col // 3)


def count_constraint_violations(board):
    """
    Count the number of constraint violations in a Sudoku board.
    A violation occurs when two cells in the same row, column, or box have the same value.
    
    Args:
        board: 9x9 list of lists representing the board
    
    Returns:
        Total number of constraint violations
    """
    violations = 0
    
    # Check rows
    for row in range(9):
        row_values = [board[row][col] for col in range(9) if board[row][col] is not None]
        # Count duplicates: if we have n occurrences of a value, that's n-1 violations
        value_counts = {}
        for val in row_values:
            value_counts[val] = value_counts.get(val, 0) + 1
        for count in value_counts.values():
            if count > 1:
                violations += count - 1
    
    # Check columns
    for col in range(9):
        col_values = [board[row][col] for row in range(9) if board[row][col] is not None]
        value_counts = {}
        for val in col_values:
            value_counts[val] = value_counts.get(val, 0) + 1
        for count in value_counts.values():
            if count > 1:
                violations += count - 1
    
    # Check boxes
    for box in range(9):
        box_row_start = (box // 3) * 3
        box_col_start = (box % 3) * 3
        box_values = []
        for row in range(box_row_start, box_row_start + 3):
            for col in range(box_col_start, box_col_start + 3):
                if board[row][col] is not None:
                    box_values.append(board[row][col])
        value_counts = {}
        for val in box_values:
            value_counts[val] = value_counts.get(val, 0) + 1
        for count in value_counts.values():
            if count > 1:
                violations += count - 1
    
    return violations


def get_cell_violations(board, row, col):
    """
    Count the number of constraint violations involving a specific cell.
    
    Args:
        board: 9x9 list of lists representing the board
        row: Row index (0-8)
        col: Column index (0-8)
    
    Returns:
        Number of violations involving this cell
    """
    if board[row][col] is None:
        return 0
    
    value = board[row][col]
    violations = 0
    
    # Check row
    for c in range(9):
        if c != col and board[row][c] == value:
            violations += 1
    
    # Check column
    for r in range(9):
        if r != row and board[r][col] == value:
            violations += 1
    
    # Check box
    box_row_start = (row // 3) * 3
    box_col_start = (col // 3) * 3
    for r in range(box_row_start, box_row_start + 3):
        for c in range(box_col_start, box_col_start + 3):
            if (r != row or c != col) and board[r][c] == value:
                violations += 1
    
    return violations


def create_initial_board(puzzle_string):
    """
    Create an initial feasible board from a puzzle string.
    A feasible board has:
    - All clue cells filled with their given values
    - All empty cells filled such that each digit 1-9 appears exactly 9 times
    
    Args:
        puzzle_string: String of 81 characters representing the puzzle
    
    Returns:
        Tuple of (board, clue_positions) where:
        - board: 9x9 list of lists
        - clue_positions: Set of (row, col) tuples for clue cells
    """
    board = parse_sudoku(puzzle_string)
    clue_positions = set()
    
    # Count clues and their values
    clue_values = []
    empty_positions = []
    
    for row in range(9):
        for col in range(9):
            if board[row][col] is not None:
                clue_positions.add((row, col))
                clue_values.append(board[row][col])
            else:
                empty_positions.append((row, col))
    
    # Calculate how many of each digit we need
    digit_counts = {i: 9 for i in range(1, 10)}
    for val in clue_values:
        digit_counts[val] -= 1
    
    # Fill empty cells randomly to satisfy digit count constraint
    digits_to_place = []
    for digit, count in digit_counts.items():
        digits_to_place.extend([digit] * count)
    
    random.shuffle(digits_to_place)
    
    for (row, col), digit in zip(empty_positions, digits_to_place):
        board[row][col] = digit
    
    return board, clue_positions


def select_cells_weighted(board, clue_positions, num_cells=2):
    """
    Select cells non-uniformly, with probability proportional to exp(violations).
    Only selects from non-clue cells.
    
    Args:
        board: 9x9 list of lists representing the board
        clue_positions: Set of (row, col) tuples for clue cells
        num_cells: Number of cells to select (default: 2)
    
    Returns:
        List of (row, col) tuples
    """
    # Get all non-clue cells
    non_clue_cells = []
    for row in range(9):
        for col in range(9):
            if (row, col) not in clue_positions:
                non_clue_cells.append((row, col))
    
    if len(non_clue_cells) < num_cells:
        return random.sample(non_clue_cells, len(non_clue_cells))
    
    # Calculate weights: exp(violations)
    weights = []
    for (row, col) in non_clue_cells:
        violations = get_cell_violations(board, row, col)
        # Use exp(violations) as weight, but add 1 to avoid zero weight
        weight = np.exp(violations)
        weights.append(weight)
    
    # Select cells using weighted sampling
    selected = []
    remaining_cells = non_clue_cells.copy()
    remaining_weights = weights.copy()
    
    for _ in range(num_cells):
        if len(remaining_cells) == 0:
            break
        
        sampler = weighted_sampler(remaining_cells, remaining_weights)
        cell = sampler()
        
        idx = remaining_cells.index(cell)
        selected.append(cell)
        remaining_cells.pop(idx)
        remaining_weights.pop(idx)
    
    return selected


def swap_cells(board, cell1, cell2):
    """
    Swap the values of two cells in the board.
    
    Args:
        board: 9x9 list of lists (will be modified)
        cell1: (row, col) tuple
        cell2: (row, col) tuple
    """
    row1, col1 = cell1
    row2, col2 = cell2
    board[row1][col1], board[row2][col2] = board[row2][col2], board[row1][col1]


def geometric_cooling_schedule(initial_temp, cooling_rate, iteration):
    """
    Geometric cooling schedule: τ_n = initial_temp * (cooling_rate ^ n)
    
    Args:
        initial_temp: Initial temperature
        cooling_rate: Cooling rate (typically 0.95-0.99)
        iteration: Current iteration number
    
    Returns:
        Current temperature
    """
    return initial_temp * (cooling_rate ** iteration)


def solve_sudoku_simulated_annealing(
    puzzle_string,
    initial_temp=10.0,
    cooling_rate=0.999,
    max_iterations=100000,
    min_temp=0.01
):
    """
    Solve a Sudoku puzzle using simulated annealing.
    
    Algorithm:
    1. Start with a feasible board (all digits appear 9 times)
    2. For each iteration:
       a. Calculate current temperature using cooling schedule
       b. Select two non-clue cells (weighted by violations)
       c. Propose swap
       d. Calculate cost change
       e. Accept or reject based on acceptance criterion
    3. Return solution when cost reaches 0 or max iterations reached
    
    Args:
        puzzle_string: String of 81 characters representing the puzzle
        initial_temp: Initial temperature (default: 10.0)
        cooling_rate: Cooling rate for geometric schedule (default: 0.999)
        max_iterations: Maximum number of iterations (default: 100000)
        min_temp: Minimum temperature threshold (default: 0.01)
    
    Returns:
        Dictionary mapping (row, col) to value, or None if not solved
    """
    # Create initial feasible board
    board, clue_positions = create_initial_board(puzzle_string)
    
    # Calculate initial cost
    current_cost = count_constraint_violations(board)
    
    # Track best solution found
    best_board = copy.deepcopy(board)
    best_cost = current_cost
    
    # Main annealing loop
    for iteration in range(max_iterations):
        # Calculate temperature
        temperature = geometric_cooling_schedule(initial_temp, cooling_rate, iteration)
        
        # Stop if temperature is too low
        if temperature < min_temp:
            break
        
        # If we found a solution, return it
        if current_cost == 0:
            # Convert board to solution dictionary
            solution = {}
            for row in range(9):
                for col in range(9):
                    solution[(row, col)] = best_board[row][col]
            return solution
        
        # Select two different non-clue cells (weighted by violations)
        cells = select_cells_weighted(board, clue_positions, num_cells=2)
        if len(cells) < 2:
            break
        
        cell1, cell2 = cells[0], cells[1]
        
        # Propose swap
        swap_cells(board, cell1, cell2)
        
        # Calculate new cost
        new_cost = count_constraint_violations(board)
        
        # Calculate cost change (note: we want to minimize cost)
        delta_cost = new_cost - current_cost
        
        # Acceptance criterion: U ≤ min{exp([c(B_n) - c(B)] / τ_n), 1}
        # Since delta_cost = c(B) - c(B_n), we have:
        # exp([c(B_n) - c(B)] / τ) = exp(-delta_cost / τ)
        accept_probability = min(np.exp(-delta_cost / temperature), 1.0)
        
        # Accept if cost decreases or with probability accept_probability
        if delta_cost < 0 or probability(accept_probability):
            # Accept the move
            current_cost = new_cost
            
            # Update best solution
            if new_cost < best_cost:
                best_board = copy.deepcopy(board)
                best_cost = new_cost
        else:
            # Reject: swap back
            swap_cells(board, cell1, cell2)
    
    # Check if we found a solution
    if best_cost == 0:
        solution = {}
        for row in range(9):
            for col in range(9):
                solution[(row, col)] = best_board[row][col]
        return solution
    
    # Return None if no solution found
    return None

