"""
Sudoku MDP: Represents Sudoku puzzle as a Markov Decision Process
for reinforcement learning.
"""

from collections import defaultdict
from mdp import MDP


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
            row.append(int(char) if char != '.' else 0)
        grid.append(row)
    return grid


def grid_to_tuple(grid):
    """Convert 9x9 grid to immutable tuple for hashing."""
    return tuple(tuple(row) for row in grid)


def tuple_to_grid(state_tuple):
    """Convert state tuple back to grid."""
    return [list(row) for row in state_tuple]


def get_neighbors(row, col):
    """Get all neighbor cells (same row, column, or box) for a given cell."""
    neighbors = set()
    
    # Same row
    for c in range(9):
        if c != col:
            neighbors.add((row, c))
    
    # Same column
    for r in range(9):
        if r != row:
            neighbors.add((r, col))
    
    # Same 3x3 box
    box_row_start = (row // 3) * 3
    box_col_start = (col // 3) * 3
    for r in range(box_row_start, box_row_start + 3):
        for c in range(box_col_start, box_col_start + 3):
            if (r, c) != (row, col):
                neighbors.add((r, c))
    
    return neighbors


def is_valid_placement(grid, row, col, value):
    """Check if placing value at (row, col) is valid."""
    if grid[row][col] != 0:
        return False  # Cell already filled
    
    # Check row
    for c in range(9):
        if grid[row][c] == value:
            return False
    
    # Check column
    for r in range(9):
        if grid[r][col] == value:
            return False
    
    # Check 3x3 box
    box_row_start = (row // 3) * 3
    box_col_start = (col // 3) * 3
    for r in range(box_row_start, box_row_start + 3):
        for c in range(box_col_start, box_col_start + 3):
            if grid[r][c] == value:
                return False
    
    return True


def count_conflicts(grid, row, col, value):
    """Count number of constraint violations if value is placed at (row, col)."""
    conflicts = 0
    
    # Check row
    for c in range(9):
        if c != col and grid[row][c] == value:
            conflicts += 1
    
    # Check column
    for r in range(9):
        if r != row and grid[r][col] == value:
            conflicts += 1
    
    # Check 3x3 box
    box_row_start = (row // 3) * 3
    box_col_start = (col // 3) * 3
    for r in range(box_row_start, box_row_start + 3):
        for c in range(box_col_start, box_col_start + 3):
            if (r, c) != (row, col) and grid[r][c] == value:
                conflicts += 1
    
    return conflicts


def is_solved(grid):
    """Check if puzzle is completely solved correctly."""
    # Check all cells are filled
    for row in grid:
        if 0 in row:
            return False
    
    # Check all constraints are satisfied
    for row in range(9):
        for col in range(9):
            value = grid[row][col]
            if count_conflicts(grid, row, col, value) > 0:
                return False
    
    return True


def get_empty_cells(grid):
    """Get list of empty cell positions."""
    empty = []
    for row in range(9):
        for col in range(9):
            if grid[row][col] == 0:
                empty.append((row, col))
    return empty


def get_valid_actions(grid):
    """Get all valid actions (row, col, value) for current state."""
    actions = []
    empty_cells = get_empty_cells(grid)
    
    for row, col in empty_cells:
        for value in range(1, 10):
            if is_valid_placement(grid, row, col, value):
                actions.append((row, col, value))
    
    return actions


class SudokuMDP(MDP):
    """
    Sudoku puzzle represented as an MDP for reinforcement learning.
    
    State: Tuple of 81 integers (0-9, where 0 = empty)
    Action: (row, col, value) tuple for placing a value
    Reward: +1000 for correct solution, -10 for constraint violation, -1 per step
    """
    
    def __init__(self, puzzle_string, max_steps=200, gamma=0.95):
        """
        Initialize Sudoku MDP.
        
        Args:
            puzzle_string: 81-character string representing initial puzzle
            max_steps: Maximum steps before timeout
            gamma: Discount factor
        """
        self.puzzle_string = puzzle_string
        self.max_steps = max_steps
        self.initial_grid = parse_sudoku(puzzle_string)
        self.initial_state = grid_to_tuple(self.initial_grid)
        
        # Build state space and transitions
        # For efficiency, we'll generate states on-the-fly rather than pre-computing
        # all possible states (which would be enormous)
        self.states = set()
        self.states.add(self.initial_state)
        
        # Actions: all possible (row, col, value) placements
        # We'll generate valid actions dynamically per state
        self.actlist = []  # Will be generated per state
        
        # Terminal states: solved puzzle or timeout
        self.terminals = set()
        
        # Transitions: deterministic (probability 1.0 for each action)
        self.transitions = {}
        
        # Rewards: computed dynamically
        self.reward = {}
        
        # Initialize transitions and rewards for initial state
        self._build_transitions(self.initial_state)
        
        super().__init__(
            init=self.initial_state,
            actlist=self.actlist,
            terminals=self.terminals,
            transitions=self.transitions,
            reward=self.reward,
            states=self.states,
            gamma=gamma
        )
    
    def _build_transitions(self, state):
        """Build transitions and rewards for a given state."""
        if state in self.transitions:
            return  # Already built
        
        grid = tuple_to_grid(state)
        
        # Check if terminal (solved)
        if is_solved(grid):
            self.terminals.add(state)
            self.transitions[state] = {None: [(1.0, state)]}
            self.reward[state] = 1000.0  # Large reward for solving
            return
        
        # Generate valid actions for this state
        valid_actions = get_valid_actions(grid)
        
        if not valid_actions:
            # No valid actions - terminal state (failed)
            self.terminals.add(state)
            self.transitions[state] = {None: [(1.0, state)]}
            self.reward[state] = -100.0  # Penalty for failure
            return
        
        # Build transitions for each action
        self.transitions[state] = {}
        for action in valid_actions:
            row, col, value = action
            
            # Apply action to get next state
            next_grid = [row[:] for row in grid]  # Deep copy
            next_grid[row][col] = value
            next_state = grid_to_tuple(next_grid)
            
            # Add to states
            self.states.add(next_state)
            
            # Transition is deterministic
            self.transitions[state][action] = [(1.0, next_state)]
            
            # Reward: small negative for step, larger negative for conflicts
            conflicts = count_conflicts(next_grid, row, col, value)
            if conflicts > 0:
                reward = -10.0 * conflicts  # Penalty for constraint violation
            else:
                reward = -1.0  # Small step penalty
            
            # Check if next state is terminal
            if is_solved(next_grid):
                self.terminals.add(next_state)
                reward = 1000.0  # Override with success reward
            
            self.reward[next_state] = reward
        
        # Store actions for this state
        if not self.actlist:
            self.actlist = valid_actions
    
    def actions(self, state):
        """Return valid actions for a state, building transitions if needed."""
        if state in self.terminals:
            return [None]
        
        if state not in self.transitions:
            self._build_transitions(state)
        
        if state in self.transitions:
            return list(self.transitions[state].keys())
        else:
            return []
    
    def T(self, state, action):
        """Transition model - build if needed."""
        if state not in self.transitions:
            self._build_transitions(state)
        
        if action is None or state in self.terminals:
            return [(1.0, state)]
        
        if state in self.transitions and action in self.transitions[state]:
            return self.transitions[state][action]
        else:
            return [(1.0, state)]  # Invalid action - stay in same state
    
    def R(self, state):
        """Return reward for state, computing if needed."""
        if state not in self.reward:
            grid = tuple_to_grid(state)
            if is_solved(grid):
                self.reward[state] = 1000.0
            else:
                self.reward[state] = -1.0  # Default step penalty
        
        return self.reward[state]
    
    def solution_to_string(self, state):
        """Convert solved state to string representation."""
        grid = tuple_to_grid(state)
        result = []
        for row in grid:
            for val in row:
                result.append(str(val))
        return ''.join(result)

