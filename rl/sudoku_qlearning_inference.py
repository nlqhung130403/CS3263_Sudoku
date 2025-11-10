"""
Sudoku Q-Learning Inference: Solve puzzles using trained Q-table without updating it
"""

import random
from sudoku_mdp import SudokuMDP, is_solved, tuple_to_grid, get_valid_actions


def solve_sudoku_with_inference(puzzle_string, shared_Q, shared_Nsa, Q_lock,
                                 max_episodes=1000, max_steps_per_episode=200,
                                 Ne=5, Rplus=50, epsilon=0.0):
    """
    Solve a Sudoku puzzle using trained Q-table WITHOUT updating it (inference only).
    
    Args:
        puzzle_string: 81-character string representing puzzle
        shared_Q: Shared dictionary for Q-values (trained, read-only)
        shared_Nsa: Shared dictionary for visit counts (for exploration function)
        Q_lock: Lock for synchronization (read-only access)
        max_episodes: Maximum number of episodes to try
        max_steps_per_episode: Maximum steps per episode
        Ne: Exploration parameter (for exploration function)
        Rplus: Optimistic Q-value (for exploration function)
        epsilon: Epsilon for epsilon-greedy (set to 0.0 for pure exploitation)
    
    Returns:
        Solution dictionary mapping (row, col) to value, or None if unsolved
    """
    # Create MDP
    mdp = SudokuMDP(puzzle_string, max_steps=max_steps_per_episode)
    
    def f(u, n):
        """Exploration function."""
        if n < Ne:
            return Rplus
        else:
            return u
    
    def actions_in_state(state):
        """Return valid actions for current state."""
        if state in mdp.terminals:
            return [None]
        
        grid = tuple_to_grid(state)
        valid_actions = get_valid_actions(grid)
        return valid_actions if valid_actions else [None]
    
    best_solution = None
    
    # Inference loop (no Q-table updates)
    for episode in range(max_episodes):
        # Run episode
        state = mdp.init
        steps = 0
        
        while steps < max_steps_per_episode:
            # Check if terminal
            if state in mdp.terminals:
                if is_solved(tuple_to_grid(state)):
                    # Solved! Convert to solution format
                    grid = tuple_to_grid(state)
                    solution = {}
                    for row in range(9):
                        for col in range(9):
                            solution[(row, col)] = grid[row][col]
                    return solution
                break
            
            # Get valid actions
            valid_actions = actions_in_state(state)
            if not valid_actions or valid_actions == [None]:
                break
            
            # Select action using epsilon-greedy (but epsilon should be 0.0 for pure inference)
            if random.random() < epsilon:
                # Explore: random valid action (only if epsilon > 0)
                action = random.choice(valid_actions)
            else:
                # Exploit: best action according to Q-values (read-only)
                best_action = None
                best_q = float('-inf')
                with Q_lock:  # Read-only lock
                    for a in valid_actions:
                        q_val = shared_Q.get((state, a), 0.0)
                        n_visits = shared_Nsa.get((state, a), 0)
                        q_val = f(q_val, n_visits)
                        if q_val > best_q:
                            best_q = q_val
                            best_action = a
                action = best_action
            
            # Apply action to get next state (no Q-table update!)
            transitions = mdp.T(state, action)
            if transitions:
                prob, next_state = transitions[0]  # Deterministic
                state = next_state
            else:
                break
            
            steps += 1
        
        # Track best solution
        grid = tuple_to_grid(state)
        if is_solved(grid):
            solution = {}
            for row in range(9):
                for col in range(9):
                    solution[(row, col)] = grid[row][col]
            best_solution = solution
    
    return best_solution

