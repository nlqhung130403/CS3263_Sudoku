"""
Sudoku Q-Learning Agent with Shared Q-Table
Allows multiple workers to update the same Q-table for knowledge transfer
"""

import random
from collections import defaultdict
from multiprocessing import Lock

from sudoku_mdp import SudokuMDP, is_solved, tuple_to_grid, get_valid_actions


class SharedSudokuQLearningAgent:
    """
    Q-Learning agent that uses a shared Q-table across multiple processes.
    Uses locks to synchronize Q-table updates.
    """
    
    def __init__(self, mdp, shared_Q, shared_Nsa, Q_lock, Ne=10, Rplus=100, 
                 alpha=None, epsilon=0.2, epsilon_decay=0.995):
        """
        Initialize Shared Sudoku Q-Learning agent.
        
        Args:
            mdp: SudokuMDP instance
            shared_Q: Shared dictionary for Q-values (from multiprocessing.Manager)
            shared_Nsa: Shared dictionary for visit counts (from multiprocessing.Manager)
            Q_lock: Lock for synchronizing Q-table updates
            Ne: Minimum visits before using actual Q-value in exploration function
            Rplus: Optimistic initial Q-value for unexplored actions
            alpha: Learning rate function (default: 1/(1+n))
            epsilon: Epsilon for epsilon-greedy exploration
            epsilon_decay: Decay factor for epsilon after each episode
        """
        self.mdp = mdp
        self.shared_Q = shared_Q  # Shared across all processes
        self.shared_Nsa = shared_Nsa  # Shared across all processes
        self.Q_lock = Q_lock  # Lock for thread-safe updates
        
        self.gamma = mdp.gamma
        self.terminals = mdp.terminals
        self.Ne = Ne
        self.Rplus = Rplus
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.episode_count = 0
        
        if alpha:
            self.alpha = alpha
        else:
            self.alpha = lambda n: 1. / (1 + n)
    
    def f(self, u, n):
        """Exploration function."""
        if n < self.Ne:
            return self.Rplus
        else:
            return u
    
    def actions_in_state(self, state):
        """Return valid actions for current state."""
        if state in self.terminals:
            return [None]
        
        grid = tuple_to_grid(state)
        valid_actions = get_valid_actions(grid)
        return valid_actions if valid_actions else [None]
    
    def get_Q(self, state, action):
        """Thread-safe Q-value retrieval."""
        key = (state, action)
        with self.Q_lock:
            return self.shared_Q.get(key, 0.0)
    
    def get_Nsa(self, state, action):
        """Thread-safe visit count retrieval."""
        key = (state, action)
        with self.Q_lock:
            return self.shared_Nsa.get(key, 0)
    
    def update_Q(self, state, action, new_value):
        """Thread-safe Q-value update."""
        key = (state, action)
        with self.Q_lock:
            self.shared_Q[key] = new_value
    
    def increment_Nsa(self, state, action):
        """Thread-safe visit count increment."""
        key = (state, action)
        with self.Q_lock:
            self.shared_Nsa[key] = self.shared_Nsa.get(key, 0) + 1
            return self.shared_Nsa[key]
    
    def decay_epsilon(self):
        """Decay epsilon after each episode."""
        # Epsilon is local to each agent instance, no lock needed
        self.epsilon = max(0.01, self.epsilon * self.epsilon_decay)
        self.episode_count += 1
    
    def reset(self):
        """Reset agent state for new episode."""
        self.s = None
        self.a = None
        self.r = None


def solve_sudoku_with_shared_qlearning(puzzle_string, shared_Q, shared_Nsa, Q_lock,
                                       max_episodes=1000, max_steps_per_episode=200,
                                       Ne=5, Rplus=50, epsilon=0.3):
    """
    Solve a Sudoku puzzle using Q-learning with shared Q-table.
    
    Args:
        puzzle_string: 81-character string representing puzzle
        shared_Q: Shared dictionary for Q-values
        shared_Nsa: Shared dictionary for visit counts
        Q_lock: Lock for synchronization
        max_episodes: Maximum number of training episodes
        max_steps_per_episode: Maximum steps per episode
        Ne: Exploration parameter
        Rplus: Optimistic Q-value
        epsilon: Epsilon for epsilon-greedy
    
    Returns:
        Solution dictionary mapping (row, col) to value, or None if unsolved
    """
    # Create MDP
    mdp = SudokuMDP(puzzle_string, max_steps=max_steps_per_episode)
    
    # Create agent with shared Q-table
    agent = SharedSudokuQLearningAgent(
        mdp, shared_Q, shared_Nsa, Q_lock, 
        Ne=Ne, Rplus=Rplus, epsilon=epsilon
    )
    
    best_solution = None
    best_reward = float('-inf')
    
    # Training loop
    for episode in range(max_episodes):
        agent.reset()
        
        # Run episode
        state = mdp.init
        prev_state = None
        prev_action = None
        steps = 0
        episode_reward = 0
        
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
            valid_actions = agent.actions_in_state(state)
            if not valid_actions or valid_actions == [None]:
                break
            
            # Select action using epsilon-greedy
            if random.random() < agent.epsilon:
                # Explore: random valid action
                action = random.choice(valid_actions)
            else:
                # Exploit: best action according to Q-values
                # Batch Q-value reads in single lock acquisition for efficiency
                best_action = None
                best_q = float('-inf')
                with agent.Q_lock:
                    for a in valid_actions:
                        q_val = agent.shared_Q.get((state, a), 0.0)
                        n_visits = agent.shared_Nsa.get((state, a), 0)
                        q_val = agent.f(q_val, n_visits)
                        if q_val > best_q:
                            best_q = q_val
                            best_action = a
                action = best_action
            
            # Update Q-value for previous state-action pair
            if prev_state is not None and prev_action is not None:
                reward = mdp.R(state)  # Reward for reaching current state
                episode_reward += reward
                
                # Batch Q-table operations in a single lock acquisition to reduce contention
                # and ensure atomicity of read-modify-write operations
                next_valid_actions = agent.actions_in_state(state)
                
                # Single lock acquisition for all Q-table operations
                with agent.Q_lock:
                    # Increment visit count
                    prev_key = (prev_state, prev_action)
                    agent.shared_Nsa[prev_key] = agent.shared_Nsa.get(prev_key, 0) + 1
                    n_visits = agent.shared_Nsa[prev_key]
                    
                    # Get current Q-value for previous state-action
                    current_q = agent.shared_Q.get(prev_key, 0.0)
                    
                    # Get max Q-value for next state (read all next state Q-values)
                    if next_valid_actions and next_valid_actions != [None]:
                        max_next_q = max(
                            agent.shared_Q.get((state, a), 0.0) 
                            for a in next_valid_actions
                        )
                    else:
                        max_next_q = 0
                    
                    # Compute new Q-value
                    learning_rate = agent.alpha(n_visits)
                    new_q = current_q + learning_rate * (
                        reward + mdp.gamma * max_next_q - current_q
                    )
                    
                    # Update Q-value atomically
                    agent.shared_Q[prev_key] = new_q
            
            # Apply action to get next state
            transitions = mdp.T(state, action)
            if transitions:
                prob, next_state = transitions[0]  # Deterministic
                prev_state = state
                prev_action = action
                state = next_state
            else:
                break
            
            steps += 1
        
        # Decay epsilon
        agent.decay_epsilon()
        
        # Track best solution
        if episode_reward > best_reward:
            best_reward = episode_reward
            grid = tuple_to_grid(state)
            if is_solved(grid):
                solution = {}
                for row in range(9):
                    for col in range(9):
                        solution[(row, col)] = grid[row][col]
                best_solution = solution
    
    return best_solution

