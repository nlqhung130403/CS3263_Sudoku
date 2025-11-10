"""
Sudoku Q-Learning Agent: Specialized Q-learning agent for Sudoku solving.
"""

import random
from collections import defaultdict

from reinforcement_learning import QLearningAgent
from sudoku_mdp import SudokuMDP, is_solved, tuple_to_grid, get_valid_actions


class SudokuQLearningAgent(QLearningAgent):
    """
    Q-Learning agent specialized for Sudoku puzzles.
    Uses epsilon-greedy exploration and learns optimal placement strategy.
    """
    
    def __init__(self, mdp, Ne=10, Rplus=100, alpha=None, epsilon=0.2, epsilon_decay=0.995):
        """
        Initialize Sudoku Q-Learning agent.
        
        Args:
            mdp: SudokuMDP instance
            Ne: Minimum visits before using actual Q-value in exploration function
            Rplus: Optimistic initial Q-value for unexplored actions
            alpha: Learning rate function (default: 1/(1+n))
            epsilon: Epsilon for epsilon-greedy exploration
            epsilon_decay: Decay factor for epsilon after each episode
        """
        super().__init__(mdp, Ne=Ne, Rplus=Rplus, alpha=alpha, epsilon=epsilon)
        self.epsilon_decay = epsilon_decay
        self.episode_count = 0
    
    def actions_in_state(self, state):
        """Return valid actions for current state."""
        if state in self.terminals:
            return [None]
        
        # Get valid actions from MDP
        grid = tuple_to_grid(state)
        valid_actions = get_valid_actions(grid)
        return valid_actions if valid_actions else [None]
    
    def update_state(self, percept):
        """Update state from percept - no modification needed."""
        return percept
    
    def decay_epsilon(self):
        """Decay epsilon after each episode."""
        self.epsilon = max(0.01, self.epsilon * self.epsilon_decay)
        self.episode_count += 1
    
    def reset(self):
        """Reset agent state for new episode."""
        self.s = None
        self.a = None
        self.r = None


def solve_sudoku_with_qlearning(puzzle_string, max_episodes=1000, max_steps_per_episode=200):
    """
    Solve a Sudoku puzzle using Q-learning.
    
    Args:
        puzzle_string: 81-character string representing puzzle
        max_episodes: Maximum number of training episodes
        max_steps_per_episode: Maximum steps per episode
    
    Returns:
        Solution dictionary mapping (row, col) to value, or None if unsolved
    """
    # Create MDP
    mdp = SudokuMDP(puzzle_string, max_steps=max_steps_per_episode)
    
    # Create agent
    agent = SudokuQLearningAgent(mdp, Ne=5, Rplus=50, epsilon=0.3)
    
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
                best_action = None
                best_q = float('-inf')
                for a in valid_actions:
                    q_val = agent.f(agent.Q[(state, a)], agent.Nsa[(state, a)])
                    if q_val > best_q:
                        best_q = q_val
                        best_action = a
                action = best_action
            
            # Update Q-value for previous state-action pair
            if prev_state is not None and prev_action is not None:
                reward = mdp.R(state)  # Reward for reaching current state
                episode_reward += reward
                
                # Q-learning update for previous state-action
                agent.Nsa[(prev_state, prev_action)] += 1
                next_valid_actions = agent.actions_in_state(state)
                if next_valid_actions and next_valid_actions != [None]:
                    max_next_q = max(agent.Q[(state, a)] for a in next_valid_actions)
                else:
                    max_next_q = 0
                
                # Q-learning update formula: Q(s,a) = Q(s,a) + alpha * (r + gamma * max Q(s',a') - Q(s,a))
                current_q = agent.Q[(prev_state, prev_action)]
                learning_rate = agent.alpha(agent.Nsa[(prev_state, prev_action)])
                agent.Q[(prev_state, prev_action)] = current_q + learning_rate * (
                    reward + mdp.gamma * max_next_q - current_q
                )
            
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

