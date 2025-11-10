# Reinforcement Learning Sudoku Solver

This directory contains a Q-Learning based Sudoku solver that uses reinforcement learning to learn optimal strategies for solving Sudoku puzzles.

## Algorithm Design

### State Representation
- **State**: A Sudoku board is represented as a tuple of 81 integers (0-9), where:
  - `0` represents an empty cell
  - `1-9` represent filled cells with their values
- States are immutable tuples, making them hashable for Q-table lookups

### Action Space
- **Action**: A tuple `(row, col, value)` representing placing `value` (1-9) at position `(row, col)`
- Only valid actions are considered: placing a value in an empty cell that doesn't violate Sudoku constraints
- Constraints checked:
  - No duplicate in the same row
  - No duplicate in the same column
  - No duplicate in the same 3x3 box

### Reward Structure
- **+1000**: Successfully solving the puzzle correctly
- **-10 × conflicts**: Penalty for constraint violations (each conflict multiplies the penalty)
- **-1**: Small step penalty to encourage efficiency
- **-100**: Penalty for reaching maximum steps without solving

### Q-Learning Algorithm
The solver uses **Q-Learning**, an off-policy temporal difference learning algorithm:

1. **Q-Table**: Stores Q-values for state-action pairs `Q[(state, action)]`
2. **Exploration Strategy**: Epsilon-greedy exploration
   - With probability `ε`: choose random valid action (explore)
   - With probability `1-ε`: choose action with highest Q-value (exploit)
   - Epsilon decays over episodes: `ε = ε × decay_factor`
3. **Q-Learning Update**:
   ```
   Q(s, a) ← Q(s, a) + α × [r + γ × max Q(s', a') - Q(s, a)]
   ```
   Where:
   - `α` (alpha): Learning rate (decreases with visits: `1/(1+n)`)
   - `γ` (gamma): Discount factor (0.95)
   - `r`: Immediate reward
   - `s'`: Next state
   - `max Q(s', a')`: Maximum Q-value in next state

4. **Training Process**:
   - Multiple episodes (default: 500) per puzzle
   - Each episode: agent explores from initial puzzle state
   - Episode ends when: puzzle solved, max steps reached, or no valid actions
   - Q-values updated after each action
   - Returns best solution found across all episodes

### MDP Formulation
The Sudoku puzzle is modeled as a **Markov Decision Process (MDP)**:
- **States**: All possible board configurations
- **Actions**: Valid placements (row, col, value)
- **Transitions**: Deterministic (placing a value leads to exactly one next state)
- **Rewards**: As defined above
- **Terminal States**: Solved puzzles or failed states

## Parallelism Design

### Architecture
The solver uses **multiprocessing with shared Q-table** to process puzzles in parallel while enabling knowledge transfer:

1. **Shared Q-Table**: All workers update the same Q-table (using `multiprocessing.Manager`)
2. **Chunking**: Input puzzles are divided into chunks (one per worker)
3. **Knowledge Transfer**: Learning from one puzzle helps solve others
4. **Synchronization**: Locks prevent race conditions in Q-table updates
5. **Temporary Files**: Each worker writes results to its own temporary CSV file
6. **Aggregation**: After all workers complete, results are concatenated into final output

### Race Condition Prevention

**Key Design Decisions:**

1. **Shared Q-Table with Locks**: 
   - All workers share the same Q-table via `multiprocessing.Manager().dict()`
   - All Q-table operations protected by `Lock()`
   - Thread-safe reads and writes prevent race conditions
   - Knowledge flows across all puzzles

2. **Synchronized Updates**:
   - Q-value updates: `with Q_lock: shared_Q[key] = value`
   - Visit count increments: `with Q_lock: shared_Nsa[key] += 1`
   - Fine-grained locking only around Q-table operations

3. **Separate Temporary Files**:
   - Each worker writes to `chunk_{id}.csv` in a unique temp directory
   - File writes are independent and don't conflict
   - No file locking needed

4. **Process Isolation**:
   - Python's `multiprocessing.Pool` creates separate processes
   - Shared memory via Manager for Q-table only
   - Other state (episode history, stats) is process-local

5. **Deterministic Chunking**:
   - Chunks are created before parallel processing starts
   - Each chunk has a unique ID
   - Shared Q-table passed to all workers

6. **Sequential Aggregation**:
   - Final result concatenation happens sequentially after all workers finish
   - Uses sorted file names to maintain puzzle order
   - No concurrent writes to final output file

### Knowledge Transfer Mechanism

**How learning flows across chunks:**

- **Early puzzles**: Workers explore randomly, building initial Q-values in shared table
- **Later puzzles**: Workers exploit learned Q-values, solving faster
- **Pattern learning**: Common strategies emerge across all puzzles
- **Transfer learning**: Easy puzzles teach strategies for hard puzzles

See `SHARED_MODEL_EXPLANATION.md` for detailed explanation.

### Performance Considerations

- **Scalability**: Linear speedup with number of cores (up to lock contention limits)
- **Memory**: Shared Q-table in shared memory, plus per-worker chunk data
- **I/O**: Parallel reads from input CSV, sequential writes to temp files
- **Load Balancing**: Chunks are approximately equal size
- **Lock Contention**: Fine-grained locking minimizes waiting time

## File Structure

```
rl/
├── __init__.py                    # Package initialization
├── utils.py                       # Utility functions (vector operations, etc.)
├── mdp.py                         # MDP base class and policy evaluation
├── reinforcement_learning.py      # Q-Learning agent base class
├── sudoku_mdp.py                  # Sudoku-specific MDP implementation
├── sudoku_qlearning.py            # Sudoku Q-Learning solver (independent Q-tables)
├── sudoku_qlearning_shared.py     # Shared Q-table Q-Learning solver (used by project_rl.py)
├── project_rl.py                  # Main script with parallel processing and shared Q-table
├── README.md                      # This file
├── DESIGN.md                      # Detailed design documentation
└── SHARED_MODEL_EXPLANATION.md    # Explanation of shared Q-table approach
```

## Usage

```bash
# Solve puzzles from CSV file
python project_rl.py --csv test_set/sudoku_test_set_random_10k.csv

# Limit to first 100 puzzles
python project_rl.py --csv test_set/sudoku_test_set_random_10k.csv --max 100

# Use specific number of workers
python project_rl.py --csv test_set/sudoku_test_set_random_10k.csv --workers 4

# Specify output file
python project_rl.py --csv test_set/sudoku_test_set_random_10k.csv --output results/my_results.csv
```

## Output Format

The script generates a CSV file with columns:
- `id`: Puzzle ID
- `puzzle`: Initial puzzle string (81 characters)
- `solution`: Expected solution string
- `clues`: Number of clues
- `difficulty`: Difficulty rating
- `solve_time`: Time taken to solve (seconds)
- `computed_solution`: Solution found by Q-learning (or ERROR message)

## Algorithm Limitations

1. **State Space**: The state space is enormous (9^81 possible states), so we can't explore all states
2. **Convergence**: Q-learning may not converge to optimal policy within limited episodes
3. **Exploration**: Requires careful balance between exploration and exploitation
4. **Reward Shaping**: Reward structure significantly affects learning performance

## Future Improvements

1. **Deep Q-Networks (DQN)**: Use neural networks to approximate Q-values
2. **Experience Replay**: Store and reuse past experiences
3. **Reward Shaping**: Fine-tune reward structure for better learning
4. **Transfer Learning**: Learn from easier puzzles to solve harder ones
5. **Curriculum Learning**: Gradually increase puzzle difficulty

