# Simulated Annealing Sudoku Solver

This directory contains a Simulated Annealing based Sudoku solver that uses combinatorial optimization to solve Sudoku puzzles.

## Algorithm Design

### State Representation
- **State**: A 9x9 matrix (board) of integers drawn from the set {1, ..., 9}
- **Feasibility Constraint**: Each integer must appear exactly nine times on the board
- **Clue Constraint**: All numerical clues (pre-filled cells) must be respected

### Move Generation (Proposal Stage)
- **Move Type**: Swap the contents of two different non-clue cells
- **Benefit**: Swapping preserves all digit counts (ensuring each digit still appears nine times)
- **Cell Selection**: Non-uniform selection prioritizing problematic areas
  - Probability of choosing a cell is proportional to `exp(i)` where `i` is the number of constraint violations involving that cell
  - Cells with more violations are more likely to be selected for swapping

### Cost Function
- **Cost**: Number of constraint violations in the board
- **Constraint Violations**: 
  - Two cells in the same row with the same value
  - Two cells in the same column with the same value
  - Two cells in the same 3x3 box with the same value
- **Goal**: Minimize cost to zero (zero violations = solved puzzle)

### Acceptance Criterion
At temperature `τ_n`, to decide whether to accept a proposed neighboring board `B` (from current board `B_n`):

1. Draw a random deviate `U` uniformly from `[0, 1]`
2. Calculate acceptance probability: `min{exp([c(B_n) - c(B)] / τ_n), 1}`
   - Where `c(B)` is the cost (number of violations) of board `B`
   - Note: `c(B_n) - c(B)` = `-delta_cost` where `delta_cost = c(B) - c(B_n)`
3. Accept if: `U ≤ min{exp([c(B_n) - c(B)] / τ_n), 1}`
   - This simplifies to: `U ≤ exp(-delta_cost / τ_n)` when `delta_cost > 0`
   - Always accept if `delta_cost ≤ 0` (cost decreases or stays same)

**Interpretation**:
- Moves that decrease cost are always accepted
- Moves that increase cost are accepted with probability that decreases as:
  - Cost increase gets larger
  - Temperature gets lower

### Cooling Schedule
- **Type**: Geometric cooling schedule
- **Formula**: `τ_n = initial_temp × (cooling_rate ^ n)`
- **Parameters**:
  - `initial_temp`: Starting temperature (default: 10.0)
  - `cooling_rate`: Geometric decay rate (default: 0.999)
  - `min_temp`: Minimum temperature threshold (default: 0.01)
- **Behavior**: Temperature starts high and slowly declines to near zero
- **At Zero Temperature**: Only favorable (cost-decreasing) or cost-neutral moves are taken

### Initial State
- Start from any feasible board:
  1. Parse puzzle string to identify clue cells
  2. Count occurrences of each digit in clues
  3. Fill empty cells randomly such that each digit 1-9 appears exactly 9 times total
  4. This ensures the board satisfies the digit count constraint from the start

### Termination Conditions
1. **Success**: Cost reaches zero (puzzle solved)
2. **Temperature**: Temperature drops below minimum threshold
3. **Iterations**: Maximum number of iterations reached (default: 100,000)

## Parallelism Design

### Architecture
The solver uses **multiprocessing with independent workers** to process puzzles in parallel:

1. **Chunking**: Input puzzles are divided into chunks (one per worker)
2. **Independent Processing**: Each worker processes its chunk independently
3. **Temporary Files**: Each worker writes results to its own temporary CSV file
4. **Aggregation**: After all workers complete, results are concatenated into final output

### Race Condition Prevention

**Key Design Decisions:**

1. **Process Isolation**:
   - Python's `multiprocessing.Pool` creates separate processes
   - Each process has its own memory space
   - No shared state between workers

2. **Separate Temporary Files**:
   - Each worker writes to `chunk_{id}.csv` in a unique temp directory
   - File writes are independent and don't conflict
   - No file locking needed

3. **Deterministic Chunking**:
   - Chunks are created before parallel processing starts
   - Each chunk has a unique ID
   - No race conditions in chunk assignment

4. **Sequential Aggregation**:
   - Final result concatenation happens sequentially after all workers finish
   - Uses sorted file names to maintain puzzle order
   - No concurrent writes to final output file

5. **No Shared State**:
   - Unlike Q-Learning (which shares Q-tables), simulated annealing doesn't require shared state
   - Each puzzle is solved independently
   - No synchronization needed

### Why No Shared State?

Simulated annealing is a **local search algorithm**:
- Each puzzle is solved independently
- No learning or knowledge transfer between puzzles
- Each run starts from scratch with a random initial state
- No benefit from sharing information across puzzles

This makes parallelism straightforward: simply divide puzzles among workers and process independently.

### Performance Considerations

- **Scalability**: Linear speedup with number of cores (no lock contention)
- **Memory**: Each worker maintains its own board state (minimal memory overhead)
- **I/O**: Parallel reads from input CSV, sequential writes to temp files, sequential aggregation
- **Load Balancing**: Chunks are approximately equal size

## File Structure

```
simu_ann/
├── __init__.py                      # Package initialization
├── utils.py                         # Utility functions (probability, weighted_sampler)
├── sudoku_simulated_annealing.py    # Core simulated annealing algorithm
├── project_sa.py                    # Main script with parallel processing
└── README.md                        # This file
```

## Usage

```bash
# Solve puzzles from CSV file
python project_sa.py --csv test_set/sudoku_test_set_random_10k.csv

# Limit to first 100 puzzles
python project_sa.py --csv test_set/sudoku_test_set_random_10k.csv --max 100

# Use specific number of workers
python project_sa.py --csv test_set/sudoku_test_set_random_10k.csv --workers 4

# Specify output file
python project_sa.py --csv test_set/sudoku_test_set_random_10k.csv --output results/my_results.csv
```

## Output Format

The script generates a CSV file with columns:
- `id`: Puzzle ID
- `puzzle`: Initial puzzle string (81 characters)
- `solution`: Expected solution string
- `clues`: Number of clues
- `difficulty`: Difficulty rating
- `solve_time`: Time taken to solve (seconds)
- `computed_solution`: Solution found by simulated annealing (or ERROR message)

## Algorithm Parameters

Default parameters can be adjusted in `solve_sudoku_simulated_annealing()`:
- `initial_temp=10.0`: Starting temperature
- `cooling_rate=0.999`: Geometric cooling rate (higher = slower cooling)
- `max_iterations=100000`: Maximum iterations per puzzle
- `min_temp=0.01`: Minimum temperature threshold

## Algorithm Limitations

1. **Local Optima**: May get stuck in local minima (though acceptance criterion helps escape)
2. **Convergence**: May not converge to solution within iteration limit
3. **Parameter Sensitivity**: Performance depends on cooling schedule parameters
4. **Initial State**: Random initial state may affect convergence time

## Comparison with Other Methods

### vs. CSP Backtracking
- **Advantage**: Can escape local minima, handles harder puzzles
- **Disadvantage**: Non-deterministic, may not find solution

### vs. Q-Learning
- **Advantage**: No training required, simpler implementation
- **Disadvantage**: No learning across puzzles, each puzzle solved independently

## Future Improvements

1. **Adaptive Cooling**: Adjust cooling rate based on progress
2. **Restart Strategy**: Restart from new random state if stuck
3. **Better Initial State**: Use heuristics to create better initial board
4. **Parameter Tuning**: Optimize temperature schedule for Sudoku
5. **Hybrid Approach**: Combine with constraint propagation for faster convergence

