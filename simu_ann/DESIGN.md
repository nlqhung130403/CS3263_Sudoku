# Simulated Annealing Algorithm Design

## Overview

This document explains the design and implementation of the Simulated Annealing algorithm for solving Sudoku puzzles, including how parallelism is handled to avoid race conditions.

## Algorithm Components

### 1. State Representation

**State**: A 9×9 matrix (board) where:
- Each cell contains an integer from {1, ..., 9}
- Each digit 1-9 appears exactly **nine times** on the board
- Clue cells (pre-filled) are fixed and cannot be modified

**Implementation**: 
- Board stored as a list of lists (9×9)
- Clue positions tracked in a set for fast lookup
- Initial board created by filling empty cells randomly to satisfy digit count constraint

### 2. Cost Function

**Cost**: Number of constraint violations

A violation occurs when:
- Two cells in the same **row** have the same value
- Two cells in the same **column** have the same value  
- Two cells in the same **3×3 box** have the same value

**Implementation**:
- `count_constraint_violations(board)`: Counts all violations
- For each duplicate value, counts `(occurrences - 1)` violations
- Goal: Minimize cost to zero (zero violations = solved puzzle)

### 3. Move Generation

**Move Type**: Swap the contents of two different non-clue cells

**Why Swapping?**
- Preserves digit counts (each digit still appears 9 times)
- Maintains feasibility constraint automatically
- Simple and efficient to implement

**Cell Selection**: Non-uniform, weighted by violations
- Probability of selecting cell `i` ∝ `exp(violations_i)`
- Cells with more violations are more likely to be selected
- Helps focus search on problematic areas

**Implementation**:
- `select_cells_weighted()`: Uses weighted sampling
- `get_cell_violations()`: Counts violations for a specific cell
- Only selects from non-clue cells

### 4. Acceptance Criterion

**Mathematical Formula**:
```
Accept if: U ≤ min{exp([c(B_n) - c(B)] / τ_n), 1}
```

Where:
- `U`: Random uniform deviate from [0, 1]
- `c(B_n)`: Cost of current board
- `c(B)`: Cost of proposed board
- `τ_n`: Temperature at iteration n

**Derivation**:
- `delta_cost = c(B) - c(B_n)` (cost change)
- `c(B_n) - c(B) = -delta_cost`
- `exp([c(B_n) - c(B)] / τ) = exp(-delta_cost / τ)`

**Behavior**:
- If `delta_cost < 0` (cost decreases): Always accept
- If `delta_cost = 0` (cost unchanged): Always accept  
- If `delta_cost > 0` (cost increases): Accept with probability `exp(-delta_cost / τ)`
  - Higher cost increase → lower acceptance probability
  - Higher temperature → higher acceptance probability

**Implementation**:
```python
accept_probability = min(np.exp(-delta_cost / temperature), 1.0)
if delta_cost < 0 or probability(accept_probability):
    # Accept move
```

### 5. Cooling Schedule

**Type**: Geometric cooling schedule

**Formula**: `τ_n = initial_temp × (cooling_rate ^ n)`

**Parameters**:
- `initial_temp = 10.0`: Starting temperature
- `cooling_rate = 0.999`: Geometric decay rate (very slow cooling)
- `min_temp = 0.01`: Minimum temperature threshold

**Behavior**:
- Starts at high temperature (exploration phase)
- Slowly decreases temperature (exploitation phase)
- At zero temperature: Only accepts cost-decreasing moves (greedy)

**Why Slow Cooling?**
- Sudoku has complex constraint structure
- Need time to escape local minima
- Slow cooling allows more exploration

### 6. Initial State Generation

**Process**:
1. Parse puzzle string to identify clue cells
2. Count occurrences of each digit in clues
3. Calculate remaining digits needed: `9 - clue_count[digit]`
4. Fill empty cells randomly with remaining digits
5. Shuffle digits to ensure randomness

**Result**: Feasible board where:
- All clue cells have correct values
- Each digit 1-9 appears exactly 9 times
- May have constraint violations (will be minimized by annealing)

## Parallelism Design

### Architecture

The solver uses **multiprocessing with independent workers**:

```
Main Process
├── Read CSV file
├── Split puzzles into chunks
├── Create worker pool (N workers)
│   ├── Worker 1 → Process chunk 1 → Write temp_file_1.csv
│   ├── Worker 2 → Process chunk 2 → Write temp_file_2.csv
│   └── ...
└── Aggregate results → Final output.csv
```

### Race Condition Prevention

#### 1. Process Isolation

**Design**: Each worker runs in a separate process with its own memory space.

**Why**: 
- Python's `multiprocessing.Pool` creates separate processes (not threads)
- Each process has independent memory
- No shared state between workers

**Implementation**:
```python
with Pool(processes=num_workers) as pool:
    results = pool.map(process_chunk, chunks)
```

#### 2. Separate Temporary Files

**Design**: Each worker writes to its own temporary file.

**Why**:
- File writes are independent
- No concurrent writes to same file
- No file locking needed

**Implementation**:
```python
temp_file = os.path.join(temp_dir, f'chunk_{chunk_id}.csv')
# Each worker writes to its own file
```

#### 3. Deterministic Chunking

**Design**: Chunks are created before parallel processing starts.

**Why**:
- No race conditions in chunk assignment
- Each chunk has unique ID
- Predictable and reproducible

**Implementation**:
```python
chunks = []
for i in range(0, total_puzzles, chunk_size):
    chunk_rows = all_rows[i:i + chunk_size]
    chunks.append((i // chunk_size, chunk_rows, temp_dir))
# All chunks created before pool.map()
```

#### 4. Sequential Aggregation

**Design**: Final result concatenation happens sequentially after all workers finish.

**Why**:
- No concurrent writes to final output file
- Maintains puzzle order
- Simple and safe

**Implementation**:
```python
# After pool.map() completes
for temp_file in sorted(temp_files):
    # Read and write sequentially
```

#### 5. No Shared State

**Design**: Unlike Q-Learning, simulated annealing doesn't require shared state.

**Why**:
- Each puzzle solved independently
- No learning or knowledge transfer
- No synchronization needed

**Comparison with Q-Learning**:
- **Q-Learning**: Shares Q-table → needs locks
- **Simulated Annealing**: No shared state → no locks needed

### Performance Characteristics

**Scalability**: 
- Linear speedup with number of cores
- No lock contention (no shared state)
- Limited only by CPU cores and memory

**Memory**:
- Each worker: ~O(81) for board state
- Minimal memory overhead
- Scales linearly with number of workers

**I/O**:
- Parallel reads: Input CSV read once by main process
- Parallel writes: Each worker writes to temp file
- Sequential aggregation: Final output written sequentially

**Load Balancing**:
- Chunks are approximately equal size
- Simple round-robin distribution
- No dynamic load balancing needed

## Comparison with Other Approaches

### vs. CSP Backtracking

| Aspect | CSP Backtracking | Simulated Annealing |
|--------|------------------|---------------------|
| **Determinism** | Deterministic | Non-deterministic |
| **Guarantee** | Finds solution if exists | May not converge |
| **Local Optima** | Avoids via backtracking | Can escape via acceptance |
| **Complexity** | Exponential worst case | Depends on parameters |

### vs. Q-Learning

| Aspect | Q-Learning | Simulated Annealing |
|--------|------------|---------------------|
| **Learning** | Learns across puzzles | No learning |
| **Shared State** | Shared Q-table | No shared state |
| **Synchronization** | Needs locks | No locks needed |
| **Training** | Requires training | No training needed |

## Algorithm Limitations

1. **Non-deterministic**: May produce different results on different runs
2. **Convergence**: May not converge to solution within iteration limit
3. **Parameter Sensitivity**: Performance depends on cooling schedule
4. **Local Optima**: May get stuck (though acceptance helps escape)

## Future Improvements

1. **Adaptive Cooling**: Adjust cooling rate based on progress
2. **Restart Strategy**: Restart from new random state if stuck
3. **Better Initial State**: Use heuristics to create better initial board
4. **Hybrid Approach**: Combine with constraint propagation
5. **Parameter Tuning**: Optimize temperature schedule for Sudoku

