# Reinforcement Learning Sudoku Solver - Design Document

## Overview

This document explains the design of the Q-Learning based Sudoku solver, including the algorithm design, MDP formulation, and parallelism strategy.

## Algorithm Design

### 1. Problem Formulation

Sudoku solving is formulated as a **Reinforcement Learning** problem where:
- An **agent** (Q-learning algorithm) learns to make optimal decisions
- The **environment** is the Sudoku puzzle board
- The **goal** is to fill all cells correctly while satisfying constraints

### 2. MDP Components

#### State Space
- **Representation**: Tuple of 81 integers `(v₁, v₂, ..., v₈₁)` where each `vᵢ ∈ {0,1,2,...,9}`
  - `0` = empty cell
  - `1-9` = filled cell with value
- **Size**: Theoretically 9^81 possible states (practically much smaller due to constraints)
- **Hashability**: Tuple representation makes states hashable for efficient Q-table lookups

#### Action Space
- **Action**: `(row, col, value)` tuple
  - `row, col ∈ {0,1,...,8}` (9x9 grid)
  - `value ∈ {1,2,...,9}`
- **Validity**: Action is valid only if:
  1. Cell `(row, col)` is empty
  2. Placing `value` doesn't violate constraints:
     - No duplicate in same row
     - No duplicate in same column  
     - No duplicate in same 3×3 box
- **Dynamic**: Valid actions depend on current state

#### Transition Model
- **Type**: Deterministic
- **Function**: `T(s, a) = s'` where:
  - `s`: Current state (board configuration)
  - `a`: Action `(row, col, value)`
  - `s'`: Next state (board with value placed at `(row, col)`)
- **Probability**: Always 1.0 (deterministic placement)

#### Reward Function
```
R(s) = {
  +1000  if puzzle is solved correctly
  -10×k  if constraint violation (k = number of conflicts)
  -1     for each step (encourages efficiency)
  -100   if max steps reached without solving
}
```

**Rationale**:
- Large positive reward encourages finding solutions
- Negative rewards discourage violations and inefficiency
- Step penalty encourages finding shorter solution paths

#### Terminal States
- **Success**: Puzzle completely filled and all constraints satisfied
- **Failure**: Maximum steps reached or no valid actions available

### 3. Q-Learning Algorithm

#### Q-Value Function
- **Definition**: `Q(s, a)` = expected cumulative reward from taking action `a` in state `s`
- **Storage**: Dictionary mapping `(state, action)` tuples to Q-values
- **Initialization**: Optimistic initialization using exploration function

#### Update Rule
```
Q(s, a) ← Q(s, a) + α × [r + γ × max Q(s', a') - Q(s, a)]
```

Where:
- `α` (alpha): Learning rate, decreases with visits: `α(n) = 1/(1+n)`
- `γ` (gamma): Discount factor = 0.95
- `r`: Immediate reward
- `s'`: Next state after action `a`
- `max Q(s', a')`: Maximum Q-value over all actions in next state

#### Exploration Strategy
**Epsilon-Greedy**:
- With probability `ε`: choose random valid action (exploration)
- With probability `1-ε`: choose action with highest Q-value (exploitation)
- `ε` decays over episodes: `ε ← ε × 0.995` (minimum 0.01)

**Exploration Function** (alternative):
- For actions visited < `Ne` times: return `Rplus` (optimistic)
- Otherwise: return actual Q-value
- Encourages exploration of less-visited actions

#### Training Process
1. **Initialization**: Create MDP and Q-learning agent
2. **Episode Loop** (default: 500 episodes):
   - Start from initial puzzle state
   - For each step:
     - Select action using epsilon-greedy
     - Apply action to get next state
     - Update Q-value for previous state-action pair
     - Check if terminal (solved or failed)
   - Decay epsilon
3. **Solution Extraction**: Return best solution found across all episodes

### 4. Implementation Details

#### State Generation
- **Lazy Evaluation**: States generated on-demand rather than pre-computing all states
- **Caching**: Transitions and rewards cached for visited states
- **Efficiency**: Only explores reachable states from initial puzzle

#### Constraint Checking
- **Row Check**: Verify no duplicate in row `row`
- **Column Check**: Verify no duplicate in column `col`
- **Box Check**: Verify no duplicate in 3×3 box containing `(row, col)`
- **Efficient**: O(1) per check using direct array access

#### Action Filtering
- Only generate actions for empty cells
- Pre-filter invalid placements before Q-value lookup
- Reduces action space significantly

## Parallelism Design

### Architecture Overview

The solver uses **multiprocessing** to parallelize puzzle solving across multiple CPU cores:

```
Input CSV
    ↓
[Chunk 1] → Worker 1 → Temp File 1
[Chunk 2] → Worker 2 → Temp File 2
[Chunk 3] → Worker 3 → Temp File 3
    ...
[Chunk N] → Worker N → Temp File N
    ↓
[Sequential Aggregation]
    ↓
Output CSV
```

### Race Condition Prevention

#### 1. Shared Q-Table with Synchronization
- **Shared Memory**: Uses `multiprocessing.Manager()` to create shared dictionaries
- **Single Q-Table**: All workers update the same Q-table for knowledge transfer
- **Locks**: Uses `Lock()` to synchronize Q-table updates and prevent race conditions
- **Thread-Safe Operations**: All Q-table reads/writes are protected by locks

#### 2. Synchronized Updates
- **Q-Value Updates**: Protected by lock to ensure atomicity
- **Visit Counts**: Incremented atomically with lock protection
- **Read Operations**: Also protected to ensure consistency
- **Lock Granularity**: Fine-grained locking only around Q-table operations

#### 3. File-Level Isolation
- **Temporary Files**: Each worker writes to `chunk_{id}.csv` in unique temp directory
- **No Concurrent Writes**: Each file written by single process
- **No File Locking**: Not needed due to isolation

#### 4. Deterministic Chunking
- Chunks created **before** parallel processing starts
- Each chunk has unique ID
- No dynamic assignment that could cause conflicts
- Chunk size: `total_puzzles // num_workers`

#### 5. Sequential Aggregation
- Final concatenation happens **after** all workers finish
- Uses `pool.map()` which waits for all workers
- Sorted file names maintain puzzle order
- Single process writes final output (no concurrent writes)

### Parallelism Flow

```python
# 1. Read and chunk (sequential)
all_rows = read_csv(input_file)
chunks = split_into_chunks(all_rows, num_workers)

# 2. Create shared Q-table (sequential)
manager = Manager()
shared_Q = manager.dict()      # Shared Q-values
shared_Nsa = manager.dict()    # Shared visit counts
Q_lock = manager.Lock()        # Lock for synchronization

# 3. Process chunks (parallel)
with Pool(processes=num_workers) as pool:
    results = pool.map(process_chunk, chunks)
    # Each worker:
    #   - Processes its chunk independently
    #   - Updates SHARED Q-table (with lock protection)
    #   - Learns from all puzzles across all chunks
    #   - Writes to its own temp file
    #   - Returns stats

# 4. Aggregate results (sequential)
for temp_file, stats in results:
    concatenate_to_output(temp_file)
    aggregate_stats(stats)
```

### Knowledge Transfer Mechanism

**How learning flows across chunks:**

1. **Shared Q-Table**: All workers read from and write to the same Q-table
   - Worker 1 learns: "Placing 5 at (0,0) is good" → Updates `Q[(state, (0,0,5))]`
   - Worker 2 can immediately use this knowledge when it encounters similar states

2. **Lock Synchronization**:
   ```python
   # Thread-safe Q-value update
   with Q_lock:
       current_q = shared_Q.get((state, action), 0.0)
       new_q = current_q + alpha * (reward + gamma * max_next_q - current_q)
       shared_Q[(state, action)] = new_q
   ```

3. **Knowledge Accumulation**:
   - Early puzzles: Agent explores randomly (high epsilon)
   - Later puzzles: Agent exploits learned Q-values (lower epsilon)
   - All workers benefit from each other's learning

4. **Transfer Learning**:
   - Patterns learned from easy puzzles help solve harder puzzles
   - Common strategies (e.g., "fill cells with fewest options first") emerge
   - Q-values converge to optimal policy across all puzzles

### Performance Characteristics

#### Scalability
- **Linear Speedup**: Up to number of CPU cores
- **Bottleneck**: I/O operations (reading CSV, writing results)
- **Memory**: Each worker loads only its chunk

#### Load Balancing
- **Equal Chunks**: Puzzles divided evenly among workers
- **Variable Solve Times**: Some puzzles harder than others
- **Natural Balancing**: Faster workers finish and wait (acceptable)

#### Resource Usage
- **CPU**: Fully utilized across all cores
- **Memory**: `O(chunk_size)` per worker
- **Disk I/O**: Parallel reads, sequential writes

### Why This Design Works

1. **Shared Learning**: All workers contribute to same Q-table
2. **Lock Protection**: Prevents race conditions in Q-table updates
3. **Knowledge Transfer**: Learning from one puzzle helps solve others
4. **Independent Processing**: Each worker processes its chunk independently
5. **Simple Aggregation**: Final step is trivial concatenation
6. **Fault Tolerance**: One worker failure doesn't corrupt shared Q-table (locks prevent partial updates)

### Lock Contention and Performance

**Potential Bottleneck**: Lock contention when multiple workers update Q-table simultaneously

**Mitigation Strategies**:
1. **Fine-Grained Locking**: Lock only around Q-table operations, not entire episodes
2. **Batch Updates**: Could batch multiple updates (future optimization)
3. **Read-Heavy**: Most operations are reads (less contention)
4. **Natural Load Balancing**: Workers process different puzzles, reducing simultaneous updates

**Performance Trade-off**:
- **Benefit**: Knowledge transfer improves solving performance over time
- **Cost**: Lock overhead slightly reduces parallel efficiency
- **Net Result**: Better overall performance due to improved learning

## Comparison with CSP Approach

| Aspect | CSP (Backtracking) | RL (Q-Learning) |
|--------|-------------------|-----------------|
| **Method** | Systematic search with constraint propagation | Learning-based exploration |
| **Guarantee** | Finds solution if exists | May not converge in limited episodes |
| **Speed** | Fast for easy puzzles | Slower (requires training) |
| **Learning** | No learning, solves each puzzle independently | Learns strategy across episodes |
| **Parallelism** | Each puzzle independent | Each puzzle independent |
| **Memory** | Low (backtracking stack) | Higher (Q-table storage) |

## Limitations and Future Work

### Current Limitations
1. **Convergence**: May not solve all puzzles within episode limit
2. **State Space**: Can't explore all possible states
3. **Reward Shaping**: Requires careful tuning
4. **No Transfer**: Each puzzle learned independently

### Potential Improvements
1. **Deep Q-Networks**: Approximate Q-values with neural networks
2. **Experience Replay**: Store and reuse past experiences
3. **Transfer Learning**: Learn general strategies across puzzles
4. **Curriculum Learning**: Start with easy puzzles, gradually increase difficulty
5. **Reward Shaping**: Better reward structure for faster learning

## Conclusion

The Q-Learning approach provides an interesting alternative to traditional CSP methods, learning strategies through exploration and exploitation. The parallel design ensures efficient utilization of multi-core systems while avoiding race conditions through process isolation and independent file handling.

