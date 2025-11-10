# Quick Start Guide: Running the RL Sudoku Solver

## Prerequisites

Make sure you have Python 3.6+ installed and are in the correct directory.

## Basic Usage

### 1. Navigate to the rl directory

```bash
cd CS3263_Sudoku/rl
```

### 2. Run with default settings (Train → Infer)

```bash
python3 project_rl.py
```

This will:
- **Phase 1**: Train on `test_set/sudoku_train_set_random_10k.csv`
- **Phase 2**: Test on `test_set/sudoku_test_set_random_10k.csv` (inference only)
- Auto-detect number of CPU cores for parallel processing
- Create output file in `results/` directory with timestamp
- Measure solve time independently for each test puzzle

### 3. Run with custom training and test files

```bash
python3 project_rl.py \
    --train ../test_set/sudoku_train_set_random_10k.csv \
    --test ../test_set/sudoku_test_set_random_10k.csv
```

### 4. Limit number of puzzles (for testing)

```bash
# Train on first 100 puzzles, test on first 50 puzzles
python3 project_rl.py \
    --train ../test_set/sudoku_train_set_random_10k.csv \
    --test ../test_set/sudoku_test_set_random_10k.csv \
    --max-train 100 \
    --max-test 50
```

### 5. Specify number of workers

```bash
# Use 4 parallel workers
python3 project_rl.py --workers 4
```

### 6. Custom output file

```bash
python3 project_rl.py --output results/my_results.csv
```

## Complete Example

```bash
# Train on 1000 puzzles, test on 500 puzzles using 8 workers
python3 project_rl.py \
    --train ../test_set/sudoku_train_set_random_10k.csv \
    --test ../test_set/sudoku_test_set_random_10k.csv \
    --max-train 1000 \
    --max-test 500 \
    --workers 8 \
    --output results/train1000_test500.csv
```

## Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--train` | Path to training CSV file | `test_set/sudoku_train_set_random_10k.csv` |
| `--test` | Path to test CSV file | `test_set/sudoku_test_set_random_10k.csv` |
| `--max-train` | Maximum number of puzzles to train on | All puzzles |
| `--max-test` | Maximum number of puzzles to test | All puzzles |
| `--workers` | Number of parallel workers | Auto-detect CPU cores |
| `--output` | Output CSV file path | Auto-generated with timestamp |

## Expected Output

The script runs in **two phases**:

**Phase 1: Training**
1. Read training puzzles from CSV file
2. Create shared Q-table (starts empty)
3. Train on puzzles in parallel (updates Q-table)
4. Show training statistics

**Phase 2: Inference**
1. Read test puzzles from CSV file
2. Use trained Q-table (read-only, no updates)
3. Solve each test puzzle independently
4. Measure solve time for each puzzle separately
5. Generate results CSV file
6. Print statistics table

Example output:
```
============================================================
PHASE 1: TRAINING
============================================================
Reading training CSV: test_set/sudoku_train_set_random_10k.csv
Loaded 1000 training puzzles
Auto-detected 16 CPU cores
Creating shared Q-table for training...
Training on 16 chunks with 16 workers...

Training completed:
  Total puzzles trained: 1000
  Successfully solved during training: 850
  Failed during training: 150
  Q-table size: 15234 state-action pairs
  Total visits: 45678

============================================================
PHASE 2: INFERENCE
============================================================
Reading test CSV: test_set/sudoku_test_set_random_10k.csv
Loaded 500 test puzzles
Using trained Q-table with 15234 state-action pairs
Auto-detected 16 CPU cores
Processing 16 chunks with 16 workers (inference only)...
Concatenating results...
Cleaning up temporary files...

Inference results saved to: results/sudoku_test_set_random_10k_rl_results_20241109_123456.csv

==================================================
Final Statistics:
==================================================
Total puzzles processed: 500
Successfully solved: 425
Failed to solve: 75
Correct solutions: 410
Incorrect solutions: 15
Success rate: 85.00%

Processing metrics...
...
```

## Troubleshooting

### Import Errors

If you get import errors, make sure you're in the `rl` directory:

```bash
cd CS3263_Sudoku/rl
python3 project_rl.py --csv ../test_set/sudoku_test_set_random_10k.csv --max 10
```

### File Not Found

Make sure the CSV file path is correct. Use relative paths from the `rl` directory:

```bash
# Correct (from rl directory)
python3 project_rl.py --csv ../test_set/sudoku_test_set_random_10k.csv

# Or use absolute path
python3 project_rl.py --csv "/full/path/to/test_set/sudoku_test_set_random_10k.csv"
```

### Memory Issues

If you run out of memory with large datasets:
- Use `--max` to limit number of puzzles
- Reduce `--workers` to use fewer parallel processes

## Testing with Small Dataset

For quick testing, start with a small number of puzzles:

```bash
python3 project_rl.py \
    --train ../test_set/sudoku_train_set_random_10k.csv \
    --test ../test_set/sudoku_test_set_random_10k.csv \
    --max-train 10 \
    --max-test 5 \
    --workers 2
```

This will:
- Train on 10 puzzles
- Test on 5 puzzles using 2 workers
- Should complete quickly for testing

