import csv
import os
import sys
import random  # <--- IMPORTED random module
from collections import defaultdict

# --- CONFIGURATION ---

# This is the large dataset you want to sample from.
# We'll use the 'sudoku_sample.csv' you provided.
# If your full dataset is in a different file (e.g., 'sudoku.csv'),
# just change this path.
SOURCE_CSV_FILE = 'sudoku-3m.csv'

# This is the new, smaller test set file that will be created.
OUTPUT_CSV_FILE = 'sudoku_test_set_random_10k.csv'  # <--- Changed output filename

# The number of puzzles to grab from each category.
PUZZLES_PER_CATEGORY = 10000

# --- END CONFIGURATION ---


def get_difficulty_category(diff_float: float) -> str:
    """
    Categorizes puzzles based on the 'difficulty' float from the CSV.
    (Matches the logic from the evaluation script).
    """
    if diff_float <= 1.0:
        return "Easy"
    elif diff_float <= 3.0:
        return "Medium"
    else:
        return "Hard"


def create_standardized_set():
    """
    Reads the source CSV and writes a new CSV with a balanced number
    of puzzles from each category.
    """
    
    # Use defaultdict to easily create new lists for categories
    puzzles_by_category = defaultdict(list)
    
    header = []
    total_read = 0

    print(f"Starting to build test set from: {SOURCE_CSV_FILE}")

    try:
        with open(SOURCE_CSV_FILE, mode='r', encoding='utf-8') as f_in:
            reader = csv.reader(f_in)
            
            try:
                # Read and store the header row
                header = next(reader)
                
                # Find the column index for 'difficulty' and 'puzzle'
                # This makes the script robust even if column order changes
                if 'difficulty' not in header or 'puzzle' not in header:
                    print(f"Error: CSV must contain 'difficulty' and 'puzzle' columns.")
                    return
                    
                difficulty_idx = header.index('difficulty')
                
            except StopIteration:
                print("Error: The CSV file is empty.")
                return

            # Read all puzzles and categorize them
            for row in reader:
                total_read += 1
                try:
                    difficulty_float = float(row[difficulty_idx])
                    category = get_difficulty_category(difficulty_float)
                    
                    # Add the puzzle to its category list
                    puzzles_by_category[category].append(row)
                    
                except (ValueError, IndexError):
                    print(f"Warning: Skipping bad row: {row}")
                
    except FileNotFoundError:
        print(f"Error: Could not find source file: {SOURCE_CSV_FILE}")
        return
    except Exception as e:
        print(f"An error occurred during reading: {e}")
        return

    print(f"Finished reading {total_read} puzzles.")
    print("--- Source File Counts ---")
    for category, puzzles in puzzles_by_category.items():
        print(f"  Found {len(puzzles)} '{category}' puzzles.")
    
    
    # --- Write the new standardized test set ---
    
    print(f"\nWriting new test set to: {OUTPUT_CSV_FILE}")
    
    total_written = 0
    final_counts = defaultdict(int)
    
    try:
        with open(OUTPUT_CSV_FILE, mode='w', encoding='utf-8', newline='') as f_out:
            writer = csv.writer(f_out)
            
            # Write the header first
            writer.writerow(header)
            
            # Write the puzzles, taking a random sample from each category
            for category, puzzles in puzzles_by_category.items():
                
                # --- MODIFIED LOGIC ---
                # Determine the number of puzzles to sample
                sample_size = min(len(puzzles), PUZZLES_PER_CATEGORY)
                
                # Take a random sample of that size
                # This replaces taking the first N elements
                puzzles_to_write = random.sample(puzzles, sample_size)
                # --- END MODIFIED LOGIC ---
                
                # Write these rows to the new file
                writer.writerows(puzzles_to_write)
                
                final_counts[category] = len(puzzles_to_write)
                total_written += len(puzzles_to_write)

    except IOError as e:
        print(f"Error writing to output file: {e}")
        return

    print(f"\nSuccessfully created new test set with {total_written} puzzles.")
    print("--- Final Test Set Counts ---")
    for category, count in final_counts.items():
        print(f"  Wrote {count} '{category}' puzzles.")
    
    if any(count < PUZZLES_PER_CATEGORY for count in final_counts.values()):
        print(f"\nNote: One or more categories had fewer than {PUZZLES_PER_CATEGORY} available puzzles.")


if __name__ == "__main__":
    create_standardized_set()