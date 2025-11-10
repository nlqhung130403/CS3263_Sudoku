#!/usr/bin/env python3
"""
Concatenate CSV chunk files from temp_mac_20251110_154624 directory.

This script reads all chunk CSV files from the specified directory,
sorts them by chunk number, and combines them into a single CSV file
with a single header row.
"""

import csv
import os
import glob
import re
import argparse
from pathlib import Path


def extract_chunk_number(filename):
    """
    Extract chunk number from filename like 'chunk_12.csv'.
    
    Args:
        filename: Name of the chunk file
        
    Returns:
        Integer chunk number, or -1 if not found
    """
    match = re.search(r'chunk_(\d+)\.csv', filename)
    if match:
        return int(match.group(1))
    return -1


def concatenate_chunks(input_dir, output_file):
    """
    Concatenate all CSV chunk files from input_dir into output_file.
    
    Args:
        input_dir: Directory containing chunk CSV files
        output_file: Path to output CSV file
    """
    # Find all chunk CSV files
    chunk_pattern = os.path.join(input_dir, 'chunk_*.csv')
    chunk_files = glob.glob(chunk_pattern)
    
    if not chunk_files:
        print(f"No chunk files found in {input_dir}")
        return False
    
    # Sort files by chunk number
    chunk_files.sort(key=lambda f: extract_chunk_number(os.path.basename(f)))
    
    print(f"Found {len(chunk_files)} chunk file(s)")
    print(f"Chunks: {[extract_chunk_number(os.path.basename(f)) for f in chunk_files]}")
    
    # Read header from first file
    header = None
    total_rows = 0
    
    with open(output_file, 'w', newline='') as outfile:
        writer = None
        
        for chunk_file in chunk_files:
            chunk_num = extract_chunk_number(os.path.basename(chunk_file))
            print(f"Processing chunk_{chunk_num}.csv...", end=' ')
            
            with open(chunk_file, 'r') as infile:
                reader = csv.reader(infile)
                
                # Read header from first file
                if header is None:
                    header = next(reader, None)
                    if header:
                        writer = csv.writer(outfile)
                        writer.writerow(header)
                else:
                    # Skip header in subsequent files
                    next(reader, None)
                
                # Write all data rows
                rows_written = 0
                for row in reader:
                    if row:  # Skip empty rows
                        writer.writerow(row)
                        rows_written += 1
                        total_rows += 1
                
                print(f"{rows_written} row(s)")
    
    print(f"\nSuccessfully concatenated {total_rows} total row(s)")
    print(f"Output saved to: {output_file}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Concatenate CSV chunk files into a single CSV file'
    )
    parser.add_argument(
        '--input-dir',
        default='temp_mac_20251110_154624',
        help='Directory containing chunk CSV files (default: temp_mac_20251110_154624)'
    )
    parser.add_argument(
        '--output',
        default=None,
        help='Output CSV file path (default: auto-generated based on input directory)'
    )
    
    args = parser.parse_args()
    
    # Get script directory (results directory)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(script_dir, args.input_dir)
    
    # Check if input directory exists
    if not os.path.exists(input_dir):
        print(f"Error: Input directory not found: {input_dir}", file=os.sys.stderr)
        os.sys.exit(1)
    
    # Generate output filename if not provided
    if args.output:
        output_file = args.output
        if not os.path.isabs(output_file):
            output_file = os.path.join(script_dir, output_file)
    else:
        # Auto-generate output filename based on input directory name
        dir_name = os.path.basename(input_dir.rstrip('/'))
        # Extract timestamp if present (e.g., temp_mac_20251110_154624 -> 20251110_154624)
        timestamp_match = re.search(r'(\d{8}_\d{6})', dir_name)
        if timestamp_match:
            timestamp = timestamp_match.group(1)
            output_file = os.path.join(script_dir, f'sudoku_train_set_random_10k_mac_results_{timestamp}_combined.csv')
        else:
            output_file = os.path.join(script_dir, f'{dir_name}_combined.csv')
    
    # Concatenate chunks
    success = concatenate_chunks(input_dir, output_file)
    
    if not success:
        os.sys.exit(1)


if __name__ == "__main__":
    main()

