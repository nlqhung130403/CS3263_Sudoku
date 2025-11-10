#!/usr/bin/env python3
"""
Sort CSV files by a specified column.

This script reads CSV files, sorts them by a specified column,
and writes the sorted data back to the file (or to a new file).
"""

import csv
import os
import argparse
from pathlib import Path


def sort_csv_file(input_file, output_file=None, sort_column='id', numeric=True, reverse=False):
    """
    Sort a CSV file by a specified column.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file (if None, overwrites input_file)
        sort_column: Column name to sort by (default: 'id')
        numeric: Whether to sort numerically (default: True)
        reverse: Whether to sort in reverse order (default: False)
    
    Returns:
        True if successful, False otherwise
    """
    if not os.path.exists(input_file):
        print(f"Error: File not found: {input_file}", file=os.sys.stderr)
        return False
    
    # Read the CSV file
    rows = []
    header = None
    
    try:
        with open(input_file, 'r', newline='') as infile:
            reader = csv.reader(infile)
            header = next(reader, None)
            
            if header is None:
                print(f"Error: Empty file: {input_file}", file=os.sys.stderr)
                return False
            
            # Find the sort column index
            try:
                sort_index = header.index(sort_column)
            except ValueError:
                print(f"Error: Column '{sort_column}' not found in {input_file}", file=os.sys.stderr)
                print(f"Available columns: {', '.join(header)}", file=os.sys.stderr)
                return False
            
            # Read all rows
            for row in reader:
                if row:  # Skip empty rows
                    rows.append(row)
        
        # Sort rows
        def sort_key(row):
            if len(row) <= sort_index:
                return float('inf') if not reverse else float('-inf')
            
            value = row[sort_index]
            if numeric:
                try:
                    # Try to convert to float for numeric sorting
                    return float(value) if value else float('inf')
                except ValueError:
                    # If conversion fails, treat as string
                    return value
            else:
                return value
        
        rows.sort(key=sort_key, reverse=reverse)
        
        # Write sorted data
        output_path = output_file if output_file else input_file
        
        with open(output_path, 'w', newline='') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(header)
            writer.writerows(rows)
        
        print(f"Successfully sorted {len(rows)} row(s) by '{sort_column}'")
        if output_path != input_file:
            print(f"Output saved to: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"Error processing {input_file}: {e}", file=os.sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Sort CSV files by a specified column'
    )
    parser.add_argument(
        'files',
        nargs='+',
        help='CSV file(s) to sort'
    )
    parser.add_argument(
        '--column',
        '-c',
        default='id',
        help='Column name to sort by (default: id)'
    )
    parser.add_argument(
        '--output',
        '-o',
        default=None,
        help='Output file path (default: overwrite input file)'
    )
    parser.add_argument(
        '--text',
        '-t',
        action='store_true',
        help='Sort as text instead of numeric (default: numeric)'
    )
    parser.add_argument(
        '--reverse',
        '-r',
        action='store_true',
        help='Sort in reverse order'
    )
    
    args = parser.parse_args()
    
    # Process each file
    success_count = 0
    for file_path in args.files:
        if sort_csv_file(file_path, args.output, args.column, not args.text, args.reverse):
            success_count += 1
    
    if success_count == 0:
        os.sys.exit(1)
    
    print(f"\nProcessed {success_count} file(s) successfully")


if __name__ == "__main__":
    main()

