#!/usr/bin/env python3
"""
Sorting script that uses the same lexicographic sorting as HerbDataset.
Reads from input.txt and writes sorted output to output.txt
"""

import os

def sort_text_file(input_file: str = "input.txt", output_file: str = "output.txt"):
    """
    Sort text lines using the same method as HerbDataset class.
    Uses lexicographic (alphabetical) sorting like sorted() function.
    """
    
    # Get the directory of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, input_file)
    output_path = os.path.join(script_dir, output_file)

    # Read input file
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found")
        return

    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Remove whitespace and empty lines, but keep original content
    lines = [line.strip() for line in lines if line.strip()]

    # Sort using the same method as HerbDataset (lexicographic sorting)
    sorted_lines = sorted(lines)

    # Write to output file
    with open(output_path, 'w', encoding='utf-8') as f:
        for line in sorted_lines:
            f.write(line + '\n')

    print(f"Sorted {len(sorted_lines)} lines from {input_file} to {output_file}")
    print(f"Sample of sorted output:")
    for i, line in enumerate(sorted_lines[:5]):
        print(f"  {i}: {line}")
    if len(sorted_lines) > 5:
        print(f"  ... and {len(sorted_lines) - 5} more lines")

if __name__ == "__main__":
    sort_text_file()
