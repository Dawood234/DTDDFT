import os
import re
import argparse

parser=argparse.ArgumentParser()
parser.add_argument('--folder_path',type=str)
parser.add_argument('--method',type=str)
args=parser.parse_args()

folder_path = args.folder_path
method=args.method

def find_last_instance(file_path, search_string):
    """Finds the last occurrence of a search string in a file."""
    with open(file_path, 'r', errors='ignore') as file:
        lines = file.readlines()
        index = None
        for i, line in enumerate(lines):
            if search_string in line:
                index = i
        return index

def print_lines_after(file_path, start_index, num_lines):
    """Prints a specified number of lines after a given index in a file."""
    with open(file_path, 'r', errors='ignore') as file:
        lines = file.readlines()
        if start_index is not None and start_index + num_lines < len(lines):
            print("".join(lines[start_index + 1:start_index + num_lines + 1]))


search_string = "(A-B)^(1/2)"
num_lines_to_print = 10

# Collect eligible files with specific prefix and suffix
eligible_files = [filename for filename in os.listdir(folder_path) if filename.startswith(f"{method}") and filename.endswith(".out")]

# Sort the eligible files by their numerical index extracted from the filename
sorted_files = sorted(eligible_files, key=lambda x: int(re.search(f"{method}_input_(\d+).out", x).group(1)))

for filename in sorted_files:
    file_path = os.path.join(folder_path, filename)
    last_instance_index = find_last_instance(file_path, search_string)
    if last_instance_index is not None:
        print(f"File: {filename}")
        print_lines_after(file_path, last_instance_index, num_lines_to_print)
        print("\n---\n")

