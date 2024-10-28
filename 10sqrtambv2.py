import os
import re
import argparse

# Argument parser setup
parser = argparse.ArgumentParser()
parser.add_argument('--folder_path', type=str, required=True, help='Path to the folder containing the files')
parser.add_argument('--method', type=str, required=True, help='Method prefix for file filtering')
args = parser.parse_args()

folder_path = args.folder_path
method = args.method
output_file = 'sqrtamb_'+folder_path

def find_last_instance(file_path, search_string):
    """Finds the last occurrence of a search string in a file."""
    with open(file_path, 'r', errors='ignore') as file:
        lines = file.readlines()
        index = None
        for i, line in enumerate(lines):
            if search_string in line:
                index = i
        return index

def save_lines_after(file_path, start_index, num_lines, output_file):
    """Saves a specified number of lines after a given index in a file to an output file."""
    with open(file_path, 'r', errors='ignore') as file:
        lines = file.readlines()
        if start_index is not None:
            with open(output_file, 'a') as out_file:
                if start_index + num_lines < len(lines):
                    out_file.write("".join(lines[start_index + 1:start_index + num_lines + 1]))
                else:
                    out_file.write("".join(lines[start_index + 1:]))
                out_file.write("\n---\n")  # Separator between different file outputs

search_string = "(A-B)^(1/2)"
num_lines_to_print = 10

# Collect eligible files with specific prefix and suffix
eligible_files = [filename for filename in os.listdir(folder_path) if filename.startswith(f"{method}") and filename.endswith(".out")]

# Sort the eligible files by their numerical index extracted from the filename
sorted_files = sorted(eligible_files, key=lambda x: int(re.search(f"{method}_input_(\d+).out", x).group(1)))

# Ensure the output file is empty before starting
with open(output_file, 'w') as out_file:
    out_file.write("")

for filename in sorted_files:
    file_path = os.path.join(folder_path, filename)
    last_instance_index = find_last_instance(file_path, search_string)
    if last_instance_index is not None:
        with open(output_file, 'a') as out_file:
            out_file.write(f"File: {filename}\n")
        save_lines_after(file_path, last_instance_index, num_lines_to_print, output_file)
