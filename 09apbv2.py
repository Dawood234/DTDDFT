import os
import re
import argparse

# Argument parser setup
parser = argparse.ArgumentParser()
parser.add_argument('--folder_path', type=str, required=True, help='Path to the folder containing the files')
args = parser.parse_args()

folder_path = args.folder_path
output_file = 'apb_'+folder_path
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

search_string = "A+B"
num_lines_to_print = 9

# Get all files in the directory, ensuring they are files
file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

# Define a sorting function that safely handles files without a match
def sort_key(file_path):
    match = re.search(r"_(\d+)\.", os.path.basename(file_path))
    if match:
        return int(match.group(1))
    else:
        return float('inf')  # Put files without a matching index at the end

# Sort the files by their numerical index extracted from the filename
sorted_files = sorted(file_paths, key=sort_key)

# Ensure the output file is empty before starting
with open(output_file, 'w') as out_file:
    out_file.write("")

for file_path in sorted_files:
    filename = os.path.basename(file_path)
    last_instance_index = find_last_instance(file_path, search_string)
    if last_instance_index is not None:
        with open(output_file, 'a') as out_file:
            out_file.write(f"File: {filename}\n")
        save_lines_after(file_path, last_instance_index, num_lines_to_print, output_file)
