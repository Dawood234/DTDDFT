import os
import argparse
import re

parser = argparse.ArgumentParser(description="Running NWChem calculations")
parser.add_argument('--folder_path', type=str, required=True, help='Path to the folder where files will be saved')
parser.add_argument('--method', type=str, required=True, choices=['tda', 'tddft'], help='Calculation method: tda or tddft')


args = parser.parse_args()

if not os.path.isdir(args.folder_path):
    print(f"Error: Input folder {args.folder_path} not found.")
    exit(1)
folder_path = args.folder_path
folder_path = folder_path.strip().replace(" ", "")

input_files = []
for filename in os.listdir(folder_path):
    if filename.startswith(f"{args.method}_input_") and filename.endswith(".nw"):
        input_files.append(filename)

def get_file_index(filename):
    """Extract numerical index from filename"""
    match = re.search(f"{args.method}_input_(\\d+)\\.nw", filename)
    if match:
        return int(match.group(1))
    return float('inf')  

sorted_input_files = sorted(input_files, key=get_file_index)

print(f"Found {len(sorted_input_files)} input files to process in order:")
for i, filename in enumerate(sorted_input_files):
    index = get_file_index(filename)
    print(f"  {i+1}: {filename} (index: {index})")

# Run NWChem for each input file in the correct order
for i, input_file in enumerate(sorted_input_files):
    output_file = os.path.splitext(input_file)[0] + ".out"
    
    file_index = get_file_index(input_file)
    print(f'\nRunning geometry #{i+1} (index {file_index}): {input_file}')
    command = f"nwchem {os.path.join(folder_path, input_file)} > {os.path.join(folder_path, output_file)}"
    print(f"Command: {command}")
    
    os.system(command)
    print(f'✓ Completed geometry #{i+1}')
print('All NWChem calculations have been run')        


