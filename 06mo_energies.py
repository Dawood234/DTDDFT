import os
import glob
import re
import argparse
import numpy as np

parser=argparse.ArgumentParser()
parser.add_argument('--method',type=str,required=True)
parser.add_argument('--folder_path',type=str,required=True)
args = parser.parse_args()

method=args.method
output_folder = args.folder_path

# Dictionary to store the extracted energies for each vector
energies = {}

# Regex patterns to match the desired line formats for vectors 14, 15, 16, and 17
patterns = {
    14: r' Vector\s+14\s+Occ=(\d+\.\d+)[Dd]([+-]\d+)\s+E=(-?\d+\.\d+)[Dd]([+-]\d+)\s+',
    15: r' Vector\s+15\s+Occ=(\d+\.\d+)[Dd]([+-]\d+)\s+E=(-?\d+\.\d+)[Dd]([+-]\d+)\s+',
    16: r' Vector\s+16\s+Occ=(\d+\.\d+)[Dd]([+-]\d+)\s+E=(-?\d+\.\d+)[Dd]([+-]\d+)\s+',
}

# Fetch all output files
files = glob.glob(os.path.join(output_folder, "*.out"))

# Sort the files by their numerical index extracted from the filename, incorporating the method name
sorted_files = sorted(files, key=lambda x: int(re.search(f"{method}_input_(\\d+).out", x).group(1)))

# Iterate over sorted output files
for file_path in sorted_files:
    with open(file_path, "r") as file:
        lines = file.readlines()
        for vector, pattern in patterns.items():
            # Initialize the list of energies for the current vector if not already present
            if vector not in energies:
                energies[vector] = []
            # Search for the final instance of the specified line for the current vector
            for line in reversed(lines):
                match = re.match(pattern, line)
                if match:
                    # Extract the energy in normal form and append to the list
                    energy = float(match.group(3)) * 10 ** int(match.group(4))
                    energies[vector].append(energy)
                    break  # Assuming you want only the final occurrence

# Create/update the data file for this combination
data_file = f"data_{output_folder}.txt"

# Read existing data or create new structure
data_dict = {}
if os.path.exists(data_file):
    with open(data_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if '=' in line and 'np.array' in line:
                var_name = line.split('=')[0].strip()
                # Extract array from the line
                array_str = line.split('np.array(')[1].split(')')[0]
                try:
                    array_data = eval(array_str)
                    data_dict[var_name] = np.array(array_data)
                except:
                    pass

# Add molecular orbital energies
for vector, energy_list in energies.items():
    data_dict[f'mo{vector}'] = np.array(energy_list)

# Write all data back to file
with open(data_file, 'w') as f:
    f.write(f"# Data file for {output_folder}\n")
    f.write(f"# All vectors stored as columns for easy analysis\n\n")
    for var_name, array_data in data_dict.items():
        f.write(f"{var_name}=np.array({array_data.tolist()})\n")

print(f"Molecular orbital energies saved to {data_file}")

# Print the extracted energies (keep for compatibility)
for vector, energy_list in energies.items():
    print(f"mo{vector}=np.array({energy_list})")

