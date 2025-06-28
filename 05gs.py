import os
import argparse
import numpy as np

parser=argparse.ArgumentParser()
parser.add_argument('--method',type=str,required=True)
parser.add_argument('--folder_path',type=str,required=True)
args = parser.parse_args()

folder= args.folder_path
method=args.method

# Function to extract the index from file name
def get_index(filename):
    try:
        return int(filename.split("_")[-1].split(".")[0])
    except ValueError:
        return float('inf')  # Return a very large number for invalid filenames

# Get the list of file names sorted by their index
file_names = sorted(os.listdir(folder), key=get_index)

# Initialize a list to store the extracted energies
energies = []

# Loop through the sorted file names
for filename in file_names:
    # Check if the file name matches the pattern (new naming: tddft_X.out)
    if filename.startswith(f"{method}_") and filename.endswith(".out") and not "input" in filename:
        # Open the file for reading
        with open(os.path.join(folder, filename), 'r') as file:
            lines = file.readlines()
            # Search for the last instance of "Total DFT energy"
            for line in reversed(lines):
                if "Total DFT energy" in line:
                    energy_value = float(line.split()[-1])
                    energies.append(energy_value)
                    break  # Break after finding the last instance

# Create/update the data file for this combination
data_file = f"data_{folder}.txt"
energies_array = np.array(energies)

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

# Add ground state energies
data_dict[f'gs_{folder}'] = energies_array

# Write all data back to file
with open(data_file, 'w') as f:
    f.write(f"# Data file for {folder}\n")
    f.write(f"# All vectors stored as columns for easy analysis\n\n")
    for var_name, array_data in data_dict.items():
        f.write(f"{var_name}=np.array({array_data.tolist()})\n")

print(f"Ground state energies saved to {data_file}")
print(f"gs_{folder}=np.array({energies})")  # Keep stdout for compatibility

