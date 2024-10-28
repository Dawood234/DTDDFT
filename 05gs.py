import os
import argparse
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
    # Check if the file name matches the pattern
    if filename.startswith(f"{method}_input_") and filename.endswith(".out"):
        # Open the file for reading
        with open(os.path.join(folder, filename), 'r') as file:
            lines = file.readlines()
            # Search for the last instance of "Total DFT energy"
            for line in reversed(lines):
                if "Total DFT energy" in line:
                    energy_value = float(line.split()[-1])
                    energies.append(energy_value)
                    break  # Break after finding the last instance

# Print the list of energies
print(f"gs_{folder}=np.array({energies})")

