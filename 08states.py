import re
import os
import numpy as np
import matplotlib.pyplot as plt
# from geom import geometries # this needs to be fixed in the next version
import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--method',type=str,required=True)
parser.add_argument('--folder_path',type=str,required=True)
args = parser.parse_args()

folder= args.folder_path
method=args.method

energies_ag = []
energies_bu = []

# Regex pattern to match the desired line format
pattern_ag = r"  Root\s+\d+\s+singlet\s+ag\s+(\d+\.\d+)\s+a\.u\.\s+(\d+\.\d+)\s+eV"
pattern_bu = r"  Root\s+\d+\s+singlet\s+bu\s+(\d+\.\d+)\s+a\.u\.\s+(\d+\.\d+)\s+eV"



# Iterate over all output files
out_files = [filename for filename in os.listdir(folder) if filename.endswith(".out")]
for i in range(len(out_files)):
    output_file = os.path.join(folder, f"{method}_input_{i}.out")
    with open(output_file, "r") as f:
        # Search for lines matching the pattern for singlet ag
        for line in f:
            match = re.match(pattern_ag, line)
            if match:
                # Extract the last number (energy in eV) and append to the list
                energy = float(match.group(2))
                energies_ag.append(energy)
                break  # Assuming you want only the first occurrence

        # Search for lines matching the pattern for singlet bu
        f.seek(0)  # Reset file pointer to start
        for line in f:
            match = re.match(pattern_bu, line)
            if match:
                # Extract the last number (energy in eV) and append to the list
                energy = float(match.group(2))
                energies_bu.append(energy)
                break  # Assuming you want only the first occurrence

# Print the extracted energies
print(f"ag_{method}=np.array({energies_ag})")
print(f"bu_{method}=np.array({energies_bu})")

# plt.plot(energies_ag,label='Ag')
# plt.plot(energies_bu,label='Bu')
# plt.legend()
# plt.show()
