import os
import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--method',type=str,required=True)
parser.add_argument('--folder_path',type=str,required=True)
args = parser.parse_args()

folder= args.folder_path
method=args.method

def get_index(filename):
    try:
        return int(filename.split("_")[-1].split(".")[0])
    except ValueError:
        return float('inf')  

file_names = sorted(os.listdir(folder), key=get_index)

energies = []

for filename in file_names:
    if filename.startswith(f"{method}_input_") and filename.endswith(".out"):
        with open(os.path.join(folder, filename), 'r') as file:
            lines = file.readlines()
            for line in reversed(lines):
                if "Total DFT energy" in line:
                    energy_value = float(line.split()[-1])
                    energies.append(energy_value)
                    break  

print(f"gs_{folder}=np.array({energies})")

