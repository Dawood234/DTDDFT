import os
import glob
import re
import argparse

parser=argparse.ArgumentParser()
parser.add_argument('--method',type=str,required=True)
parser.add_argument('--folder_path',type=str,required=True)
args = parser.parse_args()

method=args.method
output_folder = args.folder_path

energies = {}

patterns = {
    14: r' Vector\s+14\s+Occ=(\d+\.\d+)[Dd]([+-]\d+)\s+E=(-?\d+\.\d+)[Dd]([+-]\d+)\s+',
    15: r' Vector\s+15\s+Occ=(\d+\.\d+)[Dd]([+-]\d+)\s+E=(-?\d+\.\d+)[Dd]([+-]\d+)\s+',
    16: r' Vector\s+16\s+Occ=(\d+\.\d+)[Dd]([+-]\d+)\s+E=(-?\d+\.\d+)[Dd]([+-]\d+)\s+',
}

files = glob.glob(os.path.join(output_folder, "*.out"))

sorted_files = sorted(files, key=lambda x: int(re.search(f"{method}_input_(\\d+).out", x).group(1)))

for file_path in sorted_files:
    with open(file_path, "r") as file:
        lines = file.readlines()
        for vector, pattern in patterns.items():
            if vector not in energies:
                energies[vector] = []
            for line in reversed(lines):
                match = re.match(pattern, line)
                if match:
                    energy = float(match.group(3)) * 10 ** int(match.group(4))
                    energies[vector].append(energy)
                    break  

for vector, energy_list in energies.items():
    print(f"mo{vector}=np.array({energy_list})")

