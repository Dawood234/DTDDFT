import re
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--method',type=str,required=True)
parser.add_argument('--folder_path',type=str,required=True)
args = parser.parse_args()

folder= args.folder_path
method=args.method

energies_ag = []
energies_bu = []

pattern_ag = r"  Root\s+\d+\s+singlet\s+ag\s+(\d+\.\d+)\s+a\.u\.\s+(\d+\.\d+)\s+eV"
pattern_bu = r"  Root\s+\d+\s+singlet\s+bu\s+(\d+\.\d+)\s+a\.u\.\s+(\d+\.\d+)\s+eV"



out_files = [filename for filename in os.listdir(folder) if filename.endswith(".out")]
for i in range(len(out_files)):
    output_file = os.path.join(folder, f"{method}_input_{i}.out")
    with open(output_file, "r") as f:
        for line in f:
            match = re.match(pattern_ag, line)
            if match:
                energy = float(match.group(2))
                energies_ag.append(energy)
                break  
        f.seek(0)  
        for line in f:
            match = re.match(pattern_bu, line)
            if match:
                energy = float(match.group(2))
                energies_bu.append(energy)
                break 
print(f"ag_{method}=np.array({energies_ag})")
print(f"bu_{method}=np.array({energies_bu})")


