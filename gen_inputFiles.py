import os
import argparse

parser = argparse.ArgumentParser(description="Generate input files for TDDFT or TDA calculations.")
parser.add_argument('--geometry_file', type=str, required=True, help='Path to the folder where files will be saved')
parser.add_argument('--num_atoms', type=int, required=True, help='number of atoms per geometry')
parser.add_argument('--folder_path', type=str, required=True, help='Path to the folder where files will be saved')
parser.add_argument('--method', type=str, required=True, choices=['tda', 'tddft'], help='Calculation method: tda or tddft')
parser.add_argument('--xc_functional', type=str, required=True, help='Exchange-correlation functional (e.g., b3lyp)')
parser.add_argument('--basis_set', type=str, required=True, help='Basis set (e.g., cc-pVDZ)')
parser.add_argument('--unit', type=str, required=True)

args = parser.parse_args()
atoms_per_geometry=args.num_atoms
folder_path = args.folder_path
folder_path = folder_path.strip().replace(" ", "")

template_tda = f"""\
echo
print debug
start TDA calculation with {args.basis_set} - Geometry {{index}}
geometry units {args.unit}
{{geometries}}
end

basis
 * library {args.basis_set}
end

dft
 xc {args.xc_functional}
end
task dft energy

tddft
 CIS
 nroots 5
 notriplet
end
task tddft energy

fcidump
orbitals molecular
end

task dft fcidump
"""

template_tddft = f"""\
echo
print debug
start TDDFT calculation with {args.basis_set}- Geometry {{index}}
geometry units {args.unit}
{{geometries}}
end

basis
 * library {args.basis_set}
end

dft
 xc {args.xc_functional.strip()}
end
task dft energy

tddft
 nroots 5
 notriplet
end
task tddft energy

fcidump
 orbitals molecular
end

task dft fcidump
"""

template = template_tda if args.method == "tda" else template_tddft


os.makedirs(folder_path, exist_ok=True)




def read_and_split_geometries(filename, atoms_per_geometry):
    with open(filename, 'r') as file:
        lines = [line.strip() for line in file.readlines() if line.strip() != ""]

    geometries = []
    for i in range(0, len(lines), atoms_per_geometry):
        geometry = "\n".join(lines[i:i + atoms_per_geometry])
        geometries.append(geometry)

    return geometries

filename = args.geometry_file
geometries = read_and_split_geometries(filename,atoms_per_geometry)

for i, geometry in enumerate(geometries):
    indented_geometry = "\n".join(["    " + line for line in geometry.strip().split("\n")])
    input_data = template.format(index=i, geometries=indented_geometry)
    
    file_path = os.path.join(folder_path, f"{args.method}_input_{i}.nw")
    
    with open(file_path, "w") as f:
        f.write(input_data)
