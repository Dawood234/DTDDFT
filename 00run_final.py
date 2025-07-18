import subprocess
import os

method = "tddft"
xc_functional = "pbe96"
basis_set = "def2-SVP"
folder_path = f"{method}_{basis_set}_{xc_functional}"
geometry_file='geom.txt'
unit = 'angstrom'
run_calc = True
print(folder_path)

def run_script(script_name, **kwargs):
    cmd = ["python3", script_name]
    for key, value in kwargs.items():
        cmd.append(f"--{key}")
        cmd.append(str(value))
    subprocess.run(cmd)

if run_calc:
    # Generate input files
    run_script("01gen_inputFiles.py", method=method, xc_functional=xc_functional, 
               basis_set=basis_set, folder_path=folder_path, geometry_file=geometry_file, 
               unit=unit)

    # Run NWChem calculations
    run_script("02run_nwchem.py", folder_path=folder_path, method=method)

    # Clean unnecessary files
    run_script("03clean.py", folder_path=folder_path)

    # Move files
    run_script("04move_files.py", method=method, basis_set=basis_set, xc_functional=xc_functional)

    print(f"\nNWChem calculations complete for {folder_path}")

# Data extraction (only if calculation folder exists)
if os.path.exists(folder_path):
    print(f"\nExtracting data for {folder_path}...")
    
    run_script("05gs.py", method=method, folder_path=folder_path)
    
    run_script("unified_file_processor.py", folder_path=folder_path, method=method, extract_types="all")
    
    run_script("coupling_extraction.py", method=method, basis_set=basis_set, xc_functional=xc_functional)
    
    print(f"\nData extraction complete for {folder_path}")
else:
    print(f"\nSkipping data extraction - folder {folder_path} not found")
    print(f"Assuming data was already extracted to data_{folder_path}.txt")

print(f"\nComplete workflow finished for {folder_path}!")
print(f"Vector data: data_{folder_path}.txt")
print(f"Matrix data: apb_{folder_path}, sqrtamb_{folder_path}")
print(f"Analysis: Use plots_analysis/analysis.py to load and analyze data") 