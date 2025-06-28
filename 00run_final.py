import subprocess
import os

method = "tddft"
xc_functional = "pbe0"
basis_set = "cc-pVTZ"
folder_path = f"{method}_{basis_set}_{xc_functional}"
geometry_file='geom.txt'
num_atoms=10
unit = 'angstrom'
run_calc = True
collect_data = True  
print(folder_path)

def run_script(script_name, **kwargs):
    cmd = ["python3", script_name]
    for key, value in kwargs.items():
        cmd.append(f"--{key}")
        cmd.append(str(value))
    subprocess.run(cmd)

def collect_and_save_data(method, basis_set, xc_functional):
    """Verify data using the standardized approach"""
    print(f"\n=== Verifying Data for {method}_{basis_set}_{xc_functional} ===")
    
    cmd = ["python3", "collect_data_simple.py", 
           "--method", method, 
           "--basis", basis_set, 
           "--functional", xc_functional]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("Data verification successful!")
        print(result.stdout)
    else:
        print(f"Data verification failed: {result.stderr}")

if run_calc:
    # Generate input files
    run_script("01gen_inputFiles.py", method=method, xc_functional=xc_functional, 
               basis_set=basis_set, folder_path=folder_path, geometry_file=geometry_file, 
               num_atoms=num_atoms, unit=unit)

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

if collect_data:
    # Verify all data using the new standardized system
    collect_and_save_data(method, basis_set, xc_functional)

print(f"\nComplete workflow finished for {folder_path}!")
print(f"Vector data: data_{folder_path}.txt")
print(f"Matrix data: apb_{folder_path}, sqrtamb_{folder_path}")
print(f"Analysis: Use plots_analysis/analysis.py to load and analyze data") 