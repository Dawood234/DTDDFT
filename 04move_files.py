import os
import shutil
import argparse

# Define the argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--method', type=str, required=True,
                    choices=['tda', 'tddft'], help='Calculation method: tda or tddft')
parser.add_argument('--basis_set', type=str, required=True, help='Basis set (e.g., cc-pVDZ)')
parser.add_argument('--xc_functional',type=str, required=True)
# Parse the arguments
args = parser.parse_args()
xc_functional = args.xc_functional
folder=args.method+'_'+args.basis_set+'_'+xc_functional
print(folder)
current_dir = os.getcwd()
# dirname=f'{folder}'
# Create the "fcidump" folder if it doesn't exist
fcidump_dir = os.path.join(current_dir, f"{folder}_fcidump")
if not os.path.exists(fcidump_dir):
    os.makedirs(fcidump_dir)

# Loop through all files in the current directory
for filename in os.listdir(current_dir):
    file_path = os.path.join(current_dir, filename)

    # Check if it's a file with .fcidump extension
    if os.path.isfile(file_path) and filename.endswith(".fcidump"):
        # Move the file to the "fcidump" folder
        shutil.move(file_path, os.path.join(fcidump_dir, filename))
        print(f"Moved {filename} to {fcidump_dir}")
