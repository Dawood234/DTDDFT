import os
import argparse

# Define the argument parser
parser = argparse.ArgumentParser(description="Running NWChem calculations")
parser.add_argument('--folder_path', type=str, required=True, help='Path to the folder where files will be saved')
parser.add_argument('--method', type=str, required=True, choices=['tda', 'tddft'], help='Calculation method: tda or tddft')


args = parser.parse_args()

# Check if the folder exists
if not os.path.isdir(args.folder_path):
    print(f"Error: Input folder {args.folder_path} not found.")
    exit(1)
folder_path = args.folder_path
folder_path = folder_path.strip().replace(" ", "")

i=0
# Get all input files and sort them numerically by index
input_files = []
for input_file in os.listdir(folder_path):
    if input_file.startswith(f"{args.method}_input_") and input_file.endswith(".nw"):
        input_files.append(input_file)

# Sort files numerically by extracting the index from filename
def extract_index(filename):
    # Extract number from filename like "tddft_input_5.nw" -> 5
    return int(filename.split('_')[-1].split('.')[0])

input_files.sort(key=extract_index)

# Run NWChem for each input file in numerical order
for input_file in input_files:
        # Generate clean output file name: "tddft_input_5.nw" → "tddft_5.out"
        base_name = input_file.replace("_input_", "_").replace(".nw", ".out")
        output_file = base_name


        # Run NWChem
        print(f'running geometry #{i}')
        command = f"nwchem {os.path.join(folder_path, input_file)} > {os.path.join(folder_path, output_file)}"
        print(command)

        os.system(command)
        print(f'Done')
        i+=1
print('All NWChem calculations have been run')        


