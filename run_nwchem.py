import os
import argparse

parser = argparse.ArgumentParser(description="Running NWChem calculations")
parser.add_argument('--folder_path', type=str, required=True, help='Path to the folder where files will be saved')
parser.add_argument('--method', type=str, required=True, choices=['tda', 'tddft'], help='Calculation method: tda or tddft')


args = parser.parse_args()

if not os.path.isdir(args.folder_path):
    print(f"Error: Input folder {args.folder_path} not found.")
    exit(1)
folder_path = args.folder_path
folder_path = folder_path.strip().replace(" ", "")

i=0
for input_file in os.listdir(folder_path):
    if input_file.startswith(f"{args.method}_input_") and input_file.endswith(".nw"):
        output_file = os.path.splitext(input_file)[0] + ".out"


        print(f'running geometry #{i}')
        command = f"nwchem {os.path.join(folder_path, input_file)} > {os.path.join(folder_path, output_file)}"
        print(command)

        os.system(command)
        print(f'Done')
        i+=1
print('All NWChem calculations have been run')        


