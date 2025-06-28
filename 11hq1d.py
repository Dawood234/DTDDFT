import re
import os
import numpy as np
import argparse

# Define the argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--method', type=str, required=True,
                    choices=['tda', 'tddft'], help='Calculation method: tda or tddft')
parser.add_argument('--basis_set', type=str, required=True, help='Basis set (e.g., cc-pVDZ)')
parser.add_argument('--xc_functional',type=str, required=True)
# Parse the arguments
args = parser.parse_args()
basis_set = args.basis_set
xc_functional = args.xc_functional

def extract_number_from_line(pattern, line):
    # Regex to match the number followed by the specific pattern
    match = re.search(r"([\d.-]+E[+-]\d+)\s+" + pattern, line)
    if match:
        # Convert the scientific notation string directly to float
        return float(match.group(1))
    return None

def process_files(directory, pattern):
    numbers = []  # Initialize a list to store the extracted numbers as floats
    files = [f for f in os.listdir(directory) if re.match(r"\d+\.fcidump", f)]
    files.sort(key=lambda x: int(x.split('.')[0]))

    for filename in files:
        full_path = os.path.join(directory, filename)
        try:
            with open(full_path, 'r') as file:
                for line in file:
                    number = extract_number_from_line(pattern, line)
                    if number is not None:
                        numbers.append(number)  # Number is already a float here
                        break  # Remove this if you want to capture all occurrences in a file
        except FileNotFoundError:
            print(f"File {filename} not found in {directory}.")

    return numbers

if __name__ == "__main__":
    directory = args.method+'_'+basis_set+'_'+xc_functional+'_fcidump'
    pattern = "16\s+15\s+15\s+14"
    numbers = process_files(directory, pattern)
    
    # Create folder name for data file
    folder_name = f"{args.method}_{basis_set}_{xc_functional}"
    data_file = f"data_{folder_name}.txt"
    
    # Read existing data or create new structure
    data_dict = {}
    if os.path.exists(data_file):
        with open(data_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if '=' in line and 'np.array' in line:
                    var_name = line.split('=')[0].strip()
                    # Extract array from the line
                    array_str = line.split('np.array(')[1].split(')')[0]
                    try:
                        array_data = eval(array_str)
                        data_dict[var_name] = np.array(array_data)
                    except:
                        pass

    # Add coupling elements
    data_dict['Hq1d'] = np.array(numbers)

    # Write all data back to file
    with open(data_file, 'w') as f:
        f.write(f"# Data file for {folder_name}\n")
        f.write(f"# All vectors stored as columns for easy analysis\n\n")
        for var_name, array_data in data_dict.items():
            f.write(f"{var_name}=np.array({array_data.tolist()})\n")

    print(f"Hq1d coupling elements saved to {data_file}")
    
    # print("Extracted numbers as floats:")
    print(f"Hq1d=np.array({numbers})")  # Keep for compatibility

