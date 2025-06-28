import re
import os
import argparse
import numpy as np

parser=argparse.ArgumentParser()
parser.add_argument('--method',type=str,required=True)
parser.add_argument('--folder_path',type=str,required=True)
args=parser.parse_args()

def find_last_occurrence_in_file(filename, pattern):
    last_occurrence = None
    try:
        with open(filename, 'r') as file:
            for line in file:
                if re.search(pattern, line):
                    last_occurrence = line
    except FileNotFoundError:
        print(f"File {filename} not found.")
    return last_occurrence

def extract_number(line):
    match = re.search(r"E=\s*([\d.-]+)D([\d+-]+)", line)
    if match:
        number_str = f"{match.group(1)}E{match.group(2)}"
        number = float(number_str)
        return number
    else:
        return None

def process_files(folder_path, method):
    pattern = r"\s*Vector\s+17\s+Occ="
    extracted_numbers = []

    # Gather all matching files
    files = [f for f in os.listdir(folder_path) if re.match(f"{method}_input_\d+\.out", f)]
    # Sort files based on the numeric part in the filename
    files.sort(key=lambda x: int(re.search(r"(\d+)\.out$", x).group(1)))

    for filename in files:
        full_path = os.path.join(folder_path, filename)
        last_occurrence = find_last_occurrence_in_file(full_path, pattern)
        if last_occurrence:
            number = extract_number(last_occurrence)
            if number is not None:
                extracted_numbers.append(number)

    return extracted_numbers

if __name__ == "__main__":
    folder_path = args.folder_path
    method = args.method
    extracted_numbers = process_files(folder_path, method)
    
    # Create/update the data file for this combination
    data_file = f"data_{folder_path}.txt"

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

    # Add mo17 data
    data_dict['mo17'] = np.array(extracted_numbers)

    # Write all data back to file
    with open(data_file, 'w') as f:
        f.write(f"# Data file for {folder_path}\n")
        f.write(f"# All vectors stored as columns for easy analysis\n\n")
        for var_name, array_data in data_dict.items():
            f.write(f"{var_name}=np.array({array_data.tolist()})\n")

    print(f"mo17 energies saved to {data_file}")
    print(f'mo17=np.array({extracted_numbers})')
