import re
import os
import numpy as np
import argparse

# Define the argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--method', type=str, required=True,
                    choices=['tda', 'tddft'], help='Calculation method: tda or tddft')
parser.add_argument('--basis_set', type=str, required=True, help='Basis set (e.g., cc-pVDZ)')
parser.add_argument('--xc_functional', type=str, required=True, help='Exchange-correlation functional')
parser.add_argument('--coupling_types', type=str, default='Hq1d,Hq2d',
                    help='Comma-separated list of coupling types to extract (default: Hq1d,Hq2d)')
# Parse the arguments
args = parser.parse_args()

# Define coupling patterns
COUPLING_PATTERNS = {
    'Hq1d': "16\\s+15\\s+15\\s+14",
    'Hq2d': "17\\s+16\\s+16\\s+15"
}

def extract_number_from_line(pattern, line):
    """Extract number from line matching the pattern"""
    # Regex to match the number followed by the specific pattern
    match = re.search(r"([\d.-]+E[+-]\d+)\s+" + pattern, line)
    if match:
        # Convert the scientific notation string directly to float
        return float(match.group(1))
    return None

def process_files_for_pattern(directory, pattern, coupling_name):
    """Process all fcidump files for a specific coupling pattern"""
    numbers = []  # Initialize a list to store the extracted numbers as floats
    
    if not os.path.exists(directory):
        print(f"Warning: Directory {directory} not found for {coupling_name}")
        return numbers
    
    files = [f for f in os.listdir(directory) if re.match(r"\d+\.fcidump", f)]
    files.sort(key=lambda x: int(x.split('.')[0]))
    
    print(f"Processing {len(files)} fcidump files for {coupling_name}")

    for filename in files:
        full_path = os.path.join(directory, filename)
        try:
            with open(full_path, 'r') as file:
                for line in file:
                    number = extract_number_from_line(pattern, line)
                    if number is not None:
                        numbers.append(number)  # Number is already a float here
                        break  # Take only the first occurrence in each file
        except FileNotFoundError:
            print(f"File {filename} not found in {directory}.")

    return numbers

def extract_coupling_elements(method, basis_set, xc_functional, coupling_types):
    """Extract all specified coupling elements"""
    directory = f"{method}_{basis_set}_{xc_functional}_fcidump"
    results = {}
    
    for coupling_name in coupling_types:
        if coupling_name in COUPLING_PATTERNS:
            pattern = COUPLING_PATTERNS[coupling_name]
            numbers = process_files_for_pattern(directory, pattern, coupling_name)
            results[coupling_name] = numbers
            print(f"Extracted {len(numbers)} values for {coupling_name}")
        else:
            print(f"Warning: Unknown coupling type '{coupling_name}'. Available: {list(COUPLING_PATTERNS.keys())}")
    
    return results

def update_data_file(coupling_results, method, basis_set, xc_functional):
    """Update the standardized data file with coupling elements"""
    folder_name = f"{method}_{basis_set}_{xc_functional}"
    data_file = f"data_{folder_name}.txt"
    
    # Read existing data or create new structure
    data_dict = {}
    if os.path.exists(data_file):
        with open(data_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if '=' in line and 'np.array' in line and not line.strip().startswith('#'):
                    var_name = line.split('=')[0].strip()
                    # Extract array from the line
                    array_str = line.split('np.array(')[1].split(')')[0]
                    try:
                        array_data = eval(array_str)
                        data_dict[var_name] = np.array(array_data)
                    except Exception as e:
                        print(f"Warning: Could not parse existing data line: {line.strip()}: {e}")

    # Add coupling elements
    for coupling_name, numbers in coupling_results.items():
        if numbers:  # Only add if we have data
            data_dict[coupling_name] = np.array(numbers)
            print(f"Added {coupling_name} with {len(numbers)} values")
        else:
            print(f"Warning: No data found for {coupling_name}")

    # Write all data back to file
    with open(data_file, 'w') as f:
        f.write(f"# Data file for {folder_name}\n")
        f.write(f"# All vectors stored as columns for easy analysis\n\n")
        for var_name, array_data in data_dict.items():
            f.write(f"{var_name}=np.array({array_data.tolist()})\n")

    return data_file

if __name__ == "__main__":
    print(f"=== Consolidated Coupling Extraction ===")
    print(f"Method: {args.method}")
    print(f"Basis: {args.basis_set}")
    print(f"Functional: {args.xc_functional}")
    
    # Parse coupling types
    coupling_types = [x.strip() for x in args.coupling_types.split(',')]
    print(f"Coupling types: {coupling_types}")
    
    # Extract coupling elements
    coupling_results = extract_coupling_elements(args.method, args.basis_set, 
                                                args.xc_functional, coupling_types)
    
    # Update data file
    data_file = update_data_file(coupling_results, args.method, args.basis_set, 
                                args.xc_functional)
    
    print(f"\n=== Results ===")
    print(f"Coupling elements saved to {data_file}")
    
    # Print extracted coupling elements for verification
    for coupling_name in coupling_types:
        if coupling_name in coupling_results and coupling_results[coupling_name]:
            print(f"{coupling_name}=np.array({coupling_results[coupling_name]})")
        else:
            print(f"{coupling_name}=np.array([])  # No data found") 