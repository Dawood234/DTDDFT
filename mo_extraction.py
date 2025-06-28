import os
import glob
import re
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--method', type=str, required=True)
parser.add_argument('--folder_path', type=str, required=True)
parser.add_argument('--mo_numbers', type=str, default='14,15,16,17,18', 
                    help='Comma-separated list of MO numbers to extract (default: 14,15,16,17,18)')
args = parser.parse_args()

method = args.method
output_folder = args.folder_path
mo_numbers = [int(x.strip()) for x in args.mo_numbers.split(',')]

def extract_energy_from_line(line):
    """Extract energy from a line containing scientific notation in D format"""
    # Handle both formats: regex match and D-notation
    patterns = [
        r"E=\s*([\d.-]+)D([\d+-]+)",  # Format: E= -2.9171D-01
        r"E=\s*(-?\d+\.\d+)[Dd]([+-]\d+)"  # Alternative format
    ]
    
    for pattern in patterns:
        match = re.search(pattern, line)
        if match:
            number_str = f"{match.group(1)}E{match.group(2)}"
            return float(number_str)
    return None

def find_last_occurrence_in_file(filename, pattern):
    """Find the last occurrence of a pattern in a file"""
    last_occurrence = None
    try:
        with open(filename, 'r') as file:
            for line in file:
                if re.search(pattern, line):
                    last_occurrence = line
    except FileNotFoundError:
        print(f"File {filename} not found.")
    return last_occurrence

def extract_mo_energies(folder_path, method, mo_numbers):
    """Extract all specified MO energies from NWChem output files"""
    energies = {mo: [] for mo in mo_numbers}
    
    # Get and sort all output files
    files = glob.glob(os.path.join(folder_path, "*.out"))
    sorted_files = sorted(files, key=lambda x: int(re.search(f"{method}_input_(\\d+).out", x).group(1)))
    
    print(f"Processing {len(sorted_files)} files for MO energies: {mo_numbers}")
    
    for file_path in sorted_files:
        filename = os.path.basename(file_path)
        
        with open(file_path, "r") as file:
            lines = file.readlines()
            
            for mo_num in mo_numbers:
                # Create pattern for this MO
                pattern = f"\\s*Vector\\s+{mo_num}\\s+Occ="
                
                # Method 1: Search from end of file (more reliable for final values)
                last_occurrence = None
                for line in reversed(lines):
                    if re.search(pattern, line):
                        last_occurrence = line
                        break
                
                if last_occurrence:
                    energy = extract_energy_from_line(last_occurrence)
                    if energy is not None:
                        energies[mo_num].append(energy)
                    else:
                        print(f"Warning: Could not extract energy from line in {filename} for MO{mo_num}")
                        print(f"Line: {last_occurrence.strip()}")
                else:
                    print(f"Warning: MO{mo_num} not found in {filename}")
    
    return energies

def update_data_file(energies, output_folder):
    """Update the standardized data file with MO energies"""
    data_file = f"data_{output_folder}.txt"
    
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
    
    # Add MO energies
    for mo_num, energy_list in energies.items():
        if energy_list:  # Only add if we have data
            data_dict[f'mo{mo_num}'] = np.array(energy_list)
            print(f"Added mo{mo_num} with {len(energy_list)} values")
        else:
            print(f"Warning: No data found for mo{mo_num}")
    
    # Write all data back to file
    with open(data_file, 'w') as f:
        f.write(f"# Data file for {output_folder}\n")
        f.write(f"# All vectors stored as columns for easy analysis\n\n")
        for var_name, array_data in data_dict.items():
            f.write(f"{var_name}=np.array({array_data.tolist()})\n")
    
    return data_file

if __name__ == "__main__":
    print(f"=== Consolidated MO Extraction ===")
    print(f"Method: {method}")
    print(f"Folder: {output_folder}")
    print(f"MO Numbers: {mo_numbers}")
    
    # Extract all MO energies
    energies = extract_mo_energies(output_folder, method, mo_numbers)
    
    # Update data file
    data_file = update_data_file(energies, output_folder)
    
    print(f"\n=== Results ===")
    print(f"MO energies saved to {data_file}")
    
    # Print extracted energies for verification
    for mo_num in mo_numbers:
        if mo_num in energies and energies[mo_num]:
            print(f"mo{mo_num}=np.array({energies[mo_num]})")
        else:
            print(f"mo{mo_num}=np.array([])  # No data found") 