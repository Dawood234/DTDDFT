"""
MO Extraction Logic

Contains functions to extract molecular orbital energies from NWChem output file content.
"""

import re
import numpy as np

def extract_energy_from_line(line):
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

def extract_mo_energies_from_content(file_content, mo_numbers):

    energies = {}
    lines = file_content.split('\n')
    
    for mo_num in mo_numbers:
        # Create pattern for this MO
        pattern = f"\\s*Vector\\s+{mo_num}\\s+Occ="
        
        # Search from end of file (more reliable for final values)
        last_occurrence = None
        for line in reversed(lines):
            if re.search(pattern, line):
                last_occurrence = line
                break
        
        if last_occurrence:
            energy = extract_energy_from_line(last_occurrence)
            if energy is not None:
                energies[mo_num] = energy
            else:
                print(f"    Warning: Could not extract energy from MO{mo_num} line")
        else:
            print(f"    Warning: MO{mo_num} not found in file")
    
    return energies

def save_mo_energies_to_data_file(all_mo_energies, mo_numbers, data_file):

    import os
    
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
    for mo_num in mo_numbers:
        if mo_num in all_mo_energies and all_mo_energies[mo_num]:
            data_dict[f'mo{mo_num}'] = np.array(all_mo_energies[mo_num])
            print(f"  MO{mo_num}: Found {len(all_mo_energies[mo_num])} values")
        else:
            print(f"  MO{mo_num}: No data found")
    
    # Write all data back to file
    with open(data_file, 'w') as f:
        f.write(f"# Data file for {os.path.basename(data_file).replace('data_', '').replace('.txt', '')}\n")
        f.write(f"# All vectors stored as columns for easy analysis\n\n")
        for var_name, array_data in data_dict.items():
            f.write(f"{var_name}=np.array({array_data.tolist()})\n") 