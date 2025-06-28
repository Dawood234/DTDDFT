"""
States Extraction Logic

Contains functions to extract excited state energies from NWChem output file content.
Separated from file navigation logic for better modularity and efficiency.
"""

import re
import numpy as np
import os
def extract_excited_states(file_content, method):
    """
    Extract excited state energies (ag and bu) from NWChem output file content.
    
    Args:
        file_content (str): Content of the NWChem output file
        method (str): Method name for identification
        
    Returns:
        dict: Dictionary with 'ag' and 'bu' energy lists
    """
    energies_ag = []
    energies_bu = []
    
    # Regex patterns to match the desired line format
    pattern_ag = r"  Root\s+\d+\s+singlet\s+ag\s+(\d+\.\d+)\s+a\.u\.\s+(\d+\.\d+)\s+eV"
    pattern_bu = r"  Root\s+\d+\s+singlet\s+bu\s+(\d+\.\d+)\s+a\.u\.\s+(\d+\.\d+)\s+eV"
    
    lines = file_content.split('\n')
    
    # Search for ag states
    for line in lines:
        match = re.match(pattern_ag, line)
        if match:
            # Extract the energy in eV and append to the list
            energy = float(match.group(2))
            energies_ag.append(energy)
            break  # Assuming we want only the first occurrence
    
    # Search for bu states  
    for line in lines:
        match = re.match(pattern_bu, line)
        if match:
            # Extract the energy in eV and append to the list
            energy = float(match.group(2))
            energies_bu.append(energy)
            break  # Assuming we want only the first occurrence
    
    return {
        'ag': energies_ag,
        'bu': energies_bu
    }

def save_states_to_data_file(all_ag_energies, all_bu_energies, method, data_file):
    """
    Save excited state energies to the standardized data file.
    
    Args:
        all_ag_energies (list): List of ag energies from all files
        all_bu_energies (list): List of bu energies from all files  
        method (str): Method name
        data_file (str): Path to the data file
    """

    
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
    
    # Add excited state energies
    data_dict[f'ag_{method}'] = np.array(all_ag_energies)
    data_dict[f'bu_{method}'] = np.array(all_bu_energies)
    
    # Write all data back to file
    with open(data_file, 'w') as f:
        f.write(f"# Data file for {os.path.basename(data_file).replace('data_', '').replace('.txt', '')}\n")
        f.write(f"# All vectors stored as columns for easy analysis\n\n")
        for var_name, array_data in data_dict.items():
            f.write(f"{var_name}=np.array({array_data.tolist()})\n")
    
    print(f"  Excited states: Found {len(all_ag_energies)} ag and {len(all_bu_energies)} bu states") 