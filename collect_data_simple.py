#!/usr/bin/env python3
"""
Simplified Data Collection Script
Reads from standardized data files created by the modified analysis scripts
"""

import os
import re
import numpy as np
import argparse

def extract_matrices_from_file(filename):
    """Extract matrices from APB/sqrt(A-B) files"""
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found")
        return None
        
    with open(filename, 'r') as file:
        text = file.read()
    
    if not text or text.strip() == "":
        return None
        
    sections = re.split(r'\n-{3,}\n|\n-{3,}$', text, flags=re.MULTILINE) 
    matrices_text = sections[1:]  
    num_list = []
    
    for i in range(len(matrices_text)):
        if i % 2 != 0 and i != 0:
            secc = sections[i].split('\n')
            if len(secc) >= 5:
                try:
                    line1 = secc[3]
                    line2 = secc[4]
                    numline1 = line1.split()
                    numline2 = line2.split()
                    
                    if len(numline1) >= 4 and len(numline2) >= 4:
                        num = np.array([[float(numline1[2]), float(numline1[3])], 
                                      [float(numline2[2]), float(numline2[3])]])
                        num_list.append(num)
                except (IndexError, ValueError) as e:
                    print(f"Warning: Could not parse matrix section {i}: {e}")
                    continue
    
    return np.array(num_list) if num_list else None

def load_vector_data(data_file):
    """Load all vector data from a data file"""
    if not os.path.exists(data_file):
        print(f"Warning: {data_file} not found")
        return {}
    
    data_dict = {}
    with open(data_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if '=' in line and 'np.array' in line and not line.strip().startswith('#'):
                try:
                    var_name = line.split('=')[0].strip()
                    # Extract array from the line
                    array_str = line.split('np.array(')[1].split(')')[0]
                    array_data = eval(array_str)
                    data_dict[var_name] = np.array(array_data)
                except Exception as e:
                    print(f"Warning: Could not parse line: {line.strip()}: {e}")
                    continue
    
    return data_dict

def collect_combination_data(method, basis, functional):
    """Collect all data for a single method/basis/functional combination"""
    folder_name = f"{method}_{basis}_{functional}"
    print(f"Collecting data for {folder_name}...")
    
    # Load vector data from standardized data file
    data_file = f"data_{folder_name}.txt"
    vector_data = load_vector_data(data_file)
    
    # Load matrix data from existing files
    apb_file = f"apb_{folder_name}"
    sqrtamb_file = f"sqrtamb_{folder_name}"
    
    apb_matrices = extract_matrices_from_file(apb_file)
    
    if method.lower() == 'tda':
        # For TDA, sqrt(A-B) = A+B
        sqrtamb_matrices = apb_matrices
        print(f"  TDA method: Using A+B matrices for sqrt(A-B)")
    else:
        sqrtamb_matrices = extract_matrices_from_file(sqrtamb_file)
    
    # Combine all data
    all_data = vector_data.copy()
    all_data['apb'] = apb_matrices
    all_data['sqrtamb'] = sqrtamb_matrices
    
    return all_data

def format_for_extracted_data(method, basis, functional, data):
    """Format data for inclusion in extracted_data.py"""
    folder_name = f"{method}_{basis}_{functional}"
    
    output_lines = [f"\n# Data for {folder_name}"]
    
    # Vector data
    for var_name, array_data in data.items():
        if var_name not in ['apb', 'sqrtamb'] and array_data is not None:
            output_lines.append(f"{var_name}=np.array({array_data.tolist()})")
    
    # Matrix data
    if data.get('apb') is not None:
        output_lines.append(f"apb_{folder_name}={repr(data['apb'])}")
    
    if data.get('sqrtamb') is not None:
        output_lines.append(f"sqrtamb_{folder_name}={repr(data['sqrtamb'])}")
    
    return '\n'.join(output_lines)

def main():
    parser = argparse.ArgumentParser(description='Collect data from standardized files')
    parser.add_argument('--method', type=str, required=True, help='Method (tddft/tda)')
    parser.add_argument('--basis', type=str, required=True, help='Basis set')
    parser.add_argument('--functional', type=str, required=True, help='Functional')
    parser.add_argument('--format', choices=['dict', 'extracted_data'], default='dict',
                        help='Output format')
    
    args = parser.parse_args()
    
    # Collect data
    data = collect_combination_data(args.method, args.basis, args.functional)
    
    if args.format == 'extracted_data':
        # Format for extracted_data.py
        formatted_output = format_for_extracted_data(args.method, args.basis, args.functional, data)
        print(formatted_output)
    else:
        # Print summary
        print(f"\nData collected for {args.method}_{args.basis}_{args.functional}:")
        for key, value in data.items():
            if value is not None:
                if isinstance(value, np.ndarray):
                    print(f"  {key}: {value.shape} array")
                else:
                    print(f"  {key}: {type(value)}")
            else:
                print(f"  {key}: None (not found)")

if __name__ == "__main__":
    main() 