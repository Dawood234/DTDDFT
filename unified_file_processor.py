#!/usr/bin/env python3
"""
Unified File Processor

This script replaces 08states.py, matrix_extraction.py, and mo_extraction.py by:
1. Reading through NWChem output files ONCE
2. Applying all extraction logic (states + matrices + MO energies) to each file
3. Saving results to appropriate output files

This is much more efficient than reading the same large files multiple times.

Usage:
    python unified_file_processor.py --folder_path tddft_cc-pVDZ_pbe0 --method tddft
    python unified_file_processor.py --folder_path tddft_cc-pVDZ_pbe0 --method tddft --extract_types states
    python unified_file_processor.py --folder_path tddft_cc-pVDZ_pbe0 --method tddft --extract_types matrices
    python unified_file_processor.py --folder_path tddft_cc-pVDZ_pbe0 --method tddft --extract_types mos --mo_numbers "14,15,16"
    python unified_file_processor.py --folder_path tddft_cc-pVDZ_pbe0 --method tddft --extract_types all
"""

import os
import re
import argparse
import sys

# Import our extraction logic modules
from extraction_logic.states_extractor import extract_excited_states, save_states_to_data_file
from extraction_logic.matrix_extractor import (
    extract_apb_matrix, extract_sqrtamb_matrix, 
    save_matrix_data, initialize_matrix_files
)
from extraction_logic.mo_extractor import extract_mo_energies_from_content, save_mo_energies_to_data_file

def sort_output_files(folder_path, method):
    """
    Get and sort NWChem output files by their numerical index.
    
    Args:
        folder_path (str): Path to folder containing output files
        method (str): Method name for filtering files
        
    Returns:
        list: Sorted list of output file paths
    """
    try:
        # Get all .out files that match the method pattern
        eligible_files = [filename for filename in os.listdir(folder_path) 
                         if filename.startswith(f"{method}") and filename.endswith(".out")]
        
        if not eligible_files:
            print(f"Warning: No files found with pattern {method}_*.out")
            return []
        
        # Sort by numerical index
        try:
            sorted_files = sorted(eligible_files, 
                                 key=lambda x: int(re.search(f"{method}_input_(\\d+).out", x).group(1)))
        except (AttributeError, ValueError) as e:
            print(f"Warning: Error sorting files by index, using alphabetical sort: {e}")
            sorted_files = sorted(eligible_files)
        
        # Convert to full paths
        file_paths = [os.path.join(folder_path, filename) for filename in sorted_files]
        
        return file_paths
        
    except FileNotFoundError:
        print(f"Error: Folder {folder_path} not found")
        return []

def process_single_file(file_path, method, extract_states=True, extract_matrices=True, extract_mos=True, mo_numbers=None):
    """
    Process a single NWChem output file and extract all requested data.
    
    Args:
        file_path (str): Path to the output file
        method (str): Method name
        extract_states (bool): Whether to extract excited states
        extract_matrices (bool): Whether to extract matrices
        extract_mos (bool): Whether to extract MO energies
        mo_numbers (list): List of MO numbers to extract
        
    Returns:
        dict: Dictionary with extracted data
    """
    filename = os.path.basename(file_path)
    
    try:
        # Read the file content once
        with open(file_path, 'r', errors='ignore') as f:
            file_content = f.read()
        
        results = {}
        
        # Extract excited states if requested
        if extract_states:
            states_data = extract_excited_states(file_content, method)
            results['states'] = states_data
        
        # Extract matrices if requested
        if extract_matrices:
            apb_data = extract_apb_matrix(file_content, filename)
            sqrtamb_data = extract_sqrtamb_matrix(file_content, filename)
            results['matrices'] = {
                'apb': apb_data,
                'sqrtamb': sqrtamb_data
            }
        
        # Extract MO energies if requested
        if extract_mos and mo_numbers:
            mo_data = extract_mo_energies_from_content(file_content, mo_numbers)
            results['mos'] = mo_data
        
        return results
        
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return {}

def main():
    parser = argparse.ArgumentParser(description='Unified extraction of states, matrices, and MO energies from NWChem output files')
    parser.add_argument('--folder_path', type=str, required=True, 
                       help='Path to the folder containing NWChem output files')
    parser.add_argument('--method', type=str, required=True,
                       help='Method name for file filtering and identification')
    parser.add_argument('--extract_types', type=str, choices=['states', 'matrices', 'mos', 'all'], 
                       default='all', help='What to extract (default: all)')
    parser.add_argument('--mo_numbers', type=str, default='14,15,16,17,18',
                       help='Comma-separated list of MO numbers to extract (default: 14,15,16,17,18)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.folder_path):
        print(f"Error: Folder {args.folder_path} does not exist")
        sys.exit(1)
    
    extract_states = args.extract_types in ['states', 'all']
    extract_matrices = args.extract_types in ['matrices', 'all']
    extract_mos = args.extract_types in ['mos', 'all']
    
    # Parse MO numbers
    mo_numbers = []
    if extract_mos:
        try:
            mo_numbers = [int(x.strip()) for x in args.mo_numbers.split(',')]
        except ValueError:
            print(f"Error: Invalid MO numbers format: {args.mo_numbers}")
            sys.exit(1)
    
    print(f"Processing files in {args.folder_path} for method {args.method}")
    extraction_types = []
    if extract_states: extraction_types.append('states')
    if extract_matrices: extraction_types.append('matrices')
    if extract_mos: extraction_types.append(f'MOs({",".join(map(str, mo_numbers))})')
    print(f"Extracting: {', '.join(extraction_types)}")
    
    # Get sorted list of output files
    file_paths = sort_output_files(args.folder_path, args.method)
    
    if not file_paths:
        print("No files found to process")
        sys.exit(1)
    
    # Initialize data structures
    all_ag_energies = []
    all_bu_energies = []
    all_mo_energies = {mo_num: [] for mo_num in mo_numbers}
    apb_file = None
    sqrtamb_file = None
    
    if extract_matrices:
        apb_file, sqrtamb_file = initialize_matrix_files(args.folder_path)
        print(f"Initialized matrix files: {apb_file}, {sqrtamb_file}")
    
    # Process each file once
    matrices_found = {'apb': 0, 'sqrtamb': 0}
    states_found = 0
    mos_found = 0
    
    print(f"Processing {len(file_paths)} files...")
    
    for file_path in file_paths:
        filename = os.path.basename(file_path)
        print(f"  Processing {filename}...")
        
        # Extract all data from this file in one pass
        results = process_single_file(file_path, args.method, extract_states, extract_matrices, extract_mos, mo_numbers)
        
        # Handle states data
        if extract_states and 'states' in results:
            states_data = results['states']
            all_ag_energies.extend(states_data['ag'])
            all_bu_energies.extend(states_data['bu'])
            if states_data['ag'] or states_data['bu']:
                states_found += 1
        
        # Handle matrix data
        if extract_matrices and 'matrices' in results:
            matrix_data = results['matrices']
            
            # Save A+B matrix data
            if matrix_data['apb']:
                save_matrix_data(matrix_data['apb'], apb_file)
                matrices_found['apb'] += 1
            
            # Save sqrt(A-B) matrix data
            if matrix_data['sqrtamb']:
                save_matrix_data(matrix_data['sqrtamb'], sqrtamb_file)
                matrices_found['sqrtamb'] += 1
        
        # Handle MO data
        if extract_mos and 'mos' in results:
            mo_data = results['mos']
            for mo_num in mo_numbers:
                if mo_num in mo_data:
                    all_mo_energies[mo_num].append(mo_data[mo_num])
            if mo_data:
                mos_found += 1
    
    # Save vector data (states + MOs) to standardized file
    data_file = f"data_{args.folder_path}.txt"
    
    if extract_states:
        save_states_to_data_file(all_ag_energies, all_bu_energies, args.method, data_file)
        print(f"States data saved to {data_file}")
    
    if extract_mos:
        save_mo_energies_to_data_file(all_mo_energies, mo_numbers, data_file)
        print(f"MO energies saved to {data_file}")
    
    # Summary
    print(f"\n=== Processing Summary ===")
    print(f"Files processed: {len(file_paths)}")
    
    if extract_states:
        print(f"Excited states found: {states_found} files with states data")
        print(f"  AG energies: {len(all_ag_energies)}")
        print(f"  BU energies: {len(all_bu_energies)}")
    
    if extract_mos:
        print(f"MO energies found: {mos_found} files with MO data")
        for mo_num in mo_numbers:
            mo_count = len(all_mo_energies[mo_num])
            print(f"  MO{mo_num}: {mo_count} values")
    
    if extract_matrices:
        print(f"Matrix data found:")
        print(f"  A+B matrices: {matrices_found['apb']} files")
        print(f"  sqrt(A-B) matrices: {matrices_found['sqrtamb']} files")
    
    success = True
    if extract_states and states_found == 0:
        print("Warning: No excited states found")
        success = False
    if extract_mos and mos_found == 0:
        print("Warning: No MO energies found")
        success = False
    if extract_matrices and (matrices_found['apb'] == 0 or matrices_found['sqrtamb'] == 0):
        print("Warning: Some matrix types not found")
        success = False
    
    if success:
        print("Unified extraction completed successfully!")
    else:
        print("Warning: Some extractions may have failed")
        sys.exit(1)

if __name__ == "__main__":
    main() 