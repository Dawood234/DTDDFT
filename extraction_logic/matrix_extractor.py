"""
Matrix Extraction Logic

Contains functions to extract A+B and sqrt(A-B) matrices from NWChem output file content.
Separated from file navigation logic for better modularity and efficiency.
"""

import re

def find_last_occurrence_in_content(content, search_string):
    """
    Find the last occurrence of a search string in file content.
    
    Args:
        content (str): File content
        search_string (str): String to search for
        
    Returns:
        int or None: Line index of last occurrence, or None if not found
    """
    lines = content.split('\n')
    index = None
    for i, line in enumerate(lines):
        if search_string in line:
            index = i
    return index

def extract_lines_after_index(content, start_index, num_lines):
    """
    Extract a specified number of lines after a given index from content.
    
    Args:
        content (str): File content
        start_index (int): Starting line index
        num_lines (int): Number of lines to extract
        
    Returns:
        str: Extracted lines joined as string
    """
    if start_index is None:
        return ""
    
    lines = content.split('\n')
    if start_index + num_lines < len(lines):
        extracted_lines = lines[start_index + 1:start_index + num_lines + 1]
    else:
        extracted_lines = lines[start_index + 1:]
    
    return '\n'.join(extracted_lines)

def extract_apb_matrix(file_content, filename):
    """
    Extract A+B matrix from NWChem output file content.
    
    Args:
        file_content (str): Content of the NWChem output file
        filename (str): Name of the file for header
        
    Returns:
        str: Formatted matrix data with file header, or empty string if not found
    """
    search_string = "A+B"
    num_lines_to_extract = 9
    
    last_instance_index = find_last_occurrence_in_content(file_content, search_string)
    
    if last_instance_index is not None:
        matrix_data = extract_lines_after_index(file_content, last_instance_index, num_lines_to_extract)
        return f"File: {filename}\n{matrix_data}\n---\n"
    
    return ""

def extract_sqrtamb_matrix(file_content, filename):
    """
    Extract sqrt(A-B) matrix from NWChem output file content.
    
    Args:
        file_content (str): Content of the NWChem output file
        filename (str): Name of the file for header
        
    Returns:
        str: Formatted matrix data with file header, or empty string if not found
    """
    search_string = "(A-B)^(1/2)"
    num_lines_to_extract = 10
    
    last_instance_index = find_last_occurrence_in_content(file_content, search_string)
    
    if last_instance_index is not None:
        matrix_data = extract_lines_after_index(file_content, last_instance_index, num_lines_to_extract)
        return f"File: {filename}\n{matrix_data}\n---\n"
    
    return ""

def save_matrix_data(matrix_data, output_file):
    """
    Save matrix data to output file.
    
    Args:
        matrix_data (str): Matrix data to save
        output_file (str): Path to output file
    """
    if matrix_data.strip():  # Only write if there's data
        with open(output_file, 'a') as f:
            f.write(matrix_data)

def initialize_matrix_files(folder_path):
    """
    Initialize (clear) matrix output files.
    
    Args:
        folder_path (str): Path to the data folder
        
    Returns:
        tuple: (apb_file, sqrtamb_file) paths
    """
    apb_file = f'apb_{folder_path}'
    sqrtamb_file = f'sqrtamb_{folder_path}'
    
    # Clear the files
    with open(apb_file, 'w') as f:
        f.write("")
    with open(sqrtamb_file, 'w') as f:
        f.write("")
    
    return apb_file, sqrtamb_file 