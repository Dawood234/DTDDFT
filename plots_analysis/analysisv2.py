"""
Analysis v2 - Simple improvements to original analysis.py
- Clean variable naming (framework instead of method)
- Better documentation
- Fixed plot title issue
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import re

mpl.rcParams['figure.dpi']=100

# Configuration
basis_list='def2-SVP,def2-TZVP,def2-TZVPP,cc-pVDZ,cc-pVTZ'.split(',')
basis = basis_list[3]  # cc-pVDZ
func = 'pbe96'
framework = 'tddft'  # Overall theoretical framework: 'tddft' or 'tda'

print(f"Analysis v2 - Loading data for: {framework}_{basis}_{func}")

# BLA array 
bla = np.array([0.125466, 0.110464, 0.0954646, 0.0803673, 0.0652275, 0.0500952, 0.0349265, 0.0198214, 0.00452051,
                -0.0107809, -0.0138626, -0.0200338, -0.0230615, -0.0262975, -0.0295307, -0.0327402, -0.0359374, 
                -0.0390515, -0.0423162, -0.0457886, -0.0490522, -0.0523145, -0.0556133, -0.062, -0.075, -0.088, 
                -0.099, -0.111, -0.122, -0.134, -0.144])

# Data loading functions
def load_vector_data(data_file):
    """Load all vector data from a standardized data file"""
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
                    array_str = line.split('np.array(')[1].split(')')[0]
                    array_data = eval(array_str)
                    data_dict[var_name] = np.array(array_data)
                except Exception as e:
                    print(f"Warning: Could not parse line: {line.strip()}: {e}")
                    continue
    return data_dict

def get_vector_data(data_dict, possible_names):
    """Get vector data by trying multiple possible names"""
    for name in possible_names:
        if name in data_dict:
            return data_dict[name]
    print(f"Warning: None of {possible_names} found in data")
    return None

# Load data
folder_name = f"{framework}_{basis}_{func}"
data_file = f"../data_{folder_name}.txt"
vector_data = load_vector_data(data_file)

# Get all vectors
gs = get_vector_data(vector_data, [f'gs_{folder_name}', 'gs'])
mo14_vec = get_vector_data(vector_data, ['mo14'])
mo15_vec = get_vector_data(vector_data, ['mo15'])  
mo16_vec = get_vector_data(vector_data, ['mo16'])
mo17_vec = get_vector_data(vector_data, ['mo17'])
ag_tddft_vec = get_vector_data(vector_data, [f'ag_{framework}', 'ag'])
bu_tddft_vec = get_vector_data(vector_data, [f'bu_{framework}', 'bu'])
Hq1d_vec = get_vector_data(vector_data, ['Hq1d'])
Hq2d_vec = get_vector_data(vector_data, ['Hq2d'])

# Convert units
au_to_ev = 27.2114
if ag_tddft_vec is not None:
    ag_tddft_vec = ag_tddft_vec / au_to_ev
if bu_tddft_vec is not None:
    bu_tddft_vec = bu_tddft_vec / au_to_ev

print(f"Data loaded successfully for {framework}_{basis}_{func}")

# Load reference data
file_path='../ex_data.txt'
data = pd.read_csv(file_path, sep=r'\s+', comment='#')
if len(data.columns) >= 3:
    data.columns = ['x', 'bu', 'ag']
bla_ex = data['x'].to_numpy()
bu_delta = data['bu'].to_numpy()
ag_delta = data['ag'].to_numpy()

# Simple plotting function
def plot_comparison(algorithm_name='mazur_dtddft'):
    """Plot comparison between TDDFT and reference data"""
    plt.figure(figsize=(10, 6))
    
    # Shift energies relative to ground state at BLA=0
    ag_shifted = ag_tddft_vec + gs - gs[0]
    bu_shifted = bu_tddft_vec + gs - gs[0]
    
    # Plot results
    plt.plot(bla_ex, bu_delta, marker='o', markerfacecolor='None', 
             label='Bu(Ref.)', color='k', markevery=4)
    plt.plot(bla_ex, ag_delta, label='Ag(Ref.)', marker='o', 
             color='k', markevery=4)
    plt.plot(bla, ag_shifted*au_to_ev, label='Ag(ATDDFT)', color='b', 
             marker='^', markevery=4, markerfacecolor='b', linestyle='dashed')
    plt.plot(bla, bu_shifted*au_to_ev, label='Bu(ATDDFT)', color='b', 
             marker='^', markevery=4, markerfacecolor='None', linestyle='dashed')
    
    plt.gca().invert_xaxis()
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.minorticks_on()
    plt.grid(True)
    plt.legend(fontsize=13, loc=(0.03,0.01))
    plt.xlabel(r'BLA(${\AA}$)', fontsize=15)
    plt.ylabel('E (eV)', fontsize=15)
    plt.title(f'{algorithm_name}: {basis}, {func}')
    plt.show()
    
    return ag_shifted, bu_shifted

# Run analysis
if framework == 'tddft':
    ag_shifted, bu_shifted = plot_comparison('mazur_dtddft')
    print("Analysis v2 complete!")
else:
    print(f"Framework '{framework}' not fully implemented in v2") 