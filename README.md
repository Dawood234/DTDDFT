# DTDDFT Bond Length Alternation (BLA) Scan Analysis

This repository contains scripts for running dressed time-dependent density functional theory (DTDDFT) calculations with bond length alternation scanning using NWChem.

## Overview

The workflow performs systematic BLA scans on molecular systems and analyzes the results using various DTDDFT/DTDA algorithms. It includes automated data extraction, matrix processing, and comparative analysis with reference data.

## Prerequisites

- Python 3.7+
- NumPy
- Pandas
- Matplotlib
- NWChem (for quantum chemistry calculations)

## Installation

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd v1.3_clean_backup_imp
   ```

2. Install Python dependencies:
   ```bash
   pip install numpy pandas matplotlib
   ```

3. Ensure NWChem is installed and accessible in your PATH.

## File Structure

### Core Scripts
- `01gen_inputFiles.py` - Generates NWChem input files for BLA scan
- `02run_nwchem.py` - Executes NWChem calculations
- `03clean.py` - Cleans up temporary files
- `04move_files.py` - Organizes output files
- `05gs.py` - Extracts ground state energies

### Analysis Scripts
- `plots_analysis/analysis.py` - Original analysis with traditional function signatures
- `plots_analysis/analysis_refactored.py` - Refactored version with simplified class methods

### Data Extraction
- `extraction_logic/` - Modular extraction components
  - `mo_extractor.py` - Molecular orbital energy extraction
  - `matrix_extractor.py` - APB and sqrt(A-B) matrix extraction
  - `states_extractor.py` - Excited state data extraction

## Usage

### Step 1: Prepare Geometry
1. Place your molecular geometry in `geom.txt`
2. Ensure the geometry includes BLA coordinate definitions

### Step 2: Generate Input Files
```bash
python3 01gen_inputFiles.py
```
This creates NWChem input files for each BLA point in the scan.

### Step 3: Run Calculations
```bash
python3 02run_nwchem.py
```
Executes all NWChem calculations. This may take considerable time depending on system size and number of BLA points.

### Step 4: Extract Data
The unified file processor automatically extracts all required data:
```bash
python3 unified_file_processor.py
```
This creates:
- `data_tddft_cc-pVDZ_pbe96.txt` - Vector data (energies, MO values, coupling elements)
- `apb_tddft_cc-pVDZ_pbe96` - A+B matrices
- `sqrtamb_tddft_cc-pVDZ_pbe96` - sqrt(A-B) matrices

### Step 5: Run Analysis

#### Option A: Original Analysis
```bash
cd plots_analysis
python3 analysis.py
```

#### Option B: Refactored Analysis (Recommended)
```bash
cd plots_analysis
python3 analysis_refactored.py
```

The refactored version offers cleaner function signatures and easier method switching.

## Configuration

### Basis Sets and Functionals
Edit the configuration variables in analysis scripts:
```python
basis_list = 'def2-SVP,def2-TZVP,def2-TZVPP,cc-pVDZ,cc-pVTZ'.split(',')
basis = basis_list[3]  # cc-pVDZ
func = 'pbe96'
framework = 'tddft'  # or 'tda'
```

### BLA Array
The BLA points are defined in the analysis scripts. Modify as needed:
```python
bla = np.array([0.125466, 0.110464, ..., -0.144])
```

### Reference Data
Place experimental or reference data in `ex_data.txt` with format:
```
BLA_value  Bu_energy  Ag_energy
0.125466   5.85       6.75
...
```

## Analysis Methods

The refactored version supports multiple algorithms:

### DTDDFT Methods
- `'DTTDFT'` - Standard DTDDFT
- `'mazur_dtddft'` - Self-consistent DTDDFT (Mazur method)

### DTDA Methods  
- `'DTDA'` - Standard DTDA
- `'mazur_DTDA'` - Self-consistent DTDA (Mazur method)

### Switching Methods
```python
# Change algorithm by modifying the function call:
plot_dtddft('mazur_dtddft', ag_tddft_vec, bu_tddft_vec, gs)
# or
plot_dtddft('DTDA', ag_tddft_vec, bu_tddft_vec, gs)
```

## Output

### Generated Files
- Vector data files (`data_*.txt`)
- Matrix files (`apb_*`, `sqrtamb_*`)
- CSV results (`dtddft_results_*.csv`)

### Plots
- Comparison plots showing DTDDFT vs reference data
- BLA vs energy curves for different states
- TDDFT comparison data

## Troubleshooting

### Common Issues

1. **Missing data files**: Ensure all extraction steps completed successfully
2. **Matrix loading errors**: Check file paths and permissions
3. **Convergence issues**: Adjust tolerance in Mazur methods
4. **Memory issues**: Large matrix files may require sufficient RAM

### Performance Tips

1. Use the refactored analysis for better performance
2. Class methods avoid redundant parameter passing
3. Matrix data is cached automatically
4. Progress can be monitored through debug output

## File Naming Convention

Files follow the pattern: `{type}_{framework}_{basis}_{functional}`

Examples:
- `data_tddft_cc-pVDZ_pbe96.txt`
- `apb_tddft_cc-pVDZ_pbe96`
- `sqrtamb_tddft_cc-pVDZ_pbe96`

## Contributing

When modifying the code:
1. Keep the original and refactored versions in sync for core functionality
2. Update both analysis scripts when changing algorithms
3. Maintain backward compatibility with existing data files
4. Test with known reference systems

## License

This code is provided for academic and research purposes. 