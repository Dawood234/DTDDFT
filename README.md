# DTDDFT 

This repository contains a rudimentary workflow for running dressed time-dependent density functional theory (DTDDFT) calculations with bond length alternation scanning using NWChem.
This workflow performs systematic BLA scans on molecular systems and analyzes the results using various DTDDFT/DTDA algorithms. 

## Prerequisites

- Python 3.7+
- NumPy
- Pandas
- Matplotlib
- NWChem


## File Structure

### Core Workflow Scripts
- `00run_final.py` - Master orchestration script
- `01gen_inputFiles.py` - Generates NWChem input files for BLA scan
- `02run_nwchem.py` - Executes NWChem calculations with proper ordering
- `03clean.py` - Cleans up temporary files
- `04move_files.py` - Organizes output files into method-specific directories
- `05gs.py` - Extracts ground state energies

### Data Extraction
- `unified_file_processor.py` - **Main extraction script** - processes all data types in single pass
- `coupling_extraction.py` - Extracts Hq1d and Hq2d coupling elements
- `extraction_logic/` - Modular extraction functions:
  - `states_extractor.py` - Excited state energy extraction
  - `matrix_extractor.py` - A+B and sqrt(A-B) matrix extraction  
  - `mo_extractor.py` - Molecular orbital energy extraction

### Analysis Scripts
- `plots_analysis/analysisv2.py` - **Primary analysis script** with simplified class-based architecture
- `plots_analysis/analysis.py` - Original analysis (maintained for reference)

### Data Files
- `geom.txt` - Input molecular geometry
- `ex_data.txt` - Reference experimental data for comparison
- `data_*.txt` - Extracted vector data (energies, MO values, coupling elements)

## Quick Start

### Complete Workflow
```bash
# Run the entire workflow
python 00run_final.py
```
after running it, run one of the analysis files in `plots_analysis` folder to generate the plot.
### Manual Step-by-Step

#### Step 1: Prepare Geometry
Place your molecular geometry in `geom.txt` in standard XYZ format.

#### Step 2: Generate and Run Calculations
```bash
python3 01gen_inputFiles.py  # Generate NWChem inputs
python3 02run_nwchem.py      # Run calculations (may take hours/days :) if you are running it with multiple cores you need to modify the script )
```

#### Step 3: Extract All Data (Unified Approach)
```bash
# Extract states, matrices, and MO energies in single pass
python3 unified_file_processor.py

# Extract coupling elements  
python3 coupling_extraction.py
```

This creates standardized data files (the names are according to the method (tddft vs tda ) basis set and functional ):
- `data_tddft_cc-pVDZ_pbe96.txt` - All vector data (energies, MO values, coupling elements)
- `apb_tddft_cc-pVDZ_pbe96` - A+B matrices
- `sqrtamb_tddft_cc-pVDZ_pbe96` - sqrt(A-B) matrices

#### Step 4: Run Analysis
```bash
cd plots_analysis
python3 analysisv2.py  # Recommended - optimized version
```

## Configuration

### Method and Basis Set Configuration
Edit the configuration in analysis scripts:
```python
basis_list = 'def2-SVP,def2-TZVP,def2-TZVPP,cc-pVDZ,cc-pVTZ'.split(',')
basis = basis_list[3]  # cc-pVDZ  
func = 'pbe96'
framework = 'tddft'  # Overall theoretical framework: 'tddft' or 'tda'
```

### BLA Scan Points
The BLA points are defined in analysis scripts:
```python
bla = np.array([0.125466, 0.110464, 0.0954646, ..., -0.144])
```

### Reference Data Format
Place reference data in `ex_data.txt`:
```
# BLA_value  Bu_energy  Ag_energy
0.125466    5.85       6.75
0.110464    5.82       6.72
...
```

## Analysis

The `analysisv2.py` script supports multiple dressed TDDFT algorithms through a clean class-based interface:

### DTDDFT Methods
- `mazur_dtddft()` - Self-consistent DTDDFT [The diagonalization method followed by Mazur and Włodarczyk](https://onlinelibrary.wiley.com/doi/10.1002/jcc.21102)
- `DTTDFT()` - Standard DTDDFT 

### DTDA Methods (Available in original analysis.py)
- `mazur_DTDA()` - Self-consistent DTDA [The diagonalization method followed by Mazur and Włodarczyk](https://onlinelibrary.wiley.com/doi/10.1002/jcc.21102)
- `DTDA()` - Standard DTDA

