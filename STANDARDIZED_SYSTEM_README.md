# Standardized Data System - Complete Solution

## 🎯 Problem Solved

You were frustrated with having to:
1. Manually redirect script outputs: `python3 09apbv2.py > apb_file`
2. Manage dozens of intermediate text files 
3. Manually parse outputs with the `extract()` function
4. Copy-paste data into `extracted_data.py`

## ✅ Solution: Standardized Data Files

### **New File Organization**

**Vector Data** (all in one file per combination):
- `data_{method}_{basis}_{functional}.txt` - Contains ALL vector data:
  - Ground state energies (`gs_*`)
  - Molecular orbital energies (`mo14`, `mo15`, `mo16`)
  - Excited state energies (`ag_*`, `bu_*`)
  - Coupling elements (`Hq1d`, `Hq2d`)

**Matrix Data** (separate files as preferred):
- `apb_{method}_{basis}_{functional}` - A+B matrices
- `sqrtamb_{method}_{basis}_{functional}` - √(A-B) matrices

## 🔧 Modified Scripts

### **Vector Scripts (now save to standardized files):**

1. **`05gs.py`** - Ground state energies
   - **Before**: Printed to stdout
   - **After**: Saves to `data_*.txt` + prints for compatibility

2. **`06mo_energies.py`** - Molecular orbital energies  
   - **Before**: Printed to stdout
   - **After**: Saves to `data_*.txt` + prints for compatibility

3. **`08states.py`** - Excited state energies
   - **Before**: Printed to stdout
   - **After**: Saves to `data_*.txt` + prints for compatibility

4. **`11hq1d.py`** - Coupling elements (1D)
   - **Before**: Printed to stdout
   - **After**: Saves to `data_*.txt` + prints for compatibility

5. **`12hq2d.py`** - Coupling elements (2D)
   - **Before**: Printed to stdout
   - **After**: Saves to `data_*.txt` + prints for compatibility

### **Matrix Scripts (already worked correctly):**

6. **`09apbv2.py`** - A+B matrices (unchanged)
   - Saves to `apb_{folder_path}` file

7. **`10sqrtambv2.py`** - √(A-B) matrices (unchanged)
   - Saves to `sqrtamb_{folder_path}` file

## 🚀 New Tools

### **1. `collect_data_simple.py` - Single Combination Collector**
```bash
# Collect data for one combination
python3 collect_data_simple.py --method tddft --basis cc-pVDZ --functional pbe0

# Format for extracted_data.py
python3 collect_data_simple.py --method tddft --basis cc-pVDZ --functional pbe0 --format extracted_data
```

### **2. `batch_collect_all_standardized.py` - Batch Processor**
```bash
# See what data is available
python3 batch_collect_all_standardized.py --summary

# Generate complete extracted_data.py from all combinations
python3 batch_collect_all_standardized.py --collect
```

### **3. `00run_final_standardized.py` - Complete Workflow**
```bash
# Run complete workflow with new standardized data saving
python3 00run_final_standardized.py
```

### **4. `test_standardized_system.py` - Verification Tool**
```bash
# Test that everything works correctly
python3 test_standardized_system.py
```

## 📂 Example File Structure

After running calculations for `tddft_cc-pVDZ_pbe0`:

```
data_tddft_cc-pVDZ_pbe0.txt          # All vectors (gs, mo14-16, ag, bu, Hq1d, Hq2d)
apb_tddft_cc-pVDZ_pbe0               # A+B matrices  
sqrtamb_tddft_cc-pVDZ_pbe0           # √(A-B) matrices
```

### **Sample `data_tddft_cc-pVDZ_pbe0.txt`:**
```python
# Data file for tddft_cc-pVDZ_pbe0
# All vectors stored as columns for easy analysis

gs_tddft_cc-pVDZ_pbe0=np.array([-382.123, -382.124, ...])
mo14=np.array([-0.234, -0.235, ...])
mo15=np.array([-0.123, -0.124, ...])
mo16=np.array([0.045, 0.046, ...])
ag_tddft=np.array([3.45, 3.46, ...])
bu_tddft=np.array([4.12, 4.13, ...])
Hq1d=np.array([0.001, 0.002, ...])
Hq2d=np.array([0.003, 0.004, ...])
```

## 🔄 Workflow Comparison

### **OLD Workflow (Manual & Error-Prone):**
```bash
# For each combination:
python3 05gs.py --method tddft --folder_path tddft_cc-pVDZ_pbe0 > gs_output.txt
python3 06mo_energies.py --method tddft --folder_path tddft_cc-pVDZ_pbe0 > mo_output.txt
python3 08states.py --method tddft --folder_path tddft_cc-pVDZ_pbe0 > states_output.txt
python3 09apbv2.py --folder_path tddft_cc-pVDZ_pbe0 > apb_tddft_cc-pVDZ_pbe0
python3 10sqrtambv2.py --folder_path tddft_cc-pVDZ_pbe0 --method tddft > sqrtamb_tddft_cc-pVDZ_pbe0
python3 11hq1d.py --method tddft --basis_set cc-pVDZ --xc_functional pbe0 > hq1d_output.txt
python3 12hq2d.py --method tddft --basis_set cc-pVDZ --xc_functional pbe0 > hq2d_output.txt

# Then manually parse all outputs in extracted_data.py...
```

### **NEW Workflow (Automated):**
```bash
# Run calculations (scripts automatically save to standardized files)
python3 00run_final_standardized.py

# Or collect from existing data
python3 batch_collect_all_standardized.py --collect

# Done! extracted_data_auto.py is ready to use
```

## 📊 Benefits Achieved

### **1. No More File Management Headaches**
- ❌ **Before**: Dozens of intermediate files to manage
- ✅ **After**: Clean, organized data files

### **2. No More Manual Redirection**
- ❌ **Before**: `python3 script.py > output_file` for every script
- ✅ **After**: Scripts automatically save data

### **3. Error-Free Data Collection**
- ❌ **Before**: Manual copy-paste with typos and formatting errors
- ✅ **After**: Automatic parsing and compilation

### **4. Batch Processing**
- ❌ **Before**: Process one combination at a time
- ✅ **After**: Process all combinations automatically

### **5. Cleaner Workspace**
- ❌ **Before**: Workspace cluttered with intermediate files
- ✅ **After**: Only organized data files

## 🧪 Testing & Verification

Run the test suite to verify everything works:
```bash
python3 test_standardized_system.py
```

This will:
1. Test vector data saving to standardized files
2. Test data collection from standardized files  
3. Test batch processing capabilities
4. Show before/after comparison

## 🎯 Quick Start

### **For New Calculations:**
```bash
# Run complete workflow
python3 00run_final_standardized.py
```

### **For Existing Data:**
```bash
# Check what data is available
python3 batch_collect_all_standardized.py --summary

# Collect all available data
python3 batch_collect_all_standardized.py --collect
```

### **For Single Combination:**
```bash
python3 collect_data_simple.py --method tddft --basis cc-pVDZ --functional pbe0 --format extracted_data
```

## 🎉 Result

Your data extraction workflow is now **completely automated** with:
- ✅ **No manual file redirection**
- ✅ **No intermediate file management** 
- ✅ **Automatic data organization**
- ✅ **Error-free compilation**
- ✅ **Batch processing support**
- ✅ **Clean, maintainable system**

You'll never have to manually manage those text files again! 🎯 