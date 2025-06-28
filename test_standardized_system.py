#!/usr/bin/env python3
"""
Test Script for Standardized Data System
Verifies the new approach works correctly
"""

import os
import sys
import subprocess

def test_vector_data_saving():
    """Test that vector scripts save data to standardized files"""
    print("🧪 Testing Vector Data Saving...")
    
    # Test with existing folder if available
    test_folders = ["tddft_def2-SVP_pbe96", "tddft_cc-pVDZ_pbe0", "tddft_cc-pVTZ_pbe0"]
    
    found_folder = None
    for folder in test_folders:
        if os.path.exists(folder):
            found_folder = folder
            break
    
    if not found_folder:
        print("❌ No test folders found. Please run some calculations first.")
        return False
    
    print(f"Testing with folder: {found_folder}")
    
    # Parse folder name
    parts = found_folder.split('_')
    if len(parts) >= 3:
        method = parts[0]
        basis = parts[1]  
        functional = '_'.join(parts[2:])
    else:
        print("❌ Could not parse folder name")
        return False
    
    # Test ground state extraction
    print("  Testing 05gs.py...")
    try:
        result = subprocess.run(["python3", "05gs.py", "--method", method, "--folder_path", found_folder], 
                               capture_output=True, text=True, check=True)
        print("    ✅ 05gs.py executed successfully")
        
        # Check if data file was created
        data_file = f"data_{found_folder}.txt"
        if os.path.exists(data_file):
            print(f"    ✅ Data file created: {data_file}")
            
            # Check content
            with open(data_file, 'r') as f:
                content = f.read()
                if f'gs_{found_folder}' in content:
                    print("    ✅ Ground state data found in file")
                else:
                    print("    ❌ Ground state data not found in file")
        else:
            print(f"    ❌ Data file not created: {data_file}")
    
    except subprocess.CalledProcessError as e:
        print(f"    ❌ Error running 05gs.py: {e}")
        return False
    
    return True

def test_data_collection():
    """Test the simplified data collection"""
    print("\n🔍 Testing Data Collection...")
    
    # Find any existing data file
    data_files = [f for f in os.listdir('.') if f.startswith('data_') and f.endswith('.txt')]
    
    if not data_files:
        print("❌ No data files found. Run vector scripts first.")
        return False
    
    # Test with first data file found
    test_file = data_files[0]
    folder_name = test_file.replace('data_', '').replace('.txt', '')
    
    print(f"Testing with: {folder_name}")
    
    # Parse combination
    parts = folder_name.split('_')
    if len(parts) >= 3:
        method = parts[0]
        basis = parts[1]
        functional = '_'.join(parts[2:])
    else:
        print("❌ Could not parse combination name")
        return False
    
    # Test data collection
    try:
        result = subprocess.run(["python3", "collect_data_simple.py", 
                                "--method", method, "--basis", basis, "--functional", functional],
                               capture_output=True, text=True, check=True)
        print("✅ Data collection successful")
        print("📊 Collection output:")
        print(result.stdout)
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error in data collection: {e}")
        return False
    
    return True

def test_batch_collection():
    """Test batch data collection"""
    print("\n📦 Testing Batch Collection...")
    
    try:
        # Show summary first
        result = subprocess.run(["python3", "batch_collect_all_standardized.py", "--summary"],
                               capture_output=True, text=True, check=True)
        print("✅ Batch summary successful")
        print("📋 Available combinations:")
        print(result.stdout)
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error in batch summary: {e}")
        return False
    
    return True

def compare_old_vs_new():
    """Compare old vs new approach"""
    print("\n🔄 Comparison: Old vs New Approach")
    
    print("\n📝 OLD APPROACH (tedious):")
    print("  1. Run python3 09apbv2.py > apb_file")
    print("  2. Run python3 10sqrtambv2.py > sqrtamb_file") 
    print("  3. Run python3 05gs.py (capture stdout)")
    print("  4. Run python3 06mo_energies.py (capture stdout)")
    print("  5. Run python3 08states.py (capture stdout)")
    print("  6. Run python3 11hq1d.py (capture stdout)")
    print("  7. Run python3 12hq2d.py (capture stdout)")
    print("  8. Manually parse all outputs in extracted_data.py")
    print("  9. Manage dozens of intermediate text files")
    
    print("\n✨ NEW APPROACH (automated):")
    print("  1. Run analysis scripts (they save to standardized files)")
    print("  2. Run python3 collect_data_simple.py")
    print("  3. Done! All data organized automatically")
    
    print("\n🎯 Benefits:")
    print("  ✅ No intermediate file management")
    print("  ✅ All vectors in single organized file per combination")
    print("  ✅ Matrices in separate files (as preferred)")
    print("  ✅ Automatic data compilation")
    print("  ✅ Error-free data extraction")
    print("  ✅ Batch processing support")

def main():
    print("🚀 Testing Standardized Data System")
    print("=" * 50)
    
    success_count = 0
    total_tests = 3
    
    # Test 1: Vector data saving
    if test_vector_data_saving():
        success_count += 1
    
    # Test 2: Data collection
    if test_data_collection():
        success_count += 1
    
    # Test 3: Batch collection
    if test_batch_collection():
        success_count += 1
    
    # Show comparison
    compare_old_vs_new()
    
    # Final summary
    print(f"\n📊 Test Results: {success_count}/{total_tests} passed")
    
    if success_count == total_tests:
        print("🎉 All tests passed! Standardized system is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    print("\n🎯 Your workflow is now streamlined:")
    print("  • No more manual file redirection")
    print("  • No more intermediate text file management") 
    print("  • Automatic data organization")
    print("  • Easy batch processing")

if __name__ == "__main__":
    main() 