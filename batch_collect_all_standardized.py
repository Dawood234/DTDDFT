#!/usr/bin/env python3
"""
Batch Data Collection for Multiple Combinations
Uses the new standardized data file system
"""

import subprocess
import os

class StandardizedDataCollector:
    def __init__(self):
        # Define all combinations you want to collect
        self.combinations = [
            {"method": "tddft", "basis": "cc-pVDZ", "functional": "pbe0"},
            {"method": "tddft", "basis": "cc-pVTZ", "functional": "pbe0"},
            {"method": "tda", "basis": "cc-pVTZ", "functional": "pbe0"},
            {"method": "tddft", "basis": "def2-TZVP", "functional": "pbe0"},
            {"method": "tddft", "basis": "def2-SVP", "functional": "pbe96"},
            # Add more combinations as needed
        ]
    
    def check_combination_files(self, method, basis, functional):
        """Check if required files exist for a combination"""
        folder_name = f"{method}_{basis}_{functional}"
        
        # Check for data file (vectors)
        data_file = f"data_{folder_name}.txt"
        
        # Check for matrix files
        apb_file = f"apb_{folder_name}"
        sqrtamb_file = f"sqrtamb_{folder_name}"
        
        files_status = {
            'data_file': os.path.exists(data_file),
            'apb_file': os.path.exists(apb_file),
            'sqrtamb_file': os.path.exists(sqrtamb_file) or method.lower() == 'tda'
        }
        
        return files_status
    
    def collect_single_combination(self, method, basis, functional):
        """Collect data for a single combination"""
        print(f"\n--- Collecting {method}_{basis}_{functional} ---")
        
        # Check file availability
        files_status = self.check_combination_files(method, basis, functional)
        
        print(f"Files status:")
        for file_type, exists in files_status.items():
            status = "✓" if exists else "✗"
            print(f"  {file_type}: {status}")
        
        if not all(files_status.values()):
            print(f"⚠️  Some files missing for {method}_{basis}_{functional}")
            return None
        
        # Use the simplified collector
        cmd = ["python3", "collect_data_simple.py", 
               "--method", method, 
               "--basis", basis, 
               "--functional", functional,
               "--format", "extracted_data"]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"✓ Data collection successful")
            return result.stdout
        except subprocess.CalledProcessError as e:
            print(f"✗ Error collecting data: {e}")
            return None
    
    def generate_complete_extracted_data(self):
        """Generate a complete extracted_data.py file from all combinations"""
        print("🚀 Generating complete extracted_data.py file...")
        
        # Start with imports and header
        output_lines = [
            "import numpy as np",
            "",
            "# Automatically generated extracted_data.py",
            "# Generated using standardized data collection system",
            "# Vector data from data_*.txt files, matrix data from apb_*/sqrtamb_* files",
            ""
        ]
        
        successful_combinations = []
        failed_combinations = []
        
        for combo in self.combinations:
            method = combo["method"]
            basis = combo["basis"]
            functional = combo["functional"]
            
            data_output = self.collect_single_combination(method, basis, functional)
            
            if data_output:
                output_lines.append(data_output)
                successful_combinations.append(f"{method}_{basis}_{functional}")
            else:
                failed_combinations.append(f"{method}_{basis}_{functional}")
        
        # Write to file
        with open("extracted_data_auto.py", "w") as f:
            f.write('\n'.join(output_lines))
        
        # Summary
        print(f"\n📊 Collection Summary:")
        print(f"✓ Successful: {len(successful_combinations)}")
        for combo in successful_combinations:
            print(f"  - {combo}")
        
        if failed_combinations:
            print(f"✗ Failed: {len(failed_combinations)}")
            for combo in failed_combinations:
                print(f"  - {combo}")
        
        print(f"\n✅ Complete extracted_data.py saved as 'extracted_data_auto.py'")
        
        return successful_combinations, failed_combinations
    
    def show_data_summary(self):
        """Show summary of available data files"""
        print("\n📁 Available Data Files:")
        
        for combo in self.combinations:
            method = combo["method"]
            basis = combo["basis"]
            functional = combo["functional"]
            folder_name = f"{method}_{basis}_{functional}"
            
            files_status = self.check_combination_files(method, basis, functional)
            
            status_str = "✓" if all(files_status.values()) else "⚠️"
            print(f"{status_str} {folder_name}")
            
            if not all(files_status.values()):
                for file_type, exists in files_status.items():
                    if not exists:
                        print(f"    Missing: {file_type}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Batch collect data using standardized system')
    parser.add_argument('--summary', action='store_true', 
                        help='Show summary of available data files')
    parser.add_argument('--collect', action='store_true',
                        help='Collect data from all available combinations')
    
    args = parser.parse_args()
    
    collector = StandardizedDataCollector()
    
    if args.summary:
        collector.show_data_summary()
    elif args.collect:
        collector.generate_complete_extracted_data()
    else:
        print("Use --summary to see available files or --collect to generate extracted_data.py")

if __name__ == "__main__":
    main() 