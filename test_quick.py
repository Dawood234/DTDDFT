#!/usr/bin/env python3
"""Quick test to verify analysisv2.py works"""

import sys
import os

# Add plots_analysis to path
sys.path.append('plots_analysis')

try:
    print("Testing imports...")
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    print(" Basic imports OK")
    
    from analysisv2 import AnalysisConfig, AnalysisInterface
    print(" analysisv2 imports OK")
    
    # Test configuration
    config = AnalysisConfig()
    print(f" Configuration created: {config.folder_name}")
    
    # Check file paths
    files_to_check = [
        ("Data", config.data_file_path),
        ("APB", config.apb_file_path), 
        ("sqrt(A-B)", config.sqrtamb_file_path)
    ]
    
    for name, path in files_to_check:
        if os.path.exists(path):
            print(f" {name} file found: {path}")
        else:
            print(f"{name} file missing: {path}")
    
    print("\nTesting data loading...")
    from analysisv2 import DataLoader
    loader = DataLoader(config)
    
    # Test vector data loading
    vector_data = loader.load_vector_data()
    print(f"✅ Vector data loaded: {len(vector_data)} variables")
    for key in list(vector_data.keys())[:5]:  # Show first 5
        print(f"   {key}: length {len(vector_data[key])}")
    
    print("\nAll tests passed! ✅")
    print("Now try: cd plots_analysis && python3 analysisv2.py")
    
except Exception as e:
    print(f" Test failed: {e}")
    import traceback
    traceback.print_exc() 