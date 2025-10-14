#!/usr/bin/env python3
"""
Quick test script to verify data loading works
Run this from project root: python test_loader.py
"""

import sys
sys.path.append('src')

from data.load_data import RetinalDataLoader
import pandas as pd

def main():
    print("🚀 Testing Retinal Disease Data Loader")
    print("="*50)
    
    # Initialize loader with your data path
    loader = RetinalDataLoader(data_root="data/raw")  # Adjust if different
    
    # Test each dataset individually
    print("\n1. Testing APTOS dataset...")
    try:
        aptos_df = loader.load_aptos_dataset()
        print(f"   ✅ Success: {len(aptos_df)} APTOS images loaded")
    except Exception as e:
        print(f"   ❌ APTOS failed: {e}")
    
    print("\n2. Testing OCT5k dataset...")
    try:
        oct5k_df = loader.load_oct5k_dataset()
        print(f"   ✅ Success: {len(oct5k_df)} OCT5k images loaded")
    except Exception as e:
        print(f"   ❌ OCT5k failed: {e}")
    
    print("\n3. Testing MURED dataset...")
    try:
        mured_df = loader.load_mured_dataset()
        print(f"   ✅ Success: {len(mured_df)} MURED images loaded")
    except Exception as e:
        print(f"   ❌ MURED failed: {e}")
    
    print("\n4. Loading all datasets together...")
    datasets = loader.load_all_datasets()
    combined_df = loader.create_combined_dataset(datasets)
    
    if not combined_df.empty:
        stats = loader.get_dataset_statistics(combined_df)
        print(f"\n📊 Final Statistics:")
        print(f"   Total Images: {stats['total_images']}")
        print(f"   Datasets: {stats['datasets']}")
        print(f"   Disease Distribution: {stats['disease_distribution']}")
        
        # Save test results
        combined_df.to_csv("test_dataset_output.csv", index=False)
        print(f"\n✅ Test complete! Saved results to test_dataset_output.csv")
    else:
        print(f"\n❌ No datasets loaded successfully")

if __name__ == "__main__":
    main()
