#!/usr/bin/env python3
"""
Debug script to identify the source of reindexing error
"""

import pandas as pd
import os
import glob

def debug_single_file(filepath):
    """Debug a single file to identify column issues."""
    print(f"\n=== Debugging {os.path.basename(filepath)} ===")
    
    try:
        # Try different encodings
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                df = pd.read_csv(filepath, encoding=encoding, low_memory=False)
                print(f"✅ Successfully loaded with {encoding}")
                break
            except UnicodeDecodeError:
                continue
        else:
            print(f"❌ Could not decode {filepath}")
            return
        
        print(f"Shape: {df.shape}")
        print(f"Columns: {len(df.columns)}")
        
        # Check for duplicate columns
        if df.columns.duplicated().any():
            print(f"⚠️ DUPLICATE COLUMNS FOUND:")
            duplicates = df.columns[df.columns.duplicated()]
            print(f"   Duplicates: {duplicates.tolist()}")
            
        # Check column names
        print(f"Column names sample: {df.columns[:10].tolist()}")
        
        # Look for unnamed columns
        unnamed_cols = [col for col in df.columns if 'Unnamed:' in str(col)]
        print(f"Unnamed columns: {len(unnamed_cols)}")
        
        return df
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None

def main():
    base_path = "c:\\laragon\\www\\ML_KIPKuliah\\Bahan Laporan KIP Kuliah 2022 s.d 2024"
    
    # Check pendaftar files
    pendaftar_pattern = os.path.join(base_path, "CSV_Pendaftar", "**", "*.csv")
    pendaftar_files = glob.glob(pendaftar_pattern, recursive=True)
    
    # Check penerima files
    penerima_pattern = os.path.join(base_path, "CSV_Penerima", "*.csv")
    penerima_files = glob.glob(penerima_pattern, recursive=True)
    
    print("=== DEBUGGING PENDAFTAR FILES ===")
    for file in pendaftar_files[:3]:  # Check first 3 files
        debug_single_file(file)
    
    print("\n=== DEBUGGING PENERIMA FILES ===")
    for file in penerima_files[:2]:  # Check first 2 files
        debug_single_file(file)

if __name__ == "__main__":
    main()
