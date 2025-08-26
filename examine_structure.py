#!/usr/bin/env python3
"""
Debug script to examine actual data structure
"""

import pandas as pd
import os

def examine_file_structure(filepath):
    """Examine the actual structure of a CSV file."""
    print(f"\n=== Examining {os.path.basename(filepath)} ===")
    
    try:
        df = pd.read_csv(filepath, encoding='utf-8', header=None, nrows=5)
        print("First 5 rows without header processing:")
        print(df.iloc[:, :10])  # Show first 10 columns
        
        print("\nLooking for actual header row...")
        for i in range(min(5, len(df))):
            row = df.iloc[i]
            if any(col for col in row if isinstance(col, str) and any(keyword in col.lower() for keyword in ['nama', 'nik', 'pendaftaran', 'alamat'])):
                print(f"Potential header row found at index {i}:")
                print(row.dropna().head(10).tolist())
                break
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")

def main():
    base_path = "c:\\laragon\\www\\ML_KIPKuliah\\Bahan Laporan KIP Kuliah 2022 s.d 2024"
    
    # Examine one pendaftar file
    filepath = os.path.join(base_path, "CSV_Pendaftar", "2022", "Siswa_Pendaftar_SBMPN_2022.csv")
    examine_file_structure(filepath)
    
    # Examine one penerima file
    filepath = os.path.join(base_path, "CSV_Penerima", "penerima KIP Kuliah angkatan 2022.csv")
    examine_file_structure(filepath)

if __name__ == "__main__":
    main()
