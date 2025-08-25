"""
Simple runner script for KIP Kuliah analysis
Provides easy command-line interface for running the analysis
"""

import argparse
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from main import main


def run_analysis():
    """Run the complete KIP Kuliah analysis"""
    
    parser = argparse.ArgumentParser(description='KIP Kuliah Data Mining Analysis')
    parser.add_argument('--data-path', type=str, 
                       default='Bahan Laporan KIP Kuliah 2022 s.d 2024',
                       help='Path to data folder')
    parser.add_argument('--max-clusters', type=int, default=10,
                       help='Maximum number of clusters to test')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("KIP KULIAH DATA MINING ANALYSIS")
    print("=" * 60)
    print(f"Data path: {args.data_path}")
    print(f"Max clusters: {args.max_clusters}")
    print(f"Verbose: {args.verbose}")
    print()
    
    try:
        # Run main analysis
        main()
        
        print()
        print("=" * 60)
        print("ANALYSIS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("Check the 'results' folder for:")
        print("- Excel file with all results")
        print("- CSV files for individual datasets")
        print("- Summary report")
        print("- Visualization plots")
        
    except Exception as e:
        print(f"ERROR: Analysis failed - {e}")
        sys.exit(1)


if __name__ == "__main__":
    run_analysis()
