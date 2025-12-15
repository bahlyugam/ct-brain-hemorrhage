#!/usr/bin/env python3
"""
Script to extract rows present in new CSV file but not in old CSV file.
Useful for identifying new data entries between different versions of a dataset.
"""

import pandas as pd
import os
import argparse

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Extract rows from new CSV that are not present in old CSV',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python extract_new_rows.py --old data/v1.csv --new data/v2.csv --output data/new_rows.csv
  python extract_new_rows.py --old data/v1.csv --new data/v2.csv --output data/new_rows.csv --key-columns id timestamp
        '''
    )
    parser.add_argument('--old', type=str, default='data/metadata/negative_feedback_v2.csv',
                        help='Path to old CSV file (default: data/metadata/negative_feedback_v2.csv)')
    parser.add_argument('--new', type=str, default='data/metadata/negative_feedback_v3.csv',
                        help='Path to new CSV file (default: data/metadata/negative_feedback_v3.csv)')
    parser.add_argument('--output', type=str, default='data/metadata/negative_feedback_new_rows_v3.csv',
                        help='Path to output CSV file (default: data/metadata/negative_feedback_new_rows_v3.csv)')
    parser.add_argument('--key-columns', nargs='+', default=['patient_id', 'instance_no', 'image_url'],
                        help='Column names to use as composite key (default: patient_id instance_no image_url)')

    args = parser.parse_args()

    old_path = args.old
    new_path = args.new
    output_path = args.output
    key_columns = args.key_columns

    print("Reading CSV files...")
    print(f"Loading old file: {old_path}...")
    df_old = pd.read_csv(old_path)
    print(f"  - Rows in old file: {len(df_old)}")

    print(f"Loading new file: {new_path}...")
    df_new = pd.read_csv(new_path)
    print(f"  - Rows in new file: {len(df_new)}")

    # Validate that key columns exist in both dataframes
    print(f"\nUsing key columns for comparison: {key_columns}")

    for col in key_columns:
        if col not in df_old.columns:
            raise ValueError(f"Column '{col}' not found in old file")
        if col not in df_new.columns:
            raise ValueError(f"Column '{col}' not found in new file")

    # Create composite keys for both dataframes
    df_old['_composite_key'] = df_old[key_columns].astype(str).agg('||'.join, axis=1)
    df_new['_composite_key'] = df_new[key_columns].astype(str).agg('||'.join, axis=1)

    # Find rows in new file that are not in old file
    old_keys = set(df_old['_composite_key'])
    new_keys = set(df_new['_composite_key'])
    unique_new_keys = new_keys - old_keys

    print(f"\nAnalysis:")
    print(f"  - Unique keys in old file: {len(old_keys)}")
    print(f"  - Unique keys in new file: {len(new_keys)}")
    print(f"  - New rows (not in old file): {len(unique_new_keys)}")

    # Filter new file to get only rows not present in old file
    df_new_rows = df_new[df_new['_composite_key'].isin(unique_new_keys)].copy()

    # Remove the temporary composite key column
    df_new_rows = df_new_rows.drop(columns=['_composite_key'])

    # Save to output file
    print(f"\nSaving new rows to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_new_rows.to_csv(output_path, index=False)

    print(f"✓ Successfully saved {len(df_new_rows)} new rows to {output_path}")

    # Display summary statistics if data-specific columns exist
    if len(df_new_rows) > 0:
        print("\nSummary of new rows:")

        # Only show column-specific stats if columns exist
        if 'date' in df_new_rows.columns:
            print(f"  - Date range: {df_new_rows['date'].min()} to {df_new_rows['date'].max()}")
        if 'patient_id' in df_new_rows.columns:
            print(f"  - Unique patients: {df_new_rows['patient_id'].nunique()}")
        if 'modelVersion' in df_new_rows.columns:
            print(f"  - Model versions: {df_new_rows['modelVersion'].unique()}")
        if 'primary_negative_feedback_type' in df_new_rows.columns:
            print(f"\nPrimary feedback types distribution:")
            print(df_new_rows['primary_negative_feedback_type'].value_counts())

if __name__ == '__main__':
    main()
