#!/usr/bin/env python3
"""
Script to extract rows present in negative_feedback_v2.csv but not in negative_feedback.csv
"""

import pandas as pd
import os

# File paths
v1_path = 'data/metadata/negative_feedback.csv'
v2_path = 'data/metadata/negative_feedback_v2.csv'
output_path = 'data/metadata/negative_feedback_new_rows.csv'

print("Reading CSV files...")
print(f"Loading {v1_path}...")
df_v1 = pd.read_csv(v1_path)
print(f"  - Rows in v1: {len(df_v1)}")

print(f"Loading {v2_path}...")
df_v2 = pd.read_csv(v2_path)
print(f"  - Rows in v2: {len(df_v2)}")

# Define key columns to identify unique rows
# Using patient_id, instance_no, and image_url as composite key
key_columns = ['patient_id', 'instance_no', 'image_url']

print(f"\nUsing key columns for comparison: {key_columns}")

# Create composite keys for both dataframes
df_v1['_composite_key'] = df_v1[key_columns].astype(str).agg('||'.join, axis=1)
df_v2['_composite_key'] = df_v2[key_columns].astype(str).agg('||'.join, axis=1)

# Find rows in v2 that are not in v1
v1_keys = set(df_v1['_composite_key'])
v2_keys = set(df_v2['_composite_key'])
new_keys = v2_keys - v1_keys

print(f"\nAnalysis:")
print(f"  - Unique keys in v1: {len(v1_keys)}")
print(f"  - Unique keys in v2: {len(v2_keys)}")
print(f"  - New rows in v2 (not in v1): {len(new_keys)}")

# Filter v2 to get only new rows
df_new_rows = df_v2[df_v2['_composite_key'].isin(new_keys)].copy()

# Remove the temporary composite key column
df_new_rows = df_new_rows.drop(columns=['_composite_key'])

# Save to output file
print(f"\nSaving new rows to {output_path}...")
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df_new_rows.to_csv(output_path, index=False)

print(f"✓ Successfully saved {len(df_new_rows)} new rows to {output_path}")

# Display summary statistics
if len(df_new_rows) > 0:
    print("\nSummary of new rows:")
    print(f"  - Date range: {df_new_rows['date'].min()} to {df_new_rows['date'].max()}")
    print(f"  - Unique patients: {df_new_rows['patient_id'].nunique()}")
    print(f"  - Model versions: {df_new_rows['modelVersion'].unique()}")
    print(f"\nPrimary negative feedback types:")
    print(df_new_rows['primary_negative_feedback_type'].value_counts())
