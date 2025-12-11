#!/usr/bin/env python3
"""
S3 DICOM to PNG Converter (CSV Input)

This script:
1. Reads a CSV file containing patient IDs
2. Connects to an S3 bucket
3. Processes DICOM files for each patient ID directly from S3
4. Converts them to PNG format
5. Saves the PNG files in an organized folder structure
"""

import os
import io
import csv
import boto3
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import numpy as np
import cv2
from botocore.exceptions import ClientError
import tempfile
import argparse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# S3 Configuration
BUCKET_NAME = "tnsqtraining"
CSV_FILE = "data/intraparanchymal_intraventricular_subarachnoid_patient_ids.csv"
OUTPUT_FOLDER = "data/intraparenchymal_intraventricular_subarachnoid_images"  # Main output folder

# Initialize S3 client
s3 = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY"),
    aws_secret_access_key=os.getenv("AWS_SECRET_KEY")
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Convert DICOM files from S3 to PNG for multiple patients.')
    parser.add_argument('--id_column', default='patient id', help='Name of the column containing patient IDs')
    parser.add_argument('--output', default=OUTPUT_FOLDER, help='Output folder name')
    return parser.parse_args()


def read_patient_ids_from_csv(csv_file, id_column='patient id'):
    """
    Read patient IDs from a CSV file.
    
    Args:
        csv_file (str): Path to the CSV file
        id_column (str): Name of the column containing patient IDs
        
    Returns:
        list: List of patient IDs
    """
    patient_ids = []
    
    try:
        with open(csv_file, 'r') as f:
            csv_reader = csv.DictReader(f)
            if id_column not in csv_reader.fieldnames:
                raise ValueError(f"Column '{id_column}' not found in CSV. Available columns: {csv_reader.fieldnames}")
            
            for row in csv_reader:
                patient_id = row[id_column].strip()
                if patient_id:  # Skip empty IDs
                    patient_ids.append(patient_id)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return []
    
    return patient_ids


def is_dicom_file(bucket_name, file_key):
    """
    Check if a file is a DICOM file by examining its content.
    
    Args:
        bucket_name (str): Name of the S3 bucket
        file_key (str): S3 key of the file
        
    Returns:
        bool: True if file is a DICOM file, False otherwise
    """
    try:
        # Read the first 132 bytes (enough for DICOM header)
        response = s3.get_object(Bucket=bucket_name, Key=file_key, Range='bytes=0-131')
        header = response['Body'].read()
        
        # Check for DICOM magic number (DICM at offset 128)
        return header[128:132] == b'DICM'
    except Exception:
        return False


def get_patient_id_from_path(file_path):
    """
    Extract patient ID from the file path.
    Looks for a numeric ID in the path that likely represents the patient ID.
    
    Args:
        file_path (str): S3 file path
        
    Returns:
        str: Extracted patient ID
    """
    parts = file_path.strip('/').split('/')
    
    # Look for parts that contain numeric patient IDs
    for part in parts:
        # Look for patterns like "249224145 AnonymousPatient"
        if " AnonymousPatient" in part:
            return part.split(" ")[0]  # Return just the numeric part
        
        # Look for purely numeric IDs
        if part.isdigit() and len(part) > 5:  # Most patient IDs are longer than 5 digits
            return part
    
    # If we couldn't find a clear patient ID, use the folder structure
    if len(parts) >= 3:  # Assuming structure like sms/249224145/...
        return parts[1]  # Try the second component which is often the patient ID
    
    # Fallback to unknown
    return "unknown"


def dicom_to_png(dicom_data):
    """
    Convert DICOM data to PNG image data.
    
    Args:
        dicom_data (bytes): Raw DICOM file content
        
    Returns:
        tuple: (numpy array of PNG data, instance number)
    """
    # Use a temporary file since some DICOM libraries work better with files
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(dicom_data)
        temp_file_path = temp_file.name
    
    try:
        dicom = pydicom.dcmread(temp_file_path)
        instance_number = dicom.get('InstanceNumber', "unknown")
        
        if 'PixelData' not in dicom:
            raise ValueError("No pixel data found in DICOM file")
            
        pixel_array = dicom.pixel_array.astype(np.float32)
        
        # Convert to HU if possible
        if 'RescaleIntercept' in dicom and 'RescaleSlope' in dicom:
            intercept = dicom.RescaleIntercept
            slope = dicom.RescaleSlope
            hu_array = pixel_array * slope + intercept
        else:
            hu_array = pixel_array
        
        # Apply windowing if available
        if 'WindowWidth' in dicom and 'WindowCenter' in dicom:
            hu_array = apply_voi_lut(hu_array, dicom)
        
        # Normalize to 0-255 for PNG
        min_hu = np.min(hu_array)
        max_hu = np.max(hu_array)
        
        if max_hu - min_hu != 0:
            normalized_array = ((hu_array - min_hu) / (max_hu - min_hu)) * 255.0
            normalized_array = np.uint8(normalized_array)
        else:
            normalized_array = np.uint8(np.zeros_like(hu_array))
        
        # Convert to PNG in memory
        success, png_data = cv2.imencode('.png', normalized_array)
        if not success:
            raise ValueError("Failed to convert image to PNG format")
        
        return png_data, instance_number
    finally:
        # Clean up the temporary file
        os.unlink(temp_file_path)


def process_patient_dicom_files(patient_id, output_folder):
    """
    Process all DICOM files for a specific patient ID, convert to PNG, and save locally.
    
    Args:
        patient_id (str): Patient ID to process
        output_folder (str): Base output folder to save PNG files
    """
    print(f"\nProcessing patient ID: {patient_id}")
    
    # Create S3 folder prefix for this patient
    s3_folder = f"sms/{patient_id}/"
    
    # Create patient-specific output folder
    patient_folder = os.path.join(output_folder, patient_id)
    os.makedirs(patient_folder, exist_ok=True)
    
    # List all objects for this patient
    paginator = s3.get_paginator('list_objects_v2')
    all_files = []
    
    # Get all files with this patient's prefix
    for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix=s3_folder):
        if 'Contents' in page:
            for obj in page['Contents']:
                key = obj['Key']
                # Check if this is likely a DICOM file
                if key.lower().endswith(('.dcm', '.dicom')) or '.' not in os.path.basename(key):
                    all_files.append(key)
    
    if not all_files:
        print(f"No files found for patient ID {patient_id}")
        return
    
    print(f"Found {len(all_files)} potential DICOM files for patient {patient_id}")
    
    # Process each file
    for i, file_key in enumerate(all_files):
        print(f"Processing file {i+1}/{len(all_files)}: {file_key}")
        
        # Skip non-DICOM files without extension
        if not file_key.lower().endswith(('.dcm', '.dicom')):
            if not is_dicom_file(BUCKET_NAME, file_key):
                print(f"Skipping non-DICOM file: {file_key}")
                continue
        
        try:
            # Download the DICOM file
            response = s3.get_object(Bucket=BUCKET_NAME, Key=file_key)
            dicom_data = response['Body'].read()
            
            # Convert to PNG
            try:
                png_data, instance_number = dicom_to_png(dicom_data)
                
                # Extract series folder from S3 path
                path_parts = file_key.strip('/').split('/')
                if len(path_parts) > 3:  # If there's a subfolder structure
                    # Create series-specific subfolder
                    series_folder = os.path.join(patient_folder, '_'.join(path_parts[2:-1]))
                    os.makedirs(series_folder, exist_ok=True)
                else:
                    series_folder = patient_folder
                
                # Create the PNG filename
                png_filename = f"{patient_id}_{instance_number}.png"
                png_path = os.path.join(series_folder, png_filename)
                
                # Save the PNG file
                with open(png_path, 'wb') as f:
                    f.write(png_data)
                
                print(f"Converted and saved: {png_path}")
            except Exception as e:
                print(f"Error converting {file_key}: {e}")
                
        except Exception as e:
            print(f"Error processing {file_key}: {e}")


def main():
    """Main function to process all patient IDs from the CSV file."""
    args = parse_args()
    
    # Create the output directory
    os.makedirs(args.output, exist_ok=True)
    
    print(f"Reading patient IDs from CSV file: {CSV_FILE}")
    patient_ids = read_patient_ids_from_csv(CSV_FILE, args.id_column)
    
    if not patient_ids:
        print("No patient IDs found in the CSV file. Exiting.")
        return
    
    print(f"Found {len(patient_ids)} patient IDs to process")
    
    # Process each patient ID
    for idx, patient_id in enumerate(patient_ids):
        print(f"\nProcessing patient {idx+1}/{len(patient_ids)}: {patient_id}")
        try:
            process_patient_dicom_files(patient_id, args.output)
        except Exception as e:
            print(f"Error processing patient {patient_id}: {e}")
    
    print("\nConversion complete!")
    print(f"All images have been saved to the '{args.output}' folder")


if __name__ == "__main__":
    main()