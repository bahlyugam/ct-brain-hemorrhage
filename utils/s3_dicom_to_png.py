#!/usr/bin/env python3
"""
S3 DICOM to PNG Converter

This script:
1. Connects to an S3 bucket
2. Processes DICOM files directly from S3
3. Converts them to PNG format
4. Saves the PNG files locally in their respective folder structure
5. Uses patientId_instanceNo.png naming format for the files
"""

import os
import io
import boto3
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import numpy as np
import cv2
from botocore.exceptions import ClientError
import tempfile
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# S3 Configuration
PATIENT_ID = "548384722"
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
BUCKET_NAME = "tnsqaiapp"
S3_FOLDER = f"sms/{PATIENT_ID}/"
LOCAL_PNG_ROOT = f"./data/{PATIENT_ID} CT Images"  # Local directory to save PNG files

# Initialize S3 client
s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY
)


def list_all_folders(bucket_name, prefix=''):
    """
    List all folders and nested folders in an S3 bucket.
    
    Args:
        bucket_name (str): Name of the S3 bucket
        prefix (str): Optional prefix to start listing from
        
    Returns:
        list: List of all folder paths discovered
    """
    all_folders = set()
    visited_folders = set()  # Track already visited folders to prevent infinite recursion
    
    def _list_folders_recursive(current_prefix):
        if current_prefix in visited_folders:
            return  # Skip if already processed this prefix
        
        visited_folders.add(current_prefix)
        paginator = s3.get_paginator('list_objects_v2')
        
        # Get folders directly under the current prefix
        operation_parameters = {
            'Bucket': bucket_name,
            'Prefix': current_prefix,
            'Delimiter': '/'
        }
        
        # Process paginated results
        for page in paginator.paginate(**operation_parameters):
            # Add common prefixes (folders)
            if 'CommonPrefixes' in page:
                for common_prefix in page['CommonPrefixes']:
                    folder_path = common_prefix['Prefix']
                    all_folders.add(folder_path)
                    
                    # Process subfolder if not already visited
                    if folder_path not in visited_folders:
                        _list_folders_recursive(folder_path)
    
    # Start recursive folder listing
    _list_folders_recursive(prefix)
    return all_folders


def list_dicom_files(bucket_name, folder_prefix):
    """
    List all DICOM files in a specific S3 folder.
    
    Args:
        bucket_name (str): Name of the S3 bucket
        folder_prefix (str): Folder prefix to list files from
        
    Returns:
        list: List of DICOM file keys
    """
    dicom_files = []
    paginator = s3.get_paginator('list_objects_v2')
    
    for page in paginator.paginate(Bucket=bucket_name, Prefix=folder_prefix):
        if 'Contents' in page:
            for obj in page['Contents']:
                file_key = obj['Key']
                # Check if file is a DICOM file (either by extension or without extension)
                if file_key.lower().endswith(('.dcm', '.dicom')):
                    dicom_files.append(file_key)
                else:
                    # For files without extension, we'll check their content later
                    if '.' not in os.path.basename(file_key):
                        dicom_files.append(file_key)
    
    return dicom_files


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

def safe_get_pixel_array(dicom):
    pixel_bytes = dicom.PixelData
    bits_allocated = dicom.BitsAllocated
    pixel_repr = dicom.PixelRepresentation  # 0 = unsigned, 1 = signed
    shape = (dicom.Rows, dicom.Columns)

    if bits_allocated == 16:
        if pixel_repr == 0:
            dtype = np.uint16
        else:
            dtype = np.int16
    elif bits_allocated == 8:
        dtype = np.uint8
    else:
        raise ValueError(f"Unsupported BitsAllocated: {bits_allocated}")

    pixel_array = np.frombuffer(pixel_bytes, dtype=dtype).reshape(shape)
    return pixel_array.astype(np.float32)  # Convert to float for later ops


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


def process_dicom_files(bucket_name, folder_prefix):
    """
    Process all DICOM files in a folder, convert to PNG, and save locally.
    
    Args:
        bucket_name (str): Name of the S3 bucket
        folder_prefix (str): Folder prefix to process files from
    """
    print(f"Processing folder: {folder_prefix}")
    
    # Create local folder structure mirroring S3
    relative_folder = folder_prefix[len(S3_FOLDER):] if folder_prefix.startswith(S3_FOLDER) else folder_prefix
    local_folder = os.path.join(LOCAL_PNG_ROOT, relative_folder)
    
    # Get all DICOM files in this folder
    dicom_files = list_dicom_files(bucket_name, folder_prefix)
    
    if not dicom_files:
        print(f"No DICOM files found in {folder_prefix}")
        return
    
    print(f"Found {len(dicom_files)} potential DICOM files")
    
    # Process each DICOM file
    dicom_instances = []
    for file_key in dicom_files:
        # For files without .dcm extension, verify they're actually DICOM
        if not file_key.lower().endswith(('.dcm', '.dicom')):
            if not is_dicom_file(bucket_name, file_key):
                print(f"Skipping non-DICOM file: {file_key}")
                continue
        
        try:
            # Get the patient ID from the path
            patient_id = get_patient_id_from_path(file_key)
            
            # Download the DICOM file
            response = s3.get_object(Bucket=bucket_name, Key=file_key)
            dicom_data = response['Body'].read()
            
            # Convert to PNG
            try:
                png_data, instance_number = dicom_to_png(dicom_data)
                
                # Extract the local folder path that mirrors the S3 structure
                # Get the directory part of the DICOM file key
                s3_dir_path = os.path.dirname(file_key)
                if s3_dir_path.startswith(S3_FOLDER):
                    s3_dir_path = s3_dir_path[len(S3_FOLDER):]
                
                # Create the directory structure
                png_dir = os.path.join(LOCAL_PNG_ROOT, s3_dir_path)
                os.makedirs(png_dir, exist_ok=True)
                
                # Create the PNG filename
                png_filename = f"{patient_id}_{instance_number}.png"
                png_path = os.path.join(png_dir, png_filename)
                
                # Save the PNG file
                with open(png_path, 'wb') as f:
                    f.write(png_data)
                
                print(f"Converted and saved: {png_path}")
                
                # Store for sorting if needed
                dicom_instances.append((file_key, instance_number, png_path))
            except Exception as e:
                print(f"Error converting {file_key}: {e}")
                
        except Exception as e:
            print(f"Error processing {file_key}: {e}")


def main():
    """Main function to process all folders and DICOM files."""
    # Create the root local directory
    os.makedirs(LOCAL_PNG_ROOT, exist_ok=True)
    
    print(f"Starting conversion from S3 bucket: {BUCKET_NAME}, folder: {S3_FOLDER}")
    
    try:
        # Get all DICOM files directly instead of processing folder by folder
        # This avoids potential infinite loops in folder traversal
        print("Listing all DICOM files in the S3 folder...")
        
        paginator = s3.get_paginator('list_objects_v2')
        all_dicom_files = []
        
        # List all objects with the given prefix
        for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix=S3_FOLDER):
            if 'Contents' in page:
                for obj in page['Contents']:
                    key = obj['Key']
                    # Check if this is likely a DICOM file
                    if key.lower().endswith(('.dcm', '.dicom')) or '.' not in os.path.basename(key):
                        all_dicom_files.append(key)
        
        print(f"Found {len(all_dicom_files)} potential DICOM files")
        
        # Process files one by one
        for i, file_key in enumerate(all_dicom_files):
            print(f"Processing file {i+1}/{len(all_dicom_files)}: {file_key}")
            
            # Skip non-DICOM files without extension
            if not file_key.lower().endswith(('.dcm', '.dicom')):
                if not is_dicom_file(BUCKET_NAME, file_key):
                    print(f"Skipping non-DICOM file: {file_key}")
                    continue
            
            try:
                # Get the patient ID from the path
                patient_id = get_patient_id_from_path(file_key)
                
                # Download the DICOM file
                response = s3.get_object(Bucket=BUCKET_NAME, Key=file_key)
                dicom_data = response['Body'].read()
                
                # Convert to PNG
                try:
                    png_data, instance_number = dicom_to_png(dicom_data)
                    
                    # Extract the local folder path that mirrors the S3 structure
                    s3_dir_path = os.path.dirname(file_key)
                    if s3_dir_path.startswith(S3_FOLDER):
                        s3_dir_path = s3_dir_path[len(S3_FOLDER):]
                    
                    # Create the directory structure
                    png_dir = os.path.join(LOCAL_PNG_ROOT, s3_dir_path)
                    os.makedirs(png_dir, exist_ok=True)
                    
                    # Create the PNG filename
                    png_filename = f"{patient_id}_{instance_number}.png"
                    png_path = os.path.join(png_dir, png_filename)
                    
                    # Save the PNG file
                    with open(png_path, 'wb') as f:
                        f.write(png_data)
                    
                    print(f"Converted and saved: {png_path}")
                except Exception as e:
                    print(f"Error converting {file_key}: {e}")
                    
            except Exception as e:
                print(f"Error processing {file_key}: {e}")
    
    except Exception as e:
        print(f"An error occurred during processing: {e}")
    
    print("Conversion complete!")


if __name__ == "__main__":
    main()