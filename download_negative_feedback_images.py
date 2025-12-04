import os
import pandas as pd
import numpy as np
import pydicom
import requests
from PIL import Image
from urllib.parse import urlparse
import boto3
from botocore import UNSIGNED
from botocore.config import Config
from tqdm import tqdm
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_window_parameters(ds):
    """
    Extract windowing parameters from DICOM metadata.
    
    Parameters:
    - ds: pydicom dataset
    
    Returns:
    - tuple: (window_center, window_width)
    """
    window_center = None
    window_width = None
    
    # Check for Window Center and Window Width
    if hasattr(ds, 'WindowCenter'):
        window_center = ds.WindowCenter
        if isinstance(window_center, pydicom.multival.MultiValue):
            window_center = float(window_center[0])
        else:
            window_center = float(window_center)
    
    if hasattr(ds, 'WindowWidth'):
        window_width = ds.WindowWidth
        if isinstance(window_width, pydicom.multival.MultiValue):
            window_width = float(window_width[0])
        else:
            window_width = float(window_width)
    
    return window_center, window_width

def apply_window(pixel_array, window_center, window_width):
    """
    Apply window/level adjustment to the pixel array.
    
    Parameters:
    - pixel_array: numpy array of pixel values
    - window_center: center of the window
    - window_width: width of the window
    
    Returns:
    - Windowed pixel array normalized to 0-255
    """
    if window_center is None or window_width is None:
        logger.info("No windowing parameters found, using default normalization")
        # Normalize to 0-255 if no windowing parameters
        pixel_min = pixel_array.min()
        pixel_max = pixel_array.max()
        if pixel_max > pixel_min:
            return ((pixel_array - pixel_min) / (pixel_max - pixel_min) * 255).astype(np.uint8)
        else:
            return np.zeros_like(pixel_array, dtype=np.uint8)
    
    # Calculate window min and max
    window_min = window_center - window_width / 2
    window_max = window_center + window_width / 2
    
    # Apply windowing
    windowed = np.clip(pixel_array, window_min, window_max)
    
    # Normalize to 0-255
    windowed = ((windowed - window_min) / (window_max - window_min) * 255).astype(np.uint8)
    
    return windowed

def download_from_s3(url, local_path, use_credentials=False, aws_access_key=None, aws_secret_key=None):
    """
    Download file from S3 URL.
    
    Parameters:
    - url: S3 URL (https:// or s3://)
    - local_path: Local path to save the file
    - use_credentials: Whether to use AWS credentials
    - aws_access_key: AWS access key (if required)
    - aws_secret_key: AWS secret key (if required)
    
    Returns:
    - True if successful, False otherwise
    """
    try:
        # Parse S3 URL
        if url.startswith('s3://'):
            # Parse s3:// URL
            parts = url.replace('s3://', '').split('/', 1)
            bucket_name = parts[0]
            object_key = parts[1] if len(parts) > 1 else ''
        elif url.startswith('https://'):
            # Parse https:// URL (e.g., https://bucket-name.s3.region.amazonaws.com/object-key)
            parsed = urlparse(url)
            # Extract bucket name from hostname
            hostname_parts = parsed.hostname.split('.')
            if 's3' in hostname_parts:
                bucket_name = hostname_parts[0]
                object_key = parsed.path.lstrip('/')
            else:
                # Try direct download for non-S3 URLs
                response = requests.get(url, stream=True)
                response.raise_for_status()
                with open(local_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                return True
        else:
            logger.error(f"Unsupported URL format: {url}")
            return False
        
        # Configure S3 client
        if use_credentials and aws_access_key and aws_secret_key:
            s3_client = boto3.client(
                's3',
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key
            )
        else:
            # For public buckets
            s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))
        
        # Download file
        s3_client.download_file(bucket_name, object_key, local_path)
        return True
        
    except Exception as e:
        logger.error(f"Error downloading {url}: {str(e)}")
        return False

def process_dicom_to_png(dicom_path, png_path, filter_slice_thickness=True):
    """
    Convert DICOM file to PNG using proper windowing and pixel processing.
    
    Parameters:
    - dicom_path: Path to DICOM file
    - png_path: Path to save PNG file
    - filter_slice_thickness: Whether to filter by slice thickness (5.0mm or 5.5mm)
    
    Returns:
    - True if successful, False otherwise
    """
    try:
        # Read DICOM file
        ds = pydicom.dcmread(dicom_path)
        
        # Check slice thickness if filtering is enabled
        if filter_slice_thickness:
            current_slice_thickness = None
            if hasattr(ds, 'SliceThickness'):
                current_slice_thickness = float(ds.SliceThickness)
            elif hasattr(ds, 'SpacingBetweenSlices'):
                current_slice_thickness = float(ds.SpacingBetweenSlices)
            
            if current_slice_thickness is not None:
                if not (abs(current_slice_thickness - 5.0) < 0.1 or abs(current_slice_thickness - 5.5) < 0.1):
                    logger.info(f"⏭️  Skipping file {dicom_path} - slice thickness {current_slice_thickness}mm "
                              f"(not 5.0mm or 5.5mm)")
                    return False
            else:
                logger.warning(f"No slice thickness found in {dicom_path}")
        
        # Extract windowing parameters from this DICOM file
        window_center, window_width = extract_window_parameters(ds)
        
        # Handle compressed DICOM and process pixel data
        if ds.file_meta.TransferSyntaxUID.is_compressed:
            logger.info("⚠️ Image is compressed! Using pydicom decompression...")
            try:
                ds.decompress()
            except Exception as e:
                logger.error(f"❌ Decompression failed: {e}")
                return False
        
        # Get pixel array
        try:
            pixel_dtype = ds.pixel_array.dtype
            pixel_array = ds.pixel_array
        except Exception as e:
            logger.error(f"❌ Error accessing pixel array: {e}")
            return False
        
        # Handle multi-frame images (take first frame)
        if len(pixel_array.shape) > 2:
            pixel_array = pixel_array[0]
        
        # Process pixel data based on data type
        if pixel_dtype == np.uint16:
            pixel_array = pixel_array.astype(np.int32)
            pixel_array[pixel_array > 32767] -= 65536
        else:
            pixel_array = pixel_array.astype(np.float32)
        
        # Apply rescale slope and intercept if present
        if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
            pixel_array = pixel_array * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
            logger.debug(f"Applied rescaling: Slope={ds.RescaleSlope}, Intercept={ds.RescaleIntercept}")
        
        # Apply windowing using extracted parameters
        pixel_array = apply_window(pixel_array, window_center, window_width)
        
        # Convert to proper format for PNG
        if pixel_array.dtype != np.uint8:
            pixel_min = np.min(pixel_array)
            pixel_max = np.max(pixel_array)
            
            if pixel_max > pixel_min:
                pixel_array = ((pixel_array - pixel_min) / (pixel_max - pixel_min) * 255).astype(np.uint8)
            else:
                pixel_array = np.zeros_like(pixel_array, dtype=np.uint8)
        
        # Handle photometric interpretation
        if hasattr(ds, 'PhotometricInterpretation'):
            if ds.PhotometricInterpretation == "MONOCHROME1":
                # Invert the image for MONOCHROME1
                pixel_array = 255 - pixel_array
        
        # Create image from array
        original_image = Image.fromarray(pixel_array, mode='L')
        
        # Convert to RGB for better compatibility
        original_image = original_image.convert('RGB')
        
        # Save as PNG
        original_image.save(png_path)
        logger.info(f"✅ Saved PNG: {png_path}")
        
        if window_center is not None and window_width is not None:
            logger.info(f"Applied windowing - Center: {window_center}, Width: {window_width}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error processing DICOM {dicom_path}: {str(e)}")
        return False

def main(csv_path, output_folder, temp_folder='temp_dicom', 
         use_credentials=False, aws_access_key=None, aws_secret_key=None,
         filter_slice_thickness=True):
    """
    Main function to process CSV and convert DICOM files to PNG.
    
    Parameters:
    - csv_path: Path to CSV file with image_url, patient_id, and instance_no columns
    - output_folder: Folder to save PNG files
    - temp_folder: Temporary folder to download DICOM files
    - use_credentials: Whether to use AWS credentials
    - aws_access_key: AWS access key (if required)
    - aws_secret_key: AWS secret key (if required)
    - filter_slice_thickness: Whether to filter by slice thickness
    """
    
    # Create folders if they don't exist
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(temp_folder, exist_ok=True)
    
    # Read CSV file
    try:
        df = pd.read_csv(csv_path)
        # Process all rows from new_rows CSV (no sampling)
        required_columns = ['image_url', 'patient_id', 'instance_no']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"CSV file must contain columns: {missing_columns}")
        logger.info(f"Processing {len(df)} rows from {csv_path}")
    except Exception as e:
        logger.error(f"Error reading CSV file: {str(e)}")
        return
    
    # Process each URL
    successful = 0
    failed = 0
    skipped = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing DICOM files"):
        url = row['image_url']
        patient_id = row['patient_id']
        instance_no = row['instance_no']
        
        # Skip empty URLs
        if pd.isna(url) or url == '':
            logger.warning(f"Row {idx}: Empty URL, skipping")
            failed += 1
            continue
        
        # Skip rows with missing patient_id or instance_no
        if pd.isna(patient_id) or pd.isna(instance_no):
            logger.warning(f"Row {idx}: Missing patient_id or instance_no, skipping")
            failed += 1
            continue
        
        # Clean patient_id and instance_no for filename (remove invalid characters)
        patient_id_clean = str(patient_id).replace('/', '_').replace('\\', '_').replace(':', '_').replace(' ', '_')
        instance_no_clean = str(instance_no).replace('/', '_').replace('\\', '_').replace(':', '_').replace(' ', '_')
        
        # Generate file names using patient_id and instance_no
        file_id = f"{patient_id_clean}_{instance_no_clean}"
        dicom_path = os.path.join(temp_folder, f"{file_id}.dcm")
        png_path = os.path.join(output_folder, f"{file_id}.png")
        
        # Skip if PNG already exists (completely processed)
        if os.path.exists(png_path):
            logger.info(f"✅ PNG already exists: {patient_id_clean}_{instance_no_clean}.png - SKIPPING")
            successful += 1
            continue
        
        # Check if DICOM file already exists (partially processed)
        dicom_already_exists = os.path.exists(dicom_path)
        
        if dicom_already_exists:
            logger.info(f"🔄 DICOM already exists, attempting conversion: {patient_id_clean}_{instance_no_clean}.dcm")
            # Try to convert existing DICOM to PNG
            result = process_dicom_to_png(dicom_path, png_path, filter_slice_thickness)
            if result is True:
                successful += 1
                # Remove temporary DICOM file after successful conversion
                os.remove(dicom_path)
            elif result is False and filter_slice_thickness:
                skipped += 1
                # Remove DICOM file even if skipped due to slice thickness
                os.remove(dicom_path)
            else:
                failed += 1
                # Keep DICOM file for debugging if conversion failed
        else:
            # Download DICOM file
            logger.info(f"📥 Downloading for Patient {patient_id_clean}, Instance {instance_no_clean}: {url}")
            if download_from_s3(url, dicom_path, use_credentials, aws_access_key, aws_secret_key):
                # Convert to PNG
                result = process_dicom_to_png(dicom_path, png_path, filter_slice_thickness)
                if result is True:
                    successful += 1
                elif result is False and filter_slice_thickness:
                    skipped += 1
                else:
                    failed += 1
                
                # Remove temporary DICOM file
                if os.path.exists(dicom_path):
                    os.remove(dicom_path)
            else:
                failed += 1
    
    # Clean up temp folder if empty
    if not os.listdir(temp_folder):
        os.rmdir(temp_folder)
    
    logger.info(f"\n🎉 Processing complete!")
    logger.info(f"✅ Successful conversions: {successful}")
    logger.info(f"⏭️  Skipped (slice thickness): {skipped}")
    logger.info(f"❌ Failed conversions: {failed}")

if __name__ == "__main__":
    # Configuration
    CSV_FILE = "/Users/yugambahl/Desktop/brain_ct/data/metadata/negative_feedback_new_rows.csv"  # Path to your CSV file
    OUTPUT_FOLDER = "/Users/yugambahl/Desktop/brain_ct/data/training_datasets/current/negative_feedback_new_rows/png"  # Folder to save PNG files
    TEMP_FOLDER = "/Users/yugambahl/Desktop/brain_ct/data/training_datasets/current/negative_feedback_new_rows/dicom"  # Temporary folder for DICOM files

    # AWS Credentials (if needed for private S3 buckets)
    USE_CREDENTIALS = True  # Set to True if S3 bucket requires authentication
    AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")  # Loaded from .env file
    AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")  # Loaded from .env file

    # Additional options
    FILTER_SLICE_THICKNESS = True  # Set to False to disable slice thickness filtering

    # Run the conversion
    main(
        csv_path=CSV_FILE,
        output_folder=OUTPUT_FOLDER,
        temp_folder=TEMP_FOLDER,
        use_credentials=USE_CREDENTIALS,
        aws_access_key=AWS_ACCESS_KEY,
        aws_secret_key=AWS_SECRET_KEY,
        filter_slice_thickness=FILTER_SLICE_THICKNESS
    )