import os
import re
import cv2
from pathlib import Path
from tqdm import tqdm

# Input/Output paths - adjust as needed
INPUT_FOLDER = "data/selected_images"
OUTPUT_FOLDER = "data/roboflow_dataset"

# Split ratios
TRAIN_RATIO = 0.80
VAL_RATIO = 0.15
TEST_RATIO = 0.05  # This will be calculated as remainder

def extract_patient_instance_id(filename):
    """
    Extract patientId_instanceNo from a filename
    """
    match = re.search(r'(\d+)_(\d+)', filename)
    if match:
        return f"{match.group(1)}_{match.group(2)}"
    return None

def create_dataset_structure():
    """
    Create the dataset directory structure for Roboflow
    """
    # Create main dataset directory
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # Create split directories with images and labels subdirectories
    for split in ['train', 'valid', 'test']:
        os.makedirs(os.path.join(OUTPUT_FOLDER, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_FOLDER, split, 'labels'), exist_ok=True)
    
    print(f"Created dataset structure in {OUTPUT_FOLDER}")

def convert_to_yolo_format(label_path, image_width, image_height):
    """
    Convert absolute coordinates to normalized YOLO format
    
    Input format: class_id center_x center_y width height (in pixel coordinates)
    Output format: class_id center_x center_y width height (normalized 0-1)
    """
    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        normalized_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 5:  # class_id, center_x, center_y, width, height
                class_id = int(parts[0])
                
                # Parse absolute coordinates
                center_x = float(parts[1])
                center_y = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                
                # Normalize to 0-1 range
                norm_center_x = center_x / image_width
                norm_center_y = center_y / image_height
                norm_width = width / image_width
                norm_height = height / image_height
                
                # Ensure values are within 0-1 range
                norm_center_x = max(0, min(1, norm_center_x))
                norm_center_y = max(0, min(1, norm_center_y))
                norm_width = max(0, min(1, norm_width))
                norm_height = max(0, min(1, norm_height))
                
                normalized_lines.append(f"{class_id} {norm_center_x} {norm_center_y} {norm_width} {norm_height}")
            else:
                print(f"Warning: Skipping invalid annotation line: {line}")
        
        return normalized_lines
    except Exception as e:
        print(f"Error processing {label_path}: {e}")
        return []

def get_image_dimensions(image_path):
    """
    Get the width and height of an image
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        height, width = img.shape[:2]
        return width, height
    except Exception as e:
        print(f"Error getting dimensions for {image_path}: {e}")
        return None, None

def process_dataset():
    """
    Process and convert the entire dataset
    """
    all_data = []
    
    # Process each class directory
    for class_id in [0, 1, 2]:
        class_dir = os.path.join(INPUT_FOLDER, f"class_{class_id}")
        label_dir = os.path.join(INPUT_FOLDER, f"class_{class_id}_labels")
        
        if not os.path.exists(class_dir):
            print(f"Warning: Class directory {class_dir} not found")
            continue
        
        if not os.path.exists(label_dir):
            print(f"Warning: Label directory {label_dir} not found")
            continue
        
        # Get all images in this class
        for filename in os.listdir(class_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(class_dir, filename)
                image_basename = os.path.splitext(filename)[0]
                label_path = os.path.join(label_dir, f"{image_basename}.txt")
                
                if not os.path.exists(label_path):
                    print(f"Warning: Label file not found for {image_path}")
                    continue
                
                # Extract patientId_instanceNo for renaming
                patient_instance_id = extract_patient_instance_id(filename)
                if not patient_instance_id:
                    print(f"Warning: Could not extract ID from {filename}, skipping")
                    continue
                
                # Get image dimensions for normalization
                width, height = get_image_dimensions(image_path)
                if width is None or height is None:
                    continue
                
                all_data.append({
                    'image_path': image_path,
                    'label_path': label_path,
                    'patient_instance_id': patient_instance_id,
                    'class_id': class_id,
                    'width': width,
                    'height': height,
                    'extension': os.path.splitext(filename)[1]
                })
    
    # Shuffle and split data
    import random
    random.shuffle(all_data)
    
    total_count = len(all_data)
    train_end = int(total_count * TRAIN_RATIO)
    val_end = train_end + int(total_count * VAL_RATIO)
    
    splits = {
        'train': all_data[:train_end],
        'valid': all_data[train_end:val_end],
        'test': all_data[val_end:]
    }
    
    print(f"\nProcessing {total_count} images:")
    print(f"- Train: {len(splits['train'])} images")
    print(f"- Validation: {len(splits['valid'])} images")
    print(f"- Test: {len(splits['test'])} images")
    
    # Process each split
    for split_name, split_data in splits.items():
        print(f"\nProcessing {split_name} split...")
        
        class_counts = {0: 0, 1: 0, 2: 0}
        
        for item in tqdm(split_data, desc=f"{split_name}"):
            # Convert label to normalized YOLO format
            normalized_lines = convert_to_yolo_format(
                item['label_path'], 
                item['width'], 
                item['height']
            )
            
            if not normalized_lines:
                print(f"Warning: No valid annotations in {item['label_path']}")
                continue
            
            # Define output paths with patientId_instanceNo naming
            dest_image_filename = f"{item['patient_instance_id']}{item['extension']}"
            dest_label_filename = f"{item['patient_instance_id']}.txt"
            
            dest_image_path = os.path.join(OUTPUT_FOLDER, split_name, 'images', dest_image_filename)
            dest_label_path = os.path.join(OUTPUT_FOLDER, split_name, 'labels', dest_label_filename)
            
            # Copy image
            import shutil
            shutil.copy2(item['image_path'], dest_image_path)
            
            # Write normalized label
            with open(dest_label_path, 'w') as f:
                for line in normalized_lines:
                    f.write(line + '\n')
            
            # Update class counts
            class_counts[item['class_id']] += 1
        
        # Print class distribution for this split
        print(f"{split_name} class distribution:")
        for class_id, count in class_counts.items():
            percentage = count / len(split_data) * 100 if split_data else 0
            print(f"  Class {class_id}: {count} images ({percentage:.1f}%)")

def create_roboflow_instructions():
    """
    Create a file with instructions for uploading to Roboflow
    """
    instructions_path = os.path.join(OUTPUT_FOLDER, "roboflow_instructions.txt")
    
    with open(instructions_path, 'w') as f:
        f.write("INSTRUCTIONS FOR UPLOADING TO ROBOFLOW\n")
        f.write("====================================\n\n")
        
        f.write("1. CREATE A NEW PROJECT\n")
        f.write("   - Go to https://app.roboflow.com\n")
        f.write("   - Create a new project\n")
        f.write("   - Select 'Object Detection' as the project type\n")
        f.write("   - Select 'YOLOv8' as the annotation format\n\n")
        
        f.write("2. UPLOAD THE DATASET\n")
        f.write("   - Click 'Upload Dataset'\n")
        f.write("   - Select 'Upload a folder' option\n")
        f.write("   - Upload the 'train' folder first\n")
        f.write("   - Then upload the 'valid' folder\n")
        f.write("   - Finally upload the 'test' folder\n\n")
        
        f.write("3. DATASET STRUCTURE\n")
        f.write("   The dataset has been organized as follows:\n")
        f.write(f"   - Train set: {TRAIN_RATIO:.0%} of the data\n")
        f.write(f"   - Validation set: {VAL_RATIO:.0%} of the data\n")
        f.write(f"   - Test set: {TEST_RATIO:.0%} of the data\n\n")
        
        f.write("4. ANNOTATION FORMAT\n")
        f.write("   All annotation files have been converted to the YOLO format:\n")
        f.write("   - class_id: Integer representing the class (0, 1, or 2)\n")
        f.write("   - center_x: Normalized center X coordinate (0-1)\n")
        f.write("   - center_y: Normalized center Y coordinate (0-1)\n")
        f.write("   - width: Normalized width (0-1)\n")
        f.write("   - height: Normalized height (0-1)\n\n")
        
        f.write("5. FILE NAMING\n")
        f.write("   All files have been renamed to follow the pattern:\n")
        f.write("   patientId_instanceNo.extension\n\n")
        
        f.write("6. NOTE\n")
        f.write("   - Ensure your project has 3 classes: 0, 1, and 2\n")
        f.write("   - The YOLOv8 format uses normalized coordinates (0-1)\n")
    
    print(f"\nCreated instructions file: {instructions_path}")

def main():
    print("Converting annotations to Roboflow YOLO format...")
    
    # Create output directory structure
    create_dataset_structure()
    
    # Process and convert the dataset
    process_dataset()
    
    # Create instructions file
    create_roboflow_instructions()
    
    print("\nConversion complete!")
    print(f"The prepared dataset is ready at: {OUTPUT_FOLDER}")
    print("Follow the instructions in the roboflow_instructions.txt file to upload to Roboflow.")

if __name__ == "__main__":
    main()