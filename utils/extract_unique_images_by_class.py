import os
import shutil
import re
import random
from pathlib import Path
from collections import defaultdict

# Folder paths
FIRST_FOLDER = "data/ct_brain_hemorrhage.v4i.yolov8"
SECOND_FOLDER = "data/intraparenchymal_intraventricular_subarachnoid_images"
OUTPUT_FOLDER = "data/selected_images"

# Required number of images per class
IMAGES_PER_CLASS = 300

def extract_ids_from_first_folder():
    """
    Extract patientId_instanceNo from all images in the first folder
    Example: "244349529_219_png.rf.91bb8ef90126de12d0e06e59dfede18a_brightness_contrast.jpg" 
    -> "244349529_219"
    """
    patient_instance_ids = set()
    
    for root, _, files in os.walk(FIRST_FOLDER):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                # Extract pattern like "244349529_219" from filenames
                match = re.search(r'(\d+)_(\d+)', file)
                if match:
                    # First two numeric sections with underscore
                    patient_instance_id = f"{match.group(1)}_{match.group(2)}"
                    patient_instance_ids.add(patient_instance_id)
    
    print(f"Found {len(patient_instance_ids)} unique patientId_instanceNo combinations in the first folder")
    
    # Print a few examples for verification
    examples = list(patient_instance_ids)[:5] if patient_instance_ids else []
    if examples:
        print("Example patientId_instanceNo from first folder:")
        for example in examples:
            print(f"  {example}")
    
    return patient_instance_ids

def find_images_in_second_folder():
    """
    Find all images in the second folder and extract their patientId_instanceNo
    Example: "245796892/245796892 AnonymousPatient_SCMC2023055811 CT Scan Brain Plain_CT 5mm Plain/245796892_1.png"
    -> "245796892_1"
    """
    images_by_class = defaultdict(list)
    image_patient_instance_map = {}  # Maps image path to patientId_instanceNo
    
    for root, _, files in os.walk(SECOND_FOLDER):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(root, file)
                
                # Extract patientId_instanceNo from the filename
                match = re.search(r'(\d+)_(\d+)', file)
                if match:
                    patient_instance_id = f"{match.group(1)}_{match.group(2)}"
                    image_patient_instance_map[image_path] = patient_instance_id
                    
                    # Find the corresponding label file
                    label_path = find_label_file(image_path)
                    
                    if label_path and os.path.exists(label_path):
                        try:
                            with open(label_path, 'r') as f:
                                label_content = f.read().strip()
                                # YOLOv8 format: class_id x y width height
                                # Extract class ID (first number in the file)
                                class_match = re.match(r'^(\d+)', label_content)
                                if class_match:
                                    class_id = int(class_match.group(1))
                                    images_by_class[class_id].append(image_path)
                        except Exception as e:
                            print(f"Error reading label file {label_path}: {e}")
    
    total_images = sum(len(images) for images in images_by_class.values())
    print(f"Found {total_images} labeled images in the second folder:")
    for class_id, images in images_by_class.items():
        print(f"  Class {class_id}: {len(images)} images")
    
    # Print a few examples for verification
    if image_patient_instance_map:
        print("Example patientId_instanceNo from second folder:")
        examples = list(image_patient_instance_map.items())[:5]
        for path, id_value in examples:
            print(f"  {path} -> {id_value}")
    
    return images_by_class, image_patient_instance_map

def find_label_file(image_path):
    """
    Find the corresponding label file for an image
    """
    image_dir = os.path.dirname(image_path)
    image_basename = os.path.splitext(os.path.basename(image_path))[0]
    
    # Try multiple possible locations for the label file
    possible_label_locations = [
        # Same directory with _labels suffix
        os.path.join(image_dir + "_labels", f"{image_basename}.txt"),
        
        # Parent directory with _labels suffix
        os.path.join(os.path.dirname(image_dir) + "_labels", f"{image_basename}.txt"),
        
        # Replace "images" with "labels" in the path
        os.path.join(image_dir.replace("/images/", "/labels/"), f"{image_basename}.txt"),
        
        # Add "_labels" to the dirname
        os.path.join(os.path.dirname(image_dir), 
                     os.path.basename(image_dir) + "_labels", 
                     f"{image_basename}.txt")
    ]
    
    for label_path in possible_label_locations:
        if os.path.exists(label_path):
            return label_path
    
    return None

def select_unique_images(first_folder_ids, images_by_class, image_patient_instance_map):
    """
    Select images from the second folder that don't have matching patientId_instanceNo in the first folder
    """
    selected_images = {}
    duplicates_found = {}
    
    for class_id in [0, 1, 2]:
        class_images = images_by_class.get(class_id, [])
        unique_images = []
        duplicates = []
        
        for image_path in class_images:
            patient_instance_id = image_patient_instance_map.get(image_path)
            
            if patient_instance_id and patient_instance_id not in first_folder_ids:
                unique_images.append(image_path)
            elif patient_instance_id:
                duplicates.append((image_path, patient_instance_id))
        
        # Randomize the order
        random.shuffle(unique_images)
        
        # Select up to IMAGES_PER_CLASS images
        selected = unique_images[:IMAGES_PER_CLASS]
        selected_images[class_id] = selected
        duplicates_found[class_id] = duplicates
        
        print(f"Class {class_id}:")
        print(f"  - Found {len(unique_images)} unique images")
        print(f"  - Selected {len(selected)} images for extraction")
        print(f"  - Found {len(duplicates)} duplicate images (matched in first folder)")
    
    # Output some examples of duplicates for verification
    for class_id, duplicates in duplicates_found.items():
        if duplicates:
            print(f"\nExample duplicates for class {class_id} (first 3):")
            for i, (path, id_value) in enumerate(duplicates[:3]):
                print(f"  {path} -> {id_value}")
    
    return selected_images

def copy_selected_images(selected_images):
    """
    Copy selected images and their label files to the output folder
    """
    # Clear and recreate output directories
    if os.path.exists(OUTPUT_FOLDER):
        shutil.rmtree(OUTPUT_FOLDER)
    
    # Create output directories for images and labels
    for class_id in [0, 1, 2]:
        os.makedirs(os.path.join(OUTPUT_FOLDER, f"class_{class_id}"), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_FOLDER, f"class_{class_id}_labels"), exist_ok=True)
    
    copied_count = 0
    
    for class_id, images in selected_images.items():
        class_dir = os.path.join(OUTPUT_FOLDER, f"class_{class_id}")
        label_dir = os.path.join(OUTPUT_FOLDER, f"class_{class_id}_labels")
        
        for image_path in images:
            # Copy the image
            image_filename = os.path.basename(image_path)
            dest_path = os.path.join(class_dir, image_filename)
            shutil.copy2(image_path, dest_path)
            
            # Try to copy the corresponding label file
            label_path = find_label_file(image_path)
            if label_path and os.path.exists(label_path):
                label_filename = os.path.basename(label_path)
                label_dest_path = os.path.join(label_dir, label_filename)
                shutil.copy2(label_path, label_dest_path)
            
            copied_count += 1
    
    print(f"\nSuccessfully copied {copied_count} images to {OUTPUT_FOLDER}")
    
    # Count images in each class directory as verification
    for class_id in [0, 1, 2]:
        class_dir = os.path.join(OUTPUT_FOLDER, f"class_{class_id}")
        image_count = len([f for f in os.listdir(class_dir) 
                         if os.path.isfile(os.path.join(class_dir, f))])
        print(f"  Class {class_id}: {image_count} images")

def create_verification_file(first_folder_ids, selected_images, image_patient_instance_map):
    """
    Create a verification file listing all selected images and their IDs
    """
    verification_path = os.path.join(OUTPUT_FOLDER, "verification.txt")
    
    with open(verification_path, 'w') as f:
        f.write("VERIFICATION OF SELECTED IMAGES\n")
        f.write("==============================\n\n")
        
        f.write("First Folder Patient IDs:\n")
        for id_value in sorted(list(first_folder_ids))[:20]:
            f.write(f"  {id_value}\n")
        if len(first_folder_ids) > 20:
            f.write(f"  ... and {len(first_folder_ids) - 20} more\n")
        
        f.write("\nSelected Images:\n")
        for class_id, images in selected_images.items():
            f.write(f"\nClass {class_id} - {len(images)} images:\n")
            for image_path in images[:10]:
                patient_instance_id = image_patient_instance_map.get(image_path, "unknown")
                f.write(f"  {os.path.basename(image_path)} - ID: {patient_instance_id}\n")
            if len(images) > 10:
                f.write(f"  ... and {len(images) - 10} more\n")
    
    print(f"\nCreated verification file: {verification_path}")

def main():
    print("Starting extraction of unique brain scan images...")
    
    # Step 1: Extract patientId_instanceNo from first folder
    first_folder_ids = extract_ids_from_first_folder()
    
    # Step 2: Find all images in second folder with their patientId_instanceNo and class
    images_by_class, image_patient_instance_map = find_images_in_second_folder()
    
    # Step 3: Select images that don't have matching patientId_instanceNo in the first folder
    selected_images = select_unique_images(first_folder_ids, images_by_class, image_patient_instance_map)
    
    # Step 4: Copy selected images and their label files to output folder
    copy_selected_images(selected_images)
    
    # Create a verification file for checking
    create_verification_file(first_folder_ids, selected_images, image_patient_instance_map)
    
    print("\nImage extraction complete!")

if __name__ == "__main__":
    main()