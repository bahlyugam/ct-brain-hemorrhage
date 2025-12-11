"""Test RF-DETR output format to understand what model.predict() returns."""

from rfdetr import RFDETRMedium
from PIL import Image
import json

# Load model
model_path = "/Users/yugambahl/Desktop/brain_ct/models/rfdetr_medium_best_epoch72.pth"
model = RFDETRMedium(pretrain_weights=model_path)

# Find an image with hemorrhage annotations
import os
test_dir = "/Users/yugambahl/Desktop/brain_ct/data/training_datasets/filtered_4class/coco/test"

# Load annotations to find an image with detections
with open(os.path.join(test_dir, "_annotations.coco.json"), 'r') as f:
    annotations = json.load(f)

# Find first image with at least one annotation
for ann in annotations['annotations']:
    image_id = ann['image_id']
    # Get image filename
    for img in annotations['images']:
        if img['id'] == image_id:
            test_image_path = os.path.join(test_dir, img['file_name'])
            print(f"Testing with: {test_image_path}")
            print(f"Expected annotations: {ann}")
            break
    break

# Load and run inference
img = Image.open(test_image_path)
print(f"\nImage size: {img.size}")

print("\nRunning model.predict()...")
result = model.predict(img)

print(f"\nResult type: {type(result)}")
print(f"Result length: {len(result) if hasattr(result, '__len__') else 'N/A'}")

if isinstance(result, tuple):
    print(f"\nResult is a tuple with {len(result)} elements:")
    for i, item in enumerate(result):
        print(f"  Element {i}: type={type(item)}, shape/len={getattr(item, 'shape', len(item) if hasattr(item, '__len__') else 'N/A')}")
        if i == 1 and hasattr(item, '__len__') and len(item) > 0:
            print(f"  First detection: {item[0]}")
            print(f"  First detection type: {type(item[0])}")
else:
    print(f"\nResult: {result}")

print("\nDone!")
