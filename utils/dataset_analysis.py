"""
CT Brain Hemorrhage Dataset Analysis
Analyzes combined dataset with thin (0.625-1.25mm) and thick (5mm) slices
Includes detailed normal (no hemorrhage) image distribution
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
import yaml

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

# ============================================================================
# CONFIGURATION
# ============================================================================

# UPDATE THIS PATH to your dataset location
DATASET_PATH = "/Users/yugambahl/Desktop/brain_ct/data/UAT_CT_BRAIN_HEMORRHAGE.v2i.yolov8_combined"

# Class names (update if needed)
CLASS_NAMES = ['EDH', 'HC', 'IPH', 'IVH', 'SAH', 'SDH']

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_data_yaml(dataset_path):
    """Load class names from data.yaml if available"""
    yaml_path = os.path.join(dataset_path, "data.yaml")
    if os.path.exists(yaml_path):
        with open(yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)
            return data_config.get('names', CLASS_NAMES)
    return CLASS_NAMES

def analyze_split(split_path, split_name):
    """Analyze a single split (train/valid/test)"""
    img_dir = os.path.join(split_path, "images")
    label_dir = os.path.join(split_path, "labels")
    
    if not os.path.exists(img_dir) or not os.path.exists(label_dir):
        print(f"Warning: {split_name} directory not found")
        return None
    
    # Image analysis
    images = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    thin_images = [f for f in images if f.startswith('thin_')]
    thick_images = [f for f in images if f.startswith('thick_')]
    
    # Label analysis
    class_counts = defaultdict(int)
    images_per_class = defaultdict(set)
    boxes_per_image = []
    images_with_hemorrhage = 0
    images_without_hemorrhage = 0
    
    # Per-domain statistics
    thin_class_counts = defaultdict(int)
    thick_class_counts = defaultdict(int)
    thin_images_with_hem = 0
    thick_images_with_hem = 0
    thin_images_without_hem = 0
    thick_images_without_hem = 0
    
    for img_file in images:
        label_file = os.path.join(label_dir, os.path.splitext(img_file)[0] + '.txt')
        
        is_thin = img_file.startswith('thin_')
        is_thick = img_file.startswith('thick_')
        
        if os.path.exists(label_file):
            with open(label_file, 'r') as f:
                lines = f.readlines()
                
            if lines:
                images_with_hemorrhage += 1
                if is_thin:
                    thin_images_with_hem += 1
                elif is_thick:
                    thick_images_with_hem += 1
                
                num_boxes = 0
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        class_counts[class_id] += 1
                        images_per_class[class_id].add(img_file)
                        num_boxes += 1
                        
                        # Track by domain
                        if is_thin:
                            thin_class_counts[class_id] += 1
                        elif is_thick:
                            thick_class_counts[class_id] += 1
                
                boxes_per_image.append(num_boxes)
            else:
                # Empty label file = normal image
                images_without_hemorrhage += 1
                if is_thin:
                    thin_images_without_hem += 1
                elif is_thick:
                    thick_images_without_hem += 1
        else:
            # No label file = normal image
            images_without_hemorrhage += 1
            if is_thin:
                thin_images_without_hem += 1
            elif is_thick:
                thick_images_without_hem += 1
    
    return {
        'split_name': split_name,
        'total_images': len(images),
        'thin_images': len(thin_images),
        'thick_images': len(thick_images),
        'thin_pct': len(thin_images) / len(images) * 100 if images else 0,
        'thick_pct': len(thick_images) / len(images) * 100 if images else 0,
        'images_with_hemorrhage': images_with_hemorrhage,
        'images_without_hemorrhage': images_without_hemorrhage,
        'thin_with_hem': thin_images_with_hem,
        'thick_with_hem': thick_images_with_hem,
        'thin_without_hem': thin_images_without_hem,
        'thick_without_hem': thick_images_without_hem,
        'class_counts': dict(class_counts),
        'images_per_class': {k: len(v) for k, v in images_per_class.items()},
        'thin_class_counts': dict(thin_class_counts),
        'thick_class_counts': dict(thick_class_counts),
        'boxes_per_image': boxes_per_image,
        'avg_boxes_per_image': np.mean(boxes_per_image) if boxes_per_image else 0,
        'max_boxes_per_image': max(boxes_per_image) if boxes_per_image else 0,
    }

def extract_patient_ids(split_path, split_name):
    """Extract unique patient IDs from filenames"""
    img_dir = os.path.join(split_path, "images")
    if not os.path.exists(img_dir):
        return set(), set()
    
    images = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    thin_patients = set()
    thick_patients = set()
    
    for img in images:
        # Remove domain prefix
        if img.startswith('thin_'):
            patient_id = img.replace('thin_', '').split('_')[0]
            thin_patients.add(patient_id)
        elif img.startswith('thick_'):
            patient_id = img.replace('thick_', '').split('_')[0]
            thick_patients.add(patient_id)
    
    return thin_patients, thick_patients

# ============================================================================
# MAIN ANALYSIS
# ============================================================================

print("="*100)
print("CT BRAIN HEMORRHAGE DATASET ANALYSIS")
print("="*100)
print(f"\nDataset Path: {DATASET_PATH}\n")

# Load class names
class_names = load_data_yaml(DATASET_PATH)
print(f"Classes: {class_names}\n")

# Analyze each split
splits = ['train', 'valid', 'test']
split_data = {}

for split in splits:
    split_path = os.path.join(DATASET_PATH, split)
    print(f"Analyzing {split.upper()} split...")
    split_data[split] = analyze_split(split_path, split)
    print(f"  ✓ Complete\n")

# ============================================================================
# 1. OVERALL DATASET SUMMARY
# ============================================================================

print("\n" + "="*100)
print("1. OVERALL DATASET SUMMARY")
print("="*100)

total_images = sum(data['total_images'] for data in split_data.values() if data)
total_thin = sum(data['thin_images'] for data in split_data.values() if data)
total_thick = sum(data['thick_images'] for data in split_data.values() if data)
total_instances = sum(sum(data['class_counts'].values()) for data in split_data.values() if data)
total_with_hem = sum(data['images_with_hemorrhage'] for data in split_data.values() if data)
total_without_hem = sum(data['images_without_hemorrhage'] for data in split_data.values() if data)

print(f"\nTotal Images: {total_images:,}")
print(f"  - Thin slices (0.625-1.25mm): {total_thin:,} ({total_thin/total_images*100:.1f}%)")
print(f"  - Thick slices (5mm): {total_thick:,} ({total_thick/total_images*100:.1f}%)")
print(f"\nHemorrhage Distribution:")
print(f"  - Images with hemorrhage: {total_with_hem:,} ({total_with_hem/total_images*100:.1f}%)")
print(f"  - Normal images (no hemorrhage): {total_without_hem:,} ({total_without_hem/total_images*100:.1f}%)")
print(f"\nTotal Hemorrhage Instances: {total_instances:,}")
print(f"Average Instances per Image: {total_instances/total_images:.2f}")

# ============================================================================
# 2. SPLIT-WISE BREAKDOWN
# ============================================================================

print("\n" + "="*100)
print("2. SPLIT-WISE BREAKDOWN")
print("="*100)

split_summary = []
for split, data in split_data.items():
    if data:
        split_summary.append({
            'Split': split.upper(),
            'Total Images': data['total_images'],
            'Thin Images': data['thin_images'],
            'Thick Images': data['thick_images'],
            'Thin %': f"{data['thin_pct']:.1f}%",
            'Thick %': f"{data['thick_pct']:.1f}%",
            'With Hemorrhage': data['images_with_hemorrhage'],
            'Normal (No Hem)': data['images_without_hemorrhage'],
            'Total Instances': sum(data['class_counts'].values()),
            'Avg Boxes/Image': f"{data['avg_boxes_per_image']:.2f}",
        })

df_splits = pd.DataFrame(split_summary)
print("\n" + df_splits.to_string(index=False))

# ============================================================================
# 3. NORMAL IMAGES DISTRIBUTION BY DOMAIN
# ============================================================================

print("\n" + "="*100)
print("3. NORMAL IMAGES DISTRIBUTION (No Hemorrhage) BY DOMAIN")
print("="*100)

normal_summary = []
for split, data in split_data.items():
    if data:
        total_normal = data['images_without_hemorrhage']
        thin_normal = data['thin_without_hem']
        thick_normal = data['thick_without_hem']
        
        normal_summary.append({
            'Split': split.upper(),
            'Total Normal': total_normal,
            'Thin Normal': thin_normal,
            'Thick Normal': thick_normal,
            'Thin %': f"{thin_normal/total_normal*100:.1f}%" if total_normal > 0 else "0%",
            'Thick %': f"{thick_normal/total_normal*100:.1f}%" if total_normal > 0 else "0%",
            'Normal/Total %': f"{total_normal/data['total_images']*100:.1f}%",
        })

df_normal = pd.DataFrame(normal_summary)
print("\n" + df_normal.to_string(index=False))

# Calculate totals
total_thin_normal = sum(data['thin_without_hem'] for data in split_data.values() if data)
total_thick_normal = sum(data['thick_without_hem'] for data in split_data.values() if data)
total_thin_with_hem = sum(data['thin_with_hem'] for data in split_data.values() if data)
total_thick_with_hem = sum(data['thick_with_hem'] for data in split_data.values() if data)

print(f"\nOverall Normal Image Distribution:")
print(f"  - Thin slices (normal): {total_thin_normal:,} ({total_thin_normal/total_without_hem*100:.1f}% of all normal)")
print(f"  - Thick slices (normal): {total_thick_normal:,} ({total_thick_normal/total_without_hem*100:.1f}% of all normal)")
print(f"\nDomain-wise Hemorrhage vs Normal:")
print(f"  Thin slices  - With hemorrhage: {total_thin_with_hem:,} | Normal: {total_thin_normal:,} ({total_thin_normal/total_thin*100:.1f}% normal)")
print(f"  Thick slices - With hemorrhage: {total_thick_with_hem:,} | Normal: {total_thick_normal:,} ({total_thick_normal/total_thick*100:.1f}% normal)")

# ============================================================================
# 4. PER-CLASS INSTANCE DISTRIBUTION
# ============================================================================

print("\n" + "="*100)
print("4. PER-CLASS INSTANCE DISTRIBUTION")
print("="*100)

class_summary = []
for class_id, class_name in enumerate(class_names):
    train_count = split_data['train']['class_counts'].get(class_id, 0) if split_data.get('train') else 0
    val_count = split_data['valid']['class_counts'].get(class_id, 0) if split_data.get('valid') else 0
    test_count = split_data['test']['class_counts'].get(class_id, 0) if split_data.get('test') else 0
    total = train_count + val_count + test_count
    
    class_summary.append({
        'Class': class_name,
        'Class ID': class_id,
        'Train': train_count,
        'Valid': val_count,
        'Test': test_count,
        'Total': total,
        'Train %': f"{train_count/total*100:.1f}%" if total > 0 else "0%",
        'Valid %': f"{val_count/total*100:.1f}%" if total > 0 else "0%",
        'Test %': f"{test_count/total*100:.1f}%" if total > 0 else "0%",
    })

df_classes = pd.DataFrame(class_summary)
print("\n" + df_classes.to_string(index=False))

# ============================================================================
# 5. DOMAIN-SPECIFIC CLASS DISTRIBUTION
# ============================================================================

print("\n" + "="*100)
print("5. DOMAIN-SPECIFIC CLASS DISTRIBUTION (Thin vs Thick)")
print("="*100)

domain_class_summary = []
for class_id, class_name in enumerate(class_names):
    # Train split
    train_thin = split_data['train']['thin_class_counts'].get(class_id, 0) if split_data.get('train') else 0
    train_thick = split_data['train']['thick_class_counts'].get(class_id, 0) if split_data.get('train') else 0
    
    # Valid split
    val_thin = split_data['valid']['thin_class_counts'].get(class_id, 0) if split_data.get('valid') else 0
    val_thick = split_data['valid']['thick_class_counts'].get(class_id, 0) if split_data.get('valid') else 0
    
    # Test split
    test_thin = split_data['test']['thin_class_counts'].get(class_id, 0) if split_data.get('test') else 0
    test_thick = split_data['test']['thick_class_counts'].get(class_id, 0) if split_data.get('test') else 0
    
    total_thin = train_thin + val_thin + test_thin
    total_thick = train_thick + val_thick + test_thick
    total = total_thin + total_thick
    
    domain_class_summary.append({
        'Class': class_name,
        'ID': class_id,
        'Thin (Train)': train_thin,
        'Thick (Train)': train_thick,
        'Thin (Val)': val_thin,
        'Thick (Val)': val_thick,
        'Thin (Test)': test_thin,
        'Thick (Test)': test_thick,
        'Total Thin': total_thin,
        'Total Thick': total_thick,
        'Thin %': f"{total_thin/total*100:.1f}%" if total > 0 else "0%",
        'Thick %': f"{total_thick/total*100:.1f}%" if total > 0 else "0%",
    })

df_domain_classes = pd.DataFrame(domain_class_summary)
print("\n" + df_domain_classes.to_string(index=False))

# ============================================================================
# 6. PATIENT-LEVEL ANALYSIS
# ============================================================================

print("\n" + "="*100)
print("6. PATIENT-LEVEL ANALYSIS")
print("="*100)

patient_summary = []
for split in splits:
    split_path = os.path.join(DATASET_PATH, split)
    thin_patients, thick_patients = extract_patient_ids(split_path, split)
    
    # Check for overlap
    overlap = thin_patients & thick_patients
    
    patient_summary.append({
        'Split': split.upper(),
        'Thin Patients': len(thin_patients),
        'Thick Patients': len(thick_patients),
        'Total Unique': len(thin_patients | thick_patients),
        'Overlap': len(overlap),
        'Warning': '⚠️  Patient overlap!' if overlap else '✓ No overlap'
    })

df_patients = pd.DataFrame(patient_summary)
print("\n" + df_patients.to_string(index=False))

# Check cross-split patient leakage
all_thin_patients = set()
all_thick_patients = set()
for split in splits:
    split_path = os.path.join(DATASET_PATH, split)
    thin, thick = extract_patient_ids(split_path, split)
    all_thin_patients.update(thin)
    all_thick_patients.update(thick)

print(f"\nTotal Unique Thin Patients: {len(all_thin_patients)}")
print(f"Total Unique Thick Patients: {len(all_thick_patients)}")

# Check for patient leakage across splits
train_thin, train_thick = extract_patient_ids(os.path.join(DATASET_PATH, 'train'), 'train')
val_thin, val_thick = extract_patient_ids(os.path.join(DATASET_PATH, 'valid'), 'valid')
test_thin, test_thick = extract_patient_ids(os.path.join(DATASET_PATH, 'test'), 'test')

thin_leakage_tv = train_thin & val_thin
thin_leakage_tt = train_thin & test_thin
thin_leakage_vt = val_thin & test_thin

thick_leakage_tv = train_thick & val_thick
thick_leakage_tt = train_thick & test_thick
thick_leakage_vt = val_thick & test_thick

print("\nCross-Split Patient Leakage Check:")
print(f"  Thin - Train/Valid overlap: {len(thin_leakage_tv)} patients {'⚠️' if thin_leakage_tv else '✓'}")
print(f"  Thin - Train/Test overlap: {len(thin_leakage_tt)} patients {'⚠️' if thin_leakage_tt else '✓'}")
print(f"  Thin - Valid/Test overlap: {len(thin_leakage_vt)} patients {'⚠️' if thin_leakage_vt else '✓'}")
print(f"  Thick - Train/Valid overlap: {len(thick_leakage_tv)} patients {'⚠️' if thick_leakage_tv else '✓'}")
print(f"  Thick - Train/Test overlap: {len(thick_leakage_tt)} patients {'⚠️' if thick_leakage_tt else '✓'}")
print(f"  Thick - Valid/Test overlap: {len(thick_leakage_vt)} patients {'⚠️' if thick_leakage_vt else '✓'}")

# ============================================================================
# 7. VISUALIZATIONS
# ============================================================================

print("\n" + "="*100)
print("7. GENERATING VISUALIZATIONS")
print("="*100)

# Create figure with multiple subplots
fig = plt.figure(figsize=(24, 18))

# 7.1 - Images per split
ax1 = plt.subplot(4, 3, 1)
splits_list = [data['split_name'].upper() for data in split_data.values() if data]
image_counts = [data['total_images'] for data in split_data.values() if data]
bars = ax1.bar(splits_list, image_counts, color=['#3498db', '#2ecc71', '#e74c3c'])
ax1.set_title('Total Images per Split', fontsize=14, fontweight='bold')
ax1.set_ylabel('Number of Images')
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height):,}', ha='center', va='bottom', fontsize=11)

# 7.2 - Thin vs Thick distribution
ax2 = plt.subplot(4, 3, 2)
x = np.arange(len(splits_list))
width = 0.35
thin_counts = [data['thin_images'] for data in split_data.values() if data]
thick_counts = [data['thick_images'] for data in split_data.values() if data]
bars1 = ax2.bar(x - width/2, thin_counts, width, label='Thin (0.625-1.25mm)', color='#3498db')
bars2 = ax2.bar(x + width/2, thick_counts, width, label='Thick (5mm)', color='#e67e22')
ax2.set_title('Thin vs Thick Slices per Split', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(splits_list)
ax2.legend()
ax2.set_ylabel('Number of Images')

# 7.3 - Hemorrhage vs No Hemorrhage
ax3 = plt.subplot(4, 3, 3)
with_hem = [data['images_with_hemorrhage'] for data in split_data.values() if data]
without_hem = [data['images_without_hemorrhage'] for data in split_data.values() if data]
bars1 = ax3.bar(x - width/2, with_hem, width, label='With Hemorrhage', color='#e74c3c')
bars2 = ax3.bar(x + width/2, without_hem, width, label='Normal (No Hemorrhage)', color='#2ecc71')
ax3.set_title('Hemorrhage Presence per Split', fontsize=14, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(splits_list)
ax3.legend()
ax3.set_ylabel('Number of Images')

# 7.4 - Normal images by domain (NEW)
ax4 = plt.subplot(4, 3, 4)
thin_normal = [data['thin_without_hem'] for data in split_data.values() if data]
thick_normal = [data['thick_without_hem'] for data in split_data.values() if data]
bars1 = ax4.bar(x - width/2, thin_normal, width, label='Thin (Normal)', color='#3498db')
bars2 = ax4.bar(x + width/2, thick_normal, width, label='Thick (Normal)', color='#e67e22')
ax4.set_title('Normal Images Distribution by Domain', fontsize=14, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(splits_list)
ax4.legend()
ax4.set_ylabel('Number of Normal Images')

# 7.5 - Per-class distribution across splits
ax5 = plt.subplot(4, 3, 5)
class_train = [split_data['train']['class_counts'].get(i, 0) for i in range(len(class_names))]
class_val = [split_data['valid']['class_counts'].get(i, 0) for i in range(len(class_names))]
class_test = [split_data['test']['class_counts'].get(i, 0) for i in range(len(class_names))]

x_classes = np.arange(len(class_names))
width_class = 0.25
bars1 = ax5.bar(x_classes - width_class, class_train, width_class, label='Train', color='#3498db')
bars2 = ax5.bar(x_classes, class_val, width_class, label='Valid', color='#2ecc71')
bars3 = ax5.bar(x_classes + width_class, class_test, width_class, label='Test', color='#e74c3c')
ax5.set_title('Instance Distribution per Class', fontsize=14, fontweight='bold')
ax5.set_xticks(x_classes)
ax5.set_xticklabels(class_names, rotation=45)
ax5.legend()
ax5.set_ylabel('Number of Instances')

# 7.6 - Total instances per class
ax6 = plt.subplot(4, 3, 6)
total_per_class = [sum([split_data[split]['class_counts'].get(i, 0) for split in splits]) 
                   for i in range(len(class_names))]
bars = ax6.barh(class_names, total_per_class, color='#9b59b6')
ax6.set_title('Total Instances per Class', fontsize=14, fontweight='bold')
ax6.set_xlabel('Number of Instances')
for i, bar in enumerate(bars):
    width = bar.get_width()
    ax6.text(width, bar.get_y() + bar.get_height()/2.,
             f'{int(width):,}', ha='left', va='center', fontsize=10)

# 7.7 - Class imbalance visualization
ax7 = plt.subplot(4, 3, 7)
colors_pie = plt.cm.Set3(np.linspace(0, 1, len(class_names)))
wedges, texts, autotexts = ax7.pie(total_per_class, labels=class_names, autopct='%1.1f%%',
                                     colors=colors_pie, startangle=90)
ax7.set_title('Class Distribution (Overall)', fontsize=14, fontweight='bold')
for autotext in autotexts:
    autotext.set_color('black')
    autotext.set_fontsize(10)
    autotext.set_weight('bold')

# 7.8 - Thin vs Thick per class (stacked)
ax8 = plt.subplot(4, 3, 8)
thin_per_class = [sum([split_data[split]['thin_class_counts'].get(i, 0) for split in splits]) 
                  for i in range(len(class_names))]
thick_per_class = [sum([split_data[split]['thick_class_counts'].get(i, 0) for split in splits]) 
                   for i in range(len(class_names))]
x_pos = np.arange(len(class_names))
bars1 = ax8.bar(x_pos, thin_per_class, label='Thin slices', color='#3498db')
bars2 = ax8.bar(x_pos, thick_per_class, bottom=thin_per_class, label='Thick slices', color='#e67e22')
ax8.set_title('Thin vs Thick Instances per Class', fontsize=14, fontweight='bold')
ax8.set_xticks(x_pos)
ax8.set_xticklabels(class_names, rotation=45)
ax8.legend()
ax8.set_ylabel('Number of Instances')

# 7.9 - Boxes per image distribution
ax9 = plt.subplot(4, 3, 9)
all_boxes = []
for split, data in split_data.items():
    if data:
        all_boxes.extend(data['boxes_per_image'])
if all_boxes:
    ax9.hist(all_boxes, bins=range(0, max(all_boxes)+2), color='#1abc9c', edgecolor='black', alpha=0.7)
    ax9.set_title('Distribution of Boxes per Image', fontsize=14, fontweight='bold')
    ax9.set_xlabel('Number of Boxes')
    ax9.set_ylabel('Frequency')
    ax9.axvline(np.mean(all_boxes), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(all_boxes):.2f}')
    ax9.legend()

# 7.10 - Domain proportion by split (pie charts)
ax10 = plt.subplot(4, 3, 10)
train_data = split_data['train']
domain_sizes = [train_data['thin_images'], train_data['thick_images']]
colors = ['#3498db', '#e67e22']
wedges, texts, autotexts = ax10.pie(domain_sizes, labels=['Thin', 'Thick'], 
                                     autopct='%1.1f%%', colors=colors, startangle=90)
ax10.set_title('Training Set Domain Distribution', fontsize=14, fontweight='bold')
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(12)
    autotext.set_weight('bold')

# 7.11 - Hemorrhage vs Normal by domain (NEW)
ax11 = plt.subplot(4, 3, 11)
categories = ['Thin Slices', 'Thick Slices']
hem_counts = [total_thin_with_hem, total_thick_with_hem]
normal_counts = [total_thin_normal, total_thick_normal]
x_domain = np.arange(len(categories))
width_domain = 0.35
bars1 = ax11.bar(x_domain - width_domain/2, hem_counts, width_domain, label='With Hemorrhage', color='#e74c3c')
bars2 = ax11.bar(x_domain + width_domain/2, normal_counts, width_domain, label='Normal', color='#2ecc71')
ax11.set_title('Hemorrhage vs Normal by Domain', fontsize=14, fontweight='bold')
ax11.set_xticks(x_domain)
ax11.set_xticklabels(categories)
ax11.legend()
ax11.set_ylabel('Number of Images')
# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax11.text(bar.get_x() + bar.get_width()/2., height,
                 f'{int(height):,}', ha='center', va='bottom', fontsize=10)

# 7.12 - Overall Hemorrhage vs Normal pie chart (NEW)
ax12 = plt.subplot(4, 3, 12)
overall_counts = [total_with_hem, total_without_hem]
colors_hem = ['#e74c3c', '#2ecc71']
wedges, texts, autotexts = ax12.pie(overall_counts, labels=['With Hemorrhage', 'Normal'], 
                                      autopct='%1.1f%%', colors=colors_hem, startangle=90)
ax12.set_title('Overall Hemorrhage Distribution', fontsize=14, fontweight='bold')
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(12)
    autotext.set_weight('bold')

plt.tight_layout()
plt.savefig('/Users/yugambahl/Desktop/brain_ct/data/dataset_analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualizations saved as 'dataset_analysis.png'")

# ============================================================================
# 8. CLASS IMBALANCE ANALYSIS
# ============================================================================

print("\n" + "="*100)
print("8. CLASS IMBALANCE ANALYSIS")
print("="*100)

max_instances = max(total_per_class)
min_instances = min(total_per_class)
imbalance_ratio = max_instances / min_instances if min_instances > 0 else float('inf')

print(f"\nClass Imbalance Ratio: {imbalance_ratio:.2f}:1")
print(f"Most Common Class: {class_names[total_per_class.index(max_instances)]} ({max_instances} instances)")
print(f"Least Common Class: {class_names[total_per_class.index(min_instances)]} ({min_instances} instances)")

print("\nRecommended Actions:")
for i, (class_name, count) in enumerate(zip(class_names, total_per_class)):
    if count < 50:
        print(f"  ⚠️  {class_name}: CRITICALLY LOW ({count} instances) - Consider data collection or removal")
    elif count < 200:
        print(f"  ⚠️  {class_name}: LOW ({count} instances) - Consider class weights or oversampling")
    elif count > 500:
        print(f"  ✓ {class_name}: SUFFICIENT ({count} instances)")

# ============================================================================
# 9. EXPORT SUMMARY TO CSV
# ============================================================================

print("\n" + "="*100)
print("9. EXPORTING SUMMARY TO CSV")
print("="*100)

# Export split summary
df_splits.to_csv('/Users/yugambahl/Desktop/brain_ct/data/dataset_split_summary.csv', index=False)
print("✓ Split summary saved to 'dataset_split_summary.csv'")

# Export class summary
df_classes.to_csv('/Users/yugambahl/Desktop/brain_ct/data/dataset_class_summary.csv', index=False)
print("✓ Class summary saved to 'dataset_class_summary.csv'")

# Export domain-class summary
df_domain_classes.to_csv('/Users/yugambahl/Desktop/brain_ct/data/dataset_domain_class_summary.csv', index=False)
print("✓ Domain-class summary saved to 'dataset_domain_class_summary.csv'")

# Export patient summary
df_patients.to_csv('/Users/yugambahl/Desktop/brain_ct/data/dataset_patient_summary.csv', index=False)
print("✓ Patient summary saved to 'dataset_patient_summary.csv'")

# Export normal images summary (NEW)
df_normal.to_csv('/Users/yugambahl/Desktop/brain_ct/data/dataset_normal_images_summary.csv', index=False)
print("✓ Normal images summary saved to 'dataset_normal_images_summary.csv'")

print("\n" + "="*100)
print("ANALYSIS COMPLETE!")
print("="*100)
print("\nGenerated Files:")
print("  1. dataset_analysis.png - Comprehensive visualization")
print("  2. dataset_split_summary.csv - Split-wise statistics")
print("  3. dataset_class_summary.csv - Per-class distribution")
print("  4. dataset_domain_class_summary.csv - Domain-specific class distribution")
print("  5. dataset_patient_summary.csv - Patient-level analysis")
print("  6. dataset_normal_images_summary.csv - Normal images distribution")
print("\n" + "="*100)