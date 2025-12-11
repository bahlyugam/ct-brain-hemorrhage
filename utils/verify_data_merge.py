"""
Verification script to check if dataset combination worked correctly
Run AFTER combining datasets
"""

import os
from pathlib import Path
from collections import defaultdict

COMBINED_DATASET_PATH = Path("/Users/yugambahl/Desktop/brain_ct/data/UAT_CT_BRAIN_HEMORRHAGE.v2i.yolov8_combined")

CLASS_NAMES = ['EDH', 'HC', 'IPH', 'IVH', 'SAH', 'SDH']

def verify_dataset():
    print("="*80)
    print("DATASET COMBINATION VERIFICATION")
    print("="*80)
    
    issues_found = []
    
    for split in ['train', 'valid', 'test']:
        label_dir = COMBINED_DATASET_PATH / split / "labels"
        
        if not label_dir.exists():
            print(f"\n⚠️  {split} labels directory not found!")
            continue
        
        print(f"\n{split.upper()} SPLIT:")
        print("-" * 80)
        
        thin_classes = defaultdict(int)
        thick_classes = defaultdict(int)
        invalid_classes = []
        
        label_files = list(label_dir.glob("*.txt"))
        
        for label_file in label_files:
            is_thin = label_file.name.startswith('thin_')
            is_thick = label_file.name.startswith('thick_')
            
            with open(label_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        try:
                            class_idx = int(parts[0])
                            
                            # Check valid range
                            if class_idx < 0 or class_idx > 5:
                                invalid_classes.append((label_file.name, line_num, class_idx))
                                issues_found.append(f"{split}/{label_file.name}:{line_num} - Invalid class {class_idx}")
                            else:
                                if is_thin:
                                    thin_classes[class_idx] += 1
                                elif is_thick:
                                    thick_classes[class_idx] += 1
                        except ValueError:
                            issues_found.append(f"{split}/{label_file.name}:{line_num} - Cannot parse class index")
        
        # Display results
        print(f"Total label files: {len(label_files)}")
        print(f"  Thin: {sum(1 for f in label_files if f.name.startswith('thin_'))}")
        print(f"  Thick: {sum(1 for f in label_files if f.name.startswith('thick_'))}")
        
        print(f"\nClass Distribution:")
        print(f"{'Class':<6} {'Name':<6} {'Thin':<10} {'Thick':<10} {'Total':<10} {'Status':<20}")
        print("-" * 70)
        
        for class_idx, class_name in enumerate(CLASS_NAMES):
            thin_count = thin_classes[class_idx]
            thick_count = thick_classes[class_idx]
            total = thin_count + thick_count
            
            # Determine status
            status = "✓ OK"
            
            # Class 0 (EDH) should ONLY be in thick (new dataset)
            if class_idx == 0:
                if thin_count > 0:
                    status = "⚠️  FOUND IN THIN!"
                    issues_found.append(f"{split} - Class 0 (EDH) found in thin dataset (should only be in thick)")
                elif thick_count == 0:
                    status = "⚠️  NOT FOUND"
            
            # Classes 1-5 should be in thin (from old dataset after remapping)
            if class_idx >= 1:
                if thin_count == 0:
                    status = "⚠️  NOT IN THIN"
                    issues_found.append(f"{split} - Class {class_idx} ({class_name}) not found in thin dataset")
            
            print(f"{class_idx:<6} {class_name:<6} {thin_count:<10} {thick_count:<10} {total:<10} {status:<20}")
        
        if invalid_classes:
            print(f"\n⚠️  INVALID CLASS INDICES FOUND:")
            for filename, line_num, class_idx in invalid_classes[:5]:  # Show first 5
                print(f"  {filename}:{line_num} - Class {class_idx}")
            if len(invalid_classes) > 5:
                print(f"  ... and {len(invalid_classes) - 5} more")
    
    # Final summary
    print("\n" + "="*80)
    print("VERIFICATION SUMMARY")
    print("="*80)
    
    if not issues_found:
        print("✓ ALL CHECKS PASSED!")
        print("✓ Class indices are in valid range [0-5]")
        print("✓ Class 0 (EDH) only appears in thick slices")
        print("✓ Classes 1-5 appear in thin slices (remapped from old dataset)")
        print("\n🎉 Dataset combination was SUCCESSFUL!")
    else:
        print(f"⚠️  FOUND {len(issues_found)} ISSUES:")
        for issue in issues_found[:10]:  # Show first 10
            print(f"  - {issue}")
        if len(issues_found) > 10:
            print(f"  ... and {len(issues_found) - 10} more")
        print("\n❌ Please review and fix the issues above!")
    
    print("="*80)

if __name__ == "__main__":
    verify_dataset()