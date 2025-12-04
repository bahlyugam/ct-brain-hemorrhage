import modal
import os
import wandb
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from datetime import datetime
import yaml
from pathlib import Path

# Augmentation strategy configuration
USE_PREAUGMENTED = False  # Using extensive on-the-fly augmentations (RSNA winners' strategy)
ORIGINAL_DATA_DIR = "/Users/yugambahl/Desktop/brain_ct/data/training_datasets/archived/current_20251105/UAT_CT_BRAIN_HEMORRHAGE.v2i.yolov8_combined"

if USE_PREAUGMENTED:
    local_data_dir = "/Users/yugambahl/Desktop/brain_ct/data/training_datasets/archived/current_20251105/UAT_CT_BRAIN_HEMORRHAGE.v2i.yolov8_combined_augmented"
    print(f"\n{'='*80}")
    print("USING PRE-AUGMENTED DATASET (4x DATA, 8-10x FASTER TRAINING)")
    print(f"{'='*80}")
    print(f"Dataset: Pre-augmented (23,956 images)")
    print("Pre-augmented with: flip, rotation ±10°, brightness/contrast")
    print("Online augmentation: minimal (light fliplr only)")
    print("Expected training speed: 0.8-1.0 it/s")
    print(f"{'='*80}\n")
else:
    local_data_dir = ORIGINAL_DATA_DIR
    print(f"\n{'='*80}")
    print("USING EXTENSIVE ON-THE-FLY AUGMENTATIONS (RSNA WINNERS' STRATEGY)")
    print(f"{'='*80}")
    print(f"Dataset: Original (5,989 images)")
    print("On-the-fly augmentations: rotation ±10°, translation 6.25%, scale 50%,")
    print("  horizontal flip 50%, mosaic 0.8, HSV brightness/contrast")
    print("Expected training speed: 0.1-0.2 it/s (slower but better results)")
    print(f"{'='*80}\n")

# Multi-resolution training hyperparameters
BATCH_SIZE = 24  # Will be split 50-50 between thin and thick
IMAGE_SIZE = 640  # Increased from 512 for better multi-scale
EPOCHS = 200
PATIENCE = 50  # Reduced from 100 - with conservative augmentation, model should converge faster
MODEL = "yolov8m"  # Upgraded from yolov8s - optimal for 7.5K images + 6 classes
VERSION = "v2"  # Conservative augmentation + focal loss preparation (Phase 1+2)
MODEL_PATH = f"{MODEL}.pt"

# Fine-tuning parameters for multi-resolution
FREEZE_LAYERS = 0  # No freezing - all layers train (optimal for binary detection with 23,956 images)
LEARNING_RATE = 0.005  
FINAL_LR_FACTOR = 0.1
WARMUP_EPOCHS = 5

# Domain mixing ratios
THIN_SLICE_RATIO = 0.5  # 50% thin slices per batch
THICK_SLICE_RATIO = 0.5  # 50% thick slices per batch

# Create a Modal volume to store the model
# volume = modal.Volume.from_name(f"{MODEL}-{VERSION}-brain-ct-hemorrhage", create_if_missing=True)
volume = modal.Volume.from_name(f"{MODEL}-v9_conservative_focal-brain-ct-hemorrhage", create_if_missing=True)

# Define the Modal image with necessary dependencies
image = (
    modal.Image.debian_slim()
    .apt_install(
        "libgl1-mesa-glx",
        "libglib2.0-0",
        "libsm6",
        "libxext6",
        "libxrender-dev",
        "libgstreamer1.0-0",
        "gstreamer1.0-plugins-base",
        "gstreamer1.0-plugins-good",
    )
    .pip_install(
        "ultralytics",
        "wandb",
        "opencv-python-headless",
        "pandas",
        "PyYAML",
        "scikit-learn",
        "matplotlib",
        "supervision",
        "albumentations>=1.3.0",  # Evidence-based augmentations
        "grpclib==0.4.6",
        "h2<5.0.0"
    )
    .add_local_dir(local_data_dir, "/datasets")
    .add_local_file(
        "/Users/yugambahl/Desktop/brain_ct/ct_augmentations.py",
        "/root/ct_augmentations.py"
    )
    .add_local_file(
        "/Users/yugambahl/Desktop/brain_ct/yolo_augmented_dataset.py",
        "/root/yolo_augmented_dataset.py"
    )
    .add_local_file(
        "/Users/yugambahl/Desktop/brain_ct/custom_loss.py",
        "/root/custom_loss.py"
    )
)

app = modal.App(f"{MODEL}-{VERSION}-brain-ct-hemorrhage-training")
def find_latest_model(model_dir="/model"):
    """Find the latest saved model by looking for files with epoch information."""
    # First check for emergency checkpoint (from crashes)
    emergency_path = f"{model_dir}/emergency_checkpoint.pt"
    if os.path.exists(emergency_path):
        print(f"Found emergency checkpoint: {emergency_path}")
        return emergency_path
    
    # Then check for latest_checkpoint.pt (most recent)
    latest_path = f"{model_dir}/latest_checkpoint.pt"
    if os.path.exists(latest_path):
        print(f"Found latest checkpoint: {latest_path}")
        return latest_path
    
    # Then check for numbered checkpoints
    model_files = glob.glob(f"{model_dir}/checkpoint_epoch_*.pt")
    
    if model_files:
        model_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        latest_checkpoint = model_files[-1]
        print(f"Found numbered checkpoint: {latest_checkpoint}")
        return latest_checkpoint
    
    # Check for best.pt files with epoch numbers
    best_files = glob.glob(f"{model_dir}/best.pt_*.pt")
    if best_files:
        best_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        latest_best = best_files[-1]
        print(f"Found numbered best model: {latest_best}")
        return latest_best
    
    # Finally check for standard best.pt
    if os.path.exists(f"{model_dir}/best.pt"):
        print(f"Found standard best model: {model_dir}/best.pt")
        return f"{model_dir}/best.pt"
    
    print("No checkpoint found")
    return None

def get_epoch_from_model_path(model_path):
    """Extract the epoch number from model path."""
    if model_path is None:
        return 0
        
    # Handle latest_checkpoint.pt - need to get epoch from training_params.json
    if model_path.endswith("/latest_checkpoint.pt"):
        params_path = "/model/training_params.json"
        if os.path.exists(params_path):
            try:
                with open(params_path, 'r') as f:
                    params = json.load(f)
                    if 'last_epoch' in params:
                        return params['last_epoch']
            except:
                pass
        return 0
        
    # Handle standard best.pt
    if model_path.endswith("/best.pt"):
        params_path = "/model/training_params.json"
        if os.path.exists(params_path):
            try:
                with open(params_path, 'r') as f:
                    params = json.load(f)
                    if 'last_epoch' in params:
                        return params['last_epoch']
            except:
                pass
        return 0
        
    # Handle numbered checkpoints (checkpoint_epoch_X.pt or best.pt_X.pt)
    try:
        if "checkpoint_epoch_" in model_path:
            epoch = int(model_path.split('checkpoint_epoch_')[-1].split('.')[0])
        elif "best.pt_" in model_path:
            epoch = int(model_path.split('_')[-1].split('.')[0])
        else:
            epoch = int(model_path.split('_')[-1].split('.')[0])
        return epoch
    except (ValueError, IndexError):
        return 0

def save_training_params(params, path):
    """Save training parameters to a JSON file."""
    with open(path, 'w') as f:
        json.dump(params, f)

def cleanup_old_checkpoints(model_dir="/model", keep_last=5):
    """Clean up old checkpoints to prevent disk space issues."""
    try:
        # Clean up old numbered checkpoints
        checkpoint_files = glob.glob(f"{model_dir}/checkpoint_epoch_*.pt")
        if len(checkpoint_files) > keep_last:
            checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
            files_to_remove = checkpoint_files[:-keep_last]
            for file_path in files_to_remove:
                os.remove(file_path)
                print(f"Removed old checkpoint: {os.path.basename(file_path)}")
        
        # Clean up old best.pt files
        best_files = glob.glob(f"{model_dir}/best.pt_*.pt")
        if len(best_files) > keep_last:
            best_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
            files_to_remove = best_files[:-keep_last]
            for file_path in files_to_remove:
                os.remove(file_path)
                print(f"Removed old best model: {os.path.basename(file_path)}")
                
    except Exception as e:
        print(f"Warning: Failed to cleanup old checkpoints: {e}")

def load_training_params(path):
    """Load training parameters from a JSON file."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except:
        return None
    
def convert_polygons_to_boxes(label_path):
    """Convert polygon annotations to bounding boxes using supervision"""
    import supervision as sv
    
    if not os.path.exists(label_path):
        return
    
    converted_lines = []
    
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        parts = line.strip().split()
        
        if len(parts) < 5:
            continue
            
        class_id = parts[0]
        coords = [float(x) for x in parts[1:]]
        
        # If more than 4 coordinates, it's a polygon
        if len(coords) > 4:
            # Reshape to polygon format: [[x1,y1], [x2,y2], ...]
            polygon = np.array(coords).reshape(-1, 2)
            
            # Convert polygon to xyxy bounding box (normalized coordinates)
            xyxy = sv.polygon_to_xyxy(polygon)
            
            # Convert xyxy to xywh (YOLO format) - already normalized
            x1, y1, x2, y2 = xyxy
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2
            width = x2 - x1
            height = y2 - y1
            
            converted_lines.append(f"{class_id} {x_center} {y_center} {width} {height}\n")
        else:
            # Already a box, keep as is
            converted_lines.append(line)
    
    # Write converted labels back
    with open(label_path, 'w') as f:
        f.writelines(converted_lines)

def preprocess_dataset_labels(base_path="/datasets"):
    """Convert all polygon labels to bounding boxes and filter out EDH (class 0)"""
    import supervision as sv

    print("\nConverting polygon annotations to bounding boxes and filtering EDH class...")

    edh_filtered_count = 0

    for split in ['train', 'valid', 'test']:
        label_dir = os.path.join(base_path, split, 'labels')

        if not os.path.exists(label_dir):
            continue

        label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]

        print(f"Processing {len(label_files)} label files in {split}...")

        for label_file in label_files:
            label_path = os.path.join(label_dir, label_file)
            try:
                convert_polygons_to_boxes(label_path)

                # Filter out EDH (class 0) annotations
                with open(label_path, 'r') as f:
                    lines = f.readlines()

                filtered_lines = []
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        if class_id != 0:  # Skip EDH (class 0)
                            filtered_lines.append(line)
                        else:
                            edh_filtered_count += 1

                # Write back filtered labels
                with open(label_path, 'w') as f:
                    f.writelines(filtered_lines)

            except Exception as e:
                print(f"Warning: Could not process {label_file}: {e}")

    print(f"Polygon conversion complete! Filtered out {edh_filtered_count} EDH instances.\n")

def create_stratified_dataset_yaml(base_path="/datasets", output_path="/model/stratified_data.yaml"):
    """
    Create a standard dataset YAML compatible with YOLOv8.
    Mixed training will happen naturally since both domains are in the same directories.
    """
    import yaml
    
    # Read original combined data.yaml
    original_yaml = os.path.join(base_path, "data.yaml")
    with open(original_yaml, 'r') as f:
        data_config = yaml.safe_load(f)
    
    # Count images by domain
    def count_by_domain(split_dir):
        img_dir = os.path.join(base_path, split_dir, "images")
        if not os.path.exists(img_dir):
            return 0, 0
        
        thin_count = sum(1 for f in os.listdir(img_dir) if f.startswith('thin_') and f.endswith(('.jpg', '.jpeg', '.png')))
        thick_count = sum(1 for f in os.listdir(img_dir) if f.startswith('thick_') and f.endswith(('.jpg', '.jpeg', '.png')))
        
        return thin_count, thick_count
    
    train_thin, train_thick = count_by_domain("train")
    val_thin, val_thick = count_by_domain("valid")
    test_thin, test_thick = count_by_domain("test")
    
    print(f"\nDataset composition:")
    print(f"Train - Thin: {train_thin}, Thick: {train_thick}, Total: {train_thin + train_thick}")
    print(f"Val   - Thin: {val_thin}, Thick: {val_thick}, Total: {val_thin + val_thick}")
    print(f"Test  - Thin: {test_thin}, Thick: {test_thick}, Total: {test_thin + test_thick}")
    
    # Create standard YOLOv8 compatible config
    stratified_config = {
        'path': base_path,
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
        'nc': data_config['nc'],
        'names': data_config['names'],
    }
    
    # Save config
    with open(output_path, 'w') as f:
        yaml.dump(stratified_config, f, default_flow_style=False)
    
    print(f"\nCreated dataset config at {output_path}")
    print(f"Mixed training will use both thin and thick slices from combined directories")
    
    return stratified_config

@app.function(
    image=image,
    gpu="A10G",
    secrets=[modal.Secret.from_name("wandb-api-key")],
    volumes={"/model": volume},
    timeout=86400,  # 24 hours
    memory=32768,    # 32GB RAM for faster data loading
    cpu=8,          # 8 CPUs for parallel data loading (critical for speed!)
    min_containers=0  # Don't keep containers warm to save resources
)
def train_yolo(resume=False):
    global BATCH_SIZE, IMAGE_SIZE, EPOCHS, MODEL, VERSION, MODEL_PATH
    global FREEZE_LAYERS, LEARNING_RATE, FINAL_LR_FACTOR

    # Import augmentation utilities
    import sys
    sys.path.insert(0, '/root')
    from yolo_augmented_dataset import (
        get_yolo_training_hyperparameters,
        print_augmentation_summary,
        analyze_dataset_slices
    )
    
    # Check if we should auto-resume even if resume=False
    if not resume:
        latest_model = find_latest_model()
        if latest_model:
            latest_epoch = get_epoch_from_model_path(latest_model)
            if latest_epoch > 0:
                print(f"\n{'='*80}")
                print(f"AUTO-RESUME DETECTED")
                print(f"{'='*80}")
                print(f"Found existing training at epoch {latest_epoch}")
                print(f"Automatically resuming from checkpoint...")
                resume = True

    batch_size = BATCH_SIZE
    image_size = IMAGE_SIZE
    total_epochs = EPOCHS
    model_name = MODEL
    version = VERSION
    model_path = MODEL_PATH

    params_path = "/model/training_params.json"
    stratified_yaml_path = "/model/stratified_data.yaml"

    preprocess_dataset_labels("/datasets")

    # Create stratified dataset YAML
    print("Creating stratified dataset configuration...")
    stratified_config = create_stratified_dataset_yaml("/datasets", stratified_yaml_path)

    # Analyze dataset to determine slice type distribution
    print("\nAnalyzing dataset composition...")
    slice_stats = analyze_dataset_slices(stratified_yaml_path)
    print(f"Thin slices: {slice_stats['thin']} ({slice_stats['thin_ratio']*100:.1f}%)")
    print(f"Thick slices: {slice_stats['thick']} ({slice_stats['thick_ratio']*100:.1f}%)")
    print(f"Dominant slice type: {slice_stats['dominant']}")

    # Use mixed strategy for balanced datasets
    slice_type = slice_stats['dominant']
    
    # Initialize wandb
    wandb.login(key=os.environ["WANDB_API_KEY"])
    
    # Variables for tracking
    start_epoch = 0
    remaining_epochs = total_epochs
    
    # Baseline performance tracking (for catastrophic forgetting detection)
    baseline_thin_map = None
    
    # Check if we need to resume training
    if resume:
        latest_model_path = find_latest_model()
        if latest_model_path:
            print(f"Resuming training from {latest_model_path}")
            
            start_epoch = get_epoch_from_model_path(latest_model_path)
            print(f"Last completed epoch: {start_epoch}")
            
            remaining_epochs = total_epochs - start_epoch
            
            if remaining_epochs <= 0:
                print(f"Training already completed ({start_epoch}/{total_epochs} epochs). Nothing to do.")
                return
                
            print(f"Will train for {remaining_epochs} more epochs (total target: {total_epochs})")
            
            model = YOLO(latest_model_path)
            
            params = load_training_params(params_path)
            if params:
                if 'baseline_thin_map' in params:
                    baseline_thin_map = params['baseline_thin_map']
                    if baseline_thin_map is not None:
                        print(f"Loaded baseline thin slice mAP: {baseline_thin_map:.4f}")
                    else:
                        print("Baseline thin slice mAP not available (will be set after first evaluation)")
            
            run_id = params.get('wandb_run_id') if params else None
            if run_id:
                try:
                    run = wandb.init(project="brain-ct-hemorrhage", id=run_id, resume="must")
                    print(f"Resumed existing wandb run: {run_id}")
                except:
                    run = wandb.init(project="brain-ct-hemorrhage", 
                                    name=f"{model_name}_multiresolution_{version}_resumed", 
                                    config={
                                        "model": model_name,
                                        "epochs": total_epochs,
                                        "batch_size": batch_size,
                                        "image_size": image_size,
                                        "freeze_layers": FREEZE_LAYERS,
                                        "learning_rate": LEARNING_RATE,
                                        "resumed_from_epoch": start_epoch,
                                    })
            else:
                run = wandb.init(project="brain-ct-hemorrhage", 
                                name=f"{model_name}_multiresolution_{version}_resumed", 
                                config={
                                    "model": model_name,
                                    "epochs": total_epochs,
                                    "batch_size": batch_size,
                                    "image_size": image_size,
                                    "freeze_layers": FREEZE_LAYERS,
                                    "learning_rate": LEARNING_RATE,
                                    "resumed_from_epoch": start_epoch,
                                })
        else:
            print("No checkpoint found, starting fresh")
            model = YOLO(model_path)
            run = wandb.init(project="brain-ct-hemorrhage", 
                            name=f"{model_name}_multiresolution_{version}", 
                            config={
                                "model": model_name,
                                "epochs": total_epochs,
                                "batch_size": batch_size,
                                "image_size": image_size,
                                "freeze_layers": FREEZE_LAYERS,
                                "learning_rate": LEARNING_RATE,
                                "training_strategy": "mixed_multiresolution",
                                "thin_ratio": THIN_SLICE_RATIO,
                                "thick_ratio": THICK_SLICE_RATIO,
                            })
    else:
        # Start fresh training
        model = YOLO(model_path)
        run = wandb.init(project="brain-ct-hemorrhage", 
                        name=f"{model_name}_multiresolution_{version}", 
                        config={
                            "model": model_name,
                            "epochs": total_epochs,
                            "batch_size": batch_size,
                            "image_size": image_size,
                            "freeze_layers": FREEZE_LAYERS,
                            "learning_rate": LEARNING_RATE,
                            "training_strategy": "mixed_multiresolution",
                            "thin_ratio": THIN_SLICE_RATIO,
                            "thick_ratio": THICK_SLICE_RATIO,
                        })
    
    # Use wandb.watch to log model gradients
    wandb.watch(model, log="all", log_freq=10)
    
    # Save current training parameters
    training_params = {
        'batch_size': batch_size,
        'image_size': image_size,
        'epochs': total_epochs,
        'wandb_run_id': run.id,
        'last_epoch': start_epoch,
        'model': model_name,
        'version': version,
        'freeze_layers': FREEZE_LAYERS,
        'learning_rate': LEARNING_RATE,
        'baseline_thin_map': baseline_thin_map,
    }
    save_training_params(training_params, params_path)
    
    # Define callback for domain-aware logging
    def on_train_epoch_end(trainer):
        nonlocal baseline_thin_map

        metrics = trainer.metrics
        current_epoch = trainer.epoch + start_epoch

        # Unfreeze DFL layer after 10 epochs for hemorrhage-specific box learning
        if current_epoch == 10:
            dfl_unfrozen = False
            for name, param in trainer.model.named_parameters():
                if name == "model.22.dfl.conv.weight":
                    param.requires_grad = True
                    dfl_unfrozen = True
                    print(f"\n{'='*80}")
                    print(f"UNFREEZING DFL LAYER AT EPOCH {current_epoch}")
                    print(f"{'='*80}")
                    print(f"Layer '{name}' is now trainable (requires_grad=True)")
                    print(f"This allows learning hemorrhage-specific bounding box characteristics")
                    print(f"{'='*80}\n")
                    break

            if not dfl_unfrozen:
                print(f"\n⚠️  Warning: Could not find 'model.22.dfl.conv.weight' to unfreeze at epoch {current_epoch}\n")

        log_dict = {
            "epoch": current_epoch,
        }
        
        # Map metrics
        metric_map = {
            "train/box_loss": ["train/box_loss", "train/loss"],
            "train/cls_loss": ["train/cls_loss", "train/loss_cls"],
            "train/dfl_loss": ["train/dfl_loss", "train/loss_dfl"],
            "val/box_loss": ["val/box_loss", "val/loss"],
            "val/cls_loss": ["val/cls_loss", "val/loss_cls"],
            "val/dfl_loss": ["val/dfl_loss", "val/loss_dfl"],
            "metrics/precision": ["metrics/precision(B)", "metrics/precision"],
            "metrics/recall": ["metrics/recall(B)", "metrics/recall"],
            "metrics/mAP50": ["metrics/mAP50(B)", "metrics/mAP50"],
            "metrics/mAP50-95": ["metrics/mAP50-95(B)", "metrics/mAP50-95"],
            "metrics/f1": ["metrics/f1(B)", "metrics/f1"],
        }
        
        for log_name, possible_names in metric_map.items():
            for metric_name in possible_names:
                if metric_name in metrics:
                    log_dict[log_name] = metrics[metric_name]
                    break        
        wandb.log(log_dict)
        
        # Update params
        training_params['last_epoch'] = current_epoch
        if baseline_thin_map is not None:
            training_params['baseline_thin_map'] = baseline_thin_map
        save_training_params(training_params, params_path)
        
        # Save checkpoints more frequently to prevent data loss
        if current_epoch % 5 == 0 or current_epoch == total_epochs:
            checkpoint_path = f"/model/checkpoint_epoch_{current_epoch}.pt"
            model.save(checkpoint_path)
            print(f"Saved checkpoint at epoch {current_epoch}")
            
            # Also save a "latest" checkpoint for easier resuming
            latest_path = "/model/latest_checkpoint.pt"
            model.save(latest_path)
            print(f"Saved latest checkpoint at epoch {current_epoch}")
            
            # Clean up old checkpoints to prevent disk space issues
            if current_epoch % 20 == 0:  # Cleanup every 20 epochs
                cleanup_old_checkpoints()
        
        if hasattr(trainer, 'best_fitness') and trainer.best_fitness == trainer.fitness:
            best_path = f"/model/best.pt_{current_epoch}.pt"
            model.save(best_path)
            print(f"Saved best model at epoch {current_epoch}")
            
            # Also save as standard best.pt
            standard_best_path = "/model/best.pt"
            model.save(standard_best_path)
            print(f"Saved standard best model at epoch {current_epoch}")
        
    model.add_callback("on_train_epoch_end", on_train_epoch_end)

    # Print evidence-based augmentation configuration
    print_augmentation_summary(slice_type)

    print("\n" + "="*80)
    print("MULTI-RESOLUTION TRAINING CONFIGURATION")
    print("="*80)
    print(f"Strategy: Mixed training with conservative augmentations")
    print(f"Slice type: {slice_type.upper()}")
    print(f"Freeze layers: 0-{FREEZE_LAYERS-1} (fine-tune from layer {FREEZE_LAYERS})")
    print(f"Learning rate: {LEARNING_RATE} (10x lower for domain adaptation)")
    print(f"Dataset composition: {slice_stats['thin_ratio']*100:.1f}% thin / {slice_stats['thick_ratio']*100:.1f}% thick")
    print(f"Multi-scale training: Enabled")
    print(f"Patience: {PATIENCE} epochs (reduced from 100)")
    print(f"  Note: Previous run stopped at epoch 72 due to plateau")
    print(f"  Conservative augmentation should enable faster, better convergence")
    print("="*80 + "\n")

    # Get evidence-based training hyperparameters
    training_hyperparams = get_yolo_training_hyperparameters(
        slice_type=slice_type,
        use_preaugmented=USE_PREAUGMENTED,  # Use pre-augmented dataset flag
        batch_size=batch_size,
        image_size=image_size,
        epochs=remaining_epochs,
        patience=PATIENCE,
        freeze_layers=FREEZE_LAYERS,
        learning_rate=LEARNING_RATE,
        final_lr_factor=FINAL_LR_FACTOR,
    )

    # Override project and name
    training_hyperparams.update({
        'data': stratified_yaml_path,
        'name': f"{model_name}_evidence_based_aug",
        'project': "brain-ct-hemorrhage-multiresolution",
    })

    # PHASE 2: Class Imbalance Handling
    print("\n" + "="*80)
    print("PHASE 2: CLASS IMBALANCE HANDLING (PRACTICAL APPROACH)")
    print("="*80)
    print("EDH class removed from dataset (insufficient data: 125 instances)")
    print("\nRemaining class distribution (estimated after EDH removal):")
    print("  HC  (Hemorrhagic Contusion):  ~538 instances  (6.1%)  - 5.1x rarer than SAH")
    print("  IPH (Intraparenchymal):       ~2731 instances (31.1%) - 1.0x (balanced)")
    print("  IVH (Intraventricular):       ~1373 instances (15.6%) - 2.0x rarer than SAH")
    print("  SAH (Subarachnoid):           ~2757 instances (31.4%) - 1.0x (reference)")
    print("  SDH (Subdural):               ~1339 instances (15.2%) - 2.1x rarer than SAH")

    # Calculate class weights using inverse sqrt frequency (dampened to prevent over-correction)
    # After EDH removal, original classes 1-5 become 0-4
    class_counts = {
        'HC': 538,   # was class 1, now class 0
        'IPH': 2731, # was class 2, now class 1
        'IVH': 1373, # was class 3, now class 2
        'SAH': 2757, # was class 4, now class 3
        'SDH': 1339, # was class 5, now class 4
    }

    max_count = max(class_counts.values())
    class_weights_no_edh = {}
    for idx, (name, count) in enumerate(class_counts.items()):
        # Use sqrt to dampen extreme weights
        raw_weight = (max_count / count) ** 0.5
        # Cap at 2.5x to prevent instability
        class_weights_no_edh[idx] = min(raw_weight, 2.5)

    print("\n" + "="*80)
    print("CLASS WEIGHTING STRATEGY")
    print("="*80)
    print("Using inverse sqrt frequency weighting (dampened for stability)")
    print("Formula: weight = min(sqrt(max_count / class_count), 2.5)")
    print("\nCalculated class weights:")
    class_names_no_edh = ['HC', 'IPH', 'IVH', 'SAH', 'SDH']
    for class_id, weight in class_weights_no_edh.items():
        count = list(class_counts.values())[class_id]
        print(f"  Class {class_id} ({class_names_no_edh[class_id]:3s}): {weight:.2f}x (n={count})")

    # Calculate weighted cls loss multiplier
    # Average weight will scale the cls loss to account for imbalance
    avg_weight = sum(class_weights_no_edh.values()) / len(class_weights_no_edh)
    weighted_cls_loss = training_hyperparams.get('cls', 1.0) * avg_weight

    print(f"\n💡 Implementation:")
    print(f"  - Base cls weight: {training_hyperparams.get('cls', 1.0):.2f}")
    print(f"  - Average class weight: {avg_weight:.2f}")
    print(f"  - Effective cls weight: {weighted_cls_loss:.2f}")
    print(f"  - This up-weights classification loss for rare classes")
    print(f"  - Combined with label_smoothing=0.05 for regularization")

    # Apply weighted cls loss
    training_hyperparams['cls'] = weighted_cls_loss

    print("\n" + "="*80)
    print("TRAINING HYPERPARAMETERS (PHASE 1+2)")
    print("="*80)
    print(f"\nLoss weights (adjusted for class imbalance + conservative augmentation):")
    print(f"  box: {training_hyperparams.get('box', 10.0):.1f}  - Increased (localization is strong)")
    print(f"  cls: {training_hyperparams.get('cls', 1.0):.2f}  - Weighted by class imbalance ({avg_weight:.2f}x)")
    print(f"  dfl: {training_hyperparams.get('dfl', 1.5):.1f}  - Unfreezes at epoch 10")

    print(f"\nPhase 1 - Conservative augmentation (preserve fine-grained features):")
    print(f"  degrees: {training_hyperparams.get('degrees', 5.0):.1f}° (reduced from ±10° → ±5°)")
    print(f"  translate: {training_hyperparams.get('translate', 0.03):.2%} (reduced from 6.25% → 3%)")
    print(f"  scale: {training_hyperparams.get('scale', 0.2):.1f} (reduced from 50% → 20%)")
    print(f"  fliplr: {training_hyperparams.get('fliplr', 0.5):.1f}")
    print(f"  mosaic: {training_hyperparams.get('mosaic', 0.3):.1f} (reduced from 0.8 → 0.3)")
    print(f"  hsv_v: {training_hyperparams.get('hsv_v', 0.0):.3f} (disabled - was 0.015)")

    print(f"\nPhase 2 - Class imbalance handling:")
    print(f"  Rare class up-weighting: HC={class_weights_no_edh[0]:.2f}x, SDH={class_weights_no_edh[4]:.2f}x")
    print(f"  Effective cls loss: {weighted_cls_loss:.2f} (base 1.0 * avg_weight {avg_weight:.2f})")
    print(f"  Label smoothing: 0.05 (prevents overconfidence)")

    print(f"\n📊 Expected improvements from Phase 1+2:")
    print(f"  cls_loss: 5.87 → 2.0-3.0 (class weighting + reduced augmentation)")
    print(f"  mAP50:    0.52 → 0.58-0.62 (+15-20%)")
    print(f"  mAP50-95: 0.21 → 0.28-0.32 (+35-50%)")
    print(f"  Recall:   50% → 55-60% (especially for rare classes)")
    print(f"  Precision: 60% → 65-70%")
    print("="*80 + "\n")

    # Train with evidence-based hyperparameters
    results = model.train(**training_hyperparams)

    # Save final models
    final_best_path = f"/model/best.pt_{total_epochs}.pt"
    model.save(final_best_path)
    final_standard_path = '/model/best.pt'
    model.save(final_standard_path)
    
    # Log the final model
    artifact = wandb.Artifact('best_model_multiresolution', type='model')
    artifact.add_file(final_standard_path)
    run.log_artifact(artifact)
    
    run.finish()

def calculate_per_class_metrics(images, img_dir, label_dir, model, class_names, conf_threshold=0.25):
    """
    Calculate per-class sensitivity, specificity, and accuracy.
    Returns metrics for each hemorrhage class and overall binary metrics.
    """
    num_classes = len(class_names)
    
    # Per-class counters
    class_tp = np.zeros(num_classes)  # True positives per class
    class_fp = np.zeros(num_classes)  # False positives per class
    class_fn = np.zeros(num_classes)  # False negatives per class
    class_tn = np.zeros(num_classes)  # True negatives per class
    
    # Overall binary counters (any hemorrhage vs no hemorrhage)
    binary_tp = 0  # Images with hemorrhage correctly detected
    binary_tn = 0  # Images without hemorrhage correctly identified
    binary_fp = 0  # Images without hemorrhage incorrectly flagged
    binary_fn = 0  # Images with hemorrhage missed
    
    print(f"Calculating metrics for {len(images)} images...")
    
    # Process images in batches
    batch_size = 16
    for i in range(0, len(images), batch_size):
        batch_images = images[i:i+batch_size]
        
        # Get predictions
        results = model.predict(batch_images, conf=conf_threshold, verbose=False)
        
        for img_path, result in zip(batch_images, results):
            # Get label file path
            label_path = img_path.replace(img_dir, label_dir).rsplit(".", 1)[0] + ".txt"
            
            # Parse ground truth
            gt_classes_in_image = set()
            gt_boxes_per_class = {c: [] for c in range(num_classes)}
            
            has_any_hemorrhage_gt = False
            
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            gt_class = int(parts[0])
                            gt_classes_in_image.add(gt_class)
                            has_any_hemorrhage_gt = True
                            
                            # Store box coordinates for IoU matching
                            x_center, y_center, width, height = map(float, parts[1:5])
                            gt_boxes_per_class[gt_class].append([x_center, y_center, width, height])
            
            # Parse predictions
            pred_classes_in_image = set()
            pred_boxes_per_class = {c: [] for c in range(num_classes)}
            
            has_any_hemorrhage_pred = False
            
            if len(result.boxes) > 0:
                has_any_hemorrhage_pred = True
                
                for box in result.boxes:
                    pred_class = int(box.cls.item())
                    pred_conf = float(box.conf.item())
                    
                    if pred_conf >= conf_threshold:
                        pred_classes_in_image.add(pred_class)
                        
                        # Get normalized coordinates
                        img_h, img_w = result.orig_shape
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        
                        x_center = ((x1 + x2) / 2) / img_w
                        y_center = ((y1 + y2) / 2) / img_h
                        width = (x2 - x1) / img_w
                        height = (y2 - y1) / img_h
                        
                        pred_boxes_per_class[pred_class].append([x_center, y_center, width, height, pred_conf])
            
            # Binary metrics (any hemorrhage vs none)
            if has_any_hemorrhage_gt and has_any_hemorrhage_pred:
                binary_tp += 1
            elif not has_any_hemorrhage_gt and not has_any_hemorrhage_pred:
                binary_tn += 1
            elif not has_any_hemorrhage_gt and has_any_hemorrhage_pred:
                binary_fp += 1
            elif has_any_hemorrhage_gt and not has_any_hemorrhage_pred:
                binary_fn += 1
            
            # Per-class metrics
            for class_id in range(num_classes):
                gt_has_class = class_id in gt_classes_in_image
                pred_has_class = class_id in pred_classes_in_image
                
                gt_boxes = gt_boxes_per_class[class_id]
                pred_boxes = pred_boxes_per_class[class_id]
                
                if gt_boxes and pred_boxes:
                    # Match predictions to ground truth using IoU
                    matched_gt = set()
                    matched_pred = set()
                    
                    for gt_idx, gt_box in enumerate(gt_boxes):
                        best_iou = 0
                        best_pred_idx = -1
                        
                        for pred_idx, pred_box in enumerate(pred_boxes):
                            if pred_idx in matched_pred:
                                continue
                            
                            # Calculate IoU
                            iou = calculate_iou(gt_box, pred_box[:4])
                            
                            if iou > best_iou:
                                best_iou = iou
                                best_pred_idx = pred_idx
                        
                        if best_iou >= 0.5:  # IoU threshold
                            matched_gt.add(gt_idx)
                            matched_pred.add(best_pred_idx)
                            class_tp[class_id] += 1
                    
                    # Unmatched predictions are false positives
                    class_fp[class_id] += len(pred_boxes) - len(matched_pred)
                    
                    # Unmatched ground truths are false negatives
                    class_fn[class_id] += len(gt_boxes) - len(matched_gt)
                    
                elif gt_boxes and not pred_boxes:
                    # All ground truth boxes are false negatives
                    class_fn[class_id] += len(gt_boxes)
                    
                elif not gt_boxes and pred_boxes:
                    # All predictions are false positives
                    class_fp[class_id] += len(pred_boxes)
                    
                elif not gt_boxes and not pred_boxes:
                    # True negative for this class in this image
                    class_tn[class_id] += 1
    
    # Calculate per-class metrics
    class_metrics = {}
    
    for class_id in range(num_classes):
        tp = class_tp[class_id]
        fp = class_fp[class_id]
        fn = class_fn[class_id]
        tn = class_tn[class_id]
        
        # Sensitivity (Recall) = TP / (TP + FN)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # Specificity = TN / (TN + FP)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Accuracy = (TP + TN) / (TP + TN + FP + FN)
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        
        # Precision = TP / (TP + FP)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        class_metrics[class_names[class_id]] = {
            'sensitivity': sensitivity,
            'specificity': specificity,
            'accuracy': accuracy,
            'precision': precision,
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn),
            'tn': int(tn)
        }
    
    # Calculate overall binary metrics
    binary_sensitivity = binary_tp / (binary_tp + binary_fn) if (binary_tp + binary_fn) > 0 else 0
    binary_specificity = binary_tn / (binary_tn + binary_fp) if (binary_tn + binary_fp) > 0 else 0
    binary_accuracy = (binary_tp + binary_tn) / (binary_tp + binary_tn + binary_fp + binary_fn) if (binary_tp + binary_tn + binary_fp + binary_fn) > 0 else 0
    binary_precision = binary_tp / (binary_tp + binary_fp) if (binary_tp + binary_fp) > 0 else 0
    
    binary_metrics = {
        'sensitivity': binary_sensitivity,
        'specificity': binary_specificity,
        'accuracy': binary_accuracy,
        'precision': binary_precision,
        'tp': int(binary_tp),
        'fp': int(binary_fp),
        'fn': int(binary_fn),
        'tn': int(binary_tn)
    }
    
    return class_metrics, binary_metrics

def calculate_iou(box1, box2):
    """Calculate IoU between two boxes in xywh format (normalized)"""
    # Convert from center format to corner format
    x1_min = box1[0] - box1[2] / 2
    y1_min = box1[1] - box1[3] / 2
    x1_max = box1[0] + box1[2] / 2
    y1_max = box1[1] + box1[3] / 2
    
    x2_min = box2[0] - box2[2] / 2
    y2_min = box2[1] - box2[3] / 2
    x2_max = box2[0] + box2[2] / 2
    y2_max = box2[1] + box2[3] / 2
    
    # Calculate intersection
    x_left = max(x1_min, x2_min)
    y_top = max(y1_min, y2_min)
    x_right = min(x1_max, x2_max)
    y_bottom = min(y1_max, y2_max)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate union
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]
    union_area = box1_area + box2_area - intersection_area
    
    return intersection_area / union_area if union_area > 0 else 0

def print_metrics_table(class_metrics, binary_metrics, split_name):
    """Print metrics in a nice table format"""
    print(f"\n{'='*100}")
    print(f"{split_name.upper()} - PER-CLASS METRICS")
    print(f"{'='*100}")
    print(f"{'Class':<30} {'Sensitivity':<12} {'Specificity':<12} {'Accuracy':<12} {'Precision':<12} {'TP':<6} {'FP':<6} {'FN':<6} {'TN':<6}")
    print(f"{'-'*100}")

    for class_name, metrics in class_metrics.items():
        print(f"{class_name:<30} "
              f"{metrics['sensitivity']:<12.4f} "
              f"{metrics['specificity']:<12.4f} "
              f"{metrics['accuracy']:<12.4f} "
              f"{metrics['precision']:<12.4f} "
              f"{metrics['tp']:<6} "
              f"{metrics['fp']:<6} "
              f"{metrics['fn']:<6} "
              f"{metrics['tn']:<6}")

    print(f"{'='*100}")
    print(f"{split_name.upper()} - OVERALL BINARY METRICS (Any Hemorrhage vs None)")
    print(f"{'='*100}")
    print(f"{'Metric':<20} {'Value':<12} {'Count':<10}")
    print(f"{'-'*100}")
    print(f"{'Sensitivity (Recall)':<20} {binary_metrics['sensitivity']:<12.4f} TP: {binary_metrics['tp']}, FN: {binary_metrics['fn']}")
    print(f"{'Specificity':<20} {binary_metrics['specificity']:<12.4f} TN: {binary_metrics['tn']}, FP: {binary_metrics['fp']}")
    print(f"{'Accuracy':<20} {binary_metrics['accuracy']:<12.4f} Total: {binary_metrics['tp'] + binary_metrics['tn'] + binary_metrics['fp'] + binary_metrics['fn']}")
    print(f"{'Precision':<20} {binary_metrics['precision']:<12.4f}")
    print(f"{'='*100}\n")

def optimize_per_class_thresholds(model, val_images, val_img_dir, val_label_dir, class_names,
                                   conf_range=(0.05, 0.70), step=0.05):
    """
    Find optimal confidence thresholds per class to maximize sensitivity while maintaining specificity.

    This addresses the severe imbalance problem by allowing rare classes (EDH, HC, SDH)
    to use lower thresholds.

    Args:
        model: Trained YOLO model
        val_images: List of validation image paths
        val_img_dir: Validation images directory
        val_label_dir: Validation labels directory
        class_names: List of class names
        conf_range: Tuple of (min_conf, max_conf) to search
        step: Step size for threshold search

    Returns:
        Dictionary of optimal thresholds per class
    """
    print("\n" + "="*80)
    print("OPTIMIZING PER-CLASS CONFIDENCE THRESHOLDS")
    print("="*80)
    print("This will take a few minutes...")

    num_classes = len(class_names)
    thresholds_to_test = np.arange(conf_range[0], conf_range[1], step)

    # Store predictions with confidence scores for all images
    all_predictions = []
    all_ground_truths = []

    # Collect all predictions at lowest threshold
    print(f"\nCollecting predictions for {len(val_images)} validation images...")
    for img_path in val_images:
        result = model.predict(img_path, conf=conf_range[0], verbose=False)[0]

        # Get ground truth
        label_path = img_path.replace(val_img_dir, val_label_dir).rsplit(".", 1)[0] + ".txt"
        gt_boxes_per_class = {c: [] for c in range(num_classes)}

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        gt_class = int(parts[0])
                        x_center, y_center, width, height = map(float, parts[1:5])
                        gt_boxes_per_class[gt_class].append([x_center, y_center, width, height])

        # Get predictions
        pred_boxes_per_class = {c: [] for c in range(num_classes)}

        if len(result.boxes) > 0:
            img_h, img_w = result.orig_shape
            for box in result.boxes:
                pred_class = int(box.cls.item())
                pred_conf = float(box.conf.item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                x_center = ((x1 + x2) / 2) / img_w
                y_center = ((y1 + y2) / 2) / img_h
                width = (x2 - x1) / img_w
                height = (y2 - y1) / img_h

                pred_boxes_per_class[pred_class].append([x_center, y_center, width, height, pred_conf])

        all_predictions.append(pred_boxes_per_class)
        all_ground_truths.append(gt_boxes_per_class)

    # Find optimal threshold for each class
    optimal_thresholds = {}

    print(f"\nSearching optimal thresholds for each class...")
    print(f"Testing thresholds from {conf_range[0]:.2f} to {conf_range[1]:.2f} with step {step:.2f}")

    for class_id in range(num_classes):
        class_name = class_names[class_id]
        best_threshold = 0.25
        best_f1 = 0
        best_sensitivity = 0
        best_specificity = 0

        for threshold in thresholds_to_test:
            tp, fp, fn, tn = 0, 0, 0, 0

            for pred_boxes, gt_boxes in zip(all_predictions, all_ground_truths):
                pred = [box for box in pred_boxes[class_id] if box[4] >= threshold]
                gt = gt_boxes[class_id]

                if gt and pred:
                    matched_gt = set()
                    matched_pred = set()

                    for gt_idx, gt_box in enumerate(gt):
                        best_iou = 0
                        best_pred_idx = -1

                        for pred_idx, pred_box in enumerate(pred):
                            if pred_idx in matched_pred:
                                continue
                            iou = calculate_iou(gt_box, pred_box[:4])
                            if iou > best_iou:
                                best_iou = iou
                                best_pred_idx = pred_idx

                        if best_iou >= 0.5:
                            matched_gt.add(gt_idx)
                            matched_pred.add(best_pred_idx)
                            tp += 1

                    fp += len(pred) - len(matched_pred)
                    fn += len(gt) - len(matched_gt)
                elif gt and not pred:
                    fn += len(gt)
                elif not gt and pred:
                    fp += len(pred)
                elif not gt and not pred:
                    tn += 1

            # Calculate metrics
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0

            # For rare classes (EDH, HC, SDH), prioritize sensitivity
            # For common classes, balance F1 score
            if class_id in [0, 1, 5]:  # EDH, HC, SDH
                # Weight sensitivity higher for rare classes
                score = 0.7 * sensitivity + 0.3 * f1
            else:
                score = f1

            if score > best_f1:
                best_f1 = score
                best_threshold = threshold
                best_sensitivity = sensitivity
                best_specificity = specificity

        optimal_thresholds[class_name] = best_threshold

        print(f"{class_name:3s}: threshold={best_threshold:.3f}, "
              f"sensitivity={best_sensitivity:.3f}, specificity={best_specificity:.3f}, "
              f"F1/score={best_f1:.3f}")

    print("\n" + "="*80)
    print("OPTIMAL THRESHOLDS FOUND")
    print("="*80)
    for class_name, threshold in optimal_thresholds.items():
        print(f"{class_name:3s}: {threshold:.3f}")
    print("="*80 + "\n")

    return optimal_thresholds

def analyze_failure_cases(model, images, img_dir, label_dir, class_names, conf_threshold=0.25, max_cases=20):
    """
    Analyze failure cases (false negatives and false positives) to understand model weaknesses.

    Args:
        model: Trained YOLO model
        images: List of image paths
        img_dir: Images directory
        label_dir: Labels directory
        class_names: List of class names
        conf_threshold: Confidence threshold
        max_cases: Maximum number of cases to analyze per class

    Returns:
        Dictionary with failure analysis
    """
    print("\n" + "="*80)
    print("ANALYZING FAILURE CASES")
    print("="*80)

    num_classes = len(class_names)

    false_negatives = {c: [] for c in range(num_classes)}
    false_positives = {c: [] for c in range(num_classes)}

    for img_path in images:
        result = model.predict(img_path, conf=conf_threshold, verbose=False)[0]

        # Get ground truth
        label_path = img_path.replace(img_dir, label_dir).rsplit(".", 1)[0] + ".txt"
        gt_boxes_per_class = {c: [] for c in range(num_classes)}

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        gt_class = int(parts[0])
                        x_center, y_center, width, height = map(float, parts[1:5])
                        gt_boxes_per_class[gt_class].append([x_center, y_center, width, height])

        # Get predictions
        pred_boxes_per_class = {c: [] for c in range(num_classes)}

        if len(result.boxes) > 0:
            img_h, img_w = result.orig_shape
            for box in result.boxes:
                pred_class = int(box.cls.item())
                pred_conf = float(box.conf.item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                x_center = ((x1 + x2) / 2) / img_w
                y_center = ((y1 + y2) / 2) / img_h
                width = (x2 - x1) / img_w
                height = (y2 - y1) / img_h

                pred_boxes_per_class[pred_class].append([x_center, y_center, width, height, pred_conf])

        # Find false negatives and false positives
        for class_id in range(num_classes):
            pred = pred_boxes_per_class[class_id]
            gt = gt_boxes_per_class[class_id]

            if gt and not pred:
                # All ground truths are false negatives
                if len(false_negatives[class_id]) < max_cases:
                    false_negatives[class_id].append({
                        'image': os.path.basename(img_path),
                        'gt_boxes': gt,
                        'reason': 'no_detection'
                    })
            elif gt and pred:
                matched_gt = set()

                for gt_idx, gt_box in enumerate(gt):
                    best_iou = 0
                    for pred_box in pred:
                        iou = calculate_iou(gt_box, pred_box[:4])
                        if iou > best_iou:
                            best_iou = iou

                    if best_iou < 0.5:
                        if len(false_negatives[class_id]) < max_cases:
                            false_negatives[class_id].append({
                                'image': os.path.basename(img_path),
                                'gt_box': gt_box,
                                'best_iou': best_iou,
                                'reason': 'poor_localization'
                            })

            if not gt and pred:
                # All predictions are false positives
                if len(false_positives[class_id]) < max_cases:
                    for pred_box in pred:
                        false_positives[class_id].append({
                            'image': os.path.basename(img_path),
                            'pred_box': pred_box[:4],
                            'confidence': pred_box[4],
                            'reason': 'spurious_detection'
                        })

    # Print analysis
    print("\n" + "="*80)
    print("FALSE NEGATIVE ANALYSIS (Missed Detections)")
    print("="*80)

    for class_id in range(num_classes):
        class_name = class_names[class_id]
        fn_cases = false_negatives[class_id]

        if fn_cases:
            print(f"\n{class_name} - {len(fn_cases)} false negatives:")
            for i, case in enumerate(fn_cases[:5], 1):  # Show first 5
                print(f"  {i}. {case['image']}: {case['reason']}")

    print("\n" + "="*80)
    print("FALSE POSITIVE ANALYSIS (Spurious Detections)")
    print("="*80)

    for class_id in range(num_classes):
        class_name = class_names[class_id]
        fp_cases = false_positives[class_id]

        if fp_cases:
            print(f"\n{class_name} - {len(fp_cases)} false positives:")
            for i, case in enumerate(fp_cases[:5], 1):  # Show first 5
                print(f"  {i}. {case['image']}: conf={case['confidence']:.3f}")

    print("\n" + "="*80)

    return {
        'false_negatives': false_negatives,
        'false_positives': false_positives
    }

@app.function(
    image=image,
    gpu="A10G",
    secrets=[modal.Secret.from_name("wandb-api-key")],
    volumes={"/model": volume}
)
def validate_and_test_yolo():
    """Domain-aware validation and testing with detailed metrics"""
    import yaml
    
    # Preprocessing
    print("\n" + "="*80)
    print("PREPROCESSING: Converting polygon annotations to bounding boxes")
    print("="*80)
    preprocess_dataset_labels("/datasets")
    
    # Load config
    stratified_yaml_path = "/model/stratified_data.yaml"
    
    if not os.path.exists(stratified_yaml_path):
        print("Creating stratified dataset configuration...")
        create_stratified_dataset_yaml("/datasets", stratified_yaml_path)
    
    with open(stratified_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    class_names = data_config.get('names', [])
    
    wandb.login(key=os.environ["WANDB_API_KEY"])
    run = wandb.init(project="brain-ct-hemorrhage", 
                     name=f"{MODEL}_multiresolution_val_test_{VERSION}")
    
    print(f"Classes: {class_names}")
    
    # Load model
    print("Finding latest model...")
    model_files = glob.glob("/model/best.pt_*.pt")
    if model_files:
        model_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        model_path = model_files[-1]
    else:
        model_path = "/model/best.pt"
    
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)
    
    print("\n" + "="*80)
    print("VALIDATION AND TESTING WITH DETAILED METRICS")
    print("="*80)
    
    # Helper function to separate thin and thick images
    def separate_by_slice_type(images):
        thin_images = [img for img in images if os.path.basename(img).startswith('thin_')]
        thick_images = [img for img in images if os.path.basename(img).startswith('thick_')]
        return thin_images, thick_images
    
    # VALIDATION SET
    print("\n" + "="*80)
    print("=== VALIDATION SET ===")
    print("="*80)
    
    val_img_dir = "/datasets/valid/images"
    val_label_dir = "/datasets/valid/labels"
    val_images = [os.path.join(val_img_dir, f) for f in os.listdir(val_img_dir) 
                  if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # Separate thin and thick slices
    val_thin_images, val_thick_images = separate_by_slice_type(val_images)
    
    print(f"\nTotal validation images: {len(val_images)}")
    print(f"  - Thin slices: {len(val_thin_images)}")
    print(f"  - Thick slices: {len(val_thick_images)}")
    
    # Calculate overall validation metrics
    print(f"\n--- Overall Validation Metrics ---")
    val_class_metrics, val_binary_metrics = calculate_per_class_metrics(
        val_images, val_img_dir, val_label_dir, model, class_names, conf_threshold=0.25
    )
    print_metrics_table(val_class_metrics, val_binary_metrics, "Validation - Overall")
    
    # Calculate thin slice validation metrics
    if val_thin_images:
        print(f"\n--- Thin Slice Validation Metrics ---")
        val_thin_class_metrics, val_thin_binary_metrics = calculate_per_class_metrics(
            val_thin_images, val_img_dir, val_label_dir, model, class_names, conf_threshold=0.25
        )
        print_metrics_table(val_thin_class_metrics, val_thin_binary_metrics, "Validation - Thin Slices")
    
    # Calculate thick slice validation metrics
    if val_thick_images:
        print(f"\n--- Thick Slice Validation Metrics ---")
        val_thick_class_metrics, val_thick_binary_metrics = calculate_per_class_metrics(
            val_thick_images, val_img_dir, val_label_dir, model, class_names, conf_threshold=0.25
        )
        print_metrics_table(val_thick_class_metrics, val_thick_binary_metrics, "Validation - Thick Slices")
    
    # Log to wandb - Overall validation metrics
    print("\n" + "="*80)
    print("LOGGING TO WANDB - VALIDATION OVERALL METRICS")
    print("="*80)
    
    for class_name, metrics in val_class_metrics.items():
        wandb_metrics = {
            f"val_overall_class/{class_name}/sensitivity": metrics['sensitivity'],
            f"val_overall_class/{class_name}/specificity": metrics['specificity'],
            f"val_overall_class/{class_name}/accuracy": metrics['accuracy'],
            f"val_overall_class/{class_name}/precision": metrics['precision'],
        }
        print(f"Class {class_name} metrics:")
        for key, value in wandb_metrics.items():
            print(f"  {key}: {value:.4f}")
        wandb.log(wandb_metrics)
    
    binary_wandb_metrics = {
        "val_overall_binary/sensitivity": val_binary_metrics['sensitivity'],
        "val_overall_binary/specificity": val_binary_metrics['specificity'],
        "val_overall_binary/accuracy": val_binary_metrics['accuracy'],
        "val_overall_binary/precision": val_binary_metrics['precision'],
        "val_overall_binary/tp": val_binary_metrics['tp'],
        "val_overall_binary/fp": val_binary_metrics['fp'],
        "val_overall_binary/fn": val_binary_metrics['fn'],
        "val_overall_binary/tn": val_binary_metrics['tn'],
    }
    print(f"\nOverall binary metrics:")
    for key, value in binary_wandb_metrics.items():
        print(f"  {key}: {value}")
    wandb.log(binary_wandb_metrics)
    
    # Log to wandb - Thin slice validation metrics
    if val_thin_images:
        print("\n" + "="*80)
        print("LOGGING TO WANDB - VALIDATION THIN SLICE METRICS")
        print("="*80)
        
        for class_name, metrics in val_thin_class_metrics.items():
            wandb_metrics = {
                f"val_thin_class/{class_name}/sensitivity": metrics['sensitivity'],
                f"val_thin_class/{class_name}/specificity": metrics['specificity'],
                f"val_thin_class/{class_name}/accuracy": metrics['accuracy'],
                f"val_thin_class/{class_name}/precision": metrics['precision'],
            }
            print(f"Thin slice class {class_name} metrics:")
            for key, value in wandb_metrics.items():
                print(f"  {key}: {value:.4f}")
            wandb.log(wandb_metrics)
        
        thin_binary_wandb_metrics = {
            "val_thin_binary/sensitivity": val_thin_binary_metrics['sensitivity'],
            "val_thin_binary/specificity": val_thin_binary_metrics['specificity'],
            "val_thin_binary/accuracy": val_thin_binary_metrics['accuracy'],
            "val_thin_binary/precision": val_thin_binary_metrics['precision'],
            "val_thin_binary/tp": val_thin_binary_metrics['tp'],
            "val_thin_binary/fp": val_thin_binary_metrics['fp'],
            "val_thin_binary/fn": val_thin_binary_metrics['fn'],
            "val_thin_binary/tn": val_thin_binary_metrics['tn'],
        }
        print(f"\nThin slice binary metrics:")
        for key, value in thin_binary_wandb_metrics.items():
            print(f"  {key}: {value}")
        wandb.log(thin_binary_wandb_metrics)
    
    # Log to wandb - Thick slice validation metrics
    if val_thick_images:
        print("\n" + "="*80)
        print("LOGGING TO WANDB - VALIDATION THICK SLICE METRICS")
        print("="*80)
        
        for class_name, metrics in val_thick_class_metrics.items():
            wandb_metrics = {
                f"val_thick_class/{class_name}/sensitivity": metrics['sensitivity'],
                f"val_thick_class/{class_name}/specificity": metrics['specificity'],
                f"val_thick_class/{class_name}/accuracy": metrics['accuracy'],
                f"val_thick_class/{class_name}/precision": metrics['precision'],
            }
            print(f"Thick slice class {class_name} metrics:")
            for key, value in wandb_metrics.items():
                print(f"  {key}: {value:.4f}")
            wandb.log(wandb_metrics)
        
        thick_binary_wandb_metrics = {
            "val_thick_binary/sensitivity": val_thick_binary_metrics['sensitivity'],
            "val_thick_binary/specificity": val_thick_binary_metrics['specificity'],
            "val_thick_binary/accuracy": val_thick_binary_metrics['accuracy'],
            "val_thick_binary/precision": val_thick_binary_metrics['precision'],
            "val_thick_binary/tp": val_thick_binary_metrics['tp'],
            "val_thick_binary/fp": val_thick_binary_metrics['fp'],
            "val_thick_binary/fn": val_thick_binary_metrics['fn'],
            "val_thick_binary/tn": val_thick_binary_metrics['tn'],
        }
        print(f"\nThick slice binary metrics:")
        for key, value in thick_binary_wandb_metrics.items():
            print(f"  {key}: {value}")
        wandb.log(thick_binary_wandb_metrics)
    
    # TEST SET
    print("\n" + "="*80)
    print("=== TEST SET ===")
    print("="*80)
    
    test_img_dir = "/datasets/test/images"
    test_label_dir = "/datasets/test/labels"
    test_images = [os.path.join(test_img_dir, f) for f in os.listdir(test_img_dir) 
                   if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # Separate thin and thick slices
    test_thin_images, test_thick_images = separate_by_slice_type(test_images)
    
    print(f"\nTotal test images: {len(test_images)}")
    print(f"  - Thin slices: {len(test_thin_images)}")
    print(f"  - Thick slices: {len(test_thick_images)}")
    
    # Calculate overall test metrics
    print(f"\n--- Overall Test Metrics ---")
    test_class_metrics, test_binary_metrics = calculate_per_class_metrics(
        test_images, test_img_dir, test_label_dir, model, class_names, conf_threshold=0.25
    )
    print_metrics_table(test_class_metrics, test_binary_metrics, "Test - Overall")
    
    # Calculate thin slice test metrics
    if test_thin_images:
        print(f"\n--- Thin Slice Test Metrics ---")
        test_thin_class_metrics, test_thin_binary_metrics = calculate_per_class_metrics(
            test_thin_images, test_img_dir, test_label_dir, model, class_names, conf_threshold=0.25
        )
        print_metrics_table(test_thin_class_metrics, test_thin_binary_metrics, "Test - Thin Slices")
    
    # Calculate thick slice test metrics
    if test_thick_images:
        print(f"\n--- Thick Slice Test Metrics ---")
        test_thick_class_metrics, test_thick_binary_metrics = calculate_per_class_metrics(
            test_thick_images, test_img_dir, test_label_dir, model, class_names, conf_threshold=0.25
        )
        print_metrics_table(test_thick_class_metrics, test_thick_binary_metrics, "Test - Thick Slices")
    
    # Log to wandb - Overall test metrics
    print("\n" + "="*80)
    print("LOGGING TO WANDB - TEST OVERALL METRICS")
    print("="*80)
    
    for class_name, metrics in test_class_metrics.items():
        wandb_metrics = {
            f"test_overall_class/{class_name}/sensitivity": metrics['sensitivity'],
            f"test_overall_class/{class_name}/specificity": metrics['specificity'],
            f"test_overall_class/{class_name}/accuracy": metrics['accuracy'],
            f"test_overall_class/{class_name}/precision": metrics['precision'],
        }
        print(f"Test class {class_name} metrics:")
        for key, value in wandb_metrics.items():
            print(f"  {key}: {value:.4f}")
        wandb.log(wandb_metrics)
    
    test_binary_wandb_metrics = {
        "test_overall_binary/sensitivity": test_binary_metrics['sensitivity'],
        "test_overall_binary/specificity": test_binary_metrics['specificity'],
        "test_overall_binary/accuracy": test_binary_metrics['accuracy'],
        "test_overall_binary/precision": test_binary_metrics['precision'],
        "test_overall_binary/tp": test_binary_metrics['tp'],
        "test_overall_binary/fp": test_binary_metrics['fp'],
        "test_overall_binary/fn": test_binary_metrics['fn'],
        "test_overall_binary/tn": test_binary_metrics['tn'],
    }
    print(f"\nTest overall binary metrics:")
    for key, value in test_binary_wandb_metrics.items():
        print(f"  {key}: {value}")
    wandb.log(test_binary_wandb_metrics)
    
    # Log to wandb - Thin slice test metrics
    if test_thin_images:
        print("\n" + "="*80)
        print("LOGGING TO WANDB - TEST THIN SLICE METRICS")
        print("="*80)
        
        for class_name, metrics in test_thin_class_metrics.items():
            wandb_metrics = {
                f"test_thin_class/{class_name}/sensitivity": metrics['sensitivity'],
                f"test_thin_class/{class_name}/specificity": metrics['specificity'],
                f"test_thin_class/{class_name}/accuracy": metrics['accuracy'],
                f"test_thin_class/{class_name}/precision": metrics['precision'],
            }
            print(f"Test thin slice class {class_name} metrics:")
            for key, value in wandb_metrics.items():
                print(f"  {key}: {value:.4f}")
            wandb.log(wandb_metrics)
        
        test_thin_binary_wandb_metrics = {
            "test_thin_binary/sensitivity": test_thin_binary_metrics['sensitivity'],
            "test_thin_binary/specificity": test_thin_binary_metrics['specificity'],
            "test_thin_binary/accuracy": test_thin_binary_metrics['accuracy'],
            "test_thin_binary/precision": test_thin_binary_metrics['precision'],
            "test_thin_binary/tp": test_thin_binary_metrics['tp'],
            "test_thin_binary/fp": test_thin_binary_metrics['fp'],
            "test_thin_binary/fn": test_thin_binary_metrics['fn'],
            "test_thin_binary/tn": test_thin_binary_metrics['tn'],
        }
        print(f"\nTest thin slice binary metrics:")
        for key, value in test_thin_binary_wandb_metrics.items():
            print(f"  {key}: {value}")
        wandb.log(test_thin_binary_wandb_metrics)
    
    # Log to wandb - Thick slice test metrics
    if test_thick_images:
        print("\n" + "="*80)
        print("LOGGING TO WANDB - TEST THICK SLICE METRICS")
        print("="*80)
        
        for class_name, metrics in test_thick_class_metrics.items():
            wandb_metrics = {
                f"test_thick_class/{class_name}/sensitivity": metrics['sensitivity'],
                f"test_thick_class/{class_name}/specificity": metrics['specificity'],
                f"test_thick_class/{class_name}/accuracy": metrics['accuracy'],
                f"test_thick_class/{class_name}/precision": metrics['precision'],
            }
            print(f"Test thick slice class {class_name} metrics:")
            for key, value in wandb_metrics.items():
                print(f"  {key}: {value:.4f}")
            wandb.log(wandb_metrics)
        
        test_thick_binary_wandb_metrics = {
            "test_thick_binary/sensitivity": test_thick_binary_metrics['sensitivity'],
            "test_thick_binary/specificity": test_thick_binary_metrics['specificity'],
            "test_thick_binary/accuracy": test_thick_binary_metrics['accuracy'],
            "test_thick_binary/precision": test_thick_binary_metrics['precision'],
            "test_thick_binary/tp": test_thick_binary_metrics['tp'],
            "test_thick_binary/fp": test_thick_binary_metrics['fp'],
            "test_thick_binary/fn": test_thick_binary_metrics['fn'],
            "test_thick_binary/tn": test_thick_binary_metrics['tn'],
        }
        print(f"\nTest thick slice binary metrics:")
        for key, value in test_thick_binary_wandb_metrics.items():
            print(f"  {key}: {value}")
        wandb.log(test_thick_binary_wandb_metrics)
    
    # SUMMARY
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    print(f"\n{'='*50}")
    print("VALIDATION SET SUMMARY")
    print(f"{'='*50}")
    
    print(f"\nOverall Validation Binary Metrics:")
    print(f"  Sensitivity: {val_binary_metrics['sensitivity']:.4f}")
    print(f"  Specificity: {val_binary_metrics['specificity']:.4f}")
    print(f"  Accuracy: {val_binary_metrics['accuracy']:.4f}")
    print(f"  Precision: {val_binary_metrics['precision']:.4f}")
    
    if val_thin_images:
        print(f"\nThin Slice Validation Binary Metrics:")
        print(f"  Sensitivity: {val_thin_binary_metrics['sensitivity']:.4f}")
        print(f"  Specificity: {val_thin_binary_metrics['specificity']:.4f}")
        print(f"  Accuracy: {val_thin_binary_metrics['accuracy']:.4f}")
        print(f"  Precision: {val_thin_binary_metrics['precision']:.4f}")
    
    if val_thick_images:
        print(f"\nThick Slice Validation Binary Metrics:")
        print(f"  Sensitivity: {val_thick_binary_metrics['sensitivity']:.4f}")
        print(f"  Specificity: {val_thick_binary_metrics['specificity']:.4f}")
        print(f"  Accuracy: {val_thick_binary_metrics['accuracy']:.4f}")
        print(f"  Precision: {val_thick_binary_metrics['precision']:.4f}")
    
    print(f"\n{'='*50}")
    print("TEST SET SUMMARY")
    print(f"{'='*50}")
    
    print(f"\nOverall Test Binary Metrics:")
    print(f"  Sensitivity: {test_binary_metrics['sensitivity']:.4f}")
    print(f"  Specificity: {test_binary_metrics['specificity']:.4f}")
    print(f"  Accuracy: {test_binary_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_binary_metrics['precision']:.4f}")
    
    if test_thin_images:
        print(f"\nThin Slice Test Binary Metrics:")
        print(f"  Sensitivity: {test_thin_binary_metrics['sensitivity']:.4f}")
        print(f"  Specificity: {test_thin_binary_metrics['specificity']:.4f}")
        print(f"  Accuracy: {test_thin_binary_metrics['accuracy']:.4f}")
        print(f"  Precision: {test_thin_binary_metrics['precision']:.4f}")
    
    if test_thick_images:
        print(f"\nThick Slice Test Binary Metrics:")
        print(f"  Sensitivity: {test_thick_binary_metrics['sensitivity']:.4f}")
        print(f"  Specificity: {test_thick_binary_metrics['specificity']:.4f}")
        print(f"  Accuracy: {test_thick_binary_metrics['accuracy']:.4f}")
        print(f"  Precision: {test_thick_binary_metrics['precision']:.4f}")
    
    print(f"\nModel evaluation complete!")
    print("="*80 + "\n")
    
    run.finish()
    
# Keep your existing predict_and_visualize function
@app.function(
    image=image,
    gpu="A10G",
    secrets=[modal.Secret.from_name("wandb-api-key")],
    volumes={"/model": volume}
)
def predict_and_visualize():
    import cv2
    import os
    import numpy as np
    from pathlib import Path
    import yaml
    
    wandb.login(key=os.environ["WANDB_API_KEY"])
    run = wandb.init(project="brain-ct-hemorrhage", name=f"{MODEL}-{VERSION}_brain_ct_hemorrhage_errors_{VERSION}")
    
    # Load the model
    model = YOLO("/model/best.pt")
    
    # Set confidence threshold
    conf_threshold = 0.25
    
    # Find test dataset path
    test_base_path = None
    possible_paths = [
        os.path.join("/datasets", "test"),
        os.path.join("/datasets", "test".lower()),
        os.path.join("/datasets", "test".upper()),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            img_dir = os.path.join(path, "images")
            if os.path.exists(img_dir):
                test_base_path = path
                break
    
    if not test_base_path:
        print("Error: Could not find test dataset directory")
        return
    
    test_img_dir = os.path.join(test_base_path, "images")
    test_label_dir = os.path.join(test_base_path, "labels")
    
    print(f"Test images directory: {test_img_dir}")
    print(f"Test labels directory: {test_label_dir}")
    
    # Load class names from data.yaml
    yaml_path = os.path.join("/datasets", "data.yaml")
    fixed_yaml_path = os.path.join("/model", "fixed_data.yaml")
    
    if os.path.exists(fixed_yaml_path):
        yaml_path = fixed_yaml_path
    
    with open(yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    class_names = data_config.get('names', [])
    print(f"Classes: {class_names}")
    
    # Get all test images
    test_images = []
    if os.path.exists(test_img_dir):
        test_images = [os.path.join(test_img_dir, f) for f in os.listdir(test_img_dir) 
                      if f.endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    print(f"Found {len(test_images)} test images")

@app.local_entrypoint()
def main():
    """Start multi-resolution training from scratch"""
    print("Starting multi-resolution training from scratch...")
    train_yolo.remote(resume=False)
    # predict_and_visualize.remote()  # Uncomment when needed

@app.local_entrypoint()
def resume_training():
    """Resume multi-resolution training from checkpoint"""
    print("Resuming multi-resolution training...")
    train_yolo.remote(resume=True)

@app.local_entrypoint()
def run_validation():
    """Run validation separately after training completes"""
    print("Running validation and testing...")
    validate_and_test_yolo.remote()
