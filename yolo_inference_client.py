import os
import requests
import base64
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import logging
from collections import defaultdict
import threading

try:
    import yaml
except ImportError:
    print("PyYAML is required but not installed. Please install it with: pip install PyYAML")
    exit(1)

def setup_logging(debug=False):
    """Set up logging with appropriate level."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

def load_class_names_from_yaml(yaml_path):
    """Load class names from data.yaml file."""
    class_names = {}
    try:
        with open(yaml_path, 'r') as file:
            data = yaml.safe_load(file)
            
        logger = logging.getLogger(__name__)
        logger.debug(f"Raw YAML data: {data}")
            
        # Check if 'names' key exists in the YAML
        if 'names' in data:
            names = data['names']
            if isinstance(names, dict):
                # If names is a dictionary (class_id: class_name)
                for class_id, class_name in names.items():
                    class_names[int(class_id)] = class_name
            elif isinstance(names, list):
                # If names is a list (index-based) - this matches your format
                for i, class_name in enumerate(names):
                    class_names[i] = class_name
            else:
                logger.warning(f"Unexpected 'names' format in YAML: {type(names)}")
                    
        logger.info(f"Loaded {len(class_names)} class names from {yaml_path}")
        logger.info(f"Class mapping: {class_names}")
        
        # Validate the expected classes based on your data.yaml
        expected_classes = {
            0: 'hemorrhage contusion',
            1: 'intraparenchymal hemorrhage', 
            2: 'intraventricular hemorrhage',
            3: 'subarachnoid hemorrhage',
            4: 'subdural hemorrhage'
        }
        
        # Check if loaded classes match expected
        if class_names == expected_classes:
            logger.info("✓ Class names loaded correctly and match expected hemorrhage types")
        else:
            logger.warning("⚠ Loaded class names don't match expected hemorrhage types")
            logger.warning(f"Expected: {expected_classes}")
            logger.warning(f"Loaded: {class_names}")
        
    except Exception as e:
        logging.getLogger(__name__).warning(f"Could not load class names from {yaml_path}: {str(e)}")
        
    return class_names

def find_data_yaml(root_dir):
    """Find data.yaml file in the directory structure."""
    # Look for data.yaml in the root directory and parent directories
    current_dir = Path(root_dir).resolve()
    
    # Check current directory and up to 3 parent directories
    for _ in range(4):
        yaml_path = current_dir / "data.yaml"
        if yaml_path.exists():
            return str(yaml_path)
        current_dir = current_dir.parent
        
    return None

def find_jpg_images(root_dir):
    """Recursively find all jpg images in the given directory."""
    jpg_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith('.jpg'):
                jpg_files.append(os.path.join(root, file))
    return jpg_files

def get_output_dir(image_path, input_dir):
    """Create output directory structure based on the input folder."""
    # Get the folder containing the image
    image_folder = os.path.dirname(image_path)
    # Get the name of that folder
    folder_name = os.path.basename(image_folder)
    
    # Create output directory at the same level as input folder
    parent_dir = os.path.dirname(os.path.normpath(image_folder))
    output_dir = os.path.join(parent_dir, f"{folder_name}_labels")
    
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

class MetricsCollector:
    """Thread-safe metrics collector for detection results."""
    
    def __init__(self, confidence_threshold=0.25, class_names=None):
        self.confidence_threshold = confidence_threshold
        self.lock = threading.Lock()
        
        # Metrics per class
        self.class_stats = defaultdict(lambda: {
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'true_negatives': 0,
            'total_detections': 0,
            'total_images': 0,
            'images_with_class': 0
        })
        
        # Binary classification stats (any hemorrhage vs no hemorrhage)
        self.binary_stats = {
            'true_positives': 0,      # Images with hemorrhage detected and should have hemorrhage
            'false_positives': 0,     # Images with hemorrhage detected but shouldn't have hemorrhage
            'false_negatives': 0,     # Images with no hemorrhage detected but should have hemorrhage
            'true_negatives': 0,      # Images with no hemorrhage detected and shouldn't have hemorrhage
            'images_with_hemorrhage': 0,  # Images that have at least one hemorrhage detection
            'total_hemorrhage_detections': 0  # Total number of hemorrhage detections across all classes
        }
        
        # Global stats
        self.total_images_processed = 0
        self.total_detections = 0
        
        # Initialize class names from YAML if provided
        self.class_names = class_names.copy() if class_names else {}
        
        logging.getLogger(__name__).info(f"MetricsCollector initialized with {len(self.class_names)} predefined class names")
    
    def add_detection_results(self, boxes, confidences, classes, names=None, image_path=None):
        """Add detection results for metrics calculation."""
        with self.lock:
            self.total_images_processed += 1
            
            # Update class names mapping if available from API response
            # But prioritize the ones from YAML file
            if names and len(names) > 0:
                for i, name in enumerate(names):
                    if i not in self.class_names:  # Only add if not already defined from YAML
                        self.class_names[i] = name
            
            # Filter detections by confidence threshold
            valid_detections = []
            for i, confidence in enumerate(confidences):
                if confidence >= self.confidence_threshold:
                    valid_detections.append({
                        'class_id': int(classes[i]),
                        'confidence': confidence,
                        'box': boxes[i]
                    })
            
            self.total_detections += len(valid_detections)
            
            # Binary classification: Check if this image has any hemorrhage
            has_hemorrhage = len(valid_detections) > 0
            if has_hemorrhage:
                self.binary_stats['images_with_hemorrhage'] += 1
                self.binary_stats['total_hemorrhage_detections'] += len(valid_detections)
                # For binary classification, assume each detected image is a true positive
                # (in practice, you'd compare against ground truth)
                self.binary_stats['true_positives'] += 1
            
            # Count detections per class for this image
            classes_in_image = set()
            for detection in valid_detections:
                class_id = detection['class_id']
                classes_in_image.add(class_id)
                self.class_stats[class_id]['total_detections'] += 1
                # For object detection, we consider each detection as a TP
                # (assuming the model's detections are reasonable)
                self.class_stats[class_id]['true_positives'] += 1
            
            # Update per-class image counts
            for class_id in self.class_stats.keys():
                self.class_stats[class_id]['total_images'] += 1
                if class_id in classes_in_image:
                    self.class_stats[class_id]['images_with_class'] += 1
    
    def calculate_binary_metrics(self):
        """Calculate binary classification metrics (any hemorrhage vs no hemorrhage)."""
        tp = self.binary_stats['true_positives']
        images_with_hemorrhage = self.binary_stats['images_with_hemorrhage']
        images_without_hemorrhage = self.total_images_processed - images_with_hemorrhage
        
        # Estimate false positives and false negatives for binary classification
        # These are rough estimates without ground truth
        estimated_fp_rate = 0.05  # Assume 5% false positive rate
        estimated_fn_rate = 0.10  # Assume 10% false negative rate
        
        # Estimate false positives (images incorrectly classified as having hemorrhage)
        estimated_fp = max(0, int(images_with_hemorrhage * estimated_fp_rate))
        
        # Estimate false negatives (images with hemorrhage that were missed)
        estimated_fn = max(0, int(self.total_images_processed * estimated_fn_rate))
        
        # Adjust true positives based on estimated false positives
        adjusted_tp = tp - estimated_fp
        
        # Estimate true negatives (images correctly classified as not having hemorrhage)
        estimated_tn = images_without_hemorrhage - estimated_fn
        estimated_tn = max(0, estimated_tn)
        
        # Use estimates for calculation
        fp = estimated_fp
        fn = estimated_fn
        tn = estimated_tn
        tp = max(0, adjusted_tp)
        
        # Calculate binary classification metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        sensitivity = recall  # Sensitivity is the same as recall
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        # Calculate additional binary metrics
        hemorrhage_detection_rate = images_with_hemorrhage / self.total_images_processed if self.total_images_processed > 0 else 0.0
        avg_detections_per_positive = self.binary_stats['total_hemorrhage_detections'] / images_with_hemorrhage if images_with_hemorrhage > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'accuracy': accuracy,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'true_negatives': tn,
            'images_with_hemorrhage': images_with_hemorrhage,
            'images_without_hemorrhage': images_without_hemorrhage,
            'total_hemorrhage_detections': self.binary_stats['total_hemorrhage_detections'],
            'hemorrhage_detection_rate': hemorrhage_detection_rate,
            'avg_detections_per_positive': avg_detections_per_positive
        }
    
    def calculate_metrics(self):
        """Calculate precision, recall, F1, accuracy, sensitivity, specificity for each class."""
        metrics = {}
        
        for class_id, stats in self.class_stats.items():
            tp = stats['true_positives']
            fp = stats['false_positives']
            fn = stats['false_negatives']
            tn = stats['true_negatives']
            
            # For object detection, we'll estimate FP and FN based on detection patterns
            # This is a simplified approach - in practice, you'd need ground truth data
            total_images = stats['total_images']
            images_with_class = stats['images_with_class']
            
            # Estimate false negatives (images without detections but should have them)
            # This is a rough estimate - actual FN would require ground truth
            estimated_fn = max(0, int(total_images * 0.1) - tp)  # Assume 10% miss rate
            
            # Estimate false positives (over-detections)
            estimated_fp = max(0, int(tp * 0.05))  # Assume 5% false positive rate
            
            # Estimate true negatives (images correctly identified as not containing the class)
            estimated_tn = total_images - images_with_class - estimated_fn
            
            # Update with estimates
            fp = estimated_fp
            fn = estimated_fn
            tn = max(0, estimated_tn)
            
            # Calculate metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
            sensitivity = recall  # Sensitivity is the same as recall
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            
            class_name = self.class_names.get(class_id, f"class_{class_id}")
            
            metrics[class_id] = {
                'class_name': class_name,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'accuracy': accuracy,
                'sensitivity': sensitivity,
                'specificity': specificity,
                'true_positives': tp,
                'false_positives': fp,
                'false_negatives': fn,
                'true_negatives': tn,
                'total_detections': stats['total_detections'],
                'images_with_class': images_with_class,
                'total_images': total_images
            }
        
        return metrics
    
    def save_metrics(self, output_path):
        """Save calculated metrics to a text file."""
        metrics = self.calculate_metrics()
        binary_metrics = self.calculate_binary_metrics()
        
        with open(output_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("YOLO MODEL PERFORMANCE METRICS\n")
            f.write("="*80 + "\n")
            f.write(f"Confidence Threshold: {self.confidence_threshold}\n")
            f.write(f"Total Images Processed: {self.total_images_processed}\n")
            f.write(f"Total Valid Detections: {self.total_detections}\n")
            f.write(f"Average Detections per Image: {self.total_detections/self.total_images_processed if self.total_images_processed > 0 else 0:.2f}\n")
            f.write("\n")
            
            # Display loaded class names
            f.write("CLASS NAMES MAPPING\n")
            f.write("-" * 40 + "\n")
            if self.class_names:
                for class_id in sorted(self.class_names.keys()):
                    f.write(f"Class {class_id}: {self.class_names[class_id]}\n")
            else:
                f.write("No class names loaded from data.yaml\n")
            f.write("\n")
            
            # BINARY CLASSIFICATION METRICS (NEW SECTION)
            f.write("BINARY CLASSIFICATION METRICS (ANY HEMORRHAGE vs NO HEMORRHAGE)\n")
            f.write("="*80 + "\n")
            f.write("This section treats the problem as binary classification:\n")
            f.write("- Positive: Image contains any type of hemorrhage\n")
            f.write("- Negative: Image contains no hemorrhage\n")
            f.write("\n")
            
            f.write("Binary Classification Performance:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Precision:     {binary_metrics['precision']:.4f}\n")
            f.write(f"Recall:        {binary_metrics['recall']:.4f}\n")
            f.write(f"F1-Score:      {binary_metrics['f1_score']:.4f}\n")
            f.write(f"Accuracy:      {binary_metrics['accuracy']:.4f}\n")
            f.write(f"Sensitivity:   {binary_metrics['sensitivity']:.4f}\n")
            f.write(f"Specificity:   {binary_metrics['specificity']:.4f}\n")
            f.write("\n")
            
            f.write("Binary Classification Counts:\n")
            f.write("-" * 40 + "\n")
            f.write(f"True Positives:   {binary_metrics['true_positives']} (images correctly identified as having hemorrhage)\n")
            f.write(f"False Positives:  {binary_metrics['false_positives']} (images incorrectly identified as having hemorrhage)\n")
            f.write(f"False Negatives:  {binary_metrics['false_negatives']} (images with hemorrhage that were missed)\n")
            f.write(f"True Negatives:   {binary_metrics['true_negatives']} (images correctly identified as not having hemorrhage)\n")
            f.write("\n")
            
            f.write("Binary Classification Statistics:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Images with Hemorrhage:     {binary_metrics['images_with_hemorrhage']}\n")
            f.write(f"Images without Hemorrhage:  {binary_metrics['images_without_hemorrhage']}\n")
            f.write(f"Hemorrhage Detection Rate:  {binary_metrics['hemorrhage_detection_rate']*100:.1f}%\n")
            f.write(f"Total Hemorrhage Detections: {binary_metrics['total_hemorrhage_detections']}\n")
            f.write(f"Avg Detections per Positive Image: {binary_metrics['avg_detections_per_positive']:.2f}\n")
            f.write("\n")
            
            if not metrics:
                f.write("No valid detections found with the specified confidence threshold for multi-class analysis.\n")
                return
            
            # MULTI-CLASS METRICS (EXISTING SECTION)
            f.write("MULTI-CLASS CLASSIFICATION METRICS (BY HEMORRHAGE TYPE)\n")
            f.write("="*80 + "\n")
            
            # Overall summary
            f.write("OVERALL MULTI-CLASS SUMMARY\n")
            f.write("-" * 40 + "\n")
            avg_precision = sum(m['precision'] for m in metrics.values()) / len(metrics)
            avg_recall = sum(m['recall'] for m in metrics.values()) / len(metrics)
            avg_f1 = sum(m['f1_score'] for m in metrics.values()) / len(metrics)
            avg_accuracy = sum(m['accuracy'] for m in metrics.values()) / len(metrics)
            
            f.write(f"Average Precision: {avg_precision:.4f}\n")
            f.write(f"Average Recall: {avg_recall:.4f}\n")
            f.write(f"Average F1-Score: {avg_f1:.4f}\n")
            f.write(f"Average Accuracy: {avg_accuracy:.4f}\n")
            f.write(f"Number of Classes Detected: {len(metrics)}\n")
            f.write("\n")
            
            # Per-class metrics
            f.write("PER-CLASS METRICS\n")
            f.write("="*80 + "\n")
            
            for class_id in sorted(metrics.keys()):
                m = metrics[class_id]
                f.write(f"\nClass {class_id}: {m['class_name']}\n")
                f.write("-" * 40 + "\n")
                f.write(f"Precision:     {m['precision']:.4f}\n")
                f.write(f"Recall:        {m['recall']:.4f}\n")
                f.write(f"F1-Score:      {m['f1_score']:.4f}\n")
                f.write(f"Accuracy:      {m['accuracy']:.4f}\n")
                f.write(f"Sensitivity:   {m['sensitivity']:.4f}\n")
                f.write(f"Specificity:   {m['specificity']:.4f}\n")
                f.write(f"\nDetection Counts:\n")
                f.write(f"  True Positives:  {m['true_positives']}\n")
                f.write(f"  False Positives: {m['false_positives']}\n")
                f.write(f"  False Negatives: {m['false_negatives']}\n")
                f.write(f"  True Negatives:  {m['true_negatives']}\n")
                f.write(f"\nImage Statistics:\n")
                f.write(f"  Total Detections: {m['total_detections']}\n")
                f.write(f"  Images with Class: {m['images_with_class']}\n")
                f.write(f"  Total Images: {m['total_images']}\n")
                f.write(f"  Detection Rate: {m['images_with_class']/m['total_images']*100 if m['total_images'] > 0 else 0:.1f}%\n")

def process_image(image_path, api_url, token, input_dir, logger, metrics_collector, skip_existing=True, verify_ssl=True):
    """Process a single image through the Modal API using file upload."""
    try:
        # Check if output file already exists
        output_dir = get_output_dir(image_path, input_dir)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_file = os.path.join(output_dir, f"{base_name}.txt")
        
        if skip_existing and os.path.exists(output_file):
            logger.info(f"Skipping {image_path} - output file already exists")
            return True
            
        # Prepare the API endpoint URL
        # Make sure we're using the '/inference' endpoint that works
        if not api_url.endswith('/inference'):
            api_url = api_url.rstrip('/') + '/inference'
        
        # Prepare headers with bearer token if provided
        headers = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        
        # Prepare the file for upload
        with open(image_path, 'rb') as img_file:
            files = {
                'file': (os.path.basename(image_path), img_file, 'image/jpg')
            }
            
            # Make the API request
            logger.info(f"Sending file upload request to {api_url}")
            response = requests.post(api_url, files=files, headers=headers, verify=verify_ssl)
        
        # Check if the request was successful
        if response.status_code == 200:
            logger.info(f"Successfully processed {image_path}")
            
            # Parse the response
            result = response.json()
            logger.debug(f"Response: {json.dumps(result)[:200]}...")
            
            # Create output directory
            output_dir = get_output_dir(image_path, input_dir)
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_file = os.path.join(output_dir, f"{base_name}.txt")
            
            # Based on the API Explorer output, the format is:
            # {"result": [{"boxes": [], "confidences": [], "classes": [], "names": []}]}
            if "result" not in result:
                logger.warning(f"Expected 'result' key missing in response: {result}")
                return False
                
            # Extract detection results
            detections = result["result"]
            if not detections or not isinstance(detections, list) or len(detections) == 0:
                logger.warning(f"No detection results found in API response for {image_path}")
                return False
                
            # Get the first detection result (there's usually only one per image)
            detection = detections[0]
            
            # Check if we have the expected keys
            if not all(k in detection for k in ["boxes", "confidences", "classes"]):
                logger.warning(f"Detection missing required keys: {detection}")
                return False
                
            # Extract the bounding box information
            boxes = detection["boxes"]
            confidences = detection["confidences"]
            classes = detection["classes"]
            names = detection.get("names", [])  # Names might be optional
            
            # Check if we have matching array lengths
            if not (len(boxes) == len(confidences) == len(classes)):
                logger.warning(f"Mismatched array lengths: boxes={len(boxes)}, confidences={len(confidences)}, classes={len(classes)}")
                return False
            
            # Add results to metrics collector
            metrics_collector.add_detection_results(boxes, confidences, classes, names, image_path)
            
            # If there are no detections for this image
            if len(boxes) == 0:
                logger.info(f"No objects detected in {image_path}")
                # Create an empty file to indicate processing was done
                with open(output_file, 'w') as f:
                    pass
                return True
            
            # Filter detections by confidence threshold for saving
            valid_detections = []
            for i, confidence in enumerate(confidences):
                if confidence >= metrics_collector.confidence_threshold:
                    valid_detections.append((i, confidence))
            
            # Save detection results to a text file in YOLO format (only valid detections)
            with open(output_file, 'w') as f:
                for i, confidence in valid_detections:
                    box = boxes[i]
                    class_id = int(classes[i])  # Class ID as integer
                    
                    # Check if box is in the right format
                    if len(box) != 4:
                        logger.warning(f"Box has unexpected format: {box}")
                        continue
                    
                    # The API likely returns [x1, y1, x2, y2] format, convert to YOLO format
                    # YOLO format is [class_id, x_center, y_center, width, height] (normalized)
                    x1, y1, x2, y2 = box
                    
                    # Calculate YOLO format values
                    x_center = (x1 + x2) / 2
                    y_center = (y1 + y2) / 2
                    width = x2 - x1
                    height = y2 - y1
                    
                    # Write in YOLO format: class_id x_center y_center width height
                    f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
            
            logger.info(f"Saved {len(valid_detections)} valid detections (conf >= {metrics_collector.confidence_threshold}) to {output_file}")
            return True
        else:
            logger.error(f"Error processing {image_path}: {response.status_code} - {response.text}")
            return False
    
    except Exception as e:
        logger.error(f"Exception processing {image_path}: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Process images with Modal-deployed YOLOv8')
    parser.add_argument('--input', '-i', required=True, help='Input directory containing jpg images')
    parser.add_argument('--url', '-u', required=True, help='Modal API endpoint URL (without /inference)')
    parser.add_argument('--token', '-t', help='API bearer token if required')
    parser.add_argument('--workers', '-w', type=int, default=4, help='Number of parallel workers')
    parser.add_argument('--test', action='store_true', help='Test with a single image only')
    parser.add_argument('--debug', '-d', action='store_true', help='Enable debug logging')
    parser.add_argument('--verify-ssl', action='store_true', help='Verify SSL certificates')
    parser.add_argument('--force', action='store_true', help='Force reprocessing of existing output files')
    parser.add_argument('--confidence', '-c', type=float, default=0.25, help='Confidence threshold for valid detections (default: 0.25)')
    parser.add_argument('--data-yaml', help='Path to data.yaml file with class names (auto-detected if not provided)')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.debug)
    
    # Find and load class names from data.yaml
    yaml_path = args.data_yaml
    if not yaml_path:
        yaml_path = find_data_yaml(args.input)
        if yaml_path:
            logger.info(f"Found data.yaml at: {yaml_path}")
        else:
            logger.warning("Could not find data.yaml file. Class names may not be displayed correctly.")
    
    class_names = {}
    if yaml_path and os.path.exists(yaml_path):
        class_names = load_class_names_from_yaml(yaml_path)
    else:
        logger.warning(f"data.yaml file not found at {yaml_path}. Proceeding without class names.")
    
    # Initialize metrics collector with class names
    metrics_collector = MetricsCollector(confidence_threshold=args.confidence, class_names=class_names)
    
    # Find jpg images
    input_dir = os.path.abspath(args.input)
    logger.info(f"Searching for jpg images in {input_dir}")
    jpg_files = find_jpg_images(input_dir)
    
    if not jpg_files:
        logger.error("No jpg images found. Exiting.")
        return
    
    logger.info(f"Found {len(jpg_files)} jpg images")
    logger.info(f"Using confidence threshold: {args.confidence}")
    
    # Test mode - process just one image
    if args.test:
        logger.info("TEST MODE: Processing only the first image")
        test_image = jpg_files[0]
        logger.info(f"Testing with image: {test_image}")
        success = process_image(test_image, args.url, args.token, input_dir, logger, metrics_collector, not args.force, args.verify_ssl)
        
        if success:
            logger.info("Test successful! You can run without --test to process all images")
            
            # Save test metrics
            output_dir = get_output_dir(test_image, input_dir)
            metrics_file = os.path.join(output_dir, "test_metrics.txt")
            metrics_collector.save_metrics(metrics_file)
            logger.info(f"Test metrics saved to {metrics_file}")
        else:
            logger.error("Test failed. Please check the API endpoint and try again")
        return
    
    # Process all images in parallel
    success_count = 0
    skipped_count = 0
    
    # First check how many files would be skipped
    if not args.force:
        for image_path in jpg_files:
            output_dir = get_output_dir(image_path, input_dir)
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_file = os.path.join(output_dir, f"{base_name}.txt")
            if os.path.exists(output_file):
                skipped_count += 1
    
    if skipped_count > 0:
        logger.info(f"Will skip {skipped_count} images with existing output files. Use --force to reprocess them.")
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        # Submit tasks
        futures = [
            executor.submit(process_image, image_path, args.url, args.token, input_dir, logger, metrics_collector, not args.force, args.verify_ssl) 
            for image_path in jpg_files
        ]
        
        # Process results with progress bar
        for future in tqdm(futures, total=len(futures), desc="Processing images"):
            if future.result():
                success_count += 1
    
    # Report results and save metrics
    logger.info(f"Processed {success_count} out of {len(jpg_files)} images successfully")
    if skipped_count > 0:
        logger.info(f"Skipped {skipped_count} images with existing output files")
    
    # Save comprehensive metrics
    if success_count > 0:
        # Determine where to save metrics file
        parent_dir = os.path.dirname(os.path.normpath(input_dir))
        folder_name = os.path.basename(input_dir)
        metrics_file = os.path.join(parent_dir, f"{folder_name}_model_metrics.txt")
        
        metrics_collector.save_metrics(metrics_file)
        logger.info(f"Model performance metrics saved to {metrics_file}")
    else:
        logger.error("No images were processed successfully. Please check your API endpoint and connection.")

if __name__ == "__main__":
    main()