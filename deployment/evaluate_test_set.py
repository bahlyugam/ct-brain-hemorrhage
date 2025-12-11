"""
Evaluate both YOLO and RF-DETR models on the test set.

Runs inference via Modal APIs and computes COCO metrics locally.
"""

import os
import json
import requests
import argparse
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestSetLoader:
    """Load COCO test set."""

    def __init__(self, test_dir: str):
        self.test_dir = test_dir
        self.annotations_file = os.path.join(test_dir, "_annotations.coco.json")

    def load_annotations(self) -> dict:
        with open(self.annotations_file, 'r') as f:
            return json.load(f)

    def get_image_paths(self) -> list:
        """Get all test images, resolving symlinks."""
        images = []
        for file in os.listdir(self.test_dir):
            if file.endswith(('.png', '.jpg')):
                path = os.path.join(self.test_dir, file)
                # Resolve symlink if necessary
                if os.path.islink(path):
                    path = os.path.realpath(path)
                images.append(path)
        return sorted(images)


class ModalInferenceClient:
    """Client for calling Modal inference APIs."""

    def __init__(self, yolo_url: str, rfdetr_url: str):
        self.yolo_url = yolo_url
        self.rfdetr_url = rfdetr_url

    def predict_single(self, image_path: str, model: str) -> dict:
        """Run inference on single image."""
        url = self.yolo_url if model == 'yolo' else self.rfdetr_url
        endpoint = f"{url}/inference"

        try:
            with open(image_path, 'rb') as f:
                files = {'file': (os.path.basename(image_path), f, 'image/png')}
                response = requests.post(endpoint, files=files, timeout=30)

            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"API error for {image_path}: {response.status_code} - {response.text}")
                return {"result": [{"boxes": [], "confidences": [], "classes": [], "names": []}]}
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            return {"result": [{"boxes": [], "confidences": [], "classes": [], "names": []}]}

    def predict_batch(self, image_paths: list, model: str, max_workers: int = 4) -> list:
        """Run batch inference with parallel workers."""
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self.predict_single, path, model)
                for path in image_paths
            ]
            for future in tqdm(futures, desc=f"Running {model.upper()} inference"):
                try:
                    result = future.result(timeout=60)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error: {e}")
                    results.append({"result": [{"boxes": [], "confidences": [], "classes": [], "names": []}]})
        return results


class COCOEvaluator:
    """COCO evaluation using pycocotools."""

    def __init__(self, ground_truth_file: str, class_names: list):
        self.coco_gt = COCO(ground_truth_file)
        self.class_names = class_names

    def get_image_id_from_filename(self, filename: str) -> int:
        """Map filename to COCO image ID."""
        for img in self.coco_gt.dataset['images']:
            if img['file_name'] == filename:
                return img['id']
        raise ValueError(f"Image {filename} not found in annotations")

    def convert_predictions_to_coco(self, predictions: list, image_paths: list) -> list:
        """Convert API predictions to COCO format."""
        coco_predictions = []

        for pred, image_path in zip(predictions, image_paths):
            # Get image ID from filename
            filename = os.path.basename(image_path)
            image_id = self.get_image_id_from_filename(filename)

            # Extract detections from API response
            result = pred['result'][0]  # Unwrap result
            boxes = result['boxes']
            confidences = result['confidences']
            classes = result['classes']

            for bbox, conf, cls in zip(boxes, confidences, classes):
                # Convert xyxy to xywh
                x_min, y_min, x_max, y_max = bbox
                width = x_max - x_min
                height = y_max - y_min

                coco_predictions.append({
                    'image_id': image_id,
                    'category_id': int(cls),
                    'bbox': [x_min, y_min, width, height],
                    'score': float(conf)
                })

        return coco_predictions

    def evaluate(self, predictions: list, image_paths: list) -> dict:
        """Run COCO evaluation."""
        logger.info("Converting predictions to COCO format...")
        coco_preds = self.convert_predictions_to_coco(predictions, image_paths)

        if not coco_preds:
            logger.warning("No predictions to evaluate!")
            return {
                'mAP_50_95': 0.0,
                'mAP_50': 0.0,
                'mAP_75': 0.0,
                'AR_100': 0.0,
                'per_class': {}
            }

        logger.info(f"Evaluating {len(coco_preds)} predictions...")

        # Load predictions into COCO
        coco_dt = self.coco_gt.loadRes(coco_preds)

        # Run COCO evaluation
        coco_eval = COCOeval(self.coco_gt, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        # Extract metrics
        metrics = {
            'mAP_50_95': float(coco_eval.stats[0]),
            'mAP_50': float(coco_eval.stats[1]),
            'mAP_75': float(coco_eval.stats[2]),
            'mAP_small': float(coco_eval.stats[3]),
            'mAP_medium': float(coco_eval.stats[4]),
            'mAP_large': float(coco_eval.stats[5]),
            'AR_1': float(coco_eval.stats[6]),
            'AR_10': float(coco_eval.stats[7]),
            'AR_100': float(coco_eval.stats[8]),
            'AR_small': float(coco_eval.stats[9]),
            'AR_medium': float(coco_eval.stats[10]),
            'AR_large': float(coco_eval.stats[11]),
        }

        # Per-class metrics
        metrics['per_class'] = {}
        for class_id, class_name in enumerate(self.class_names):
            # AP@50 for this class
            ap_50 = coco_eval.eval['precision'][0, :, class_id, 0, 2].mean()
            # AP@50-95 for this class
            ap_50_95 = coco_eval.eval['precision'][:, :, class_id, 0, 2].mean()

            metrics['per_class'][class_name] = {
                'ap_50': float(ap_50) if not np.isnan(ap_50) else 0.0,
                'ap_50_95': float(ap_50_95) if not np.isnan(ap_50_95) else 0.0,
            }

        return metrics


class ComparisonReporter:
    """Generate comparison report between YOLO and RF-DETR."""

    def __init__(self, yolo_metrics: dict, rfdetr_metrics: dict):
        self.yolo_metrics = yolo_metrics
        self.rfdetr_metrics = rfdetr_metrics

    def generate_text_report(self) -> str:
        """Generate text comparison report."""
        report = []
        report.append("=" * 80)
        report.append("BRAIN CT HEMORRHAGE DETECTION - MODEL COMPARISON")
        report.append("=" * 80)
        report.append(f"Test Set: 279 images, 341 annotations")
        report.append("")

        report.append("OVERALL METRICS")
        report.append("-" * 80)
        for metric in ['mAP_50_95', 'mAP_50', 'mAP_75', 'AR_100']:
            yolo_val = self.yolo_metrics.get(metric, 0.0)
            rfdetr_val = self.rfdetr_metrics.get(metric, 0.0)
            delta = rfdetr_val - yolo_val
            winner = "RF-DETR" if delta > 0.001 else "YOLO" if delta < -0.001 else "TIE"

            report.append(
                f"{metric:12s}: YOLO={yolo_val:.4f}  RF-DETR={rfdetr_val:.4f}  "
                f"Δ={delta:+.4f}  Winner={winner}"
            )

        report.append("")
        report.append("PER-CLASS METRICS (AP@50)")
        report.append("-" * 80)

        for class_name in ['IPH', 'IVH', 'SAH', 'SDH']:
            yolo_ap = self.yolo_metrics.get('per_class', {}).get(class_name, {}).get('ap_50', 0.0)
            rfdetr_ap = self.rfdetr_metrics.get('per_class', {}).get(class_name, {}).get('ap_50', 0.0)
            delta = rfdetr_ap - yolo_ap
            winner = "RF-DETR" if delta > 0.01 else "YOLO" if delta < -0.01 else "TIE"

            report.append(
                f"{class_name:6s}: YOLO={yolo_ap:.4f}  RF-DETR={rfdetr_ap:.4f}  "
                f"Δ={delta:+.4f}  Winner={winner}"
            )

        report.append("=" * 80)
        return '\n'.join(report)

    def generate_csv_report(self) -> str:
        """Generate CSV comparison report."""
        import io
        output = io.StringIO()

        # Overall metrics
        output.write("Metric,YOLO,RF-DETR,Delta,Winner\n")
        for metric in ['mAP_50_95', 'mAP_50', 'mAP_75', 'AR_100']:
            yolo_val = self.yolo_metrics.get(metric, 0.0)
            rfdetr_val = self.rfdetr_metrics.get(metric, 0.0)
            delta = rfdetr_val - yolo_val
            winner = "RF-DETR" if delta > 0.001 else "YOLO" if delta < -0.001 else "TIE"
            output.write(f"{metric},{yolo_val:.4f},{rfdetr_val:.4f},{delta:+.4f},{winner}\n")

        # Per-class metrics
        output.write("\nClass,YOLO_AP50,RF-DETR_AP50,Delta,Winner\n")
        for class_name in ['IPH', 'IVH', 'SAH', 'SDH']:
            yolo_ap = self.yolo_metrics.get('per_class', {}).get(class_name, {}).get('ap_50', 0.0)
            rfdetr_ap = self.rfdetr_metrics.get('per_class', {}).get(class_name, {}).get('ap_50', 0.0)
            delta = rfdetr_ap - yolo_ap
            winner = "RF-DETR" if delta > 0.01 else "YOLO" if delta < -0.01 else "TIE"
            output.write(f"{class_name},{yolo_ap:.4f},{rfdetr_ap:.4f},{delta:+.4f},{winner}\n")

        return output.getvalue()


def main():
    parser = argparse.ArgumentParser(description="Evaluate both models on test set")
    parser.add_argument('--yolo-url',
                       default='https://tnsqai-2--inference-ct-brain-hemorrhage-yolov8m-v2-fastapi-app.modal.run',
                       help='YOLO Modal API URL')
    parser.add_argument('--rfdetr-url',
                       default='https://tnsqai-2--inference-ct-brain-hemorrhage-rfdetr-medium-v3-b8c1a6.modal.run',
                       help='RF-DETR Modal API URL')
    parser.add_argument('--test-dir',
                       default='/Users/yugambahl/Desktop/brain_ct/data/training_datasets/filtered_4class/coco/test',
                       help='Test dataset directory')
    parser.add_argument('--output-dir', default='./evaluation_results',
                       help='Output directory for results')
    parser.add_argument('--max-workers', type=int, default=4,
                       help='Number of parallel workers for inference')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print("BRAIN CT HEMORRHAGE DETECTION - MODEL EVALUATION")
    print("=" * 80)
    print(f"YOLO URL:    {args.yolo_url}")
    print(f"RF-DETR URL: {args.rfdetr_url}")
    print(f"Test Dir:    {args.test_dir}")
    print(f"Output Dir:  {args.output_dir}")
    print("=" * 80 + "\n")

    # Step 1: Load test set
    print("[1/6] Loading test set...")
    loader = TestSetLoader(args.test_dir)
    image_paths = loader.get_image_paths()
    print(f"  ✓ Loaded {len(image_paths)} images")

    # Step 2: Run YOLO inference
    print(f"\n[2/6] Running YOLO inference (workers={args.max_workers})...")
    client = ModalInferenceClient(args.yolo_url, args.rfdetr_url)
    yolo_predictions = client.predict_batch(image_paths, model='yolo', max_workers=args.max_workers)

    yolo_pred_file = os.path.join(args.output_dir, 'yolo_predictions.json')
    with open(yolo_pred_file, 'w') as f:
        json.dump(yolo_predictions, f, indent=2)
    print(f"  ✓ Saved predictions to {yolo_pred_file}")

    # Step 3: Run RF-DETR inference
    print(f"\n[3/6] Running RF-DETR inference (workers={args.max_workers})...")
    rfdetr_predictions = client.predict_batch(image_paths, model='rfdetr', max_workers=args.max_workers)

    rfdetr_pred_file = os.path.join(args.output_dir, 'rfdetr_predictions.json')
    with open(rfdetr_pred_file, 'w') as f:
        json.dump(rfdetr_predictions, f, indent=2)
    print(f"  ✓ Saved predictions to {rfdetr_pred_file}")

    # Step 4: Compute YOLO COCO metrics
    print("\n[4/6] Computing YOLO COCO metrics...")
    evaluator = COCOEvaluator(
        os.path.join(args.test_dir, '_annotations.coco.json'),
        ['IPH', 'IVH', 'SAH', 'SDH']
    )

    yolo_metrics = evaluator.evaluate(yolo_predictions, image_paths)

    yolo_metrics_file = os.path.join(args.output_dir, 'yolo_metrics.json')
    with open(yolo_metrics_file, 'w') as f:
        json.dump(yolo_metrics, f, indent=2)
    print(f"  ✓ YOLO mAP@50: {yolo_metrics['mAP_50']:.4f}")
    print(f"  ✓ YOLO mAP@50-95: {yolo_metrics['mAP_50_95']:.4f}")
    print(f"  ✓ Saved metrics to {yolo_metrics_file}")

    # Step 5: Compute RF-DETR COCO metrics
    print("\n[5/6] Computing RF-DETR COCO metrics...")
    rfdetr_metrics = evaluator.evaluate(rfdetr_predictions, image_paths)

    rfdetr_metrics_file = os.path.join(args.output_dir, 'rfdetr_metrics.json')
    with open(rfdetr_metrics_file, 'w') as f:
        json.dump(rfdetr_metrics, f, indent=2)
    print(f"  ✓ RF-DETR mAP@50: {rfdetr_metrics['mAP_50']:.4f}")
    print(f"  ✓ RF-DETR mAP@50-95: {rfdetr_metrics['mAP_50_95']:.4f}")
    print(f"  ✓ Saved metrics to {rfdetr_metrics_file}")

    # Step 6: Generate comparison report
    print("\n[6/6] Generating comparison report...")
    reporter = ComparisonReporter(yolo_metrics, rfdetr_metrics)

    # Text report
    report_text = reporter.generate_text_report()
    report_file = os.path.join(args.output_dir, 'comparison_report.txt')
    with open(report_file, 'w') as f:
        f.write(report_text)

    # CSV report
    csv_report = reporter.generate_csv_report()
    csv_file = os.path.join(args.output_dir, 'comparison_report.csv')
    with open(csv_file, 'w') as f:
        f.write(csv_report)

    print(f"  ✓ Saved text report to {report_file}")
    print(f"  ✓ Saved CSV report to {csv_file}")

    print("\n" + report_text)
    print(f"\n✅ Evaluation complete! Results saved to: {args.output_dir}\n")


if __name__ == "__main__":
    main()
