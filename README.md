# Brain CT Hemorrhage Detection

A machine learning project for detecting brain hemorrhages in CT scans using YOLO object detection.

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/bahlyugam/ct-brain-hemorrhage.git
cd ct-brain-hemorrhage
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- `python-dotenv` - For loading environment variables
- `boto3` - For AWS S3 integration
- `pydicom` - For DICOM file processing
- `opencv-python` - For image processing
- `pandas`, `numpy` - For data manipulation
- `ultralytics` - For YOLO model training and inference
- `wandb` - For experiment tracking (optional)
- `openai` - For OpenAI API integration (optional)
- `roboflow` - For Roboflow dataset management (optional)

### 3. Configure Environment Variables

Copy the example environment file and add your credentials:

```bash
cp .env.example .env
```

Then edit `.env` and add your actual API keys:

```bash
# AWS Credentials
AWS_ACCESS_KEY=your_aws_access_key_here
AWS_SECRET_KEY=your_aws_secret_key_here

# OpenAI API Key (optional)
OPENAI_API_KEY=your_openai_api_key_here

# Roboflow API Key (optional)
ROBOFLOW_API_KEY=your_roboflow_api_key_here

# Weights & Biases API Key (optional)
WANDB_API_KEY=your_wandb_api_key_here
```

**Important:** Never commit the `.env` file to git. It's already in `.gitignore`.

### 4. Data Directory Structure

The project expects data to be organized as follows:

```
data/
├── raw_data/           # Raw DICOM files
├── metadata/           # CSV and JSON metadata files
├── training_datasets/  # Processed training data
└── documentation/      # Dataset documentation
```

Note: Data files are excluded from git via `.gitignore`.

## Project Structure

- **Training Scripts:**
  - [train.py](train.py) - Main YOLO training script
  - [yolov8s_GRE.py](yolov8s_GRE.py) - YOLOv8 training with custom configurations

- **Data Processing:**
  - [s3_dicom_to_png.py](s3_dicom_to_png.py) - Convert DICOM files from S3 to PNG
  - [bulk_s3_to_png.py](bulk_s3_to_png.py) - Batch DICOM to PNG conversion
  - [download_negative_feedback_images.py](download_negative_feedback_images.py) - Download images from S3
  - [yolo_augmented_dataset.py](yolo_augmented_dataset.py) - Data augmentation for YOLO

- **Analysis & Utilities:**
  - [analyze_dataset.py](analyze_dataset.py) - Dataset analysis and statistics
  - [dataset_analysis.py](dataset_analysis.py) - Comprehensive dataset analysis
  - [combine_datasets.py](combine_datasets.py) - Merge multiple datasets

- **Deployment:**
  - [deploy_yolo.py](deploy_yolo.py) - YOLO model deployment
  - [yolo_inference_client.py](yolo_inference_client.py) - Inference client

## Usage

### Training a Model

```bash
python train.py
```

### Converting DICOM to PNG

```bash
python s3_dicom_to_png.py
```

### Running Inference

```bash
python deploy_yolo.py
```

## Security Notes

- All credentials are managed via environment variables
- Never commit the `.env` file or any files containing secrets
- AWS keys, API keys, and other sensitive data should only be stored in `.env`
- If you accidentally commit secrets, rotate them immediately

## Documentation

- [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) - Project overview and architecture
- [IMPROVEMENTS_SUMMARY.md](IMPROVEMENTS_SUMMARY.md) - Summary of improvements made
- [QUICK_START_IMPROVEMENTS.md](QUICK_START_IMPROVEMENTS.md) - Quick start guide

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]
