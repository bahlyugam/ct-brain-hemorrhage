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

```
brain_ct/
├── train.py                    # Main training script
├── ct_augmentations.py         # CT-specific augmentation techniques
├── analyze_dataset.py          # Dataset analysis
├── docs/                       # Documentation
│   ├── SETUP_INSTRUCTIONS.md
│   ├── WANDB_METRICS_GUIDE.md
│   ├── PROJECT_OVERVIEW.md
│   ├── PHASE3_SUMMARY.md
│   └── ...
├── deployment/                 # Deployment scripts
│   ├── deploy_yolo.py
│   ├── deploy_rfdetr.py
│   ├── evaluate_test_set.py
│   └── test_rfdetr_output.py
├── utils/                      # Utility scripts
│   ├── s3_dicom_to_png.py
│   ├── bulk_s3_to_png.py
│   ├── dataset_analysis.py
│   ├── yolo_inference_client.py
│   └── ...
├── models/                     # Model implementations
│   ├── yolo_model.py
│   ├── rfdetr_model.py
│   └── base_model.py
├── scripts/                    # Additional scripts
├── evaluation_results/         # Evaluation outputs
└── data/                       # Data directory (gitignored)
```

## Usage

### Training a Model

```bash
python train.py
```

### Converting DICOM to PNG

```bash
python utils/s3_dicom_to_png.py
```

### Running Inference

```bash
# YOLO deployment
python deployment/deploy_yolo.py

# RF-DETR deployment
python deployment/deploy_rfdetr.py

# Evaluate on test set
python deployment/evaluate_test_set.py
```

## Security Notes

- All credentials are managed via environment variables
- Never commit the `.env` file or any files containing secrets
- AWS keys, API keys, and other sensitive data should only be stored in `.env`
- If you accidentally commit secrets, rotate them immediately

## Documentation

For detailed documentation, see the [docs/](docs/) directory:

- [docs/SETUP_INSTRUCTIONS.md](docs/SETUP_INSTRUCTIONS.md) - Detailed setup instructions
- [docs/WANDB_METRICS_GUIDE.md](docs/WANDB_METRICS_GUIDE.md) - Guide to using Weights & Biases
- [docs/PROJECT_OVERVIEW.md](docs/PROJECT_OVERVIEW.md) - Project overview and architecture
- [docs/IMPROVEMENTS_SUMMARY.md](docs/IMPROVEMENTS_SUMMARY.md) - Summary of improvements made
- [docs/PHASE3_SUMMARY.md](docs/PHASE3_SUMMARY.md) - Phase 3 development summary

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]
