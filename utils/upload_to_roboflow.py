import os
import roboflow
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

rf = roboflow.Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))

# Get a workspace
workspace = rf.workspace("https://app.roboflow.com/tnsqai-maigj")

# Upload dataset to a new/existing project
workspace.upload_dataset(
    "data/roboflow_dataset",  # This is your dataset path
    "ct_brain_hemorrhage",  # This will either create or get a dataset with the given ID
    num_workers=10,  # Number of images to upload concurrently
    project_license="MIT",
    project_type="object-detection",
    batch_name=None,
    num_retries=0
)