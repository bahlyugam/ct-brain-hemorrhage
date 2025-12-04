import boto3
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# AWS credentials and S3 bucket details
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
BUCKET_NAME = "tnsqaiapp"
S3_FOLDER = "sms/543050876/"
LOCAL_DOWNLOAD_PATH = "/Users/yugambahl/Desktop/brain_ct/data"

# Initialize S3 client
s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY
)

def download_folder_from_s3(bucket_name, s3_folder, local_path):
    if not os.path.exists(local_path):
        os.makedirs(local_path)

    # List all objects in the folder
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=s3_folder)

    if "Contents" not in response:
        print("No files found in the specified S3 folder.")
        return

    for obj in response["Contents"]:
        s3_file_path = obj["Key"]
        relative_path = s3_file_path[len(s3_folder):]  # Remove folder prefix
        local_file_path = os.path.join(local_path, relative_path)

        # Create directories if they don’t exist
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

        print(f"Downloading {s3_file_path} to {local_file_path}...")
        s3.download_file(bucket_name, s3_file_path, local_file_path)

    print("Download complete!")

# Call the function
download_folder_from_s3(BUCKET_NAME, S3_FOLDER, LOCAL_DOWNLOAD_PATH)