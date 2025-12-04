import boto3
import os

def list_s3_folders(bucket_name, output_file):
    s3 = boto3.client('s3')
    paginator = s3.get_paginator('list_objects_v2')
    result_set = paginator.paginate(Bucket=bucket_name)
    
    folders = set()
    
    for page in result_set:
        if 'Contents' in page:
            for obj in page['Contents']:
                folder_path = os.path.dirname(obj['Key'])
                if folder_path:
                    folders.add(folder_path)
    
    with open(output_file, 'w') as f:
        for folder in sorted(folders):
            f.write(folder + '\n')
    
    print(f"Saved folder list to {output_file}")

if __name__ == "__main__":
    bucket_name = "tnsqtraining"  # Replace with your bucket name
    output_file = "s3_folders.txt"  # Output file to save folder names
    
    list_s3_folders(bucket_name, output_file)
