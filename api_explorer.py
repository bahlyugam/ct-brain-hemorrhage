import requests
import json
import argparse
import sys
import os
from urllib.parse import urljoin

def explore_api(api_url):
    """Try to explore the API and determine its structure."""
    print(f"Exploring API at {api_url}...")
    
    # Check for trailing slash and normalize
    if not api_url.endswith('/'):
        api_url += '/'
    
    # Common FastAPI endpoints to check
    common_paths = [
        "",                   # Root path
        "docs",               # Swagger docs
        "redoc",              # ReDoc
        "openapi.json",       # OpenAPI schema
    ]
    
    for path in common_paths:
        full_url = urljoin(api_url, path)
        print(f"Checking {full_url}...")
        
        try:
            response = requests.get(full_url, timeout=10)
            print(f"  Status: {response.status_code}")
            
            if response.status_code == 200:
                content_type = response.headers.get('Content-Type', '')
                print(f"  Content-Type: {content_type}")
                
                if 'application/json' in content_type:
                    print(f"  JSON Response: {json.dumps(response.json(), indent=2)[:500]}...")
                elif 'text/html' in content_type:
                    print(f"  HTML Response: Found HTML content ({len(response.text)} characters)")
                    
                    # Check if it's swagger docs
                    if 'swagger' in response.text.lower() or 'openapi' in response.text.lower():
                        print("  This appears to be API documentation (Swagger/OpenAPI)")
                        
                        # Try to extract endpoints from HTML
                        import re
                        endpoints = re.findall(r'"path":"([^"]+)"', response.text)
                        if endpoints:
                            print("  Detected API endpoints:")
                            for ep in set(endpoints):
                                print(f"    - {ep}")
                else:
                    print(f"  Response preview: {response.text[:200]}...")
            else:
                print(f"  Non-success response: {response.text[:200]}...")
                
        except requests.exceptions.RequestException as e:
            print(f"  Error: {str(e)}")

def test_image_upload(api_url, image_path):
    """Test uploading an image to various potential endpoints."""
    print(f"\nTesting image upload with {image_path}...")
    
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        return
    
    # If URL doesn't end with slash, add it
    if not api_url.endswith('/'):
        api_url += '/'
    
    # Potential API endpoints for YOLOv8 models
    endpoints = [
        "predict",            # YOLOv8 FastAPI standard
        "detect",             # Common for object detection
        "inference",          # Generic ML term
        "uploadfile",         # FastAPI convention
        "upload",             # Simple upload
        "process",            # Generic processing
        "yolo/predict",       # Nested path
        "model/predict",      # Another nested path
        ""                    # Root path
    ]
    
    # First try with multipart/form-data (file upload)
    print("\nTesting multipart/form-data uploads:")
    for endpoint in endpoints:
        full_url = urljoin(api_url, endpoint)
        print(f"Trying {full_url}...")
        
        try:
            with open(image_path, 'rb') as f:
                files = {'file': (os.path.basename(image_path), f, 'image/png')}
                response = requests.post(full_url, files=files, timeout=30)
                
                print(f"  Status: {response.status_code}")
                print(f"  Headers: {dict(response.headers)}")
                
                if response.status_code == 200:
                    try:
                        json_response = response.json()
                        print(f"  JSON Response: {json.dumps(json_response, indent=2)[:500]}...")
                        print("  SUCCESS! This endpoint works with file upload.")
                        print(f"  Use this endpoint: {endpoint}")
                        return
                    except:
                        print(f"  Non-JSON Response: {response.text[:200]}...")
                else:
                    print(f"  Response: {response.text[:200]}...")
        except Exception as e:
            print(f"  Error: {str(e)}")
    
    # Then try with JSON/base64
    print("\nTesting JSON/base64 uploads:")
    import base64
    
    with open(image_path, 'rb') as f:
        img_data = base64.b64encode(f.read()).decode('utf-8')
    
    # Try different JSON structures
    json_formats = [
        {"image": img_data},
        {"file": img_data},
        {"data": img_data},
        {"base64": img_data},
        {"image_data": img_data}
    ]
    
    for endpoint in endpoints:
        full_url = urljoin(api_url, endpoint)
        for json_format in json_formats:
            print(f"Trying {full_url} with format {json_format.keys()}...")
            
            try:
                response = requests.post(
                    full_url, 
                    json=json_format,
                    headers={"Content-Type": "application/json"},
                    timeout=30
                )
                
                print(f"  Status: {response.status_code}")
                
                if response.status_code == 200:
                    try:
                        json_response = response.json()
                        print(f"  JSON Response: {json.dumps(json_response, indent=2)[:500]}...")
                        print("  SUCCESS! This endpoint works with JSON/base64.")
                        print(f"  Use this endpoint: {endpoint} with format {json_format.keys()}")
                        return
                    except:
                        print(f"  Non-JSON Response: {response.text[:200]}...")
                else:
                    print(f"  Response: {response.text[:50]}...")
            except Exception as e:
                print(f"  Error: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Explore a FastAPI YOLOv8 endpoint')
    parser.add_argument('--url', '-u', required=True, help='API base URL')
    parser.add_argument('--image', '-i', help='Path to a test image')
    
    args = parser.parse_args()
    
    # First explore the API structure
    explore_api(args.url)
    
    # Then test image upload if an image was provided
    if args.image:
        test_image_upload(args.url, args.image)
    else:
        print("\nNo test image provided. Use --image to test image uploads.")

if __name__ == "__main__":
    main()