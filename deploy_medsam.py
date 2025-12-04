import modal
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from pydicom import dcmread
from PIL import Image, ImageDraw
import io
import os
import logging
import numpy as np
import torch
from segment_anything import sam_model_registry
from skimage import transform
import torch.nn.functional as F
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from fastapi.responses import JSONResponse
import base64
import ast

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = modal.App("inference-ct-brain-hemorrhage-medsam")

volume = modal.Volume.from_name("medsam-ct-brain-hemorrhage", create_if_missing=True)

image = modal.Image.debian_slim().apt_install("libopencv-dev","git").pip_install(
    "fastapi",
    "pillow",
    "python-multipart",
    "pydicom",
    "numpy",
    "torch",
    "scipy",
    "matplotlib",
    "scikit-image",
    "git+https://github.com/bowang-lab/MedSAM.git",
    "torchvision"
)

@app.function(
    image=image,
    gpu="t4",
    volumes={"/root/model": volume},
    timeout=600,
    scaledown_window=300,
)
@modal.asgi_app()
def fastapi_app():
    app = FastAPI()

    MedSAM_CKPT_PATH = "/root/model/medsam_vit_b.pth"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    medsam_model = sam_model_registry['vit_b'](checkpoint=MedSAM_CKPT_PATH)
    medsam_model = medsam_model.to(device)
    medsam_model.eval()
    logger.info(f"MedSAM model loaded successfully on {device}")

    def medsam_inference(medsam_model, img_embed, box_1024, H, W):
        box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=img_embed.device)
        if len(box_torch.shape) == 2:
            box_torch = box_torch[:, None, :]  # (B, 1, 4)
    
        with torch.no_grad():  # Ensure no gradients are computed
            sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
                points=None,
                boxes=box_torch,
                masks=None,
            )
            low_res_logits, *_ = medsam_model.mask_decoder(
                image_embeddings=img_embed,  # (B, 256, 64, 64)
                image_pe=medsam_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
                sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
                dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
                multimask_output=False,
            )
            low_res_pred = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)
            low_res_pred = F.interpolate(
                low_res_pred,
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            )  # (1, 1, gt.shape)
    
        low_res_pred = low_res_pred.squeeze().cpu().numpy()  # (256, 256)
        medsam_seg = (low_res_pred > 0.5).astype(np.uint8)
        return medsam_seg

    def find_longest_length_breadth(segmentation, dicom_file):
        ds = dcmread(dicom_file)
        pixel_spacing = ds.PixelSpacing

        coords = np.column_stack(np.where(segmentation > 0))

        distances = pdist(coords, 'euclidean')
        dist_matrix = squareform(distances)

        max_dist_idx = np.unravel_index(np.argmax(dist_matrix, axis=None), dist_matrix.shape)
        longest_length_pixels = dist_matrix[max_dist_idx]

        min_row, min_col = np.min(coords, axis=0)
        max_row, max_col = np.max(coords, axis=0)

        longest_length_mm = longest_length_pixels * pixel_spacing[0]  # Calculate in mm
        longest_breadth_mm = (max_col - min_col) * pixel_spacing[1]  # Calculate in mm
        return longest_length_mm, longest_breadth_mm

    def calculate_area_mm2(segmentation, dicom_file, width, height):
        ds = dcmread(dicom_file)
        pixel_spacing = ds.PixelSpacing
        slice_thickness = ds.SliceThickness  # Get slice thickness from DICOM file
        width_mm = width * pixel_spacing[0]  # Calculate in mm
        height_mm = height * pixel_spacing[1]  # Calculate in mm
        pixel_area_mm2 = pixel_spacing[0] * pixel_spacing[1]
        num_pixels = np.sum(segmentation)
        area_mm2 = num_pixels * pixel_area_mm2
        volume_mm3 = area_mm2 * slice_thickness  # Calculate volume in mm^3
        return area_mm2, volume_mm3

    def create_mask_image(segmentation_mask, original_size):
        """
        Create a clean segmentation mask image (just the mask, no background image)
        
        Args:
            segmentation_mask: numpy array of the segmentation (0s and 1s)
            original_size: tuple of (width, height)
            
        Returns:
            PIL Image: Clean mask image (RGBA format)
        """
        try:
            # Create RGBA image (Red, Green, Blue, Alpha)
            mask_rgba = np.zeros((*segmentation_mask.shape, 4), dtype=np.uint8)
            
            # Where mask is True (segmentation exists), set to white with full alpha
            mask_rgba[segmentation_mask > 0] = [255, 255, 255, 255]  # White mask
            # Where mask is False (no segmentation), set to transparent
            mask_rgba[segmentation_mask == 0] = [0, 0, 0, 0]  # Transparent
            
            # Convert to PIL Image
            mask_image = Image.fromarray(mask_rgba, 'RGBA')
            
            # Ensure correct size
            if mask_image.size != original_size:
                mask_image = mask_image.resize(original_size, Image.NEAREST)  # Use NEAREST to preserve binary mask
            
            return mask_image
            
        except Exception as e:
            logger.error(f"Error creating mask image: {e}")
            # Return transparent image as fallback
            return Image.new('RGBA', original_size, (0, 0, 0, 0))

    def create_colored_mask_visualization(segmentation_mask, original_size, color=(255, 100, 100, 150)):
        """
        Create a colored visualization of just the segmentation mask for preview
        
        Args:
            segmentation_mask: numpy array of the segmentation
            original_size: tuple of (width, height)  
            color: tuple of (R, G, B, A) for the mask color
            
        Returns:
            PIL Image: Colored mask visualization
        """
        try:
            # Create RGBA image
            colored_mask = np.zeros((*segmentation_mask.shape, 4), dtype=np.uint8)
            
            # Apply color where mask exists
            colored_mask[segmentation_mask > 0] = color
            # Transparent elsewhere
            colored_mask[segmentation_mask == 0] = [0, 0, 0, 0]
            
            # Convert to PIL Image
            mask_image = Image.fromarray(colored_mask, 'RGBA')
            
            # Resize if needed
            if mask_image.size != original_size:
                mask_image = mask_image.resize(original_size, Image.NEAREST)
            
            return mask_image
            
        except Exception as e:
            logger.error(f"Error creating colored mask: {e}")
            return Image.new('RGBA', original_size, (0, 0, 0, 0))

    @app.get("/health")
    async def health_check():
        return {"status": "healthy", "message": "Model loaded successfully"}

    @app.post("/inference")
    async def inference(dicom_file: UploadFile = File(...), image_file: UploadFile = File(...), bbox: str = Form(...)):
        try:
            dicom_content = await dicom_file.read()
            with open("/tmp/temp.dcm", "wb") as f:
                f.write(dicom_content)
            
            image_content = await image_file.read()
            img = Image.open(io.BytesIO(image_content))
            img_np = np.array(img)

            if len(img_np.shape) == 2:
                img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
            else:
                img_3c = img_np
            H, W, _ = img_3c.shape
            original_size = (W, H)  # PIL uses (width, height)

            img_1024 = transform.resize(img_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True).astype(np.uint8)
            img_1024 = (img_1024 - img_1024.min()) / np.clip(
                img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
            )  # normalize to [0, 1], (H, W, 3)
            img_1024_tensor = torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(device)

            bbox_list = ast.literal_eval(bbox)
            if not isinstance(bbox_list, list) or not all(isinstance(x, int) for x in bbox_list):
                raise ValueError("bbox must be a list of integers")
        
            # Convert to numpy array with shape (1, 4)
            box_np = np.array([bbox_list])
        
            logger.info(f"box_np: {box_np}")
            
            width = box_np[0, 2] - box_np[0, 0]
            height = box_np[0, 3] - box_np[0, 1]
            box_1024 = box_np / np.array([W, H, W, H]) * 1024

            # Run MedSAM inference
            with torch.no_grad():
                image_embedding = medsam_model.image_encoder(img_1024_tensor)  # (1, 256, 64, 64)

            medsam_seg = medsam_inference(medsam_model, image_embedding, box_1024, H, W)
        
            # Calculate measurements
            area_mm2, volume_mm3 = calculate_area_mm2(medsam_seg, "/tmp/temp.dcm", width, height)
            longest_length_mm, longest_breadth_mm = find_longest_length_breadth(medsam_seg, "/tmp/temp.dcm")

            # Create clean mask image (binary mask)
            clean_mask = create_mask_image(medsam_seg, original_size)
            
            # Create colored preview (optional - for debugging/preview)
            colored_preview = create_colored_mask_visualization(medsam_seg, original_size)

            # Convert masks to base64
            # 1. Clean binary mask (what you'll use for overlaying)
            mask_buffer = io.BytesIO()
            clean_mask.save(mask_buffer, format='PNG')
            mask_buffer.seek(0)
            clean_mask_b64 = base64.b64encode(mask_buffer.getvalue()).decode()

            # 2. Colored preview (for debugging/visualization)
            preview_buffer = io.BytesIO()
            colored_preview.save(preview_buffer, format='PNG')
            preview_buffer.seek(0)
            colored_preview_b64 = base64.b64encode(preview_buffer.getvalue()).decode()

            result = {
                "area_mm2": float(area_mm2),
                "volume_mm3": float(volume_mm3),
                "longest_length_mm": float(longest_length_mm),
                "longest_breadth_mm": float(longest_breadth_mm),
                "segmentation_mask": clean_mask_b64,  # Clean binary mask for your overlaying
                "colored_preview": colored_preview_b64,  # Optional colored preview
                "mask_dimensions": {
                    "width": original_size[0],
                    "height": original_size[1]
                }
            }

            return JSONResponse(content=result)

        except Exception as e:
            logger.error(f"An error occurred during inference: {e}")
            raise HTTPException(status_code=500, detail=f"An error occurred during inference: {str(e)}")

    return app

if __name__ == "__main__":
    app.serve()