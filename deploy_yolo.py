import modal
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import io
import os
import logging

MODEL="yolov8m"
VERSION="v2"

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = modal.App(f"inference-ct-brain-hemorrhage-{MODEL}-{VERSION}")

volume = modal.Volume.from_name(f"{MODEL}-v9_conservative_focal-brain-ct-hemorrhage", create_if_missing=True)

image = modal.Image.debian_slim().apt_install("libopencv-dev").pip_install(
    "ultralytics",
    "fastapi",
    "pillow",
    "python-multipart"
)

@app.function(
    image=image,
    gpu="t4",
    volumes={"/root/model": volume},
    timeout=600,
    scaledown_window=300,
    # keep_warm=1  # Keep 1 container always warm
)
@modal.asgi_app()
def fastapi_app():
    app = FastAPI()
    from ultralytics import YOLO
    
    # Load the model at the module level
    model_path = "/root/model/best.pt_83.pt"
    if not os.path.exists(model_path):
        logger.error(f"Model file not found at {model_path}")
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    model = YOLO(model_path)
    logger.info(f"Model loaded successfully: {model}")

    @app.get("/health")
    async def health_check():
        return {"status": "healthy", "message": "Model loaded successfully"}

    @app.post("/inference")
    async def inference(file: UploadFile = File(...)):
        try:
            contents = await file.read()
            img = Image.open(io.BytesIO(contents))
        except Exception as e:
            logger.error(f"Invalid image file: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

        try:
            results = model(img, conf=0.7, iou=0.7, save=False, imgsz=640)
            
            processed_results = []
            for result in results:
                boxes = result.boxes.xyxy.tolist()
                boxes_int = [(int(box[0]), int(box[1]), int(box[2]), int(box[3])) for box in boxes]
                confidences = result.boxes.conf.tolist()
                classes = result.boxes.cls.tolist()
                names = [model.names[int(c)] for c in classes]

                
                processed_results.append({
                    "boxes": boxes_int,
                    "confidences": confidences,
                    "classes": classes,
                    "names": names
                })

            return {"result": processed_results}
        except Exception as e:
            logger.error(f"Error during inference: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error during inference: {str(e)}")

    return app

if __name__ == "__main__":
    app.serve()