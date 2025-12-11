import modal

MODEL = "rfdetr-medium"
VERSION = "v3"


app = modal.App(f"inference-ct-brain-hemorrhage-{MODEL}-{VERSION}")

volume = modal.Volume.from_name(f"{MODEL}-brain-ct-hemorrhage-{VERSION}", create_if_missing=True)

image = (
    modal.Image.debian_slim()
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "rfdetr==1.2.1",
        "fastapi",
        "pillow",
        "python-multipart",
        "torch"
    )
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
    from fastapi import FastAPI, File, UploadFile, HTTPException
    from PIL import Image
    import io
    import os
    import logging

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    app = FastAPI()

    # Import RF-DETR
    from rfdetr import RFDETRMedium

    # Load the model with fine-tuned checkpoint
    model_path = "/root/model/model/rfdetr_medium_best_epoch72.pth"

    logger.info(f"Loading model from {model_path}")
    logger.info(f"Volume contents: {os.listdir('/root/model')}")

    # Initialize model with pretrained weights (fine-tuned checkpoint)
    model = RFDETRMedium(pretrain_weights=model_path)

    logger.info(f"RF-DETR model loaded successfully from {model_path}")

    # Class names mapping
    class_names = ['IPH', 'IVH', 'SAH', 'SDH']

    @app.get("/health")
    async def health_check():
        return {"status": "healthy", "message": "RF-DETR model loaded successfully", "model": "rfdetr-medium"}

    @app.post("/inference")
    async def inference(file: UploadFile = File(...)):
        try:
            contents = await file.read()
            img = Image.open(io.BytesIO(contents))
        except Exception as e:
            logger.error(f"Invalid image file: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

        try:
            # Run RF-DETR inference
            # API: model.predict(image) - returns supervision.Detections object
            detections = model.predict(img)

            logger.info(f"Prediction result type: {type(detections)}")
            logger.info(f"Number of detections: {len(detections)}")

            # Convert supervision.Detections to YOLO format
            boxes = []
            confidences = []
            classes = []
            names = []

            # supervision.Detections has direct attributes: xyxy, confidence, class_id
            for i in range(len(detections)):
                bbox = detections.xyxy[i]  # numpy array [x1, y1, x2, y2]
                conf = float(detections.confidence[i])
                cls = int(detections.class_id[i])

                # Filter by confidence threshold
                if conf < 0.01:
                    continue

                # Convert to integers
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                boxes.append((x1, y1, x2, y2))
                confidences.append(conf)
                classes.append(cls)
                names.append(class_names[cls])

            logger.info(f"Filtered detections: {len(boxes)} (threshold: 0.01)")

            # Match YOLO response format exactly
            processed_results = [{
                "boxes": boxes,
                "confidences": confidences,
                "classes": classes,
                "names": names
            }]

            return {"result": processed_results}
        except Exception as e:
            import traceback
            logger.error(f"Error during inference: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"Error during inference: {str(e)}")

    return app

if __name__ == "__main__":
    app.serve()
