import modal
import os
import wandb
from ultralytics import YOLO

# Create a Modal volume to store the model
volume = modal.Volume.from_name("yolo-GRE", create_if_missing=True)

# Define the Modal image with necessary dependencies
image = (
    modal.Image.debian_slim()
    .apt_install(
        "libgl1-mesa-glx",
        "libglib2.0-0",
        "libsm6",
        "libxext6",
        "libxrender-dev",
        "libgstreamer1.0-0",
        "gstreamer1.0-plugins-base",
        "gstreamer1.0-plugins-good",
    )
    .pip_install("ultralytics", "wandb", "opencv-python-headless", "pandas")
)

# Define the local data directory and mount it in the Modal container
local_data_dir = "/Users/administrator/Documents/modal_training_vgg16/yolov8_GRE"
data_mount = modal.Mount.from_local_dir(local_data_dir, remote_path="/data")

app = modal.App("yolo-GRE")

@app.function(
    image=image,
    mounts=[data_mount],
    gpu="A10G",
    secrets=[modal.Secret.from_name("wandb-secret")],
    volumes={"/model": volume},
    timeout=3600*24
)
def train_yolo():
    wandb.login(key=os.environ["WANDB_API_KEY"])
    
    # Initialize wandb run
    run = wandb.init(project="brain-GRE", name="yolov8_GRE", config={
        "model": "yolov8s",
        "epochs":50,
        "batch_size": 16,
        "image_size": 512
    })
    
    model = YOLO("yolov8s.pt")
    
    # Use wandb.watch to log model gradients and parameters
    wandb.watch(model, log="all", log_freq=10)
    
    # Define a callback to log metrics
    def on_train_epoch_end(trainer):
        metrics = trainer.metrics
        log_dict = {
            "epoch": trainer.epoch,
        }
        
        # Map of expected metric names to possible alternatives
        metric_map = {
            "train/box_loss": ["train/box_loss", "train/loss"],
            "train/cls_loss": ["train/cls_loss", "train/loss_cls"],
            "train/dfl_loss": ["train/dfl_loss", "train/loss_dfl"],
            "metrics/precision": ["metrics/precision(B)", "metrics/precision"],
            "metrics/recall": ["metrics/recall(B)", "metrics/recall"],
            "metrics/mAP50": ["metrics/mAP50(B)", "metrics/mAP50"],
            "metrics/mAP50-95": ["metrics/mAP50-95(B)", "metrics/mAP50-95"],
            "val/box_loss": ["val/box_loss", "val/loss"],
            "val/cls_loss": ["val/cls_loss", "val/loss_cls"],
            "val/dfl_loss": ["val/dfl_loss", "val/loss_dfl"],
        }
        
        for log_name, possible_names in metric_map.items():
            for metric_name in possible_names:
                if metric_name in metrics:
                    log_dict[log_name] = metrics[metric_name]
                    break
        
        wandb.log(log_dict)
    
    # Add the callback to the model
    model.add_callback("on_train_epoch_end", on_train_epoch_end)
    
    # Train the model
    results = model.train(
        data="/data/data.yaml",
        epochs=50,
        imgsz=512,
        batch=16,
        name="yolov8s_GRE",
        project="brain_GRE",
        save_period=10,
        plots=True,
        device=0
        #patience=7
    )
    
    # Save the best model to the volume
    model.save('/model/best.pt')
    
    # Log the final model as an artifact
    artifact = wandb.Artifact('best_model', type='model')
    artifact.add_file('/model/best.pt')
    run.log_artifact(artifact)
    
    # Close the wandb run
    run.finish()

@app.function(
    image=image,
    mounts=[data_mount],
    gpu="A10G",
    secrets=[modal.Secret.from_name("wandb-secret")],
    volumes={"/model": volume}
)
def validate_and_test_yolo():
    wandb.login(key=os.environ["WANDB_API_KEY"])
    run = wandb.init(project="brain-GRE", name="yolov8s_GRE_val")
    
    model = YOLO("/model/best.pt")  # Load the model from the volume
    
    # Validation
    validation_results = model.val(
        data="/data/data.yaml",
        imgsz=512,
        batch=8,
        conf=0.7,
        iou=0.6,
    )
    
    wandb.log({
        "val/mAP50-95": validation_results.box.map,
        "val/mAP50": validation_results.box.map50,
        "val/mAP75": validation_results.box.map75,
        "val/Inference_Speed": validation_results.speed['inference'],
    })
    
    # Test
    test_results = model.val(
        data="/data/data.yaml",
        split='test',
        imgsz=512,
        batch=4,
        conf=0.7,
        iou=0.6
    )
    
    wandb.log({
        "test/mAP50-95": test_results.box.map,
        "test/mAP50": test_results.box.map50,
        "test/mAP75": test_results.box.map75,
    })
    
    print (f"test_results.box.map: {test_results.box.map}")
    print (f"test_results.box.map50: {test_results.box.map50}")
    print (f"test_results.box.map75: {test_results.box.map75}")
    run.finish()

@app.function(
    image=image,
    mounts=[data_mount],
    gpu="A10G",
    secrets=[modal.Secret.from_name("wandb-secret")],
    volumes={"/model": volume}
)
def predict_and_visualize():
    import cv2
    
    wandb.login(key=os.environ["WANDB_API_KEY"])
    run = wandb.init(project="brain-GRE", name="yolov8s_GRE_pred")
    
    model = YOLO("/model/best.pt")  # Load the model from the volume
    
    test_results = model.predict(
        source="/data/test/images",
        imgsz=512,
        conf=0.5,
        iou=0.6,
        save=False
    )
    
    for i, result in enumerate(test_results):
        img = result.plot()
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        wandb.log({f"Prediction {i+1}": wandb.Image(img_rgb)})
    
    run.finish()

@app.local_entrypoint()
def main():
    #train_yolo.remote()
    validate_and_test_yolo.remote()
    predict_and_visualize.remote()

if __name__ == "__main__":
    modal.run(main)
