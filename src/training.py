from ultralytics import YOLO
from pathlib import Path
import os


PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# Path to the dataset.yaml you just created
DATA_YAML = PROJECT_ROOT / "data" / "yolo_dataset" / "dataset.yaml"

# Where to save the trained model
MODEL_OUTPUT_DIR = PROJECT_ROOT / "models" / "detector"

def train():
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Loading Config: {DATA_YAML}")
    
    if not DATA_YAML.exists():
        print(f"ERROR: Could not find dataset.yaml at {DATA_YAML}")
        return

    # 1. Initialize the Model
    # We use 'yolov8n.pt' (Nano) because it's fast and your tables are large objects.
    # It will download automatically on the first run.
    model = YOLO("yolov8n.pt") 

    # 2. Start Training
    print("Starting Training...")
    # epochs=50 is standard for small datasets (30 images)
    # imgsz=640 is the standard YOLO resolution
    results = model.train(
        data=str(DATA_YAML),
        epochs=50,
        imgsz=1280,
        batch=4,           # Keep low for safety, increase if you have a GPU
        project=str(MODEL_OUTPUT_DIR),
        name="receipt_detector_v1",
        exist_ok=True      # Overwrite folder if it exists
    )
    
    # 3. Validation results
    print("Training Complete. Validating...")
    metrics = model.val()
    print(f"Final mAP50: {metrics.box.map50}")
    print(f"Model saved to: {MODEL_OUTPUT_DIR / 'receipt_detector_v1' / 'weights' / 'best.pt'}")

if __name__ == "__main__":
    train()