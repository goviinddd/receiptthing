from ultralytics import YOLO
from pathlib import Path
import os

# --- CONFIGURATION ---
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# 1. Input Source: Use the validation images for a quick check
INPUT_FOLDER = PROJECT_ROOT / "data" / "yolo_dataset" / "images" / "train"

# 2. Model: Your trained detector
MODEL_PATH = PROJECT_ROOT / "models" / "detector" / "receipt_detector_v1" / "weights" / "best.pt"

# 3. Output Destination: Folder where plotted images will be saved
OUTPUT_BASE_DIR = PROJECT_ROOT / "data" / "visual_output"
OUTPUT_SUBDIR = "yolo_predictions" # The results will be saved inside a folder named 'yolo_predictions'

def visualize_batch_results():
    """Runs the YOLO model on a batch of images and saves the results."""
    
    if not MODEL_PATH.exists():
        print(f"Error: Model not found at {MODEL_PATH}")
        return
        
    if not INPUT_FOLDER.exists():
        print(f"Error: Input folder not found at {INPUT_FOLDER}")
        print("Please check the INPUT_FOLDER variable.")
        return

    # Load the trained model
    print(f"Loading model from: {MODEL_PATH.name}")
    model = YOLO(MODEL_PATH)
    
    print(f"Starting batch visualization on images in: {INPUT_FOLDER.name}")
    
    # The predict method handles the batch processing, plotting, and saving automatically.
    results = model.predict(
        source=str(INPUT_FOLDER), 
        conf=0.25,        # Minimum confidence to draw a box
        save=True,          # Save the results to disk
        project=str(OUTPUT_BASE_DIR), # Base directory for results
        name=OUTPUT_SUBDIR, # Subdirectory name (e.g., visual_output/yolo_predictions)
        imgsz=1280,         # Use the high resolution you trained with
        exist_ok=True
    )
    
    final_output_path = OUTPUT_BASE_DIR / OUTPUT_SUBDIR
    print("\n" + "=" * 60)
    print("✅ Batch Visualization Complete!")
    print(f"Results saved to: {final_output_path}")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    visualize_batch_results()