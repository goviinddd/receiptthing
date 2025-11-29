import json
import os
import shutil
import yaml
from pathlib import Path
from sklearn.model_selection import train_test_split

# --- CONFIG (Updated with your paths) ---
# We use .resolve() to ensure paths work regardless of where you run the script from
PROJECT_ROOT = Path(__file__).parent.parent.resolve() 
JSON_PATH = PROJECT_ROOT / "dataset" / "result.json" 
IMAGES_DIR = PROJECT_ROOT / "dataset" / "images"      
OUTPUT_DIR = PROJECT_ROOT / "dataset" / "yolo_dataset"

def convert_coco_to_yolo():
    # 1. Load COCO JSON
    if not JSON_PATH.exists():
        print(f"ERROR: Could not find JSON at {JSON_PATH}")
        return

    with open(JSON_PATH, 'r') as f:
        data = json.load(f)

    # 2. Map Categories
    categories = {cat['id']: cat['name'] for cat in data['categories']}
    cat_to_id = {cat_id: i for i, cat_id in enumerate(categories.keys())}
    class_names = [categories[id] for id in categories]
    print(f"Classes found: {class_names}")

    # 3. Create Directory Structure
    for split in ['train', 'val']:
        (OUTPUT_DIR / 'images' / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / 'labels' / split).mkdir(parents=True, exist_ok=True)

    # 4. Group Annotations
    image_map = {img['id']: img for img in data['images']}
    ann_map = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in ann_map: ann_map[img_id] = []
        ann_map[img_id].append(ann)

    # 5. Process Images
    image_ids = list(image_map.keys())
    train_ids, val_ids = train_test_split(image_ids, test_size=0.2, random_state=42)

    found_count = 0
    missing_count = 0

    for img_id in image_ids:
        img_info = image_map[img_id]
        
        # --- ROBUST FILE FINDING (The Fix) ---
        # Label Studio name in JSON: "54ca7a7e-1_p1.jpg"
        # Real File on Disk:         "1_p1.jpg"
        
        ls_filename = Path(img_info['file_name']).name 
        
        # 1. Try exact match
        found_files = list(IMAGES_DIR.glob(ls_filename))
        
        # 2. Try removing the hash prefix (everything before first '-')
        if not found_files and '-' in ls_filename:
            clean_name = ls_filename.split('-', 1)[1] 
            found_files = list(IMAGES_DIR.glob(clean_name))
        
        # 3. Try matching just the end of the filename (fallback)
        if not found_files:
             clean_name = ls_filename.split('-', 1)[-1]
             found_files = list(IMAGES_DIR.glob(f"*{clean_name}"))

        if not found_files:
            print(f"Warning: Could not find image {ls_filename} in {IMAGES_DIR}")
            missing_count += 1
            continue
            
        src_path = found_files[0]
        found_count += 1
        
        # Determine Split
        split = 'train' if img_id in train_ids else 'val'
        
        # Copy Image (Save as clean name)
        dst_img_path = OUTPUT_DIR / 'images' / split / src_path.name
        shutil.copy(src_path, dst_img_path)

        # Create Label File
        label_path = OUTPUT_DIR / 'labels' / split / f"{src_path.stem}.txt"
        
        with open(label_path, 'w') as lf:
            if img_id in ann_map:
                for ann in ann_map[img_id]:
                    # Convert to YOLO format
                    bbox = ann['bbox']
                    img_w, img_h = img_info['width'], img_info['height']
                    
                    x_center = (bbox[0] + bbox[2]/2) / img_w
                    y_center = (bbox[1] + bbox[3]/2) / img_h
                    width = bbox[2] / img_w
                    height = bbox[3] / img_h
                    
                    cls_idx = cat_to_id[ann['category_id']]
                    lf.write(f"{cls_idx} {x_center} {y_center} {width} {height}\n")

    # 6. Create YAML
    yaml_content = {
        'path': str(OUTPUT_DIR.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'names': {i: name for i, name in enumerate(class_names)}
    }
    
    with open(OUTPUT_DIR / "dataset.yaml", 'w') as yf:
        yaml.dump(yaml_content, yf)

    print(f"Done! Found {found_count} images, Missing {missing_count}.")
    print(f"Data ready at {OUTPUT_DIR}")

if __name__ == "__main__":
    convert_coco_to_yolo()