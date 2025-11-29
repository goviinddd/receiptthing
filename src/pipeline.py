import cv2
import numpy as np
import torch
from pathlib import Path
from PIL import Image, ImageDraw

# Import our new modules
from src.modules.detector import TableDetector
from src.modules.ocr import TextReader
from src.modules.extractor import TableParser

class ReceiptPipeline:
    def __init__(self):
        self.ROOT = Path(__file__).parent.parent.resolve()
        
        # Paths
        yolo_path = self.ROOT / "models" / "detector" / "receipt_detector_v1" / "weights" / "best.pt"
        donut_path = self.ROOT / "models" / "extractor"
        
        # Device Check
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize Modules
        self.detector = TableDetector(yolo_path, device)
        self.reader = TextReader()
        self.extractor = TableParser(donut_path, device)
        
        print(">>> PIPELINE READY <<<")

    def process(self, image_path):
        # 1. Load Image
        pil_image = Image.open(image_path).convert("RGB")
        cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        # 2. Run Detection
        detections = self.detector.detect(pil_image)
        
        debug_image = pil_image.copy()
        draw = ImageDraw.Draw(debug_image)
        
        # 3. Processing Variables
        po_candidates = []
        table_boxes = []
        
        # 4. Route Detections to Correct Modules
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            
            # --- PO NUMBER (Class 0) -> OCR MODULE ---
            if det['class_id'] == 0:
                draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
                
                # Crop for Paddle
                po_crop = cv_image[y1:y2, x1:x2]
                text = self.reader.read_region(po_crop)
                score = self.reader.validate_po(text)
                
                if score > 0:
                    po_candidates.append({"text": text, "score": score, "bbox": [x1, y1, x2, y2]})
                    draw.text((x1, y1-15), text, fill="red")

            # --- TABLE (Class 1) -> COLLECT FOR MERGING ---
            elif det['class_id'] == 1:
                table_boxes.append([x1, y1, x2, y2])
        
        # 5. Select Best PO
        final_po = "Not Detected"
        if po_candidates:
            best = sorted(po_candidates, key=lambda x: x['score'], reverse=True)[0]
            final_po = best['text']
            draw.rectangle(best['bbox'], outline="green", width=5)

        # 6. Merge Table Boxes & Extract
        final_json = {}
        if table_boxes:
            # Union Logic
            ux1 = min([b[0] for b in table_boxes])
            uy1 = min([b[1] for b in table_boxes])
            ux2 = max([b[2] for b in table_boxes])
            uy2 = max([b[3] for b in table_boxes])
            
            # Padding
            pad_x, pad_y = 20, 10
            crop_box = (
                max(0, ux1 - pad_x), max(0, uy1 - pad_y),
                min(pil_image.width, ux2 + pad_x), min(pil_image.height, uy2 + pad_y)
            )
            
            draw.rectangle(crop_box, outline="green", width=5)
            table_crop = pil_image.crop(crop_box)
            
            # --- EXTRACTOR MODULE ---
            final_json = self.extractor.extract_table(table_crop)
        else:
            final_json = {"error": "No table detected"}

        # 7. Final Package
        final_json['po_number'] = final_po
        final_json['debug_image'] = debug_image
        
        return final_json