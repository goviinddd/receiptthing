from paddleocr import PaddleOCR
import numpy as np
import cv2
import re
import logging

# Suppress Paddle logs
logging.getLogger("ppocr").setLevel(logging.ERROR)

class TextReader:
    def __init__(self, lang='en'):
        print("[OCR] Loading PaddleOCR...")
        # Initialize once to save memory
        self.reader = PaddleOCR(use_angle_cls=True, lang=lang)

    def read_region(self, image_crop: np.ndarray):
        """
        Input: OpenCV Image (BGR)
        Output: Cleaned Text String
        """
        result = self.reader.predict(image_crop)
        if not result or not result[0]:
            return ""
        
        # Merge all text lines found in the crop
        text = " ".join([line[1][0] for line in result[0]])
        return text.strip()

    def validate_po(self, text):
        """
        Heuristic filter to check if text looks like a PO Number
        """
        text_upper = text.upper()
        banned = ["INVOICE", "TAX", "DATE", "PAGE", "NO:", "SHIPMENT"]
        
        if any(x in text_upper for x in banned): return 0
        if re.search(r'[A-Z0-9-]{5,}', text_upper): return 100
        if re.search(r'\d{4,}', text): return 50
        return 0