import cv2
import numpy as np
import torch

# PyTorch 2.6+ defaults to weights_only=True, which breaks ultralytics==8.2.0.
# We monkey-patch torch.load to force weights_only=False for YOLO loading.
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from ultralytics import YOLO
from utils import format_bbox_label

class SmartShelfDetector:
    def __init__(self, model_path="yolov8n.pt"):
        """Initialize the YOLOv8 model."""
        try:
            self.model = YOLO(model_path)
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None

    def preprocess_image(self, image):
        """
        Apply Gaussian blur and Contrast normalization (CLAHE).
        Assumes image is an RGB numpy array.
        """
        try:
            # Convert to LAB color space for CLAHE on L-channel
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l_channel, a, b = cv2.split(lab)

            # Apply CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l_channel)

            # Merge back
            merged = cv2.merge((cl, a, b))
            enhanced_image = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)

            # Apply Gaussian Blur
            blurred_image = cv2.GaussianBlur(enhanced_image, (3, 3), 0)
            return blurred_image
        except Exception as e:
            print(f"Error during preprocessing: {e}")
            return image

    def detect(self, image, conf_threshold=0.45):
        """
        Accept an image (RGB numpy array).
        Run inference with confidence threshold.
        Return annotated image, list of detections.
        """
        if self.model is None:
            return image, []

        try:
            # Preprocess
            processed_image = self.preprocess_image(image)
            
            # YOLO expects BGR numpy arrays (OpenCV default format)
            # Since our image is RGB, we must convert it to BGR for inference
            inference_image = cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)
            
            # Inference
            results = self.model(inference_image, conf=conf_threshold, verbose=False)
            
            detections = []
            annotated_image = processed_image.copy()
            
            result = results[0]
            boxes = result.boxes
            
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    cls_id = int(box.cls[0].cpu().numpy())
                    cls_name = result.names[cls_id]
                    
                    detections.append({
                        "class": cls_name,
                        "confidence": conf,
                        "bbox": [x1, y1, x2, y2]
                    })
                    
                    # Draw green bounding box (in stock)
                    color = (0, 255, 0) # Green in RGB
                    cv2.rectangle(annotated_image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    
                    # Draw label
                    label = format_bbox_label(cls_name, conf)
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(annotated_image, (int(x1), int(y1) - th - 5), (int(x1) + tw, int(y1)), color, -1)
                    cv2.putText(annotated_image, label, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                
            return annotated_image, detections

        except Exception as e:
            print(f"Error during detection: {e}")
            return image, []
