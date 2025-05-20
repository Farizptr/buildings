import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ultralytics import YOLO

def load_model(model_path="../models/best.pt"):
    """
    Load YOLOv8 model from the specified path
    
    Args:
        model_path: Path to the YOLOv8 model file (.pt)
        
    Returns:
        Loaded YOLOv8 model
    """
    # Check if model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found")
    
    # Load YOLOv8 model
    try:
        model = YOLO(model_path)
        print(f"YOLOv8 model loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def detect_buildings(model, image_path, conf=0.25):
    """
    Detect buildings in an image using the loaded YOLOv8 model
    
    Args:
        model: Loaded YOLOv8 model
        image_path: Path to the image file
        conf: Confidence threshold
        
    Returns:
        results: Model detection results
        img: Original image as numpy array
    """
    # Check if image file exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file {image_path} not found")
    
    # Load image
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img)
    
    # Run detection
    results = model.predict(image_path, conf=conf)
    
    return results[0], img_array

