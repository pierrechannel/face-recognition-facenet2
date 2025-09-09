# app/face_utils.py
import torch
import cv2
import numpy as np
from torchvision import transforms
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optimized transform for Raspberry Pi
transform = transforms.Compose([
    transforms.Resize((160, 160)),  # Standard size for face recognition models
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def read_image_from_bytes(file_bytes):
    """
    Read and decode image from bytes with error handling and optimization.
    
    Args:
        file_bytes: Image data in bytes format
        
    Returns:
        RGB image as numpy array or None if decoding fails
    """
    try:
        np_arr = np.frombuffer(file_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image is None:
            logger.error("Failed to decode image from bytes")
            return None
        
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except Exception as e:
        logger.error(f"Error reading image from bytes: {e}")
        return None

def preprocess_face(image):
    """
    Preprocess face image for model input.
    
    Args:
        image: RGB image as numpy array
        
    Returns:
        Preprocessed tensor ready for model input
    """
    if image is None:
        logger.error("Cannot preprocess None image")
        return None
    
    try:
        # Convert numpy array to PIL Image for transformation
        from PIL import Image
        pil_image = Image.fromarray(image)
        return transform(pil_image).unsqueeze(0)  # Add batch dimension
    except Exception as e:
        logger.error(f"Error preprocessing face: {e}")
        return None

def get_embedding(model, image_tensor, device='cpu'):
    """
    Generate embedding from face image using the provided model.
    
    Args:
        model: Pre-trained model for feature extraction
        image_tensor: Preprocessed image tensor
        device: Device to run inference on ('cpu' or 'cuda')
        
    Returns:
        Feature embedding as numpy array or None if error occurs
    """
    if image_tensor is None:
        logger.error("Cannot get embedding from None tensor")
        return None
    
    try:
        # Set model to evaluation mode and move to device
        model.eval()
        model.to(device)
        
        with torch.no_grad():
            # Use torch.inference_mode() for better performance if available
            if hasattr(torch, 'inference_mode'):
                with torch.inference_mode():
                    embedding = model(image_tensor.to(device)).cpu().numpy()[0]
            else:
                embedding = model(image_tensor.to(device)).cpu().numpy()[0]
        
        return embedding
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        return None

