# app/face_utils.py
import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
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
    Accepts either PIL Image or numpy array.
    
    Args:
        image: PIL Image or numpy array
        
    Returns:
        Preprocessed tensor ready for model input
    """
    if image is None:
        logger.error("Cannot preprocess None image")
        return None
    
    try:
        # Convert numpy array to PIL Image if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Ensure it's a PIL Image
        if not isinstance(image, Image.Image):
            logger.error(f"Expected PIL Image or numpy array, got {type(image)}")
            return None
            
        return transform(image).unsqueeze(0)  # Add batch dimension
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

# Additional utility functions for Raspberry Pi optimization
def optimize_model_for_rpi(model):
    """
    Optimize model for Raspberry Pi deployment.
    
    Args:
        model: PyTorch model to optimize
        
    Returns:
        Optimized model
    """
    try:
        # Use half precision for faster inference and less memory
        if torch.cuda.is_available():
            model.half()  # Convert to half precision
        else:
            # On CPU, use more efficient settings
            model.float()
            torch.set_num_threads(2)  # Limit threads for Raspberry Pi
        
        # Use JIT compilation if possible
        try:
            model = torch.jit.script(model)
            logger.info("Model compiled with TorchScript")
        except Exception as e:
            logger.warning(f"TorchScript compilation failed: {e}")
        
        return model
    except Exception as e:
        logger.error(f"Error optimizing model: {e}")
        return model

def resize_image(image, max_size=640):
    """
    Resize image while maintaining aspect ratio to reduce processing time.
    
    Args:
        image: Input image
        max_size: Maximum dimension size
        
    Returns:
        Resized image
    """
    if image is None:
        return None
    
    h, w = image.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    return image