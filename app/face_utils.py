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
    print(f"read_image_from_bytes: Processing {len(file_bytes)} bytes")
    try:
        np_arr = np.frombuffer(file_bytes, np.uint8)
        print(f"read_image_from_bytes: Created numpy array of shape {np_arr.shape}")
        
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image is None:
            logger.error("Failed to decode image from bytes")
            print("ERROR: Failed to decode image from bytes")
            return None
        
        print(f"read_image_from_bytes: Successfully decoded image with shape {image.shape}")
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print(f"read_image_from_bytes: Converted BGR to RGB")
        return rgb_image
        
    except Exception as e:
        logger.error(f"Error reading image from bytes: {e}")
        print(f"EXCEPTION in read_image_from_bytes: {e}")
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
    print(f"preprocess_face: Input type = {type(image)}")
    
    if image is None:
        logger.error("Cannot preprocess None image")
        print("ERROR: Cannot preprocess None image")
        return None
    
    try:
        # Convert numpy array to PIL Image if needed
        if isinstance(image, np.ndarray):
            print(f"preprocess_face: Converting numpy array to PIL Image, shape {image.shape}")
            image = Image.fromarray(image)
        
        # Ensure it's a PIL Image
        if not isinstance(image, Image.Image):
            logger.error(f"Expected PIL Image or numpy array, got {type(image)}")
            print(f"ERROR: Expected PIL Image or numpy array, got {type(image)}")
            return None
        
        print(f"preprocess_face: Applying transform to image size {image.size}")
        processed = transform(image).unsqueeze(0)  # Add batch dimension
        print(f"preprocess_face: Output tensor shape = {processed.shape}")
        return processed
        
    except Exception as e:
        logger.error(f"Error preprocessing face: {e}")
        print(f"EXCEPTION in preprocess_face: {e}")
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
    print(f"get_embedding: Input tensor shape = {image_tensor.shape if image_tensor is not None else 'None'}")
    print(f"get_embedding: Using device = {device}")
    
    if image_tensor is None:
        logger.error("Cannot get embedding from None tensor")
        print("ERROR: Cannot get embedding from None tensor")
        return None
    
    try:
        # Set model to evaluation mode and move to device
        model.eval()
        model.to(device)
        print(f"get_embedding: Model moved to {device}")
        
        with torch.no_grad():
            # Use torch.inference_mode() for better performance if available
            if hasattr(torch, 'inference_mode'):
                print("get_embedding: Using torch.inference_mode()")
                with torch.inference_mode():
                    embedding = model(image_tensor.to(device)).cpu().numpy()[0]
            else:
                print("get_embedding: Using torch.no_grad()")
                embedding = model(image_tensor.to(device)).cpu().numpy()[0]
        
        print(f"get_embedding: Generated embedding of shape {embedding.shape}")
        return embedding
        
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        print(f"EXCEPTION in get_embedding: {e}")
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
    print("optimize_model_for_rpi: Starting model optimization")
    try:
        # Use half precision for faster inference and less memory
        if torch.cuda.is_available():
            print("optimize_model_for_rpi: CUDA available, converting to half precision")
            model.half()  # Convert to half precision
        else:
            # On CPU, use more efficient settings
            print("optimize_model_for_rpi: Using CPU, setting float precision and limiting threads")
            model.float()
            torch.set_num_threads(2)  # Limit threads for Raspberry Pi
            print(f"optimize_model_for_rpi: Set number of threads to {torch.get_num_threads()}")
        
        # Use JIT compilation if possible
        try:
            print("optimize_model_for_rpi: Attempting TorchScript compilation")
            model = torch.jit.script(model)
            logger.info("Model compiled with TorchScript")
            print("SUCCESS: Model compiled with TorchScript")
        except Exception as e:
            logger.warning(f"TorchScript compilation failed: {e}")
            print(f"WARNING: TorchScript compilation failed: {e}")
        
        print("optimize_model_for_rpi: Optimization completed")
        return model
        
    except Exception as e:
        logger.error(f"Error optimizing model: {e}")
        print(f"EXCEPTION in optimize_model_for_rpi: {e}")
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
    print(f"resize_image: Input image shape = {image.shape if image is not None else 'None'}")
    
    if image is None:
        print("resize_image: Input image is None")
        return None
    
    h, w = image.shape[:2]
    print(f"resize_image: Original dimensions: {w}x{h}")
    
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        print(f"resize_image: Resizing to {new_w}x{new_h} (scale: {scale:.2f})")
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        print(f"resize_image: Resized image shape = {image.shape}")
    else:
        print("resize_image: No resizing needed - image within max size")
    
    return image