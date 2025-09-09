# app/face_utils.py
import torch
import cv2
import numpy as np
from torchvision import transforms


transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])


def read_image_from_bytes(file_bytes):
    np_arr = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def preprocess_face(image):
    return transform(image).unsqueeze(0)  # batch of 1

def get_embedding(model, image_tensor, device='cpu'):
    model = model.to(device)
    model.eval()
    
    print(f"Input tensor shape: {image_tensor.shape}")
    print(f"Input tensor range: [{image_tensor.min():.3f}, {image_tensor.max():.3f}]")
    
    with torch.no_grad():
        output = model(image_tensor.to(device))
        print(f"Model output type: {type(output)}")
        
        # Handle different output formats
        if isinstance(output, tuple):
            print(f"Model output is tuple, length: {len(output)}")
            output = output[0]  # Take first element
        
        print(f"Final output shape: {output.shape}")
        print(f"Final output range: [{output.min():.6f}, {output.max():.6f}]")
        print(f"Contains NaN: {torch.any(torch.isnan(output))}")
        
        embedding = output.cpu().numpy()[0]
        print(f"Final embedding shape: {embedding.shape}")
        return embedding