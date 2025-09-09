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
    with torch.no_grad():
        return model(image_tensor.to(device)).cpu().numpy()[0]