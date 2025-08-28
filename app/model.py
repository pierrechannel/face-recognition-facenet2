# app/model.py
import torch
from facenet_pytorch import InceptionResnetV1

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_facenet_model():
    model = InceptionResnetV1(pretrained='vggface2', classify=False)
    model.load_state_dict(torch.load('facenet_africain_finetuned.pth', map_location=device))
    model.eval().to(device)
    return model
