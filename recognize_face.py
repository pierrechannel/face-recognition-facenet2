import os
import torch
import numpy as np
from app.model import load_facenet_model
from app.face_utils import read_image_from_bytes, preprocess_face, get_embedding
from app.face_detect import extract_faces
from app.db import load_embedding

# Configuration
MODEL_PATH = "facenet_africain_finetuned.pth"
EMBEDDINGS_DIR = "embeddings"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
THRESHOLD = 0.8  # Seuil de reconnaissance (à ajuster)

# Chargement du modèle
def load_model(path):
    model = load_facenet_model()
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

# Calcul de distance euclidienne avec conversion sûre
def euclidean_distance(t1, t2):
    if isinstance(t1, np.ndarray):
        t1 = torch.tensor(t1)
    if isinstance(t2, np.ndarray):
        t2 = torch.tensor(t2)
    return torch.norm(t1 - t2).item()

# Chargement de toutes les embeddings sauvegardées
def get_all_saved_embeddings():
    embeddings = {}
    for file in os.listdir(EMBEDDINGS_DIR):
        if file.endswith(".pt"):
            name = file[:-3]  # Enlève l'extension .pt
            embedding = load_embedding(name)
            embeddings[name] = embedding
    return embeddings

# Reconnaissance de visage à partir d'une image
def recognize_face(model, image_path):
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    image = read_image_from_bytes(image_bytes)
    faces = extract_faces(image)

    if not faces:
        print("Aucun visage détecté dans l'image.")
        return

    face_tensor = preprocess_face(faces[0])
    embedding = get_embedding(model, face_tensor, DEVICE)

    saved_embeddings = get_all_saved_embeddings()

    if not saved_embeddings:
        print("Aucune embedding enregistrée dans le dossier.")
        return

    # Comparaison avec toutes les embeddings
    distances = {}
    for name, saved_emb in saved_embeddings.items():
        dist = euclidean_distance(embedding, saved_emb)
        distances[name] = dist

    # Trouver le match le plus proche
    best_match = min(distances, key=distances.get)
    best_distance = distances[best_match]

    if best_distance < THRESHOLD:
        print(f"Visage reconnu : {best_match} (distance = {best_distance:.3f})")
    else:
        print(f"Visage inconnu (distance minimale = {best_distance:.3f})")

# 📌 Point d'entrée
if __name__ == "__main__":
    IMAGE_TO_RECOGNIZE = r"C:\Users\HP\Pictures\tripleFace\tripleFace\WIN_20250819_10_04_29_Pro.jpg"  # 🔁 Mets ici le chemin de ton image à reconnaître
    model = load_model(MODEL_PATH)
    recognize_face(model, IMAGE_TO_RECOGNIZE)
