from facenet_pytorch import MTCNN
from PIL import Image
import torch
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(keep_all=True, device=device)

def extract_faces(image, return_multiple=False):
    """
    Détecte les visages dans une image (PIL ou np.ndarray) et retourne les visages recadrés.
    """
    # Si image est un tableau NumPy, convertis-le en image PIL
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    boxes, _ = mtcnn.detect(image)

    if boxes is None:
        return []

    faces = []
    for box in boxes:
        x1, y1, x2, y2 = [int(coord) for coord in box]
        face = image.crop((x1, y1, x2, y2))
        faces.append(face)

    return faces if return_multiple else faces[:1]




    ###########################################################""
# import cv2

# # Utilisation du détecteur Haar Cascade de visage frontal de OpenCV
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# def extract_faces_cv2(frame, target_size=(160, 160)):
#     """
#     Détecte les visages sur une image BGR (OpenCV) et retourne les visages rognés + les coordonnées.

#     Args:
#         frame: image BGR provenant de la webcam
#         target_size: taille à laquelle redimensionner les visages

#     Returns:
#         faces: liste de visages (images RGB redimensionnées)
#         boxes: liste de bounding boxes (x, y, w, h)
#     """
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     detections = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

#     faces = []
#     boxes = []

#     for (x, y, w, h) in detections:
#         face = frame[y:y+h, x:x+w]
#         face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
#         face_resized = cv2.resize(face_rgb, target_size)
#         faces.append(face_resized)
#         boxes.append((x, y, w, h))

#     return faces, boxes

