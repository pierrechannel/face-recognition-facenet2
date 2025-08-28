# app/db.py
import os
import torch

DB_DIR = "embeddings"

def save_embedding(name, embedding):
    os.makedirs(DB_DIR, exist_ok=True)
    path = os.path.join(DB_DIR, f"{name}.pt")
    torch.save(torch.tensor(embedding), path)

def load_embedding(name):
    path = os.path.join(DB_DIR, f"{name}.pt")
    if not os.path.exists(path):
        return None
    return torch.load(path)
