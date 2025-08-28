import os
import torch
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox
from app.model import load_facenet_model
from app.face_utils import read_image_from_bytes, preprocess_face, get_embedding
from app.face_detect import extract_faces
from app.db import save_embedding

MODEL_PATH = "facenet_africain_finetuned.pth"
EMBEDDINGS_DIR = "embeddings"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_model(path):
    model = load_facenet_model()
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

def register_face(name: str, image_paths: list):
    """
    Register multiple face images for a person and save their embeddings.
    
    Args:
        name (str): Name of the person
        image_paths (list): List of image file paths
    """
    model = load_model(MODEL_PATH)
    all_embeddings = []
    successful_images = 0
    
    for i, image_path in enumerate(image_paths):
        try:
            with open(image_path, "rb") as f:
                image_bytes = f.read()

            image = read_image_from_bytes(image_bytes)
            faces = extract_faces(image)

            if not faces:
                print(f"Aucun visage détecté dans l'image {i+1}: {os.path.basename(image_path)}")
                continue

            face_tensor = preprocess_face(faces[0])
            embedding = get_embedding(model, face_tensor, DEVICE)
            
            # Convert embedding to numpy array if it's a tensor
            if isinstance(embedding, torch.Tensor):
                embedding = embedding.cpu().detach().numpy()
                
            all_embeddings.append(embedding)
            successful_images += 1
            
            print(f"Embedding extrait pour {name} depuis l'image {i+1}: {os.path.basename(image_path)}")
            
        except Exception as e:
            print(f"Erreur lors du traitement de l'image {image_path}: {str(e)}")
    
    if not all_embeddings:
        print("Aucun embedding n'a pu être extrait des images fournies.")
        return 0
    
    # Calculate average embedding using numpy
    avg_embedding = np.mean(np.stack(all_embeddings), axis=0)
    
    # Convert back to tensor if needed by save_embedding function
    if hasattr(torch, 'from_numpy'):
        avg_embedding_tensor = torch.from_numpy(avg_embedding)
    else:
        avg_embedding_tensor = torch.tensor(avg_embedding)
    
    # Save the embedding
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
    save_embedding(name, avg_embedding_tensor)
    
    print(f"{successful_images} embeddings combinés enregistrés pour {name} dans '{EMBEDDINGS_DIR}/{name}.pt'.")
    return successful_images

def select_images():
    """Open a file dialog to select multiple images"""
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    
    file_paths = filedialog.askopenfilenames(
        title="Sélectionnez les images de la personne",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
    )
    
    return list(file_paths)

def display_selected_images(image_paths):
    """Create a simple preview of selected images"""
    if not image_paths:
        return
    
    # Create a preview window
    preview = tk.Toplevel()
    preview.title("Aperçu des images sélectionnées")
    
    # Create a scrollable frame
    canvas = tk.Canvas(preview)
    scrollbar = tk.Scrollbar(preview, orient="horizontal", command=canvas.xview)
    scrollable_frame = tk.Frame(canvas)
    
    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )
    
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(xscrollcommand=scrollbar.set)
    
    # Load and display thumbnails
    for i, path in enumerate(image_paths):
        try:
            img = Image.open(path)
            img.thumbnail((100, 100))
            photo = ImageTk.PhotoImage(img)
            
            label = tk.Label(scrollable_frame, image=photo, text=os.path.basename(path), compound="top")
            label.image = photo  # Keep a reference
            label.grid(row=0, column=i, padx=5, pady=5)
        except Exception as e:
            print(f"Impossible de charger l'aperçu pour {path}: {str(e)}")
    
    canvas.pack(side="top", fill="both", expand=True)
    scrollbar.pack(side="bottom", fill="x")
    
    # Add a close button
    close_btn = tk.Button(preview, text="Fermer", command=preview.destroy)
    close_btn.pack(pady=5)

if __name__ == "__main__":
    # Create a simple GUI
    root = tk.Tk()
    root.title("Enregistrement de visages")
    root.geometry("500x300")
    
    # Use a class to manage state
    class AppState:
        def __init__(self):
            self.selected_images = []
    
    state = AppState()
    
    # Name entry
    tk.Label(root, text="Nom de la personne:").pack(pady=10)
    name_entry = tk.Entry(root, width=40)
    name_entry.pack(pady=5)
    
    # Image selection
    def on_select_images():
        state.selected_images = select_images()
        if state.selected_images:
            status_label.config(text=f"{len(state.selected_images)} image(s) sélectionnée(s)")
            # Show preview button only if images are selected
            preview_btn.pack(pady=5)
        else:
            status_label.config(text="Aucune image sélectionnée")
            preview_btn.pack_forget()
    
    def on_show_preview():
        display_selected_images(state.selected_images)
    
    def on_register():
        name = name_entry.get().strip()
        if not name:
            messagebox.showerror("Erreur", "Veuillez entrer un nom.")
            return
        
        if not state.selected_images:
            messagebox.showerror("Erreur", "Veuillez sélectionner au moins une image.")
            return
        
        # Register the face
        success_count = register_face(name, state.selected_images)
        
        if success_count > 0:
            messagebox.showinfo("Succès", f"{success_count} image(s) enregistrée(s) pour {name}.")
            # Reset the form
            name_entry.delete(0, tk.END)
            state.selected_images = []
            status_label.config(text="Aucune image sélectionnée")
            preview_btn.pack_forget()
        else:
            messagebox.showerror("Erreur", "Aucun visage n'a pu être détecté dans les images sélectionnées.")
    
    select_btn = tk.Button(root, text="Sélectionner des images", command=on_select_images)
    select_btn.pack(pady=10)
    
    status_label = tk.Label(root, text="Aucune image sélectionnée")
    status_label.pack(pady=5)
    
    preview_btn = tk.Button(root, text="Aperçu des images", command=on_show_preview)
    
    register_btn = tk.Button(root, text="Enregistrer", command=on_register, bg="green", fg="white")
    register_btn.pack(pady=20)
    
    root.mainloop()