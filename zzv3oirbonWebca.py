import os
import cv2
import torch
import time
import winsound
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
from datetime import datetime
import threading
from collections import deque
import platform

from app.model import load_facenet_model
from app.face_utils import preprocess_face, get_embedding
from app.db import load_embedding

class FacialRecognitionUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🔒 Secure Access Control System")
        
        # Full screen setup
        self.root.attributes('-fullscreen', True)
        self.root.configure(bg='#0a0e14')
        
        # Bind ESC key to exit full screen
        self.root.bind('<Escape>', self.toggle_fullscreen)
        
        # Platform-specific configuration
        self.is_raspberry_pi = platform.machine() in ('armv7l', 'armv6l', 'aarch64')
        
        # Configuration
        self.MODEL_PATH = "facenet_africain_finetuned.pth"
        self.EMBEDDINGS_DIR = "embeddings"
        self.CAPTURE_DIR = "captures"
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.THRESHOLD = 0.7
        self.DETECTION_DELAY = 3
        
        # State variables
        self.cap = None
        self.model = None
        self.saved_embeddings = {}
        self.face_detector = None
        self.is_running = False
        self.face_detected = False
        self.detection_start_time = None
        self.detection_history = deque(maxlen=50)
        self.door_locked = True
        
        # Create directories
        os.makedirs(self.CAPTURE_DIR, exist_ok=True)
        
        self.setup_ui()
        self.load_resources()
        
    def toggle_fullscreen(self, event=None):
        self.root.attributes('-fullscreen', not self.root.attributes('-fullscreen'))
        
    def exit_app(self):
        self.stop_recognition()
        self.root.destroy()
        
    def setup_ui(self):
        # Main container with dark theme
        main_frame = tk.Frame(self.root, bg='#0a0e14')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Header with minimal information
        header_frame = tk.Frame(main_frame, bg='#0a0e14', height=80)
        header_frame.pack(fill=tk.X, pady=(0, 20))
        header_frame.pack_propagate(False)
        
        # Logo and title
        title_frame = tk.Frame(header_frame, bg='#0a0e14')
        title_frame.pack(side=tk.LEFT, padx=20)
        
        # Icon placeholder (you can replace with actual logo)
        icon_label = tk.Label(
            title_frame,
            text="🔒",
            font=('Arial', 28),
            fg='#47d7ac',
            bg='#0a0e14'
        )
        icon_label.pack(side=tk.LEFT)
        
        title_label = tk.Label(
            title_frame, 
            text="Secure Access Control",
            font=('Arial', 24, 'bold'),
            fg='#ffffff',
            bg='#0a0e14'
        )
        title_label.pack(side=tk.LEFT, padx=10)
        
        # Exit button with modern design
        exit_btn = tk.Button(
            header_frame,
            text="✕",
            command=self.exit_app,
            font=('Arial', 16, 'bold'),
            bg='#ff5f56',
            fg='white',
            width=3,
            height=1,
            relief=tk.FLAT,
            bd=0,
            cursor='hand2'
        )
        exit_btn.pack(side=tk.RIGHT, padx=10)
        
        # Main content area
        content_frame = tk.Frame(main_frame, bg='#0a0e14')
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel for video feed
        left_panel = tk.Frame(content_frame, bg='#0a0e14', width=800)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        left_panel.pack_propagate(False)
        
        # Right panel for status and controls
        right_panel = tk.Frame(content_frame, bg='#151a21', width=400)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(20, 0))
        right_panel.pack_propagate(False)
        
        # Setup video panel
        self.setup_video_panel(left_panel)
        
        # Setup status panel
        self.setup_status_panel(right_panel)
        
    def setup_video_panel(self, parent):
        # Video container with subtle border
        video_container = tk.Frame(parent, bg='#1c2129', padx=2, pady=2)
        video_container.pack(fill=tk.BOTH, expand=True)
        
        # Video label
        video_label = tk.Label(
            video_container,
            text="LIVE FEED",
            font=('Arial', 12, 'bold'),
            fg='#7a828e',
            bg='#1c2129',
            pady=10
        )
        video_label.pack(fill=tk.X)
        
        # Separator
        separator = ttk.Separator(video_container, orient='horizontal')
        separator.pack(fill=tk.X, padx=10)
        
        # Video canvas
        self.video_canvas = tk.Canvas(
            video_container,
            bg='#000000',
            highlightthickness=0
        )
        self.video_canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Initial placeholder text
        self.video_canvas.create_text(
            self.root.winfo_screenwidth() // 2, 
            self.root.winfo_screenheight() // 2,
            text="INITIALIZING CAMERA...",
            fill='#7a828e',
            font=('Arial', 14)
        )
        
    def setup_status_panel(self, parent):
        # Status panel title
        status_title = tk.Label(
            parent,
            text="SYSTEM STATUS",
            font=('Arial', 14, 'bold'),
            fg='#47d7ac',
            bg='#151a21',
            pady=20
        )
        status_title.pack(fill=tk.X)
        
        # Separator
        separator = ttk.Separator(parent, orient='horizontal')
        separator.pack(fill=tk.X, padx=20)
        
        # Status cards container
        status_cards = tk.Frame(parent, bg='#151a21')
        status_cards.pack(fill=tk.X, padx=20, pady=20)
        
        # Door status card
        door_card = tk.Frame(status_cards, bg='#1c2129', padx=15, pady=15)
        door_card.pack(fill=tk.X, pady=(0, 15))
        
        door_icon = tk.Label(
            door_card,
            text="🚪",
            font=('Arial', 20),
            fg='#47d7ac',
            bg='#1c2129'
        )
        door_icon.pack(side=tk.LEFT)
        
        door_text_frame = tk.Frame(door_card, bg='#1c2129')
        door_text_frame.pack(side=tk.LEFT, padx=(10, 0))
        
        door_title = tk.Label(
            door_text_frame,
            text="DOOR STATUS",
            font=('Arial', 10),
            fg='#7a828e',
            bg='#1c2129',
            anchor='w'
        )
        door_title.pack(fill=tk.X)
        
        self.door_status = tk.Label(
            door_text_frame,
            text="LOCKED",
            font=('Arial', 16, 'bold'),
            fg='#ff6b6b',
            bg='#1c2129',
            anchor='w'
        )
        self.door_status.pack(fill=tk.X)
        
        # Model status card
        model_card = tk.Frame(status_cards, bg='#1c2129', padx=15, pady=15)
        model_card.pack(fill=tk.X, pady=(0, 15))
        
        model_icon = tk.Label(
            model_card,
            text="🧠",
            font=('Arial', 20),
            fg='#47d7ac',
            bg='#1c2129'
        )
        model_icon.pack(side=tk.LEFT)
        
        model_text_frame = tk.Frame(model_card, bg='#1c2129')
        model_text_frame.pack(side=tk.LEFT, padx=(10, 0))
        
        model_title = tk.Label(
            model_text_frame,
            text="AI MODEL",
            font=('Arial', 10),
            fg='#7a828e',
            bg='#1c2129',
            anchor='w'
        )
        model_title.pack(fill=tk.X)
        
        self.model_status = tk.Label(
            model_text_frame,
            text="LOADING...",
            font=('Arial', 12, 'bold'),
            fg='#f0c674',
            bg='#1c2129',
            anchor='w'
        )
        self.model_status.pack(fill=tk.X)
        
        # Detection status card
        detection_card = tk.Frame(status_cards, bg='#1c2129', padx=15, pady=15)
        detection_card.pack(fill=tk.X, pady=(0, 15))
        
        detection_icon = tk.Label(
            detection_card,
            text="👁️",
            font=('Arial', 20),
            fg='#47d7ac',
            bg='#1c2129'
        )
        detection_icon.pack(side=tk.LEFT)
        
        detection_text_frame = tk.Frame(detection_card, bg='#1c2129')
        detection_text_frame.pack(side=tk.LEFT, padx=(10, 0))
        
        detection_title = tk.Label(
            detection_text_frame,
            text="DETECTION STATUS",
            font=('Arial', 10),
            fg='#7a828e',
            bg='#1c2129',
            anchor='w'
        )
        detection_title.pack(fill=tk.X)
        
        self.detection_status = tk.Label(
            detection_text_frame,
            text="IDLE",
            font=('Arial', 12, 'bold'),
            fg='#7a828e',
            bg='#1c2129',
            anchor='w'
        )
        self.detection_status.pack(fill=tk.X)
        
        # Recognition status card
        recognition_card = tk.Frame(status_cards, bg='#1c2129', padx=15, pady=15)
        recognition_card.pack(fill=tk.X, pady=(0, 20))
        
        recognition_icon = tk.Label(
            recognition_card,
            text="👤",
            font=('Arial', 20),
            fg='#47d7ac',
            bg='#1c2129'
        )
        recognition_icon.pack(side=tk.LEFT)
        
        recognition_text_frame = tk.Frame(recognition_card, bg='#1c2129')
        recognition_text_frame.pack(side=tk.LEFT, padx=(10, 0))
        
        recognition_title = tk.Label(
            recognition_text_frame,
            text="LAST RECOGNITION",
            font=('Arial', 10),
            fg='#7a828e',
            bg='#1c2129',
            anchor='w'
        )
        recognition_title.pack(fill=tk.X)
        
        self.recognition_status = tk.Label(
            recognition_text_frame,
            text="NONE",
            font=('Arial', 12, 'bold'),
            fg='#7a828e',
            bg='#1c2129',
            anchor='w'
        )
        self.recognition_status.pack(fill=tk.X)
        
        # Control buttons frame
        controls_frame = tk.Frame(parent, bg='#151a21', pady=20)
        controls_frame.pack(fill=tk.X, padx=20)
        
        self.start_btn = tk.Button(
            controls_frame,
            text="START RECOGNITION",
            command=self.start_recognition,
            font=('Arial', 12, 'bold'),
            bg='#27ae60',
            fg='white',
            padx=20,
            pady=12,
            relief=tk.FLAT,
            bd=0,
            cursor='hand2',
            state=tk.DISABLED
        )
        self.start_btn.pack(fill=tk.X, pady=(0, 10))
        
        self.stop_btn = tk.Button(
            controls_frame,
            text="STOP RECOGNITION",
            command=self.stop_recognition,
            font=('Arial', 12, 'bold'),
            bg='#c0392b',
            fg='white',
            padx=20,
            pady=12,
            relief=tk.FLAT,
            bd=0,
            cursor='hand2',
            state=tk.DISABLED
        )
        self.stop_btn.pack(fill=tk.X)
        
        # Stats frame
        stats_frame = tk.Frame(parent, bg='#151a21')
        stats_frame.pack(fill=tk.X, padx=20, pady=(0, 20))
        
        stats_title = tk.Label(
            stats_frame,
            text="SYSTEM INFORMATION",
            font=('Arial', 12, 'bold'),
            fg='#7a828e',
            bg='#151a21',
            anchor='w'
        )
        stats_title.pack(fill=tk.X, pady=(0, 10))
        
        # Stats items
        stats_grid = tk.Frame(stats_frame, bg='#151a21')
        stats_grid.pack(fill=tk.X)
        
        # Device info
        device_label = tk.Label(
            stats_grid,
            text="Device:",
            font=('Arial', 10),
            fg='#7a828e',
            bg='#151a21',
            anchor='w'
        )
        device_label.grid(row=0, column=0, sticky='w', pady=2)
        
        device_value = tk.Label(
            stats_grid,
            text="CUDA" if torch.cuda.is_available() else "CPU",
            font=('Arial', 10, 'bold'),
            fg='#ffffff',
            bg='#151a21',
            anchor='w'
        )
        device_value.grid(row=0, column=1, sticky='w', padx=(10, 0), pady=2)
        
        # Platform info
        platform_label = tk.Label(
            stats_grid,
            text="Platform:",
            font=('Arial', 10),
            fg='#7a828e',
            bg='#151a21',
            anchor='w'
        )
        platform_label.grid(row=1, column=0, sticky='w', pady=2)
        
        platform_value = tk.Label(
            stats_grid,
            text="Raspberry Pi" if self.is_raspberry_pi else "Desktop",
            font=('Arial', 10, 'bold'),
            fg='#ffffff',
            bg='#151a21',
            anchor='w'
        )
        platform_value.grid(row=1, column=1, sticky='w', padx=(10, 0), pady=2)
        
        # Known faces info
        faces_label = tk.Label(
            stats_grid,
            text="Known Faces:",
            font=('Arial', 10),
            fg='#7a828e',
            bg='#151a21',
            anchor='w'
        )
        faces_label.grid(row=2, column=0, sticky='w', pady=2)
        
        self.faces_value = tk.Label(
            stats_grid,
            text="0",
            font=('Arial', 10, 'bold'),
            fg='#ffffff',
            bg='#151a21',
            anchor='w'
        )
        self.faces_value.grid(row=2, column=1, sticky='w', padx=(10, 0), pady=2)
        
        # Configure column weights
        stats_grid.columnconfigure(0, weight=1)
        stats_grid.columnconfigure(1, weight=2)
        
    def load_resources(self):
        """Load model and embeddings in a separate thread"""
        def load():
            try:
                # Load model
                self.model = self.load_model(self.MODEL_PATH)
                self.root.after(0, lambda: self.model_status.config(
                    text="LOADED", fg='#27ae60'))
                
                # Load embeddings
                self.saved_embeddings = self.get_all_saved_embeddings()
                self.root.after(0, lambda: self.faces_value.config(
                    text=str(len(self.saved_embeddings))))
                
                # Load face detector
                if self.is_raspberry_pi:
                    # Use more efficient detector for Raspberry Pi
                    self.face_detector = cv2.CascadeClassifier(
                        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
                else:
                    # Use more accurate detector for desktop
                    self.face_detector = cv2.CascadeClassifier(
                        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
                
                # Update UI
                self.root.after(0, self.update_ui_after_load)
                
            except Exception as e:
                error_msg = f"Error loading resources: {str(e)}"
                self.root.after(0, lambda: messagebox.showerror("Error", error_msg))
        
        threading.Thread(target=load, daemon=True).start()
    
    def update_ui_after_load(self):
        self.model_status.config(text="LOADED", fg='#27ae60')
        self.start_btn.config(state=tk.NORMAL)
        self.faces_value.config(text=str(len(self.saved_embeddings)))
    
    def load_model(self, path):
        model = load_facenet_model()
        model.load_state_dict(torch.load(path, map_location=self.DEVICE))
        model.to(self.DEVICE)
        model.eval()
        return model
    
    def get_all_saved_embeddings(self):
        embeddings = {}
        if os.path.exists(self.EMBEDDINGS_DIR):
            for file in os.listdir(self.EMBEDDINGS_DIR):
                if file.endswith(".pt"):
                    name = file[:-3]
                    embedding = load_embedding(name)
                    embeddings[name] = embedding
        return embeddings
    
    def start_recognition(self):
        if self.model is None or self.face_detector is None:
            messagebox.showerror("Error", "Resources not loaded yet. Please wait.")
            return
            
        try:
            # Try different camera indices
            for camera_id in [0, 1, 2]:
                if self.is_raspberry_pi:
                    self.cap = cv2.VideoCapture(camera_id)
                else:
                    self.cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
                
                if self.cap.isOpened():
                    break
            
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Cannot open any camera")
                return
            
            # Set camera properties based on platform
            if self.is_raspberry_pi:
                # Lower resolution for better performance on Raspberry Pi
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap.set(cv2.CAP_PROP_FPS, 15)
            else:
                # Higher resolution for desktop
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            self.is_running = True
            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            
            self.detection_status.config(text="SCANNING", fg='#f0c674')
            
            self.video_loop()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start camera: {str(e)}")
    
    def stop_recognition(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        
        self.detection_status.config(text="IDLE", fg='#7a828e')
        self.recognition_status.config(text="NONE", fg='#7a828e')
        
        # Clear video canvas
        self.video_canvas.delete("all")
        self.video_canvas.create_text(
            self.video_canvas.winfo_width() // 2, 
            self.video_canvas.winfo_height() // 2,
            text="CAMERA DISCONNECTED",
            fill='#7a828e',
            font=('Arial', 14)
        )
    
    def video_loop(self):
        if not self.is_running or not self.cap:
            return
        
        ret, frame = self.cap.read()
        if not ret:
            self.root.after(10, self.video_loop)
            return
        
        # Process frame
        processed_frame = self.process_frame(frame)
        
        # Convert to PhotoImage and display
        rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        
        # Resize to fit canvas
        canvas_width = self.video_canvas.winfo_width()
        canvas_height = self.video_canvas.winfo_height()
        
        if canvas_width > 1 and canvas_height > 1:
            pil_image = pil_image.resize((canvas_width, canvas_height), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(pil_image)
            
            self.video_canvas.delete("all")
            self.video_canvas.create_image(canvas_width // 2, canvas_height // 2, image=photo)
            self.video_canvas.image = photo  # Keep a reference
        
        # Schedule next frame
        self.root.after(30, self.video_loop)  # ~30 FPS
    
    def process_frame(self, frame):
        # Resize frame for faster processing if needed
        if self.is_raspberry_pi:
            processing_frame = cv2.resize(frame, (320, 240))
        else:
            processing_frame = frame.copy()
            
        gray = cv2.cvtColor(processing_frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        current_time = time.time()
        
        if len(faces) > 0:
            if not self.face_detected:
                self.detection_start_time = current_time
                self.face_detected = True
                self.detection_status.config(
                    text="FACE DETECTED", 
                    fg='#f0c674'
                )
            else:
                elapsed = current_time - self.detection_start_time
                if elapsed >= self.DETECTION_DELAY:
                    # Scale faces back to original frame size if needed
                    if self.is_raspberry_pi:
                        scale_x = frame.shape[1] / processing_frame.shape[1]
                        scale_y = frame.shape[0] / processing_frame.shape[0]
                        faces = [(int(x * scale_x), int(y * scale_y), 
                                 int(w * scale_x), int(h * scale_y)) for (x, y, w, h) in faces]
                    
                    for (x, y, w, h) in faces:
                        name, distance = self.recognize_face(frame, (x, y, w, h))
                        
                        # Draw enhanced bounding box
                        self.draw_enhanced_bbox(frame, (x, y, w, h), name, distance)
                        
                        # Handle door unlocking
                        if name != "UNKNOWN":
                            self.unlock_door(name, distance)
                        else:
                            self.deny_access()
                else:
                    # Show countdown
                    remaining = int(self.DETECTION_DELAY - elapsed) + 1
                    self.detection_status.config(
                        text=f"PROCESSING IN {remaining}s",
                        fg='#f0c674'
                    )
                    
                    # Draw simple bounding box during countdown
                    for (x, y, w, h) in faces:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                        cv2.putText(frame, f"Processing in {remaining}s", 
                                   (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                                   0.6, (0, 255, 255), 2)
        else:
            if self.face_detected:
                self.face_detected = False
                self.detection_start_time = None
                self.detection_status.config(
                    text="SCANNING", 
                    fg='#f0c674'
                )
        
        return frame
    
    def draw_enhanced_bbox(self, frame, box, name, distance):
        x, y, w, h = box
        
        # Choose colors based on recognition
        if name != "UNKNOWN":
            box_color = (0, 255, 0)  # Green for recognized
            text_color = (0, 255, 0)
            status = "ACCESS GRANTED"
        else:
            box_color = (0, 0, 255)  # Red for unknown
            text_color = (0, 0, 255)
            status = "ACCESS DENIED"
        
        # Draw main bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 3)
        
        # Draw corner accents
        corner_length = 20
        cv2.line(frame, (x, y), (x + corner_length, y), box_color, 5)
        cv2.line(frame, (x, y), (x, y + corner_length), box_color, 5)
        cv2.line(frame, (x + w, y), (x + w - corner_length, y), box_color, 5)
        cv2.line(frame, (x + w, y), (x + w, y + corner_length), box_color, 5)
        cv2.line(frame, (x, y + h), (x + corner_length, y + h), box_color, 5)
        cv2.line(frame, (x, y + h), (x, y + h - corner_length), box_color, 5)
        cv2.line(frame, (x + w, y + h), (x + w - corner_length, y + h), box_color, 5)
        cv2.line(frame, (x + w, y + h), (x + w, y + h - corner_length), box_color, 5)
        
        # Create info panel
        label = f"{name}"
        confidence = f"Confidence: {(1-distance)*100:.1f}%"
        
        # Background for text
        cv2.rectangle(frame, (x, y - 90), (x + 300, y), (0, 0, 0), -1)
        cv2.rectangle(frame, (x, y - 90), (x + 300, y), box_color, 2)
        
        # Draw text
        cv2.putText(frame, status, (x + 5, y - 65), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
        cv2.putText(frame, label, (x + 5, y - 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, confidence, (x + 5, y - 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def recognize_face(self, frame, box):
        try:
            x, y, w, h = box
            face_img = frame[y:y+h, x:x+w]
            
            # Ensure we have a valid face image
            if face_img.size == 0:
                return "UNKNOWN", 1.0
                
            face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
            face_tensor = preprocess_face(face_pil)
            embedding = get_embedding(self.model, face_tensor, self.DEVICE)
            
            distances = {name: self.euclidean_distance(embedding, emb)
                        for name, emb in self.saved_embeddings.items()}
            
            if distances:
                best_match = min(distances, key=distances.get)
                best_distance = distances[best_match]
                
                if best_distance < self.THRESHOLD:
                    return best_match, best_distance
            
            return "UNKNOWN", 1.0
        except Exception as e:
            print(f"Recognition error: {e}")
            return "ERROR", 1.0
    
    def euclidean_distance(self, t1, t2):
        if isinstance(t1, np.ndarray):
            t1 = torch.tensor(t1)
        if isinstance(t2, np.ndarray):
            t2 = torch.tensor(t2)
        return torch.norm(t1 - t2).item()
    
    def unlock_door(self, name, distance):
        if self.door_locked:
            self.door_locked = False
            self.door_status.config(text="UNLOCKED", fg='#27ae60')
            
            # Update recognition status
            timestamp = datetime.now().strftime("%H:%M:%S")
            confidence = f"{(1-distance)*100:.1f}%"
            self.recognition_status.config(
                text=f"{name} ({confidence})", 
                fg='#27ae60'
            )
            
            # Play success sound
            try:
                winsound.Beep(1000, 500)
            except:
                pass  # Ignore sound errors
            
            # Log the access
            self.log_access(name, "GRANTED", distance)
            
            # Auto-relock after 5 seconds
            self.root.after(5000, self.relock_door)
    
    def deny_access(self):
        self.door_status.config(text="LOCKED", fg='#ff6b6b')
        self.recognition_status.config(text="UNKNOWN PERSON", fg='#ff6b6b')
        
        # Play denial sound
        try:
            winsound.Beep(500, 1000)
        except:
            pass  # Ignore sound errors
        
        # Log the denied access
        self.log_access("UNKNOWN", "DENIED", 1.0)
    
    def relock_door(self):
        self.door_locked = True
        self.door_status.config(text="LOCKED", fg='#ff6b6b')
        
        # Play relock sound
        try:
            winsound.Beep(800, 300)
        except:
            pass  # Ignore sound errors
    
    def log_access(self, name, status, distance):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        confidence = f"{(1-distance)*100:.1f}%" if status == "GRANTED" else "N/A"
        log_entry = f"[{timestamp}] {name} - {status} - {confidence}"
        
        self.detection_history.append({
            'name': name,
            'time': timestamp,
            'status': status,
            'distance': distance
        })
        
        # Save to log file
        with open("access_log.txt", "a") as f:
            f.write(log_entry + "\n")

def main():
    root = tk.Tk()
    app = FacialRecognitionUI(root)
    
    def on_closing():
        app.stop_recognition()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()