import os
import math
import cv2
import torch
import time
import numpy as np
from PIL import Image
from datetime import datetime
import threading
from collections import deque
import platform
import base64
import logging
from contextlib import contextmanager

from app.model import load_facenet_model
from app.face_utils import preprocess_face, get_embedding
from app.db import load_embedding

logger = logging.getLogger(__name__)

class FacialRecognitionAPI:
    def __init__(self):
        # Configuration
        self.MODEL_PATH = "facenet_africain_finetuned.pth"
        self.EMBEDDINGS_DIR = "embeddings"
        self.CAPTURE_DIR = "captures"
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Platform detection
        self.is_raspberry_pi = platform.machine() in ('armv7l', 'armv6l', 'aarch64')
        
        # Ajustement du seuil pour Raspberry Pi
        if self.is_raspberry_pi:
            self.THRESHOLD = 0.75  # Seuil équilibré pour Raspberry Pi
            print(f"Raspberry Pi détecté - Seuil ajusté à {self.THRESHOLD}")
        else:
            self.THRESHOLD = 0.7
            
        self.DETECTION_DELAY = 3
        self.door_unlock_available_until = 0
        
        # State variables
        self.cap = None
        self.model = None
        self.saved_embeddings = {}
        self.face_detector = None
        self.is_running = False
        self.door_locked = True
        self.last_recognition = None
        self.detection_history = deque(maxlen=100)
        self.current_frame = None
        self.frame_lock = threading.Lock()
        self.annotated_frame = None
        self.stream_active = False
        self.stream_with_recognition = False
        self.fps_counter = 0
        self.fps_start_time = time.time()
        
        # Create directories
        os.makedirs(self.CAPTURE_DIR, exist_ok=True)
        
        # Load resources
        self.load_resources()
    
    def load_resources(self): 
        """Load model and embeddings"""
        try:
            logger.info("Loading facial recognition model...")
            self.model = self.load_model(self.MODEL_PATH)
            
            logger.info("Loading saved embeddings...")
            self.saved_embeddings = self.get_all_saved_embeddings()
            
            # Afficher les embeddings chargés pour debug
            print(f"Embeddings chargés: {list(self.saved_embeddings.keys())}")
            for name, embedding in self.saved_embeddings.items():
                print(f"  {name}: shape={embedding.shape if hasattr(embedding, 'shape') else 'N/A'}")
            
            logger.info("Loading face detector...")
            self.face_detector = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            
            logger.info(f"System ready - {len(self.saved_embeddings)} known faces loaded")
            
        except Exception as e:
            logger.error(f"Error loading resources: {str(e)}")
            raise
    
    def load_model(self, path):
        """Load the facial recognition model"""
        model = load_facenet_model()
        
        # Pour Raspberry Pi, charger avec des options spécifiques
        if self.is_raspberry_pi:
            # Utiliser map_location='cpu' explicitement sur Raspberry Pi
            checkpoint = torch.load(path, map_location='cpu')
            model.load_state_dict(checkpoint)
            print("Modèle chargé en mode CPU pour Raspberry Pi")
        else:
            model.load_state_dict(torch.load(path, map_location=self.DEVICE))
            
        model.to(self.DEVICE)
        model.eval()
        
        # Sur Raspberry Pi, définir le modèle en mode inference strict
        if self.is_raspberry_pi:
            torch.set_grad_enabled(False)
            
        return model
    
    def get_all_saved_embeddings(self):
        """Load all saved embeddings from disk"""
        embeddings = {}
        if os.path.exists(self.EMBEDDINGS_DIR):
            for file in os.listdir(self.EMBEDDINGS_DIR):
                if file.endswith(".pt"):
                    name = file[:-3]
                    try:
                        embedding = load_embedding(name)
                        # Assurer que l'embedding est un tensor
                        if not isinstance(embedding, torch.Tensor):
                            embedding = torch.tensor(embedding, dtype=torch.float32)
                        # Normaliser l'embedding
                        embedding = torch.nn.functional.normalize(embedding, p=2, dim=0)
                        embeddings[name] = embedding
                        print(f"Embedding chargé pour {name}: {embedding.shape}")
                    except Exception as e:
                        print(f"Erreur lors du chargement de {name}: {e}")
        return embeddings
    
    @contextmanager
    def camera_context(self, camera_id=0):
        """Context manager for camera operations"""
        cap = None
        try:
            if self.is_raspberry_pi:
                cap = cv2.VideoCapture(camera_id)
                # Configuration optimisée pour Raspberry Pi
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)  # Réduction de la résolution
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
                cap.set(cv2.CAP_PROP_FPS, 10)  # FPS réduit
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Buffer réduit
            else:
                cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                cap.set(cv2.CAP_PROP_FPS, 30)
            
            if not cap.isOpened():
                raise Exception(f"Cannot open camera {camera_id}")
            
            yield cap
            
        finally:
            if cap:
                cap.release()
    
    def start_camera_stream(self, enable_recognition=False):
        """Start continuous camera streaming with optional recognition"""
        def stream():
            try:
                # Try different camera indices
                for camera_id in [0, 1, 2]:
                    try:
                        with self.camera_context(camera_id) as cap:
                            self.cap = cap
                            self.stream_with_recognition = enable_recognition
                            logger.info(f"Camera stream started on camera {camera_id} (recognition: {enable_recognition})")
                            
                            while self.is_running:
                                ret, frame = cap.read()
                                if ret:
                                    # Update frame counter for FPS calculation
                                    self.fps_counter += 1
                                    current_time = time.time()
                                    if current_time - self.fps_start_time >= 1.0:
                                        self.fps_start_time = current_time
                                        self.fps_counter = 0
                                    
                                    with self.frame_lock:
                                        self.current_frame = frame.copy()
                                        
                                        # Process recognition if enabled
                                        if self.stream_with_recognition:
                                            self.annotated_frame = self.annotate_frame_with_recognition(frame.copy())
                                        else:
                                            self.annotated_frame = frame.copy()
                                            
                                # Délai adapté à la plateforme
                                sleep_time = 0.1 if self.is_raspberry_pi else 0.033
                                time.sleep(sleep_time)
                            break
                    except Exception as e:
                        logger.warning(f"Failed to open camera {camera_id}: {e}")
                        continue
                else:
                    logger.error("No cameras available")
                    
            except Exception as e:
                logger.error(f"Camera stream error: {e}")
            finally:
                self.cap = None
                with self.frame_lock:
                    self.current_frame = None
                    self.annotated_frame = None
        
        if not self.is_running:
            self.is_running = True
            self.stream_active = True
            threading.Thread(target=stream, daemon=True).start()
    
    def stop_camera_stream(self):
        """Stop camera streaming"""
        self.is_running = False
        self.stream_active = False
        self.stream_with_recognition = False
        if self.cap:
            self.cap = None
    
    def get_current_frame(self, annotated=False):
        """Get the current frame from camera stream"""
        with self.frame_lock:
            if annotated and self.annotated_frame is not None:
                return self.annotated_frame.copy()
            return self.current_frame.copy() if self.current_frame is not None else None
    
    def detect_and_recognize_faces(self, frame):
        """Detect and recognize faces in a frame"""
        if frame is None:
            return []
        
        # Traitement adapté à la plateforme
        if self.is_raspberry_pi:
            # Résolution très réduite pour le traitement
            processing_frame = cv2.resize(frame, (240, 180))
            scale_x = frame.shape[1] / processing_frame.shape[1]
            scale_y = frame.shape[0] / processing_frame.shape[0]
        else:
            processing_frame = frame.copy()
            scale_x = scale_y = 1.0
        
        gray = cv2.cvtColor(processing_frame, cv2.COLOR_BGR2GRAY)
        
        # Paramètres de détection ajustés pour Raspberry Pi
        if self.is_raspberry_pi:
            faces = self.face_detector.detectMultiScale(
                gray, 
                scaleFactor=1.2,  # Plus agressif
                minNeighbors=3,   # Moins strict
                minSize=(20, 20), # Taille minimum réduite
                maxSize=(200, 200)  # Taille maximum ajoutée
            )
        else:
            faces = self.face_detector.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        results = []
        for (x, y, w, h) in faces:
            # Scale back to original frame size
            x, y, w, h = int(x * scale_x), int(y * scale_y), int(w * scale_x), int(h * scale_y)
            
            # Recognize face
            name, distance = self.recognize_face(frame, (x, y, w, h))
            
            results.append({
                'bbox': [x, y, w, h],
                'name': name,
                'confidence': max(0, (1 - distance) * 100),
                'distance': distance,
                'recognized': name != "UNKNOWN"
            })
        
        return results
    
    def annotate_frame_with_recognition(self, frame):
        """Annotate frame with face recognition results"""
        faces = self.detect_and_recognize_faces(frame)
        
        for face in faces:
            x, y, w, h = face['bbox']
            name = face['name']
            confidence = face['confidence']
            distance = face['distance']
            
            # Debug: afficher les distances calculées
            print(f"Face détectée: {name}, distance: {distance:.4f}, confidence: {confidence:.1f}%, seuil: {self.THRESHOLD}")
            
            # Choose colors based on recognition
            if face['recognized']:
                box_color = (0, 255, 0)  # Green for recognized
                text_color = (0, 255, 0)
                status = "ACCESS GRANTED"
                self.door_locked = False
            else:
                box_color = (0, 0, 255)  # Red for unknown
                text_color = (0, 0, 255)
                status = "ACCESS DENIED"
                self.door_locked = True
            
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
            confidence_text = f"Conf: {confidence:.1f}%"
            distance_text = f"Dist: {distance:.3f}"
            
            # Calculate text area size
            panel_width = max(300, len(status) * 12)
            panel_height = 110  # Augmenté pour inclure la distance
            
            # Background for text
            cv2.rectangle(frame, (x, y - panel_height), (x + panel_width, y), (0, 0, 0), -1)
            cv2.rectangle(frame, (x, y - panel_height), (x + panel_width, y), box_color, 2)
            
            # Draw text
            cv2.putText(frame, status, (x + 5, y - 85), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
            cv2.putText(frame, label, (x + 5, y - 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, confidence_text, (x + 5, y - 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, distance_text, (x + 5, y - 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add system info overlay
        overlay_text = [
            f"Door: {'LOCKED' if self.door_locked else 'UNLOCKED'}",
            f"Faces: {len(faces)}",
            f"Known: {len(self.saved_embeddings)}",
            f"Threshold: {self.THRESHOLD}",
            f"Platform: {'RPi' if self.is_raspberry_pi else 'PC'}",
            f"Time: {datetime.now().strftime('%H:%M:%S')}"
        ]
        
        # Background for system info
        cv2.rectangle(frame, (10, 10), (300, 160), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (300, 160), (100, 100, 100), 2)
        
        for i, text in enumerate(overlay_text):
            cv2.putText(frame, text, (20, 35 + i * 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def recognize_face(self, frame, box):
        """Recognize a face in the given bounding box"""
        try:
            x, y, w, h = box
            
            # Debug: dimensions de la frame
            print(f"Frame shape: {frame.shape}")
            print(f"Bounding box: {x}, {y}, {w}, {h}")
            
            # Ajouter une marge autour du visage
            margin = int(max(w, h) * 0.1)
            x_start = max(0, x - margin)
            y_start = max(0, y - margin)
            x_end = min(frame.shape[1], x + w + margin)
            y_end = min(frame.shape[0], y + h + margin)
            
            print(f"After margin - start: ({x_start}, {y_start}), end: ({x_end}, {y_end})")
            
            face_img = frame[y_start:y_end, x_start:x_end]
            
            if face_img.size == 0:
                print("ERROR: Face image is empty!")
                return "UNKNOWN", 1.0
            
            print(f"Face image shape: {face_img.shape}")
            
            # Sur Raspberry Pi, améliorer la qualité de l'image
            if self.is_raspberry_pi and (face_img.shape[0] < 100 or face_img.shape[1] < 100):
                face_img = cv2.resize(face_img, (160, 160), interpolation=cv2.INTER_CUBIC)
                print(f"Resized face image shape: {face_img.shape}")
            
            # Amélioration de l'image
            face_img = cv2.convertScaleAbs(face_img, alpha=1.1, beta=10)
            
            face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
            face_tensor = preprocess_face(face_pil)
            
            print(f"Face tensor shape: {face_tensor.shape}")
            print(f"Face tensor device before: {face_tensor.device}")
            
            # S'assurer que le tensor est sur le bon device
            if self.DEVICE == 'cuda' and torch.cuda.is_available():
                face_tensor = face_tensor.to(self.DEVICE)
            else:
                face_tensor = face_tensor.to('cpu')
                
            print(f"Face tensor device after: {face_tensor.device}")
            
            embedding = get_embedding(self.model, face_tensor, self.DEVICE)
            print(f"Embedding généré: type={type(embedding)}")
            
            # Debug approfondi de l'embedding
            if hasattr(embedding, 'shape'):
                print(f"Embedding shape: {embedding.shape}")
            if hasattr(embedding, 'dtype'):
                print(f"Embedding dtype: {embedding.dtype}")
            if hasattr(embedding, 'device'):
                print(f"Embedding device: {embedding.device}")
            
            # Vérifier les valeurs concrètes
            if hasattr(embedding, '__len__') and len(embedding) > 0:
                print(f"Embedding first 5 values: {embedding[:5] if hasattr(embedding, '__getitem__') else 'N/A'}")
            
            # S'assurer que l'embedding est un tensor PyTorch
            if isinstance(embedding, np.ndarray):
                embedding = torch.tensor(embedding, dtype=torch.float32)
                print("Converted numpy array to tensor")
            
            # S'assurer que l'embedding est sur le bon device
            if self.DEVICE == 'cuda' and torch.cuda.is_available():
                embedding = embedding.to(self.DEVICE)
            else:
                embedding = embedding.to('cpu')
                
            print(f"Final embedding device: {embedding.device}")
            
            # Aplatir si nécessaire
            if len(embedding.shape) > 1:
                embedding = embedding.flatten()
                print(f"Flattened embedding shape: {embedding.shape}")
                
            # Vérifier les NaN avant normalisation
            if torch.isnan(embedding).any():
                print("CRITICAL: NaN detected in embedding before normalization!")
                nan_indices = torch.isnan(embedding).nonzero()
                print(f"NaN indices: {nan_indices}")
                return "ERROR", 1.0
            
            # Normaliser l'embedding généré
            embedding = torch.nn.functional.normalize(embedding.unsqueeze(0), p=2, dim=1).squeeze(0)
            print(f"After normalization shape: {embedding.shape}")
            
            # Vérifier les NaN après normalisation
            if torch.isnan(embedding).any():
                print("CRITICAL: NaN detected in embedding after normalization!")
                return "ERROR", 1.0
            
            # Calculer les distances avec tous les embeddings sauvegardés
            distances = {}
            for name, saved_embedding in self.saved_embeddings.items():
                print(f"\nComparing with {name}:")
                print(f"Saved embedding shape: {saved_embedding.shape}")
                print(f"Saved embedding device: {saved_embedding.device}")
                
                # Vérifier les NaN dans l'embedding sauvegardé
                if torch.isnan(saved_embedding).any():
                    print(f"CRITICAL: NaN in saved embedding {name}!")
                    continue
                    
                # S'assurer que les devices sont compatibles
                if embedding.device != saved_embedding.device:
                    saved_embedding = saved_embedding.to(embedding.device)
                    print(f"Moved saved embedding to device: {saved_embedding.device}")
                
                distance = self.cosine_distance(embedding, saved_embedding)
                distances[name] = distance
                print(f"Distance avec {name}: {distance}")
                
                # Debug de la distance
                if math.isnan(distance):
                    print(f"NaN distance calculated for {name}!")
            
            if distances:
                best_match = min(distances, key=distances.get)
                best_distance = distances[best_match]
                
                print(f"Meilleure correspondance: {best_match} avec distance {best_distance} (seuil: {self.THRESHOLD})")
                
                if best_distance < self.THRESHOLD:
                    return best_match, best_distance
                else:
                    print(f"Distance {best_distance} > seuil {self.THRESHOLD}, marqué comme UNKNOWN")
            
            return "UNKNOWN", 1.0
            
        except Exception as e:
            logger.error(f"Recognition error: {e}")
            print(f"Erreur de reconnaissance: {e}")
            import traceback
            traceback.print_exc()
            return "ERROR", 1.0
    def euclidean_distance(self, t1, t2):
        """Calculate Euclidean distance between two tensors"""
        if isinstance(t1, np.ndarray):
            t1 = torch.tensor(t1, dtype=torch.float32)
        if isinstance(t2, np.ndarray):
            t2 = torch.tensor(t2, dtype=torch.float32)
        return torch.norm(t1 - t2).item()
    
    def cosine_distance(self, t1, t2):
        """Calculate cosine distance between two tensors"""
        if isinstance(t1, np.ndarray):
            t1 = torch.tensor(t1, dtype=torch.float32)
        if isinstance(t2, np.ndarray):
            t2 = torch.tensor(t2, dtype=torch.float32)
        
        # Calculer la similarité cosine puis convertir en distance
        cosine_similarity = torch.nn.functional.cosine_similarity(t1.unsqueeze(0), t2.unsqueeze(0))
        return (1 - cosine_similarity).item()
    
    def process_access_request(self, recognized_faces):
        """Process access request based on recognized faces"""
        access_granted = False
        recognized_person = None
        
        for face in recognized_faces:
            if face['recognized'] and face['confidence'] > (1 - self.THRESHOLD) * 100:
                access_granted = True
                recognized_person = face['name']
                self.unlock_door(face['name'], face['distance'])
                break
        
        if not access_granted and recognized_faces:
            self.deny_access()
        
        return {
            'access_granted': access_granted,
            'person': recognized_person,
            'faces_detected': len(recognized_faces),
            'recognized_faces': [f for f in recognized_faces if f['recognized']],
            'unlock_window_valid': self.is_unlock_window_valid()
        }
    
    def unlock_door(self, name, distance):
        """Unlock the door for recognized person"""
        # Unlock the door
        self.door_locked = False
        
        # Update last recognition
        self.last_recognition = {
            'name': name,
            'timestamp': datetime.now().isoformat(),
            'confidence': (1 - distance) * 100,
            'status': 'GRANTED'
        }
        
        # Log the access
        self.log_access(name, "GRANTED", distance)
        
        logger.info(f"Access granted to {name}")
        
        # Set a timestamp for when the unlock window expires (15 seconds total)
        self.door_unlock_available_until = time.time() + 15
        
        # Cancel any existing auto-relock timer and start a new one
        if hasattr(self, '_relock_timer') and self._relock_timer:
            self._relock_timer.cancel()
        
        # Wait 5 seconds before starting the relock countdown
        # This gives the person time to enter
        self._relock_timer = threading.Timer(5.0, self.start_relock_countdown)
        self._relock_timer.start()
        
    def deny_access(self):
        """Deny access for unknown person"""
        self.last_recognition = {
            'name': 'UNKNOWN',
            'timestamp': datetime.now().isoformat(),
            'confidence': 0,
            'status': 'DENIED'
        }
        
        # Log the denied access
        self.log_access("UNKNOWN", "DENIED", 1.0)
        
        logger.info("Access denied - unknown person")
    
    def relock_door(self):
        """Relock the door after a timeout"""
        self.door_locked = True
        self.door_unlock_available_until = 0  # Clear any remaining unlock window
        if hasattr(self, '_relock_timer'):
            self._relock_timer = None
        logger.info("Door automatically relocked")
        
    def is_unlock_window_valid(self):
        """Check if the unlock window is still valid"""
        return time.time() <= self.door_unlock_available_until
    
    def log_access(self, name, status, distance):
        """Log access attempt"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        confidence = f"{(1-distance)*100:.1f}%" if status == "GRANTED" else "N/A"
        
        log_entry = {
            'name': name,
            'timestamp': timestamp,
            'status': status,
            'confidence': confidence,
            'distance': distance
        }
        
        self.detection_history.append(log_entry)
        
        # Save to log file
        with open("access_log.txt", "a") as f:
            f.write(f"[{timestamp}] {name} - {status} - {confidence} - Distance: {distance:.4f}\n")
    
    def frame_to_base64(self, frame):
        """Convert frame to base64 string"""
        _, buffer = cv2.imencode('.jpg', frame)
        img_str = base64.b64encode(buffer).decode('utf-8')
        return img_str
    
    def generate_video_stream(self, annotated=False):
        """Generate video stream for HTTP streaming"""
        while self.stream_active:
            frame = self.get_current_frame(annotated=annotated)
            if frame is not None:
                # Encode frame as JPEG
                quality = 70 if self.is_raspberry_pi else 85  # Qualité réduite sur Pi
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
                frame_bytes = buffer.tobytes()
                
                # Yield frame in multipart format
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            sleep_time = 0.1 if self.is_raspberry_pi else 0.033
            time.sleep(sleep_time)
    
    def start_relock_countdown(self):
        """Start the relock countdown (called after initial delay)"""
        # Wait additional 10 seconds before relocking
        self._relock_timer = threading.Timer(10.0, self.relock_door)
        self._relock_timer.start()
        
    def adjust_threshold_for_lighting(self, frame):
        """Ajuster le seuil en fonction des conditions d'éclairage"""
        if self.is_raspberry_pi:
            # Analyser la luminosité moyenne de l'image
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            
            # Ajuster le seuil selon la luminosité
            if brightness < 80:  # Faible luminosité
                self.THRESHOLD = 0.9
            elif brightness > 180:  # Forte luminosité
                self.THRESHOLD = 0.8
            else:  # Luminosité normale
                self.THRESHOLD = 0.85