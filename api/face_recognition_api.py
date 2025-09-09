import os
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
from app.face_utils import preprocess_face, get_embedding, resize_image
from app.db import load_embedding

logger = logging.getLogger(__name__)

class FacialRecognitionAPI:
    def __init__(self):
        # Configuration
        self.MODEL_PATH = "facenet_africain_finetuned.pth"
        self.EMBEDDINGS_DIR = "embeddings"
        self.CAPTURE_DIR = "captures"
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.THRESHOLD = 0.7
        self.DETECTION_DELAY = 3
        self.door_unlock_available_until = 0
        
        # Platform detection
        self.is_raspberry_pi = platform.machine() in ('armv7l', 'armv6l', 'aarch64')
        
        # Raspberry Pi specific optimizations
        if self.is_raspberry_pi:
            logger.info("Raspberry Pi detected - applying optimizations")
            # Use smaller frame size for RPi
            self.FRAME_WIDTH = 320
            self.FRAME_HEIGHT = 240
            self.FPS = 10
            # Adjust threshold for potentially lower quality camera
            self.THRESHOLD = 0.75
            # Limit torch threads
            torch.set_num_threads(1)
        else:
            self.FRAME_WIDTH = 640
            self.FRAME_HEIGHT = 480
            self.FPS = 15
        
        # State variables
        self.cap = None
        self.model = None
        self.saved_embeddings = {}
        self.face_detector = None
        self.is_running = False
        self.door_locked = True
        self.last_recognition = None
        self.detection_history = deque(maxlen=50)  # Reduced for RPi
        self.current_frame = None
        self.frame_lock = threading.Lock()
        self.annotated_frame = None
        self.stream_active = False
        self.stream_with_recognition = False
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.last_processing_time = 0
        self.processing_interval = 0.5  # Process frames less frequently on RPi
        
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
            
            logger.info("Loading face detector...")
            # Use more efficient face detector for Raspberry Pi
            if self.is_raspberry_pi:
                # Try to load a more efficient detector
                try:
                    # Try to use OpenCV's DNN face detector if available
                    proto_path = "deploy.prototxt"
                    model_path = "res10_300x300_ssd_iter_140000.caffemodel"
                    if os.path.exists(proto_path) and os.path.exists(model_path):
                        self.face_detector = cv2.dnn.readNetFromCaffe(proto_path, model_path)
                        logger.info("Using DNN face detector")
                    else:
                        # Fall back to Haar cascade
                        self.face_detector = cv2.CascadeClassifier(
                            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
                        logger.info("Using Haar cascade face detector")
                except:
                    self.face_detector = cv2.CascadeClassifier(
                        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
                    logger.info("Using Haar cascade face detector (fallback)")
            else:
                self.face_detector = cv2.CascadeClassifier(
                    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            
            logger.info(f"System ready - {len(self.saved_embeddings)} known faces loaded")
            
        except Exception as e:
            logger.error(f"Error loading resources: {str(e)}")
            raise
    
    def load_model(self, path):
        """Load the facial recognition model with optimizations"""
        model = load_facenet_model()
        model.load_state_dict(torch.load(path, map_location=self.DEVICE))
        
        # Apply Raspberry Pi optimizations
        if self.is_raspberry_pi:
            # Use float32 instead of half precision for better CPU compatibility
            model.float()
            # Set to evaluation mode
            model.eval()
            # Try to use torchscript if available
            try:
                model = torch.jit.script(model)
                logger.info("Model compiled with TorchScript")
            except Exception as e:
                logger.warning(f"TorchScript compilation failed: {e}")
        else:
            model.to(self.DEVICE)
            model.eval()
            
        return model
    
    def get_all_saved_embeddings(self):
        """Load all saved embeddings from disk"""
        embeddings = {}
        if os.path.exists(self.EMBEDDINGS_DIR):
            for file in os.listdir(self.EMBEDDINGS_DIR):
                if file.endswith(".pt"):
                    name = file[:-3]
                    embedding = load_embedding(name)
                    embeddings[name] = embedding
        return embeddings
    
    @contextmanager
    def camera_context(self, camera_id=0):
        """Context manager for camera operations"""
        cap = None
        try:
            if self.is_raspberry_pi:
                # Use different backend for Raspberry Pi
                cap = cv2.VideoCapture(camera_id)
            else:
                cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
            
            if not cap.isOpened():
                raise Exception(f"Cannot open camera {camera_id}")
            
            # Set camera properties based on platform
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.FRAME_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.FRAME_HEIGHT)
            cap.set(cv2.CAP_PROP_FPS, self.FPS)
            
            # Additional Raspberry Pi camera optimizations
            if self.is_raspberry_pi:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer size
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # Better compression
            
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
                                        logger.debug(f"FPS: {self.fps_counter}")
                                        self.fps_counter = 0
                                        self.fps_start_time = current_time
                                    
                                    with self.frame_lock:
                                        self.current_frame = frame.copy()
                                        
                                        # Process recognition if enabled, but less frequently on RPi
                                        if self.stream_with_recognition:
                                            current_time = time.time()
                                            if current_time - self.last_processing_time >= self.processing_interval:
                                                self.annotated_frame = self.annotate_frame_with_recognition(frame.copy())
                                                self.last_processing_time = current_time
                                            elif self.annotated_frame is not None:
                                                # Reuse the last annotated frame if not time to process yet
                                                pass
                                        else:
                                            self.annotated_frame = frame.copy()
                                            
                                time.sleep(0.1 if self.is_raspberry_pi else 0.033)  # Adjust for RPi
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
        """Detect and recognize faces in a frame with RPi optimizations"""
        if frame is None:
            return []
        
        # Resize frame for faster processing on RPi
        processing_frame = resize_image(frame, max_size=320 if self.is_raspberry_pi else 640)
        scale_x = frame.shape[1] / processing_frame.shape[1]
        scale_y = frame.shape[0] / processing_frame.shape[0]
        
        # Use appropriate face detection method
        if isinstance(self.face_detector, cv2.dnn_Net):
            # DNN-based face detection
            blob = cv2.dnn.blobFromImage(
                processing_frame, 1.0, (300, 300), [104, 117, 123], False, False)
            self.face_detector.setInput(blob)
            detections = self.face_detector.forward()
            
            faces = []
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:  # Confidence threshold
                    x1 = int(detections[0, 0, i, 3] * processing_frame.shape[1])
                    y1 = int(detections[0, 0, i, 4] * processing_frame.shape[0])
                    x2 = int(detections[0, 0, i, 5] * processing_frame.shape[1])
                    y2 = int(detections[0, 0, i, 6] * processing_frame.shape[0])
                    w = x2 - x1
                    h = y2 - y1
                    faces.append((x1, y1, w, h))
        else:
            # Haar cascade face detection
            gray = cv2.cvtColor(processing_frame, cv2.COLOR_BGR2GRAY)
            # Use more aggressive scaling on RPi for speed
            scale_factor = 1.05 if self.is_raspberry_pi else 1.1
            min_neighbors = 3 if self.is_raspberry_pi else 5
            faces = self.face_detector.detectMultiScale(
                gray, scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=(30, 30))
        
        results = []
        for (x, y, w, h) in faces:
            # Scale back to original frame size
            x, y, w, h = int(x * scale_x), int(y * scale_y), int(w * scale_x), int(h * scale_y)
            
            # Recognize face
            name, distance = self.recognize_face(frame, (x, y, w, h))
            
            results.append({
                'bbox': [x, y, w, h],
                'name': name,
                'confidence': (1 - distance) * 100 if distance < 1.0 else 0,
                'distance': distance,
                'recognized': name != "UNKNOWN"
            })
        
        return results
    
    def annotate_frame_with_recognition(self, frame):
        """Annotate frame with face recognition results - optimized for RPi"""
        faces = self.detect_and_recognize_faces(frame)
        
        for face in faces:
            x, y, w, h = face['bbox']
            name = face['name']
            confidence = face['confidence']
            
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
            
            # Draw main bounding box (simpler for RPi)
            cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
            
            # Draw label background
            label = f"{name}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(frame, (x, y - label_size[1] - 10), (x + label_size[0] + 10, y), box_color, -1)
            
            # Draw text
            cv2.putText(frame, label, (x + 5, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Add simplified system info overlay for RPi
        overlay_text = [
            f"Door: {'LOCKED' if self.door_locked else 'UNLOCKED'}",
            f"Faces: {len(faces)}",
            f"Time: {datetime.now().strftime('%H:%M:%S')}"
        ]
        
        # Background for system info
        for i, text in enumerate(overlay_text):
            cv2.putText(frame, text, (10, 20 + i * 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def recognize_face(self, frame, box):
        """Recognize a face in the given bounding box with RPi optimizations"""
        try:
            x, y, w, h = box
            face_img = frame[y:y+h, x:x+w]
            
            if face_img.size == 0:
                return "UNKNOWN", 1.0
            
            # Convert BGR to RGB (OpenCV uses BGR, PIL uses RGB)
            face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            
            # Pass the numpy array directly to preprocess_face
            # The updated preprocess_face function can handle numpy arrays
            face_tensor = preprocess_face(face_img_rgb)
            
            if face_tensor is None:
                return "UNKNOWN", 1.0
                
            embedding = get_embedding(self.model, face_tensor, self.DEVICE)
            
            if embedding is None:
                return "UNKNOWN", 1.0
                
            distances = {name: self.euclidean_distance(embedding, emb)
                        for name, emb in self.saved_embeddings.items()}
            
            if distances:
                best_match = min(distances, key=distances.get)
                best_distance = distances[best_match]
                
                if best_distance < self.THRESHOLD:
                    return best_match, best_distance
            
            return "UNKNOWN", 1.0
        except Exception as e:
            logger.error(f"Recognition error: {e}")
            return "ERROR", 1.0
    def euclidean_distance(self, t1, t2):
        """Calculate Euclidean distance between two tensors"""
        if isinstance(t1, np.ndarray):
            t1 = torch.tensor(t1)
        if isinstance(t2, np.ndarray):
            t2 = torch.tensor(t2)
        return torch.norm(t1 - t2).item()
    
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
            f.write(f"[{timestamp}] {name} - {status} - {confidence}\n")
    
    def frame_to_base64(self, frame):
        """Convert frame to base64 string with lower quality for RPi"""
        quality = 50 if self.is_raspberry_pi else 80
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        img_str = base64.b64encode(buffer).decode('utf-8')
        return img_str
    
    def generate_video_stream(self, annotated=False):
        """Generate video stream for HTTP streaming - optimized for RPi"""
        while self.stream_active:
            frame = self.get_current_frame(annotated=annotated)
            if frame is not None:
                # Lower quality encoding for RPi
                quality = 50 if self.is_raspberry_pi else 85
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
                frame_bytes = buffer.tobytes()
                
                # Yield frame in multipart format
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            # Longer sleep interval for RPi to reduce CPU load
            time.sleep(0.1 if self.is_raspberry_pi else 0.033)
    
    def start_relock_countdown(self):
        """Start the relock countdown (called after initial delay)"""
        # Wait additional 10 seconds before relocking
        self._relock_timer = threading.Timer(10.0, self.relock_door)
        self._relock_timer.start()