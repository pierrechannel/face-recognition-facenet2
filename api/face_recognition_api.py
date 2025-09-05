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
        self.THRESHOLD = 0.7
        self.DETECTION_DELAY = 3
        self.door_unlock_available_until = 0
        
        # Platform detection
        self.is_raspberry_pi = platform.machine() in ('armv7l', 'armv6l', 'aarch64')
        
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
        model.load_state_dict(torch.load(path, map_location=self.DEVICE))
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
    
    def analyze_lighting_conditions(self, frame):
        """Analyze current lighting conditions of the frame"""
        if frame is None:
            return {'brightness': 0.5, 'contrast': 0.5, 'lighting_level': 'normal'}
        
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate histogram
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        
        # Calculate lighting metrics
        mean_brightness = np.mean(gray)
        std_brightness = np.std(gray)
        
        # Analyze histogram distribution
        dark_pixels = np.sum(hist[:85])    # Pixels in range 0-84 (dark)
        mid_pixels = np.sum(hist[85:170])  # Pixels in range 85-169 (medium)
        bright_pixels = np.sum(hist[170:]) # Pixels in range 170-255 (bright)
        
        total_pixels = gray.shape[0] * gray.shape[1]
        dark_ratio = dark_pixels / total_pixels
        bright_ratio = bright_pixels / total_pixels
        
        # Determine lighting conditions
        if mean_brightness < 80 or dark_ratio > 0.6:
            lighting_level = 'dark'
        elif mean_brightness > 180 or bright_ratio > 0.4:
            lighting_level = 'bright'
        elif std_brightness < 30:
            lighting_level = 'low_contrast'
        else:
            lighting_level = 'normal'
        
        return {
            'brightness': mean_brightness / 255.0,
            'contrast': std_brightness / 127.5,  # Normalized contrast
            'lighting_level': lighting_level,
            'dark_ratio': dark_ratio,
            'bright_ratio': bright_ratio,
            'std_brightness': std_brightness
        }

    def adjust_camera_settings(self, cap, lighting_conditions):
        """Dynamically adjust camera settings based on lighting conditions"""
        lighting_level = lighting_conditions['lighting_level']
        brightness = lighting_conditions['brightness']
        contrast = lighting_conditions['contrast']
        
        try:
            if lighting_level == 'dark':
                # Dark environment - increase exposure, brightness
                if not self.is_raspberry_pi:
                    cap.set(cv2.CAP_PROP_EXPOSURE, -1)  # Auto exposure
                    cap.set(cv2.CAP_PROP_GAIN, 50)      # Increase gain
                cap.set(cv2.CAP_PROP_BRIGHTNESS, min(0.8, brightness + 0.3))
                cap.set(cv2.CAP_PROP_CONTRAST, min(1.0, contrast + 0.2))
                cap.set(cv2.CAP_PROP_SATURATION, 0.6)   # Moderate saturation
                
            elif lighting_level == 'bright':
                # Bright environment - reduce exposure, adjust for overexposure
                if not self.is_raspberry_pi:
                    cap.set(cv2.CAP_PROP_EXPOSURE, -8)  # Lower exposure
                    cap.set(cv2.CAP_PROP_GAIN, 10)      # Reduce gain
                cap.set(cv2.CAP_PROP_BRIGHTNESS, max(0.2, brightness - 0.2))
                cap.set(cv2.CAP_PROP_CONTRAST, max(0.3, contrast - 0.1))
                cap.set(cv2.CAP_PROP_SATURATION, 0.7)   # Higher saturation
                
            elif lighting_level == 'low_contrast':
                # Low contrast - enhance contrast and saturation
                cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.5)
                cap.set(cv2.CAP_PROP_CONTRAST, min(1.0, contrast + 0.3))
                cap.set(cv2.CAP_PROP_SATURATION, 0.8)   # Higher saturation for low contrast
                
            else:  # normal lighting
                # Normal lighting - balanced settings
                cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.5)
                cap.set(cv2.CAP_PROP_CONTRAST, 0.6)
                cap.set(cv2.CAP_PROP_SATURATION, 0.6)
            
            logger.info(f"Camera settings adjusted for {lighting_level} lighting")
            
        except Exception as e:
            logger.warning(f"Could not adjust camera settings: {e}")

    def enhance_frame_for_recognition(self, frame):
        """Enhance frame based on lighting conditions for better recognition"""
        if frame is None:
            return frame
        
        # Analyze current lighting
        lighting_conditions = self.analyze_lighting_conditions(frame)
        lighting_level = lighting_conditions['lighting_level']
        
        enhanced_frame = frame.copy()
        
        try:
            if lighting_level == 'dark':
                # For dark conditions: increase brightness and contrast
                enhanced_frame = cv2.convertScaleAbs(enhanced_frame, alpha=1.3, beta=30)
                
                # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
                lab = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2LAB)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                lab[:,:,0] = clahe.apply(lab[:,:,0])
                enhanced_frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
                
            elif lighting_level == 'bright':
                # For bright conditions: reduce brightness, increase contrast
                enhanced_frame = cv2.convertScaleAbs(enhanced_frame, alpha=1.1, beta=-20)
                
            elif lighting_level == 'low_contrast':
                # For low contrast: apply histogram equalization
                # Convert to YUV color space
                yuv = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2YUV)
                yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
                enhanced_frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
                
                # Increase saturation
                hsv = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2HSV)
                hsv[:,:,1] = cv2.multiply(hsv[:,:,1], 1.2)
                enhanced_frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            
            # Apply gamma correction based on lighting
            if lighting_conditions['brightness'] < 0.3:
                # Dark image - apply gamma correction to brighten
                gamma = 1.5
                lookup_table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in np.arange(0, 256)]).astype("uint8")
                enhanced_frame = cv2.LUT(enhanced_frame, lookup_table)
            elif lighting_conditions['brightness'] > 0.8:
                # Bright image - apply gamma correction to darken
                gamma = 0.7
                lookup_table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in np.arange(0, 256)]).astype("uint8")
                enhanced_frame = cv2.LUT(enhanced_frame, lookup_table)
            
            return enhanced_frame
            
        except Exception as e:
            logger.warning(f"Frame enhancement failed: {e}")
            return frame

    # CORRECTION 1: Update your camera_context method
    @contextmanager
    def camera_context(self, camera_id=0):
        """Context manager for camera operations with dynamic lighting adaptation"""
        cap = None
        try:
            if self.is_raspberry_pi:
                cap = cv2.VideoCapture(camera_id)
            else:
                cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
            
            if not cap.isOpened():
                raise Exception(f"Cannot open camera {camera_id}")
            
            # Set initial camera properties
            if self.is_raspberry_pi:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 15)
            else:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                cap.set(cv2.CAP_PROP_FPS, 30)
            
            # Enable auto exposure and white balance
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
            cap.set(cv2.CAP_PROP_AUTO_WB, 1)
            
            # Set initial balanced properties (these will be dynamically adjusted)
            cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.5)
            cap.set(cv2.CAP_PROP_CONTRAST, 0.5)
            cap.set(cv2.CAP_PROP_SATURATION, 0.6)
            
            # ADD: Warm-up period for camera settings
            for _ in range(10):
                ret, frame = cap.read()
                if ret:
                    # Analyze lighting and adjust settings during warm-up
                    lighting_conditions = self.analyze_lighting_conditions(frame)
                    self.adjust_camera_settings(cap, lighting_conditions)
                    break
            
            yield cap
            
        finally:
            if cap:
                cap.release()

    # CORRECTION 2: Update your start_camera_stream method
    def start_camera_stream(self, enable_recognition=False):
        """Start continuous camera streaming with adaptive lighting"""
        def stream():
            try:
                for camera_id in [0, 1, 2]:
                    try:
                        with self.camera_context(camera_id) as cap:
                            self.cap = cap
                            self.stream_with_recognition = enable_recognition
                            logger.info(f"Camera stream started on camera {camera_id} (recognition: {enable_recognition})")
                            
                            # ADD: Lighting adjustment counter
                            lighting_adjustment_counter = 0
                            
                            while self.is_running:
                                ret, frame = cap.read()
                                if ret:
                                    # UPDATE: Periodically adjust camera settings (every 30 frames)
                                    lighting_adjustment_counter += 1
                                    if lighting_adjustment_counter >= 30:
                                        lighting_conditions = self.analyze_lighting_conditions(frame)
                                        self.adjust_camera_settings(cap, lighting_conditions)
                                        lighting_adjustment_counter = 0
                                    
                                    # UPDATE: Apply frame enhancement for recognition
                                    if self.stream_with_recognition:
                                        enhanced_frame = self.enhance_frame_for_recognition(frame)
                                    else:
                                        enhanced_frame = frame
                                    
                                    self.fps_counter += 1
                                    current_time = time.time()
                                    if current_time - self.fps_start_time >= 1.0:
                                        self.fps_start_time = current_time
                                        self.fps_counter = 0
                                    
                                    with self.frame_lock:
                                        self.current_frame = enhanced_frame.copy()
                                        
                                        if self.stream_with_recognition:
                                            self.annotated_frame = self.annotate_frame_with_recognition(enhanced_frame.copy())
                                        else:
                                            self.annotated_frame = enhanced_frame.copy()
                                
                                time.sleep(0.033)  # ~30 FPS
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

    # CORRECTION 3: Update your detect_and_recognize_faces method
    def detect_and_recognize_faces(self, frame):
        """Detect and recognize faces in a frame with lighting enhancement"""
        if frame is None:
            return []
        
        # ADD: Apply lighting enhancement before processing
        enhanced_frame = self.enhance_frame_for_recognition(frame)
        
        # Resize frame for faster processing if needed
        if self.is_raspberry_pi:
            processing_frame = cv2.resize(enhanced_frame, (320, 240))
            scale_x = enhanced_frame.shape[1] / processing_frame.shape[1]
            scale_y = enhanced_frame.shape[0] / processing_frame.shape[0]
        else:
            processing_frame = enhanced_frame.copy()
            scale_x = scale_y = 1.0
        
        gray = cv2.cvtColor(processing_frame, cv2.COLOR_BGR2GRAY)
        
        # ADD: Apply additional preprocessing for face detection
        # Histogram equalization for better face detection in varying lighting
        gray = cv2.equalizeHist(gray)
        
        faces = self.face_detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        results = []
        for (x, y, w, h) in faces:
            # Scale back to original frame size
            x, y, w, h = int(x * scale_x), int(y * scale_y), int(w * scale_x), int(h * scale_y)
            
            # Use enhanced frame for recognition
            name, distance = self.recognize_face(enhanced_frame, (x, y, w, h))
            
            results.append({
                'bbox': [x, y, w, h],
                'name': name,
                'confidence': (1 - distance) * 100 if distance < 1.0 else 0,
                'distance': distance,
                'recognized': name != "UNKNOWN"
            })
        
        return results   
   
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
    

    def annotate_frame_with_recognition(self, frame):
        """Annotate frame with face recognition results"""
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
            confidence_text = f"Confidence: {confidence:.1f}%"
            
            # Calculate text area size
            panel_width = max(300, len(status) * 12)
            panel_height = 90
            
            # Background for text
            cv2.rectangle(frame, (x, y - panel_height), (x + panel_width, y), (0, 0, 0), -1)
            cv2.rectangle(frame, (x, y - panel_height), (x + panel_width, y), box_color, 2)
            
            # Draw text
            cv2.putText(frame, status, (x + 5, y - 65), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
            cv2.putText(frame, label, (x + 5, y - 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, confidence_text, (x + 5, y - 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add system info overlay
        overlay_text = [
            f"Door: {'LOCKED' if self.door_locked else 'UNLOCKED'}",
            f"Faces: {len(faces)}",
            f"Known: {len(self.saved_embeddings)}",
            f"Time: {datetime.now().strftime('%H:%M:%S')}"
        ]
        
        # Background for system info
        cv2.rectangle(frame, (10, 10), (300, 120), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (300, 120), (100, 100, 100), 2)
        
        for i, text in enumerate(overlay_text):
            cv2.putText(frame, text, (20, 35 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def recognize_face(self, frame, box):
        """Recognize a face in the given bounding box"""
        try:
            x, y, w, h = box
            face_img = frame[y:y+h, x:x+w]
            
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
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                frame_bytes = buffer.tobytes()
                
                # Yield frame in multipart format
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            time.sleep(0.033)  # ~30 FPS
    
    def start_relock_countdown(self):
        """Start the relock countdown (called after initial delay)"""
        # Wait additional 10 seconds before relocking
        self._relock_timer = threading.Timer(10.0, self.relock_door)
        self._relock_timer.start()