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
        self.camera_error = None
        self.active_camera_id = None
        
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
    
    def test_camera_access(self):
        """Test camera access and return available cameras"""
        available_cameras = []
        
        # Test multiple camera backends and indices
        backends = [cv2.CAP_DSHOW, cv2.CAP_V4L2, cv2.CAP_ANY] if not self.is_raspberry_pi else [cv2.CAP_V4L2, cv2.CAP_ANY]
        
        for camera_id in range(5):  # Test cameras 0-4
            for backend in backends:
                try:
                    cap = cv2.VideoCapture(camera_id, backend)
                    if cap.isOpened():
                        ret, frame = cap.read()
                        if ret and frame is not None:
                            available_cameras.append({
                                'id': camera_id,
                                'backend': backend,
                                'resolution': (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 
                                             int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                            })
                            logger.info(f"Camera {camera_id} with backend {backend} is working")
                        cap.release()
                        break  # Found working camera, try next ID
                    cap.release()
                except Exception as e:
                    logger.debug(f"Camera {camera_id} with backend {backend} failed: {e}")
                    continue
        
        return available_cameras
    
    @contextmanager
    def camera_context(self, camera_id=0, backend=None):
        """Context manager for camera operations with enhanced error handling"""
        cap = None
        try:
            # Determine backend
            if backend is None:
                if self.is_raspberry_pi:
                    backend = cv2.CAP_V4L2
                else:
                    backend = cv2.CAP_DSHOW
            
            logger.info(f"Attempting to open camera {camera_id} with backend {backend}")
            cap = cv2.VideoCapture(camera_id, backend)
            
            if not cap.isOpened():
                raise Exception(f"Cannot open camera {camera_id} with backend {backend}")
            
            # Test if we can actually read frames
            ret, test_frame = cap.read()
            if not ret or test_frame is None:
                raise Exception(f"Camera {camera_id} opened but cannot read frames")
            
            # Set camera properties with error handling
            try:
                if self.is_raspberry_pi:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    cap.set(cv2.CAP_PROP_FPS, 15)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                else:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    cap.set(cv2.CAP_PROP_FPS, 30)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                # Verify settings
                actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                actual_fps = cap.get(cv2.CAP_PROP_FPS)
                
                logger.info(f"Camera {camera_id} configured: {actual_width}x{actual_height} @ {actual_fps} FPS")
                
            except Exception as e:
                logger.warning(f"Could not set camera properties: {e}")
            
            self.active_camera_id = camera_id
            yield cap
            
        except Exception as e:
            logger.error(f"Camera context error: {e}")
            self.camera_error = str(e)
            raise
        finally:
            if cap:
                cap.release()
            self.active_camera_id = None
    
    def start_camera_stream(self, enable_recognition=False):
        """Start continuous camera streaming with optional recognition"""
        def stream():
            try:
                # First, test available cameras
                available_cameras = self.test_camera_access()
                
                if not available_cameras:
                    self.camera_error = "No cameras available"
                    logger.error("No cameras found")
                    return
                
                logger.info(f"Found {len(available_cameras)} available cameras")
                
                # Try each available camera
                for camera_info in available_cameras:
                    try:
                        camera_id = camera_info['id']
                        backend = camera_info['backend']
                        
                        with self.camera_context(camera_id, backend) as cap:
                            self.cap = cap
                            self.stream_with_recognition = enable_recognition
                            self.camera_error = None
                            
                            logger.info(f"Camera stream started on camera {camera_id} (recognition: {enable_recognition})")
                            
                            frame_count = 0
                            last_frame_time = time.time()
                            
                            while self.is_running:
                                try:
                                    ret, frame = cap.read()
                                    current_time = time.time()
                                    
                                    if not ret or frame is None:
                                        logger.warning("Failed to read frame")
                                        # Try to reinitialize camera
                                        time.sleep(0.1)
                                        continue
                                    
                                    # Check if frame is all black (common issue)
                                    if np.mean(frame) < 1.0:
                                        logger.warning("Received black frame")
                                        time.sleep(0.1)
                                        continue
                                    
                                    frame_count += 1
                                    
                                    # Calculate actual FPS
                                    if current_time - last_frame_time >= 1.0:
                                        actual_fps = frame_count / (current_time - last_frame_time)
                                        logger.debug(f"Actual FPS: {actual_fps:.1f}")
                                        frame_count = 0
                                        last_frame_time = current_time
                                    
                                    with self.frame_lock:
                                        self.current_frame = frame.copy()
                                        
                                        # Process recognition if enabled
                                        if self.stream_with_recognition:
                                            self.annotated_frame = self.annotate_frame_with_recognition(frame.copy())
                                        else:
                                            self.annotated_frame = frame.copy()
                                    
                                    # Adaptive sleep based on processing time
                                    processing_time = time.time() - current_time
                                    target_interval = 0.033  # ~30 FPS
                                    sleep_time = max(0.001, target_interval - processing_time)
                                    time.sleep(sleep_time)
                                    
                                except Exception as e:
                                    logger.error(f"Frame processing error: {e}")
                                    time.sleep(0.1)
                                    continue
                            
                            logger.info("Camera stream stopped normally")
                            return  # Successfully streamed, exit function
                            
                    except Exception as e:
                        logger.warning(f"Camera {camera_info['id']} failed: {e}")
                        continue
                
                # If we get here, all cameras failed
                self.camera_error = "All cameras failed to initialize"
                logger.error("All available cameras failed")
                        
            except Exception as e:
                logger.error(f"Camera stream error: {e}")
                self.camera_error = str(e)
            finally:
                self.cap = None
                with self.frame_lock:
                    self.current_frame = None
                    self.annotated_frame = None
        
        if not self.is_running:
            self.is_running = True
            self.stream_active = True
            threading.Thread(target=stream, daemon=True).start()
            
            # Give the camera a moment to initialize
            time.sleep(2)
            
            # Check if camera started successfully
            if self.camera_error:
                logger.error(f"Camera failed to start: {self.camera_error}")
    
    def stop_camera_stream(self):
        """Stop camera streaming"""
        self.is_running = False
        self.stream_active = False
        self.stream_with_recognition = False
        if self.cap:
            self.cap = None
        time.sleep(0.5)  # Give time for threads to clean up
    
    def get_current_frame(self, annotated=False):
        """Get the current frame from camera stream"""
        with self.frame_lock:
            if annotated and self.annotated_frame is not None:
                return self.annotated_frame.copy()
            return self.current_frame.copy() if self.current_frame is not None else None
    
    def get_camera_status(self):
        """Get current camera status"""
        return {
            'active': self.stream_active,
            'camera_id': self.active_camera_id,
            'has_frame': self.current_frame is not None,
            'error': self.camera_error,
            'recognition_enabled': self.stream_with_recognition
        }
    
    def capture_test_frame(self, camera_id=0):
        """Capture a single test frame to verify camera functionality"""
        try:
            with self.camera_context(camera_id) as cap:
                ret, frame = cap.read()
                if ret and frame is not None:
                    # Save test frame
                    test_path = os.path.join(self.CAPTURE_DIR, f"test_frame_{camera_id}.jpg")
                    cv2.imwrite(test_path, frame)
                    logger.info(f"Test frame saved to {test_path}")
                    return frame, None
                else:
                    return None, "Failed to capture frame"
        except Exception as e:
            return None, str(e)
    
    def detect_and_recognize_faces(self, frame):
        """Detect and recognize faces in a frame"""
        if frame is None:
            return []
        
        # Resize frame for faster processing if needed
        if self.is_raspberry_pi:
            processing_frame = cv2.resize(frame, (320, 240))
            scale_x = frame.shape[1] / processing_frame.shape[1]
            scale_y = frame.shape[0] / processing_frame.shape[0]
        else:
            processing_frame = frame.copy()
            scale_x = scale_y = 1.0
        
        gray = cv2.cvtColor(processing_frame, cv2.COLOR_BGR2GRAY)
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
                'confidence': (1 - distance) * 100 if distance < 1.0 else 0,
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
            f"Camera: {self.active_camera_id or 'None'}",
            f"Time: {datetime.now().strftime('%H:%M:%S')}"
        ]
        
        # Background for system info
        cv2.rectangle(frame, (10, 10), (300, 140), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (300, 140), (100, 100, 100), 2)
        
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
        if frame is None:
            return None
        _, buffer = cv2.imencode('.jpg', frame)
        img_str = base64.b64encode(buffer).decode('utf-8')
        return img_str
    
    def generate_video_stream(self, annotated=False):
        """Generate video stream for HTTP streaming"""
        while self.stream_active:
            frame = self.get_current_frame(annotated=annotated)
            if frame is not None:
                try:
                    # Encode frame as JPEG
                    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    frame_bytes = buffer.tobytes()
                    
                    # Yield frame in multipart format
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                except Exception as e:
                    logger.error(f"Frame encoding error: {e}")
            else:
                # Send a black frame or error frame if no camera feed
                black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(black_frame, "Camera Not Available", (50, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                if self.camera_error:
                    cv2.putText(black_frame, f"Error: {self.camera_error}", (50, 280), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                
                _, buffer = cv2.imencode('.jpg', black_frame)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            time.sleep(0.033)  # ~30 FPS
    
    def start_relock_countdown(self):
        """Start the relock countdown (called after initial delay)"""
        # Wait additional 10 seconds before relocking
        self._relock_timer = threading.Timer(10.0, self.relock_door)
        self._relock_timer.start()