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
from numpy.linalg import norm

from app.model import load_facenet_model
from app.face_utils import preprocess_face, get_embedding
from app.db import load_embedding

logger = logging.getLogger(__name__)

class FacialRecognitionAPI:
    def __init__(self):
        print("🚀 Initializing FacialRecognitionAPI...")
        
        # Configuration
        self.MODEL_PATH = "facenet_africain_finetuned.pth"
        self.EMBEDDINGS_DIR = "embeddings"
        self.CAPTURE_DIR = "captures"
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.THRESHOLD = 0.7
        self.DETECTION_DELAY = 3
        self.door_unlock_available_until = 0
        
        print(f"📱 Device: {self.DEVICE}")
        print(f"🎯 Threshold: {self.THRESHOLD}")
        
        # Platform detection
        self.is_raspberry_pi = platform.machine() in ('armv7l', 'armv6l', 'aarch64')
        print(f"🍓 Raspberry Pi detected: {self.is_raspberry_pi}")
        print(f"💻 Platform: {platform.machine()}")
        
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
        print(f"📁 Creating capture directory: {self.CAPTURE_DIR}")
        os.makedirs(self.CAPTURE_DIR, exist_ok=True)
        
        # Load resources
        self.load_resources()
        print("✅ FacialRecognitionAPI initialized successfully!")
    
    def load_resources(self): 
        """Load model and embeddings"""
        try:
            print("🔄 Loading facial recognition resources...")
            
            print(f"🧠 Loading model from: {self.MODEL_PATH}")
            logger.info("Loading facial recognition model...")
            self.model = self.load_model(self.MODEL_PATH)
            print("✅ Model loaded successfully!")
            
            print(f"💾 Loading embeddings from: {self.EMBEDDINGS_DIR}")
            logger.info("Loading saved embeddings...")
            self.saved_embeddings = self.get_all_saved_embeddings()
            print(f"✅ {len(self.saved_embeddings)} embeddings loaded!")
            
            print("👤 Loading face detector...")
            logger.info("Loading face detector...")
            self.face_detector = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            print("✅ Face detector loaded!")
            
            print(f"🎉 System ready - {len(self.saved_embeddings)} known faces loaded")
            logger.info(f"System ready - {len(self.saved_embeddings)} known faces loaded")
            
        except Exception as e:
            print(f"❌ Error loading resources: {str(e)}")
            logger.error(f"Error loading resources: {str(e)}")
            raise
    
    def load_model(self, path):
        """Load the facial recognition model"""
        print(f"🔄 Loading model from path: {path}")
        print(f"📊 Model exists: {os.path.exists(path)}")
        
        model = load_facenet_model()
        print("🏗️ Model architecture loaded")
        
        model.load_state_dict(torch.load(path, map_location=self.DEVICE))
        print(f"⚙️ Model weights loaded to device: {self.DEVICE}")
        
        model.to(self.DEVICE)
        model.eval()
        print("✅ Model set to evaluation mode")
        
        return model
    
    def get_all_saved_embeddings(self):
        """Load all saved embeddings from disk"""
        print(f"🔍 Scanning embeddings directory: {self.EMBEDDINGS_DIR}")
        embeddings = {}
        
        if os.path.exists(self.EMBEDDINGS_DIR):
            files = os.listdir(self.EMBEDDINGS_DIR)
            print(f"📂 Found {len(files)} files in embeddings directory")
            
            for file in files:
                if file.endswith(".pt"):
                    print(f"📥 Loading embedding: {file}")
                    name = file[:-3]
                    embedding = load_embedding(name)
                    embeddings[name] = embedding
                    print(f"✅ Loaded embedding for: {name}")
        else:
            print(f"⚠️ Embeddings directory doesn't exist: {self.EMBEDDINGS_DIR}")
            
        print(f"📊 Total embeddings loaded: {len(embeddings)}")
        return embeddings
    
    @contextmanager
    def camera_context(self, camera_id=0):
        """Context manager for camera operations"""
        print(f"📹 Initializing camera {camera_id}...")
        cap = None
        try:
            if self.is_raspberry_pi:
                print("🍓 Using Raspberry Pi camera configuration")
                cap = cv2.VideoCapture(camera_id)
            else:
                print("💻 Using desktop camera configuration (DSHOW)")
                cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
            
            if not cap.isOpened():
                print(f"❌ Cannot open camera {camera_id}")
                raise Exception(f"Cannot open camera {camera_id}")
            
            print(f"✅ Camera {camera_id} opened successfully")
            
            # Set camera properties
            if self.is_raspberry_pi:
                print("🔧 Setting Raspberry Pi camera properties...")
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 15)
                print("📏 Resolution: 640x480 @ 15fps")
            else:
                print("🔧 Setting desktop camera properties...")
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                cap.set(cv2.CAP_PROP_FPS, 30)
                print("📏 Resolution: 1280x720 @ 30fps")
            
            yield cap
            
        finally:
            if cap:
                print(f"🔒 Releasing camera {camera_id}")
                cap.release()
    
    def start_camera_stream(self, enable_recognition=False):
        """Start continuous camera streaming with optional recognition"""
        print(f"🎬 Starting camera stream (recognition: {enable_recognition})")
        
        def stream():
            try:
                # Try different camera indices
                print("🔍 Searching for available cameras...")
                for camera_id in [0, 1, 2]:
                    print(f"🎯 Trying camera {camera_id}...")
                    try:
                        with self.camera_context(camera_id) as cap:
                            self.cap = cap
                            self.stream_with_recognition = enable_recognition
                            print(f"🎉 Camera stream started on camera {camera_id} (recognition: {enable_recognition})")
                            logger.info(f"Camera stream started on camera {camera_id} (recognition: {enable_recognition})")
                            
                            frame_count = 0
                            while self.is_running:
                                ret, frame = cap.read()
                                if ret:
                                    frame_count += 1
                                    if frame_count % 150 == 0:  # Print every 5 seconds at 30fps
                                        print(f"📸 Frame {frame_count} captured ({frame.shape[1]}x{frame.shape[0]})")
                                    
                                    # Update frame counter for FPS calculation
                                    self.fps_counter += 1
                                    current_time = time.time()
                                    if current_time - self.fps_start_time >= 1.0:
                                        print(f"🎬 FPS: {self.fps_counter}")
                                        self.fps_start_time = current_time
                                        self.fps_counter = 0
                                    
                                    with self.frame_lock:
                                        self.current_frame = frame.copy()
                                        
                                        # Process recognition if enabled
                                        if self.stream_with_recognition:
                                            if frame_count % 30 == 0:  # Print every second
                                                print("🔍 Processing frame for recognition...")
                                            self.annotated_frame = self.annotate_frame_with_recognition(frame.copy())
                                        else:
                                            self.annotated_frame = frame.copy()
                                else:
                                    print("⚠️ Failed to capture frame")
                                            
                                time.sleep(0.033)  # ~30 FPS
                            
                            print("🛑 Camera stream loop ended")
                            break
                            
                    except Exception as e:
                        print(f"❌ Failed to open camera {camera_id}: {e}")
                        logger.warning(f"Failed to open camera {camera_id}: {e}")
                        continue
                else:
                    print("❌ No cameras available")
                    logger.error("No cameras available")
                    
            except Exception as e:
                print(f"❌ Camera stream error: {e}")
                logger.error(f"Camera stream error: {e}")
            finally:
                print("🔄 Cleaning up camera resources...")
                self.cap = None
                with self.frame_lock:
                    self.current_frame = None
                    self.annotated_frame = None
                print("✅ Camera cleanup complete")
        
        if not self.is_running:
            print("▶️ Starting camera stream thread...")
            self.is_running = True
            self.stream_active = True
            threading.Thread(target=stream, daemon=True).start()
        else:
            print("⚠️ Camera stream already running")
    
    def stop_camera_stream(self):
        """Stop camera streaming"""
        print("🛑 Stopping camera stream...")
        self.is_running = False
        self.stream_active = False
        self.stream_with_recognition = False
        if self.cap:
            self.cap = None
        print("✅ Camera stream stopped")
    
    def get_current_frame(self, annotated=False):
        """Get the current frame from camera stream"""
        with self.frame_lock:
            if annotated and self.annotated_frame is not None:
                return self.annotated_frame.copy()
            return self.current_frame.copy() if self.current_frame is not None else None
    
    def detect_and_recognize_faces(self, frame):
        """Detect and recognize faces in a frame"""
        print("🔍 Starting face detection and recognition...")
        
        if frame is None:
            print("❌ No frame provided for face detection")
            return []
        
        print(f"📏 Input frame shape: {frame.shape}")
        
        # Resize frame for faster processing if needed
        if self.is_raspberry_pi:
            print("🍓 Resizing frame for Raspberry Pi processing...")
            processing_frame = cv2.resize(frame, (320, 240))
            scale_x = frame.shape[1] / processing_frame.shape[1]
            scale_y = frame.shape[0] / processing_frame.shape[0]
            print(f"📐 Scale factors: x={scale_x:.2f}, y={scale_y:.2f}")
        else:
            processing_frame = frame.copy()
            scale_x = scale_y = 1.0
        
        print("🔄 Converting to grayscale...")
        gray = cv2.cvtColor(processing_frame, cv2.COLOR_BGR2GRAY)
        
        print("👤 Detecting faces...")
        faces = self.face_detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        print(f"👥 Found {len(faces)} faces")
        
        results = []
        for i, (x, y, w, h) in enumerate(faces):
            print(f"🎯 Processing face {i+1}/{len(faces)} at ({x}, {y}, {w}, {h})")
            
            # Scale back to original frame size
            x, y, w, h = int(x * scale_x), int(y * scale_y), int(w * scale_x), int(h * scale_y)
            print(f"📐 Scaled coordinates: ({x}, {y}, {w}, {h})")
            
            # Recognize face
            print(f"🔍 Recognizing face {i+1}...")
            name, distance = self.recognize_face(frame, (x, y, w, h))
            print(f"🏷️ Face {i+1} result: {name} (distance: {distance:.3f})")
            
            face_result = {
                'bbox': [x, y, w, h],
                'name': name,
                'confidence': (1 - distance) * 100 if distance < 1.0 else 0,
                'distance': distance,
                'recognized': name != "UNKNOWN"
            }
            
            results.append(face_result)
            print(f"✅ Face {i+1} processed: {face_result}")
        
        print(f"🎉 Face detection complete: {len(results)} faces processed")
        return results
    
    def annotate_frame_with_recognition(self, frame):
        """Annotate frame with face recognition results"""
        print("🎨 Annotating frame with recognition results...")
        faces = self.detect_and_recognize_faces(frame)
        
        for i, face in enumerate(faces):
            print(f"🖼️ Annotating face {i+1}/{len(faces)}")
            x, y, w, h = face['bbox']
            name = face['name']
            confidence = face['confidence']
            
            # Choose colors based on recognition
            if face['recognized']:
                box_color = (0, 255, 0)  # Green for recognized
                text_color = (0, 255, 0)
                status = "ACCESS GRANTED"
                self.door_locked = False
                print(f"✅ Access granted for {name}")
            else:
                box_color = (0, 0, 255)  # Red for unknown
                text_color = (0, 0, 255)
                status = "ACCESS DENIED"
                self.door_locked = True
                print(f"❌ Access denied for unknown person")
            
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
        
        print("✅ Frame annotation complete")
        return frame
    
    def recognize_face(self, frame, box):
        """Recognize a face in the given bounding box"""
        print(f"🔍 Recognizing face in bounding box: {box}")
        try:
            x, y, w, h = box
            face_img = frame[y:y+h, x:x+w]
            print(f"👤 Extracted face image shape: {face_img.shape}")
            
            if face_img.size == 0:
                print("❌ Empty face image extracted")
                return "UNKNOWN", 1.0
            
            print("🔄 Converting face to PIL format...")
            face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
            
            print("⚙️ Preprocessing face...")
            face_tensor = preprocess_face(face_pil)
            
            print(f"🧠 Getting embedding using model on {self.DEVICE}...")
            embedding = get_embedding(self.model, face_tensor, self.DEVICE)
            print(f"📊 Embedding shape: {embedding.shape}")
            
            print(f"📏 Calculating distances to {len(self.saved_embeddings)} known faces...")
            distances = {name: self.euclidean_distance(embedding, emb)
                        for name, emb in self.saved_embeddings.items()}
            
            print(f"📊 Distance calculations: {distances}")
            
            if distances:
                best_match = min(distances, key=distances.get)
                best_distance = distances[best_match]
                
                print(f"🎯 Best match: {best_match} (distance: {best_distance:.3f})")
                print(f"🔒 Threshold: {self.THRESHOLD}")
                
                if best_distance < self.THRESHOLD:
                    print(f"✅ Face recognized as: {best_match}")
                    return best_match, best_distance
                else:
                    print(f"❌ Distance {best_distance:.3f} exceeds threshold {self.THRESHOLD}")
            else:
                print("⚠️ No saved embeddings to compare against")
            
            print("❓ Face not recognized - returning UNKNOWN")
            return "UNKNOWN", 1.0
            
        except Exception as e:
            print(f"❌ Recognition error: {e}")
            logger.error(f"Recognition error: {e}")
            return "ERROR", 1.0
    
    def euclidean_distance(self, t1, t2):
        """Calculate Euclidean distance between two tensors"""
        if isinstance(t1, np.ndarray):
            t1 = torch.tensor(t1)
        if isinstance(t2, np.ndarray):
            t2 = torch.tensor(t2)
        distance = torch.norm(t1 - t2).item()
        return distance
    
    def process_access_request(self, recognized_faces):
        """Process access request based on recognized faces"""
        print(f"🚪 Processing access request for {len(recognized_faces)} faces...")
        
        access_granted = False
        recognized_person = None
        
        for i, face in enumerate(recognized_faces):
            print(f"🔍 Evaluating face {i+1}: {face['name']} (confidence: {face['confidence']:.1f}%)")
            
            if face['recognized'] and face['confidence'] > (1 - self.THRESHOLD) * 100:
                print(f"✅ Access criteria met for {face['name']}")
                access_granted = True
                recognized_person = face['name']
                self.unlock_door(face['name'], face['distance'])
                break
            else:
                print(f"❌ Access criteria not met for {face['name']}")
        
        if not access_granted and recognized_faces:
            print("🚫 No valid faces found - denying access")
            self.deny_access()
        
        result = {
            'access_granted': access_granted,
            'person': recognized_person,
            'faces_detected': len(recognized_faces),
            'recognized_faces': [f for f in recognized_faces if f['recognized']],
            'unlock_window_valid': self.is_unlock_window_valid()
        }
        
        print(f"🎉 Access request processed: {result}")
        return result
    
    def unlock_door(self, name, distance):
        """Unlock the door for recognized person"""
        print(f"🔓 UNLOCKING DOOR for {name}!")
        
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
        
        print(f"📝 Access logged for {name}")
        logger.info(f"Access granted to {name}")
        
        # Set a timestamp for when the unlock window expires (15 seconds total)
        self.door_unlock_available_until = time.time() + 15
        print(f"⏰ Door unlock window valid until: {datetime.fromtimestamp(self.door_unlock_available_until)}")
        
        # Cancel any existing auto-relock timer and start a new one
        if hasattr(self, '_relock_timer') and self._relock_timer:
            print("⏰ Cancelling existing relock timer")
            self._relock_timer.cancel()
        
        # Wait 5 seconds before starting the relock countdown
        # This gives the person time to enter
        print("⏰ Starting initial 5-second entry window...")
        self._relock_timer = threading.Timer(5.0, self.start_relock_countdown)
        self._relock_timer.start()
        
    def deny_access(self):
        """Deny access for unknown person"""
        print("🚫 ACCESS DENIED - Unknown person")
        
        self.last_recognition = {
            'name': 'UNKNOWN',
            'timestamp': datetime.now().isoformat(),
            'confidence': 0,
            'status': 'DENIED'
        }
        
        # Log the denied access
        self.log_access("UNKNOWN", "DENIED", 1.0)
        
        print("📝 Access denial logged")
        logger.info("Access denied - unknown person")
    
    def relock_door(self):
        """Relock the door after a timeout"""
        print("🔒 DOOR AUTOMATICALLY RELOCKED!")
        self.door_locked = True
        self.door_unlock_available_until = 0  # Clear any remaining unlock window
        if hasattr(self, '_relock_timer'):
            self._relock_timer = None
        logger.info("Door automatically relocked")
        
    def is_unlock_window_valid(self):
        """Check if the unlock window is still valid"""
        valid = time.time() <= self.door_unlock_available_until
        if self.door_unlock_available_until > 0:
            remaining = max(0, self.door_unlock_available_until - time.time())
            print(f"⏰ Unlock window valid: {valid} ({remaining:.1f}s remaining)")
        return valid
    
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
        
        print(f"📝 Logging access: {log_entry}")
        self.detection_history.append(log_entry)
        
        # Save to log file
        try:
            with open("access_log.txt", "a") as f:
                f.write(f"[{timestamp}] {name} - {status} - {confidence}\n")
            print("✅ Access logged to file")
        except Exception as e:
            print(f"❌ Failed to write to log file: {e}")
    
    def frame_to_base64(self, frame):
        """Convert frame to base64 string"""
        print("🔄 Converting frame to base64...")
        _, buffer = cv2.imencode('.jpg', frame)
        img_str = base64.b64encode(buffer).decode('utf-8')
        print(f"✅ Frame converted to base64 ({len(img_str)} chars)")
        return img_str
    
    def generate_video_stream(self, annotated=False):
        """Generate video stream for HTTP streaming"""
        print(f"🎬 Starting video stream generator (annotated: {annotated})")
        stream_count = 0
        
        while self.stream_active:
            frame = self.get_current_frame(annotated=annotated)
            if frame is not None:
                stream_count += 1
                if stream_count % 150 == 0:  # Print every 5 seconds at 30fps
                    print(f"📺 Streaming frame {stream_count}")
                
                # Encode frame as JPEG
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                frame_bytes = buffer.tobytes()
                
                # Yield frame in multipart format
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            else:
                if stream_count % 30 == 0:  # Print warning every second
                    print("⚠️ No frame available for streaming")
            
            time.sleep(0.033)  # ~30 FPS
        
        print("🛑 Video stream generator stopped")
    
    def start_relock_countdown(self):
        """Start the relock countdown (called after initial delay)"""
        print("⏰ Starting 10-second relock countdown...")
        # Wait additional 10 seconds before relocking
        self._relock_timer = threading.Timer(10.0, self.relock_door)
        self._relock_timer.start()
        print("✅ Relock timer started")