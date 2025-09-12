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
import tempfile
import pyttsx3  # Simple offline TTS library
import paho.mqtt.client as mqtt  # Add MQTT client
import json  # For JSON serialization

from app.model import load_facenet_model
from app.face_utils import preprocess_face, get_embedding
from app.db import load_embedding

logger = logging.getLogger(__name__)
import gc
import psutil
import threading

class FacialRecognitionAPI:
    def __init__(self):
        logger.info("Initializing FacialRecognitionAPI...")
        
        # Configuration
        self.MODEL_PATH = "facenet_africain_finetuned.pth"
        self.EMBEDDINGS_DIR = "embeddings"
        self.CAPTURE_DIR = "captures"
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.THRESHOLD = 0.7
        self.DETECTION_DELAY = 3
        self.door_unlock_available_until = 0
        
        # MQTT Configuration
        self.MQTT_BROKER = "de7ccf69e70f4ffea025b868e4f08223.s1.eu.hivemq.cloud"  # HiveMQ public broker
        self.MQTT_PORT = 8883
        self.MQTT_TOPIC_IMAGE = "facial_recognition/images"
        self.MQTT_TOPIC_RESULTS = "facial_recognition/results"
        self.MQTT_CLIENT_ID = f"facial_recognition_{int(time.time())}"
        self.MQTT_QOS = 1  # At least once delivery
        
        # Audio settings
        self.ENABLE_AUDIO = True
        self.AUDIO_LANGUAGE = 'fr'  # French by default
        
        logger.info(f"Device: {self.DEVICE}, Threshold: {self.THRESHOLD}")
        
        # Platform detection
        self.is_raspberry_pi = platform.machine() in ('armv7l', 'armv6l', 'aarch64')
        logger.info(f"Raspberry Pi detected: {self.is_raspberry_pi}, Platform: {platform.machine()}")
        
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
        self.debug_mode = False  # Control verbose logging
        
        # MQTT client
        self.mqtt_client = None
        self.mqtt_connected = False
        
        # Camera fallback configuration
        self.camera_configs = [
            {'id': 0, 'backend': cv2.CAP_DSHOW if not self.is_raspberry_pi else cv2.CAP_V4L2},
            {'id': 1, 'backend': cv2.CAP_DSHOW if not self.is_raspberry_pi else cv2.CAP_V4L2},
            {'id': 0, 'backend': cv2.CAP_ANY},
            {'id': 1, 'backend': cv2.CAP_ANY},
            {'id': 2, 'backend': cv2.CAP_ANY},
        ]
        self._timer_lock = threading.Lock()
        self._last_mqtt_image = 0
        self._last_resource_check = 0
        
        # Create directories
        os.makedirs(self.CAPTURE_DIR, exist_ok=True)
        
        # Initialize audio system
        self._init_audio_system()
        
        # Initialize MQTT
        self._init_mqtt()
        
        # Load resources
        self.load_resources()
        logger.info("FacialRecognitionAPI initialized successfully!")
    
    def _init_mqtt(self):
        """Initialize MQTT client and connection"""
        try:
            self.mqtt_client = mqtt.Client(client_id=self.MQTT_CLIENT_ID)
            self.mqtt_client.on_connect = self._on_mqtt_connect
            self.mqtt_client.on_disconnect = self._on_mqtt_disconnect
            
            # Connect to broker
            self.mqtt_client.connect(self.MQTT_BROKER, self.MQTT_PORT, keepalive=60)
            self.mqtt_client.loop_start()
            logger.info(f"MQTT client initialized, connecting to {self.MQTT_BROKER}:{self.MQTT_PORT}")
            
        except Exception as e:
            logger.error(f"Failed to initialize MQTT client: {e}")
            self.mqtt_client = None
    
    def _on_mqtt_connect(self, client, userdata, flags, rc):
        """Callback for when MQTT client connects"""
        if rc == 0:
            self.mqtt_connected = True
            logger.info("Connected to MQTT broker successfully")
        else:
            self.mqtt_connected = False
            logger.error(f"Failed to connect to MQTT broker, return code: {rc}")
    
    def _on_mqtt_disconnect(self, client, userdata, rc):
        """Callback for when MQTT client disconnects"""
        self.mqtt_connected = False
        if rc != 0:
            logger.warning(f"Unexpected MQTT disconnection, return code: {rc}")
        else:
            logger.info("Disconnected from MQTT broker")
    
    def _publish_mqtt_message(self, topic, payload, retain=False):
        """Publish a message to MQTT broker"""
        if not self.mqtt_connected or self.mqtt_client is None:
            logger.warning("MQTT client not connected, cannot publish message")
            return False
        
        try:
            result = self.mqtt_client.publish(
                topic, 
                payload, 
                qos=self.MQTT_QOS, 
                retain=retain
            )
            
            # Wait for the message to be published
            result.wait_for_publish(timeout=2.0)
            
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                self._debug_log(f"Message published to {topic}")
                return True
            else:
                logger.error(f"Failed to publish message to {topic}, return code: {result.rc}")
                return False
                
        except Exception as e:
            logger.error(f"Error publishing MQTT message: {e}")
            return False
    
    
    def _publish_recognition_results(self, results):
        """Publish recognition results to MQTT"""
        try:
            # Prepare message payload
            timestamp = datetime.now().isoformat()
            message = {
                "timestamp": timestamp,
                "results": results,
                "device_id": self.MQTT_CLIENT_ID
            }
            
            # Convert to JSON
            payload = json.dumps(message)
            
            # Publish to MQTT
            success = self._publish_mqtt_message(self.MQTT_TOPIC_RESULTS, payload)
            
            if success:
                self._debug_log("Recognition results published to MQTT")
            else:
                logger.warning("Failed to publish recognition results to MQTT")
                
        except Exception as e:
            logger.error(f"Error preparing MQTT results message: {e}")

    def _publish_processed_image(self, frame, recognition_results):
        """CORRECTED: Memory-safe MQTT image publishing with rate limiting"""
        try:
            # Rate limiting - only publish every 2 seconds minimum
            current_time = time.time()
            if current_time - self._last_mqtt_image < 2.0:
                return False
            
            self._last_mqtt_image = current_time
            
            # Compress image to reduce memory usage
            compressed_frame = cv2.resize(frame, (640, 480))
            img_base64 = self.frame_to_base64(compressed_frame)
            
            message = {
                "timestamp": datetime.now().isoformat(),
                "image": img_base64,
                "results": recognition_results,
                "device_id": self.MQTT_CLIENT_ID
            }
            
            payload = json.dumps(message)
            success = self._publish_mqtt_message(self.MQTT_TOPIC_IMAGE, payload)
            
            # Immediate cleanup
            del img_base64, message, payload, compressed_frame
            
            return success
            
        except Exception as e:
            logger.error(f"Error in MQTT image publish: {e}")
            return False
    def _init_audio_system(self):
        """Initialize pyttsx3 for audio playback"""
        try:
            self.tts_engine = pyttsx3.init()
            
            # Set voice properties
            voices = self.tts_engine.getProperty('voices')
            
            # Try to find a French voice if available
            if self.AUDIO_LANGUAGE == 'fr':
                for voice in voices:
                    if 'french' in voice.name.lower() or 'fr' in voice.id.lower():
                        self.tts_engine.setProperty('voice', voice.id)
                        break
            
            # Set speech rate (words per minute)
            self.tts_engine.setProperty('rate', 150)
            
            # Set volume (0.0 to 1.0)
            self.tts_engine.setProperty('volume', 0.9)
            
            logger.info("Audio system initialized with pyttsx3")
        except Exception as e:
            logger.warning(f"Failed to initialize audio system: {e}")
            self.ENABLE_AUDIO = False
    
    def speak_message(self, text, language=None):
        """Convert text to speech and play it"""
        if not self.ENABLE_AUDIO:
            return
            
        try:
            # Run TTS in a thread to avoid blocking
            def speak():
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
            
            threading.Thread(target=speak, daemon=True).start()
            
        except Exception as e:
            logger.error(f"Text-to-speech error: {e}")
    
    def _speak_access_message(self, name, granted):
        """Speak access message in appropriate language"""
        if granted:
            if name in ['UNKNOWN', 'ERROR']:
                message = "Accès autorisé"
            else:
                message = f"Bonjour {name}, accès autorisé"
        else:
            message = "Accès refusé, personne non reconnue"
        
        # Play message
        self.speak_message(message)
    
    def set_debug_mode(self, enabled):
        """Enable/disable debug mode for verbose logging"""
        self.debug_mode = enabled
        logger.info(f"Debug mode {'enabled' if enabled else 'disabled'}")
    
    def _debug_log(self, message):
        """Log debug messages only if debug mode is enabled"""
        if self.debug_mode:
            print(f"[DEBUG] {message}")
    
    def load_resources(self): 
        """Load model and embeddings"""
        try:
            logger.info("Loading facial recognition resources...")
            
            logger.info(f"Loading model from: {self.MODEL_PATH}")
            self.model = self.load_model(self.MODEL_PATH)
            
            logger.info(f"Loading embeddings from: {self.EMBEDDINGS_DIR}")
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
        self._debug_log(f"Loading model from path: {path}")
        self._debug_log(f"Model exists: {os.path.exists(path)}")
        
        model = load_facenet_model()
        model.load_state_dict(torch.load(path, map_location=self.DEVICE))
        model.to(self.DEVICE)
        model.eval()
        
        logger.info("Model loaded and set to evaluation mode")
        return model
    
    def get_all_saved_embeddings(self):
        """Load all saved embeddings from disk"""
        embeddings = {}
        
        if os.path.exists(self.EMBEDDINGS_DIR):
            files = [f for f in os.listdir(self.EMBEDDINGS_DIR) if f.endswith(".pt")]
            logger.info(f"Found {len(files)} embedding files")
            
            for file in files:
                name = file[:-3]
                embedding = load_embedding(name)
                embeddings[name] = embedding
                self._debug_log(f"Loaded embedding for: {name}")
        else:
            logger.warning(f"Embeddings directory doesn't exist: {self.EMBEDDINGS_DIR}")
            
        logger.info(f"Total embeddings loaded: {len(embeddings)}")
        return embeddings
    
    def _try_camera_config(self, config):
        """Try to open camera with specific configuration"""
        try:
            camera_id = config['id']
            backend = config['backend']
            
            self._debug_log(f"Trying camera {camera_id} with backend {backend}")
            
            # Add timeout and retry parameters
            cap = cv2.VideoCapture(camera_id, backend)
            
            # Set timeout properties if supported
            if hasattr(cv2, 'CAP_PROP_OPEN_TIMEOUT_MSEC'):
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)  # 5 second timeout
            
            if not cap.isOpened():
                return None
            
            # Test if we can read multiple frames to ensure stability
            successful_frames = 0
            for _ in range(5):  # Try to read 5 frames
                ret, frame = cap.read()
                if ret and frame is not None:
                    successful_frames += 1
                time.sleep(0.1)  # Small delay between tests
            
            if successful_frames < 3:  # Require at least 3 successful frames
                cap.release()
                return None
            
            # Set camera properties with error handling
            try:
                if self.is_raspberry_pi:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    cap.set(cv2.CAP_PROP_FPS, 15)
                else:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    cap.set(cv2.CAP_PROP_FPS, 30)
            except Exception as prop_error:
                self._debug_log(f"Could not set all camera properties: {prop_error}")
                # Continue anyway - some properties might be set
            
            logger.info(f"Camera {camera_id} opened successfully with backend {backend}")
            return cap
            
        except Exception as e:
            self._debug_log(f"Failed to open camera {config['id']} with backend {config['backend']}: {e}")
            return None
    @contextmanager
    def camera_context(self):
        """Context manager for camera operations with fallback"""
        cap = None
        successful_config = None
        
        try:
            # Try each camera configuration
            for config in self.camera_configs:
                cap = self._try_camera_config(config)
                if cap is not None:
                    successful_config = config
                    break
            
            if cap is None:
                raise Exception("No cameras available with any configuration")
            
            logger.info(f"Using camera {successful_config['id']} with backend {successful_config['backend']}")
            yield cap
            
        finally:
            if cap:
                self._debug_log(f"Releasing camera {successful_config['id'] if successful_config else 'unknown'}")
                cap.release()
    
    def start_camera_stream(self, enable_recognition=False):
        """CORRECTED: Memory-safe camera streaming"""
        logger.info(f"Starting camera stream (recognition: {enable_recognition})")
        
        def stream():
            try:
                with self.camera_context() as cap:
                    self.cap = cap
                    self.stream_with_recognition = enable_recognition
                    
                    frame_count = 0
                    consecutive_failures = 0
                    max_consecutive_failures = 10
                    last_fps_log = time.time()
                    last_mqtt_publish = 0
                    mqtt_publish_interval = 2
                    
                    while self.is_running:
                        ret, frame = cap.read()
                        if ret and frame is not None:
                            consecutive_failures = 0
                            frame_count += 1
                            
                            # Resource monitoring every 50 frames
                            if frame_count % 50 == 0:
                                self._monitor_system_resources()
                            
                            # Garbage collection every 100 frames
                            if frame_count % 100 == 0:
                                gc.collect()
                            
                            current_time = time.time()
                            if current_time - last_fps_log >= 5.0:
                                fps = frame_count / (current_time - self.fps_start_time + 0.001)
                                logger.info(f"Camera FPS: {fps:.1f}, Frame: {frame_count}")
                                last_fps_log = current_time
                            
                            with self.frame_lock:
                                # Clean up previous frames
                                if hasattr(self, 'current_frame') and self.current_frame is not None:
                                    del self.current_frame
                                if hasattr(self, 'annotated_frame') and self.annotated_frame is not None:
                                    del self.annotated_frame
                                
                                self.current_frame = frame.copy()
                                
                                if self.stream_with_recognition:
                                    annotated_frame, recognition_results = self.annotate_frame_with_recognition(frame.copy())
                                    self.annotated_frame = annotated_frame
                                    
                                    # Publish to MQTT at controlled intervals
                                    if current_time - last_mqtt_publish >= mqtt_publish_interval:
                                        self._publish_processed_image(annotated_frame, recognition_results)
                                        last_mqtt_publish = current_time
                                    
                                    # Clean up local reference
                                    del annotated_frame
                                else:
                                    self.annotated_frame = frame.copy()
                        else:
                            consecutive_failures += 1
                            if consecutive_failures % 3 == 0:
                                healthy, message = self.check_camera_health()
                                if not healthy:
                                    logger.warning(f"Camera health check failed: {message}")
                            
                            if consecutive_failures >= max_consecutive_failures:
                                logger.error("Too many consecutive failures, restarting camera...")
                                break
                        
                        time.sleep(0.033)  # ~30 FPS
                        
            except Exception as e:
                    logger.error(f"Camera stream error: {e}")
            finally:
                    self._cleanup_camera_resources()
                    
            if not self.is_running:
                self.is_running = True
                self.stream_active = True
                threading.Thread(target=stream, daemon=True).start() 
                  
    def reinitialize_camera(self):
        """Attempt to reinitialize the camera connection"""
        logger.info("Attempting to reinitialize camera...")
        
        # Clean up existing resources
        self._cleanup_camera_resources()
        
        # Try to reopen camera
        try:
            with self.camera_context() as cap:
                self.cap = cap
                logger.info("Camera reinitialized successfully")
                return True
        except Exception as e:
            logger.error(f"Failed to reinitialize camera: {e}")
            return False
        
    def check_camera_health(self):
        try:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                return False, "Failed to read frame"
            
            if frame.size == 0:
                del frame  # Clean up
                return False, "Empty frame"
            
            del frame  # Always clean up
            return True, "Camera is healthy"
        except Exception as e:
            return False, f"Health check error: {e}"
    
    def _cleanup_camera_resources(self):
        """Clean up camera resources"""
        self._debug_log("Cleaning up camera resources...")
        self.cap = None
        with self.frame_lock:
            self.current_frame = None
            self.annotated_frame = None
        logger.info("Camera cleanup complete")
    
    def stop_camera_stream(self):
        """Stop camera streaming"""
        logger.info("Stopping camera stream...")
        self.is_running = False
        self.stream_active = False
        self.stream_with_recognition = False
        if self.cap:
            self.cap = None
        logger.info("Camera stream stopped")
    
    def get_current_frame(self, annotated=False):
        """Get the current frame from camera stream"""
        with self.frame_lock:
            if annotated and self.annotated_frame is not None:
                return self.annotated_frame.copy()
            return self.current_frame.copy() if self.current_frame is not None else None
    
    def detect_and_recognize_faces(self, frame):
        """CORRECTED: Memory-safe face detection"""
        processing_frame = None
        gray = None
        
        try:
            if frame is None:
                return []
            
            # Resize for processing
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
            for i, (x, y, w, h) in enumerate(faces):
                # Scale back to original frame size
                x, y, w, h = int(x * scale_x), int(y * scale_y), int(w * scale_x), int(h * scale_y)
                
                # Recognize face
                name, distance = self.recognize_face(frame, (x, y, w, h))
                
                face_result = {
                    'bbox': [x, y, w, h],
                    'name': name,
                    'confidence': (1 - distance) * 100 if distance < 1.0 else 0,
                    'distance': distance,
                    'recognized': name != "UNKNOWN"
                }
                results.append(face_result)
            
            return results
            
        finally:
            # Always clean up temporary arrays
            if processing_frame is not None:
                del processing_frame
            if gray is not None:
                del gray
            
    def annotate_frame_with_recognition(self, frame):
        """Annotate frame with face recognition results and return both frame and results"""
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
                if self.door_locked:  # Only speak when status changes
                    
                    self._speak_access_message(name, True)
            else:
                box_color = (0, 0, 255)  # Red for unknown
                text_color = (0, 0, 255)
                status = "ACCESS DENIED"
                self.door_locked = True
                if not self.door_locked:  # Only speak when status changes
                    
                    self._speak_access_message(name, False)
            
            # Draw main bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 3)
            
            # Draw corner accents
            corner_length = 20
            corners = [
                ((x, y), (x + corner_length, y), (x, y + corner_length)),
                ((x + w, y), (x + w - corner_length, y), (x + w, y + corner_length)),
                ((x, y + h), (x + corner_length, y + h), (x, y + h - corner_length)),
                ((x + w, y + h), (x + w - corner_length, y + h), (x + w, y + h - corner_length))
            ]
            
            for corner in corners:
                cv2.line(frame, corner[0], corner[1], box_color, 5)
                cv2.line(frame, corner[0], corner[2], box_color, 5)
            
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
        
        return frame, faces
    
    def recognize_face(self, frame, box):
        """Recognize a face in the given bounding box"""
        try:
            x, y, w, h = box
            face_img = frame[y:y+h, x:x+w]
            
            if face_img.size == 0:
                logger.warning("Empty face image extracted")
                return "UNKNOWN", 1.0
            
            face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
            face_tensor = preprocess_face(face_pil)
            embedding = get_embedding(self.model, face_tensor, self.DEVICE)
            
            distances = {name: self.euclidean_distance(embedding, emb)
                        for name, emb in self.saved_embeddings.items()}
            
            if distances:
                best_match = min(distances, key=distances.get)
                best_distance = distances[best_match]
                
                self._debug_log(f"Best match: {best_match} (distance: {best_distance:.3f})")
                
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
        distance = torch.norm(t1 - t2).item()
        return distance
    
    def process_access_request(self, recognized_faces):
        """Process access request based on recognized faces"""
        logger.info(f"Processing access request for {len(recognized_faces)} faces...")
        
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
        
        result = {
            'access_granted': access_granted,
            'person': recognized_person,
            'faces_detected': len(recognized_faces),
            'recognized_faces': [f for f in recognized_faces if f['recognized']],
            'unlock_window_valid': self.is_unlock_window_valid()
        }
        
        # Publish results to MQTT
        self._publish_recognition_results(result)
        
        logger.info(f"Access request result: {result}")
        return result
    
    def unlock_door(self, name, distance):
        """CORRECTED: Thread-safe door unlocking with proper timer management"""
        logger.info(f"UNLOCKING DOOR for {name}!")
        
        self.door_locked = False
        self.last_recognition = {
            'name': name,
            'timestamp': datetime.now().isoformat(),
            'confidence': (1 - distance) * 100,
            'status': 'GRANTED'
        }
        
        self.log_access(name, "GRANTED", distance)
        self._speak_access_message(name, True)
        self.door_unlock_available_until = time.time() + 15
        
        # Thread-safe timer management
        with self._timer_lock:
            if hasattr(self, '_relock_timer') and self._relock_timer:
                self._relock_timer.cancel()
                # Wait for timer to finish (non-blocking)
                try:
                    self._relock_timer.join(timeout=0.5)
                except:
                    pass
            
            self._relock_timer = threading.Timer(5.0, self.start_relock_countdown)
            self._relock_timer.daemon = True  # Prevent hanging on exit
            self._relock_timer.start()
            
    def _cleanup_camera_resources(self):
        """CORRECTED: Enhanced camera cleanup"""
        logger.info("Cleaning up camera resources...")
        self.cap = None
        
        with self.frame_lock:
            if hasattr(self, 'current_frame') and self.current_frame is not None:
                del self.current_frame
                self.current_frame = None
            if hasattr(self, 'annotated_frame') and self.annotated_frame is not None:
                del self.annotated_frame
                self.annotated_frame = None
        
        # Force garbage collection
        gc.collect()
        logger.info("Camera cleanup complete")

    def _monitor_system_resources(self):
        """NEW: Monitor system resources and take action if needed"""
        try:
            current_time = time.time()
            if current_time - self._last_resource_check < 5.0:  # Check every 5 seconds max
                return
            
            self._last_resource_check = current_time
            
            memory_percent = psutil.virtual_memory().percent
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            if memory_percent > 85:
                logger.warning(f"High memory usage: {memory_percent}% - forcing cleanup")
                gc.collect()
                
                # If still high, clear some buffers
                if psutil.virtual_memory().percent > 90:
                    with self.frame_lock:
                        if hasattr(self, 'current_frame'):
                            self.current_frame = None
                        if hasattr(self, 'annotated_frame'):
                            self.annotated_frame = None
                    gc.collect()
            
            if cpu_percent > 95:
                logger.warning(f"High CPU usage: {cpu_percent}%")
                time.sleep(0.1)  # Brief pause to let system recover
                
        except Exception as e:
            logger.error(f"Resource monitoring error: {e}")

    def shutdown(self):
        """NEW: Graceful shutdown with complete cleanup"""
        logger.info("Initiating graceful shutdown...")
        
        # Stop camera stream
        self.is_running = False
        self.stream_active = False
        
        # Wait a moment for threads to stop
        time.sleep(1.0)
        
        # Cancel timers
        with self._timer_lock:
            if hasattr(self, '_relock_timer') and self._relock_timer:
                self._relock_timer.cancel()
        
        # Disconnect MQTT
        if self.mqtt_client:
            try:
                self.mqtt_client.loop_stop()
                self.mqtt_client.disconnect()
            except:
                pass
        
        # Final cleanup
        self._cleanup_camera_resources()
        
        # Force final garbage collection
        gc.collect()
        
        logger.info("Shutdown complete")

    def get_current_frame(self, annotated=False):
        """CORRECTED: Safe frame retrieval with memory management"""
        with self.frame_lock:
            try:
                if annotated and hasattr(self, 'annotated_frame') and self.annotated_frame is not None:
                    return self.annotated_frame.copy()
                elif hasattr(self, 'current_frame') and self.current_frame is not None:
                    return self.current_frame.copy()
                else:
                    return None
            except Exception as e:
                logger.error(f"Error getting frame: {e}")
                return None       
    def deny_access(self):
        """Deny access for unknown person"""
        logger.info("ACCESS DENIED - Unknown person")
        
        self.last_recognition = {
            'name': 'UNKNOWN',
            'timestamp': datetime.now().isoformat(),
            'confidence': 0,
            'status': 'DENIED'
        }
        
        # Log the denied access
        self.log_access("UNKNOWN", "DENIED", 1.0)
        
        # Speak denial message
        self._speak_access_message("UNKNOWN", False)
    
    def relock_door(self):
        """Relock the door after a timeout"""
        logger.info("DOOR AUTOMATICALLY RELOCKED!")
        self.door_locked = True
        self.door_unlock_available_until = 0
        if hasattr(self, '_relock_timer'):
            self._relock_timer = None
        
        # Speak relock message
        self.speak_message("Porte verrouillée")
        
    def is_unlock_window_valid(self):
        """Check if the unlock window is still valid"""
        valid = time.time() <= self.door_unlock_available_until
        if self.door_unlock_available_until > 0:
            remaining = max(0, self.door_unlock_available_until - time.time())
            self._debug_log(f"Unlock window valid: {valid} ({remaining:.1f}s remaining)")
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
        
        self.detection_history.append(log_entry)
        
        # Save to log file
        try:
            with open("access_log.txt", "a") as f:
                f.write(f"[{timestamp}] {name} - {status} - {confidence}\n")
            self._debug_log("Access logged to file")
        except Exception as e:
            logger.error(f"Failed to write to log file: {e}")
    
    def frame_to_base64(self, frame):
        """Convert frame to base64 string"""
        _, buffer = cv2.imencode('.jpg', frame)
        img_str = base64.b64encode(buffer).decode('utf-8')
        return img_str
    
    def generate_video_stream(self, annotated=False):
        """Generate video stream for HTTP streaming"""
        logger.info(f"Starting video stream generator (annotated: {annotated})")
        stream_count = 0
        last_log_time = time.time()
        
        while self.stream_active:
            frame = self.get_current_frame(annotated=annotated)
            if frame is not None:
                stream_count += 1
                
                # Log streaming status every 10 seconds
                current_time = time.time()
                if current_time - last_log_time >= 10.0:
                    logger.info(f"Streaming frame {stream_count}")
                    last_log_time = current_time
                
                # Encode frame as JPEG
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                frame_bytes = buffer.tobytes()
                
                # Yield frame in multipart format
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            else:
                # Only log occasionally when no frame is available
                if stream_count % 30 == 0:
                    logger.warning("No frame available for streaming")
            
            time.sleep(0.033)  # ~30 FPS
        
        logger.info("Video stream generator stopped")
    
    def start_relock_countdown(self):
        """Start the relock countdown (called after initial delay)"""
        logger.info("Starting 10-second relock countdown...")
        # Wait additional 10 seconds before relocking
        self._relock_timer = threading.Timer(10.0, self.relock_door)
        self._relock_timer.start()
    
    def set_audio_language(self, language):
        """Set the language for audio messages"""
        self.AUDIO_LANGUAGE = language
        logger.info(f"Audio language set to: {language}")
    
    def toggle_audio(self, enabled):
        """Enable or disable audio messages"""
        self.ENABLE_AUDIO = enabled
        logger.info(f"Audio {'enabled' if enabled else 'disabled'}")
    
    def disconnect_mqtt(self):
        """Disconnect from MQTT broker"""
        if self.mqtt_client:
            self.mqtt_client.loop_stop()
            self.mqtt_client.disconnect()
            logger.info("MQTT client disconnected")