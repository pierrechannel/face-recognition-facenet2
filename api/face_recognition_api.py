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
    
    def _publish_processed_image(self, frame, recognition_results):
        """Publish processed image and results to MQTT"""
        if frame is None:
            return
        
        try:
            # Convert frame to base64
            img_base64 = self.frame_to_base64(frame)
            
            # Prepare message payload
            timestamp = datetime.now().isoformat()
            message = {
                "timestamp": timestamp,
                "image": img_base64,
                "results": recognition_results,
                "device_id": self.MQTT_CLIENT_ID
            }
            
            # Convert to JSON
            payload = json.dumps(message)
            
            # Publish to MQTT
            success = self._publish_mqtt_message(self.MQTT_TOPIC_IMAGE, payload)
            
            if success:
                self._debug_log("Processed image published to MQTT")
            else:
                logger.warning("Failed to publish processed image to MQTT")
                
        except Exception as e:
            logger.error(f"Error preparing MQTT image message: {e}")
    
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
            
            cap = cv2.VideoCapture(camera_id, backend)
            
            if not cap.isOpened():
                return None
            
            # Test if we can read a frame
            ret, frame = cap.read()
            if not ret or frame is None:
                cap.release()
                return None
            
            # Set camera properties
            if self.is_raspberry_pi:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 15)
            else:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                cap.set(cv2.CAP_PROP_FPS, 30)
            
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
        """Start continuous camera streaming with optional recognition"""
        logger.info(f"Starting camera stream (recognition: {enable_recognition})")
        
        def stream():
            try:
                with self.camera_context() as cap:
                    self.cap = cap
                    self.stream_with_recognition = enable_recognition
                    logger.info(f"Camera stream started (recognition: {enable_recognition})")
                    
                    frame_count = 0
                    last_fps_log = time.time()
                    last_mqtt_publish = 0
                    mqtt_publish_interval = 2  # Publish to MQTT every 2 seconds
                    
                    while self.is_running:
                        ret, frame = cap.read()
                        if ret:
                            frame_count += 1
                            
                            # Log FPS every 5 seconds
                            current_time = time.time()
                            if current_time - last_fps_log >= 5.0:
                                fps = frame_count / (current_time - self.fps_start_time + 0.001)
                                logger.info(f"Camera FPS: {fps:.1f}, Frame: {frame_count}")
                                last_fps_log = current_time
                            
                            with self.frame_lock:
                                self.current_frame = frame.copy()
                                
                                # Process recognition if enabled
                                if self.stream_with_recognition:
                                    annotated_frame, recognition_results = self.annotate_frame_with_recognition(frame.copy())
                                    self.annotated_frame = annotated_frame
                                    
                                    # Publish to MQTT at regular intervals
                                    if current_time - last_mqtt_publish >= mqtt_publish_interval:
                                        self._publish_processed_image(annotated_frame, recognition_results)
                                        last_mqtt_publish = current_time
                                else:
                                    self.annotated_frame = frame.copy()
                        else:
                            logger.warning("Failed to capture frame")
                                    
                        time.sleep(0.033)  # ~30 FPS
                    
                    logger.info("Camera stream loop ended")
                    
            except Exception as e:
                logger.error(f"Camera stream error: {e}")
            finally:
                self._cleanup_camera_resources()
        
        if not self.is_running:
            logger.info("Starting camera stream thread...")
            self.is_running = True
            self.stream_active = True
            threading.Thread(target=stream, daemon=True).start()
        else:
            logger.warning("Camera stream already running")
    
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
        """Detect and recognize faces in a frame"""
        self._debug_log("Starting face detection and recognition...")
        
        if frame is None:
            logger.warning("No frame provided for face detection")
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
        
        self._debug_log(f"Found {len(faces)} faces")
        
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
            self._debug_log(f"Face {i+1} processed: {name} (confidence: {face_result['confidence']:.1f}%)")
        
        if len(results) > 0:
            logger.info(f"Face detection complete: {len(results)} faces processed")
        
        return results
    
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
                if self.door_locked:  # Only speak when status changes
                    self.door_locked = False
                    self._speak_access_message(name, True)
            else:
                box_color = (0, 0, 255)  # Red for unknown
                text_color = (0, 0, 255)
                status = "ACCESS DENIED"
                if not self.door_locked:  # Only speak when status changes
                    self.door_locked = True
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
        """Unlock the door for recognized person"""
        logger.info(f"UNLOCKING DOOR for {name}!")
        
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
        
        # Speak welcome message
        self._speak_access_message(name, True)
        
        # Set a timestamp for when the unlock window expires (15 seconds total)
        self.door_unlock_available_until = time.time() + 15
        
        # Cancel any existing auto-relock timer and start a new one
        if hasattr(self, '_relock_timer') and self._relock_timer:
            self._relock_timer.cancel()
        
        # Wait 5 seconds before starting the relock countdown
        self._relock_timer = threading.Timer(5.0, self.start_relock_countdown)
        self._relock_timer.start()
        
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