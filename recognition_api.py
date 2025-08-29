import os
import cv2
import torch
import time
import numpy as np
from flask import Flask, request, jsonify, Response
from PIL import Image
from datetime import datetime
import threading
from collections import deque
import platform
import base64
import io
import json
from werkzeug.serving import make_server
import logging
from contextlib import contextmanager
from flask_cors import CORS

from app.model import load_facenet_model
from app.face_utils import preprocess_face, get_embedding
from app.db import load_embedding

app = Flask(__name__)
CORS(app)  # Enable CORS for web clients

# Configure logging
logging.basicConfig(level=logging.INFO)
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
        self.annotated_frame = None  # Frame with recognition annotations
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
    
    @contextmanager
    def camera_context(self, camera_id=0):
        """Context manager for camera operations"""
        cap = None
        try:
            if self.is_raspberry_pi:
                cap = cv2.VideoCapture(camera_id)
            else:
                cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
            
            if not cap.isOpened():
                raise Exception(f"Cannot open camera {camera_id}")
            
            # Set camera properties
            if self.is_raspberry_pi:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 15)
            else:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                cap.set(cv2.CAP_PROP_FPS, 30)
            
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
        # Always update the door state when a known person is recognized
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
        
        # Set a timestamp for when the unlock window expires
        self.door_unlock_available_until = time.time() + 10  # 10 seconds window
        
        # Cancel any existing auto-relock timer and start a new one
        if hasattr(self, '_relock_timer') and self._relock_timer:
            self._relock_timer.cancel()
        
        self._relock_timer = threading.Timer(5.0, self.relock_door)
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
        """Relock the door"""
        self.door_locked = True
        self.door_unlock_available_until = 0  # Clear any remaining unlock window
        if hasattr(self, '_relock_timer'):
            self._relock_timer = None
        logger.info("Door automatically relocked")
        # 6. Add a method to check if unlock window is still valid
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
    
    
    def setup_esp32_endpoints(self):
        """Setup endpoints that ESP32 will call"""
        # This is just for documentation - the routes are already defined above
        pass

# Initialize the API
face_api = FacialRecognitionAPI()

# API Routes
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': face_api.model is not None,
        'known_faces': len(face_api.saved_embeddings),
        'camera_active': face_api.is_running,
        'device': face_api.DEVICE,
        'platform': 'Raspberry Pi' if face_api.is_raspberry_pi else 'Desktop'
    })

@app.route('/status', methods=['GET'])
def get_status():
    """Get current system status"""
    return jsonify({
        'door_locked': face_api.door_locked,
        'camera_active': face_api.is_running,
        'last_recognition': face_api.last_recognition,
        'known_faces_count': len(face_api.saved_embeddings),
        'threshold': face_api.THRESHOLD
    })


@app.route('/camera/start', methods=['POST'])
def start_camera():
    """Start camera stream"""
    try:
        # Handle both JSON and form data
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form.to_dict()
        
        enable_recognition = data.get('recognition', 'false').lower() == 'true'
        
        face_api.start_camera_stream(enable_recognition=enable_recognition)
        return jsonify({
            'message': f'Camera stream started (recognition: {enable_recognition})', 
            'success': True,
            'recognition_enabled': enable_recognition
        })
    except Exception as e:
        return jsonify({'message': f'Failed to start camera: {str(e)}', 'success': False}), 500
    
@app.route('/camera/stop', methods=['POST'])
def stop_camera():
    """Stop camera stream"""
    face_api.stop_camera_stream()
    return jsonify({'message': 'Camera stream stopped', 'success': True})

@app.route('/camera/capture', methods=['GET'])
def capture_frame():
    """Capture current frame"""
    # Check for annotated parameter
    annotated = request.args.get('annotated', 'false').lower() == 'true'
    frame = face_api.get_current_frame(annotated=annotated)
    
    if frame is None:
        return jsonify({'message': 'No frame available', 'success': False}), 404
    
    # Convert frame to base64
    img_base64 = face_api.frame_to_base64(frame)
    
    return jsonify({
        'success': True,
        'image': img_base64,
        'annotated': annotated,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/recognize', methods=['POST'])
def recognize_faces():
    """Recognize faces in current frame or uploaded image"""
    try:
        if 'image' in request.files:
            # Process uploaded image
            file = request.files['image']
            image = Image.open(file.stream)
            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        else:
            # Use current camera frame
            frame = face_api.get_current_frame()
            if frame is None:
                return jsonify({
                    'message': 'No frame available. Start camera first.',
                    'success': False
                }), 404
        
        # Detect and recognize faces
        faces = face_api.detect_and_recognize_faces(frame)
        
        return jsonify({
            'success': True,
            'faces': faces,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'message': f'Recognition failed: {str(e)}',
            'success': False
        }), 500

@app.route('/access/check', methods=['POST'])
def check_access():
    """Check access for current frame and potentially unlock door"""
    try:
        frame = face_api.get_current_frame()
        if frame is None:
            return jsonify({
                'message': 'No frame available. Start camera first.',
                'success': False
            }), 404
        
        # Detect and recognize faces
        faces = face_api.detect_and_recognize_faces(frame)
        
        # Process access request
        access_result = face_api.process_access_request(faces)
        
        return jsonify({
            'success': True,
            'access_result': access_result,
            'all_faces': faces,
            'door_locked': face_api.door_locked,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'message': f'Access check failed: {str(e)}',
            'success': False
        }), 500

@app.route('/door/lock', methods=['POST'])
def lock_door():
    """Manually lock the door"""
    face_api.door_locked = True
    return jsonify({
        'message': 'Door locked',
        'success': True,
        'door_locked': True
    })

@app.route('/door/unlock', methods=['POST'])
def unlock_door_manual():
    """Manually unlock the door"""
    face_api.door_locked = False
    # Auto-relock after 5 seconds
    threading.Timer(5.0, face_api.relock_door).start()
    return jsonify({
        'message': 'Door unlocked (will auto-relock in 5 seconds)',
        'success': True,
        'door_locked': False
    })

@app.route('/logs', methods=['GET'])
def get_logs():
    """Get access logs"""
    limit = request.args.get('limit', 50, type=int)
    logs = list(face_api.detection_history)[-limit:]
    
    return jsonify({
        'success': True,
        'logs': logs,
        'total_count': len(face_api.detection_history)
    })

@app.route('/faces/list', methods=['GET'])
def list_known_faces():
    """List all known faces"""
    return jsonify({
        'success': True,
        'known_faces': list(face_api.saved_embeddings.keys()),
        'count': len(face_api.saved_embeddings)
    })

@app.route('/config', methods=['GET', 'POST'])
def handle_config():
    """Get or update configuration"""
    if request.method == 'GET':
        return jsonify({
            'success': True,
            'config': {
                'threshold': face_api.THRESHOLD,
                'detection_delay': face_api.DETECTION_DELAY,
                'device': face_api.DEVICE,
                'model_path': face_api.MODEL_PATH
            }
        })
    else:
        # Update configuration
        data = request.json
        if 'threshold' in data:
            face_api.THRESHOLD = float(data['threshold'])
        if 'detection_delay' in data:
            face_api.DETECTION_DELAY = int(data['detection_delay'])
        
        return jsonify({
            'success': True,
            'message': 'Configuration updated'
        })

# Livestream endpoints
@app.route('/stream/video')
def video_stream():
    """Live video stream without recognition annotations"""
    if not face_api.stream_active:
        return Response("Camera not active", status=404, mimetype='text/plain')
    
    return Response(
        face_api.generate_video_stream(annotated=False),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/stream/recognition')
def recognition_stream():
    """Live video stream with recognition annotations"""
    if not face_api.stream_active:
        return Response("Camera not active", status=404, mimetype='text/plain')
    
    return Response(
        face_api.generate_video_stream(annotated=True),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/stream/status')
def stream_status():
    """Get streaming status"""
    return jsonify({
        'stream_active': face_api.stream_active,
        'recognition_enabled': face_api.stream_with_recognition,
        'camera_connected': face_api.cap is not None,
        'current_frame_available': face_api.current_frame is not None
    })

@app.route('/stream/toggle-recognition', methods=['POST'])
def toggle_recognition():
    """Toggle recognition processing on current stream"""
    if not face_api.stream_active:
        return jsonify({
            'message': 'Camera stream not active',
            'success': False
        }), 404
    
    face_api.stream_with_recognition = not face_api.stream_with_recognition
    
    return jsonify({
        'success': True,
        'message': f'Recognition {"enabled" if face_api.stream_with_recognition else "disabled"}',
        'recognition_enabled': face_api.stream_with_recognition
    })

# Add this route to your Flask app for ESP32 to check door status
@app.route('/esp32/door-status', methods=['GET'])
def esp32_door_status():
    """Endpoint for ESP32 to check if door should be opened"""
    try:
        current_time = time.time()
        
        # Check if we have a valid unlock window
        should_unlock = (
            face_api.last_recognition and 
            face_api.last_recognition.get('status') == 'GRANTED' and
            current_time <= face_api.door_unlock_available_until and
            not face_api.door_locked
        )
        
        if should_unlock:
            # Get person info before clearing
            person_name = face_api.last_recognition.get('name', 'KNOWN')
            
            # Clear the unlock window to prevent multiple unlocks
            face_api.door_unlock_available_until = 0
            
            return jsonify({
                'open_door': True,
                'person': person_name,
                'timestamp': datetime.now().isoformat(),
                'message': 'Access granted'
            })
        
        return jsonify({
            'open_door': False,
            'message': 'No access granted recently',
            'timestamp': datetime.now().isoformat(),
            'door_locked': face_api.door_locked
        })
        
    except Exception as e:
        return jsonify({
            'open_door': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/esp32/simple-status', methods=['GET'])
def esp32_simple_status():
    """Simplified status endpoint for ESP32"""
    try:
        current_time = time.time()
        
        # Check if access was recently granted and window is still valid
        should_open = (
            face_api.last_recognition and 
            face_api.last_recognition.get('status') == 'GRANTED' and
            current_time <= face_api.door_unlock_available_until and
            not face_api.door_locked
        )
        
        if should_open:
            person_name = face_api.last_recognition.get('name', 'KNOWN')
            
            # Clear the unlock window
            face_api.door_unlock_available_until = 0
            
            response_data = {
                'open': 1,
                'message': 'ACCESS_GRANTED',
                'person': person_name
            }
        else:
            response_data = {
                'open': 0,
                'message': 'WAITING',
                'door_locked': face_api.door_locked
            }
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({'open': 0, 'error': str(e)}), 500
@app.route('/esp32/status', methods=['GET'])
def esp32_status():
    """Endpoint for ESP32 to get system status"""
    return jsonify({
        'system_active': True,
        'door_locked': face_api.door_locked,
        'camera_active': face_api.is_running,
        'known_faces': len(face_api.saved_embeddings),
        'timestamp': datetime.now().isoformat()
    })

# Add this route for manual door control from ESP32
@app.route('/esp32/control', methods=['POST'])
def esp32_control():
    """Endpoint for ESP32 to control door manually"""
    try:
        data = request.get_json()
        command = data.get('command', '').upper()
        
        if command == 'OPEN':
            face_api.door_locked = False
            # Auto-relock after 5 seconds
            threading.Timer(5.0, face_api.relock_door).start()
            return jsonify({
                'success': True,
                'message': 'Door unlocked',
                'door_locked': False
            })
            
        elif command == 'CLOSE':
            face_api.door_locked = True
            return jsonify({
                'success': True,
                'message': 'Door locked',
                'door_locked': True
            })
            
        elif command == 'STATUS':
            return jsonify({
                'success': True,
                'door_locked': face_api.door_locked,
                'message': 'Door status retrieved'
            })
            
        else:
            return jsonify({
                'success': False,
                'message': f'Unknown command: {command}'
            }), 400
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500



# HTML endpoint for easy testing
@app.route('/viewer')
def viewer():
    """Simple HTML viewer for testing streams"""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Facial Recognition Stream Viewer</title>
        <style>
            body { 
                font-family: Arial, sans-serif; 
                margin: 20px;
                background-color: #0a0e14;
                color: white;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
            }
            .stream-container {
                display: flex;
                gap: 20px;
                margin: 20px 0;
            }
            .stream {
                flex: 1;
                text-align: center;
            }
            .stream img {
                width: 100%;
                max-width: 600px;
                border: 2px solid #47d7ac;
                border-radius: 8px;
            }
            .controls {
                margin: 20px 0;
                padding: 20px;
                background-color: #151a21;
                border-radius: 8px;
            }
            button {
                background-color: #47d7ac;
                color: white;
                border: none;
                padding: 10px 20px;
                margin: 5px;
                border-radius: 5px;
                cursor: pointer;
                font-size: 14px;
            }
            button:hover {
                background-color: #3dbf99;
            }
            button.danger {
                background-color: #ff6b6b;
            }
            button.danger:hover {
                background-color: #ff5252;
            }
            .status {
                margin: 10px 0;
                padding: 15px;
                background-color: #1c2129;
                border-radius: 5px;
            }
            h1 { color: #47d7ac; }
            h3 { color: #f0c674; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🔒 Facial Recognition Stream Viewer</h1>
            
            <div class="controls">
                <button onclick="startCamera()">Start Camera</button>
                <button onclick="stopCamera()" class="danger">Stop Camera</button>
                <button onclick="toggleRecognition()">Toggle Recognition</button>
                <button onclick="checkAccess()">Check Access</button>
                <button onclick="updateStatus()">Refresh Status</button>
            </div>
            
            <div id="status" class="status">
                Loading status...
            </div>
            
            <div class="stream-container">
                <div class="stream">
                    <h3>Raw Video Stream</h3>
                    <img id="rawStream" src="/stream/video" onerror="this.src='data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNjAwIiBoZWlnaHQ9IjQwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjMzMzIi8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGZvbnQtZmFtaWx5PSJBcmlhbCIgZm9udC1zaXplPSIxOCIgZmlsbD0iI2NjYyIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZHk9Ii4zZW0iPkNhbWVyYSBOb3QgQWN0aXZlPC90ZXh0Pjwvc3ZnPg=='">
                </div>
                <div class="stream">
                    <h3>Recognition Stream</h3>
                    <img id="recognitionStream" src="/stream/recognition" onerror="this.src='data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNjAwIiBoZWlnaHQ9IjQwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjMzMzIi8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGZvbnQtZmFtaWx5PSJBcmlhbCIgZm9udC1zaXplPSIxOCIgZmlsbD0iI2NjYyIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZHk9Ii4zZW0iPkNhbWVyYSBOb3QgQWN0aXZlPC90ZXh0Pjwvc3ZnPg=='">
                </div>
            </div>
        </div>
        
        <script>
            async function startCamera() {
                try {
                    const response = await fetch('/camera/start', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ recognition: true })
                    });
                    const data = await response.json();
                    alert(data.message);
                    updateStatus();
                    setTimeout(() => {
                        document.getElementById('rawStream').src = '/stream/video?' + new Date().getTime();
                        document.getElementById('recognitionStream').src = '/stream/recognition?' + new Date().getTime();
                    }, 1000);
                } catch (error) {
                    alert('Error starting camera: ' + error.message);
                }
            }
            
            async function stopCamera() {
                try {
                    const response = await fetch('/camera/stop', { method: 'POST' });
                    const data = await response.json();
                    alert(data.message);
                    updateStatus();
                } catch (error) {
                    alert('Error stopping camera: ' + error.message);
                }
            }
            
            async function toggleRecognition() {
                try {
                    const response = await fetch('/stream/toggle-recognition', { method: 'POST' });
                    const data = await response.json();
                    alert(data.message);
                    updateStatus();
                } catch (error) {
                    alert('Error toggling recognition: ' + error.message);
                }
            }
            
            async function checkAccess() {
                try {
                    const response = await fetch('/access/check', { method: 'POST' });
                    const data = await response.json();
                    const result = data.access_result;
                    const message = result.access_granted ? 
                        `Access granted to ${result.person}` : 
                        `Access denied (${result.faces_detected} faces detected)`;
                    alert(message);
                    updateStatus();
                } catch (error) {
                    alert('Error checking access: ' + error.message);
                }
            }
            
            async function updateStatus() {
                try {
                    const response = await fetch('/status');
                    const data = await response.json();
                    const streamResponse = await fetch('/stream/status');
                    const streamData = await streamResponse.json();
                    
                    document.getElementById('status').innerHTML = `
                        <strong>System Status:</strong><br>
                        Door: ${data.door_locked ? 'LOCKED' : 'UNLOCKED'}<br>
                        Camera: ${data.camera_active ? 'ACTIVE' : 'INACTIVE'}<br>
                        Stream: ${streamData.stream_active ? 'ACTIVE' : 'INACTIVE'}<br>
                        Recognition: ${streamData.recognition_enabled ? 'ENABLED' : 'DISABLED'}<br>
                        Known Faces: ${data.known_faces_count}<br>
                        Last Recognition: ${data.last_recognition ? data.last_recognition.name + ' (' + data.last_recognition.status + ')' : 'None'}
                    `;
                } catch (error) {
                    document.getElementById('status').innerHTML = 'Error loading status';
                }
            }
            
            // Update status on load
            updateStatus();
            
            // Auto-refresh status every 5 seconds
            setInterval(updateStatus, 5000);
        </script>
    </body>
    </html>
    """
    return html

@app.route('/dashboard')
@app.route('/viewer2')  # Keep backward compatibility
def dashboard():
    """Modern Bootstrap-based dashboard for facial recognition system"""
  
    from flask import render_template
    return render_template('dashboard.html')

if __name__ == '__main__':
    # Run the Flask app
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting Facial Recognition API on port {port}")
    logger.info(f"Known faces: {len(face_api.saved_embeddings)}")
    logger.info(f"Device: {face_api.DEVICE}")
    
    app.run(host='0.0.0.0', port=port, debug=debug, threaded=True)