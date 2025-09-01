import time
import threading
import numpy as np
from datetime import datetime
from PIL import Image
from flask import request, jsonify, Response, render_template
import cv2

def register_routes(app, face_api):
    """Register all Flask routes with the app instance"""
    
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

    # ESP32 specific endpoints
    @app.route('/esp32/door-status', methods=['GET'])
    def esp32_door_status():
        """Endpoint for ESP32 to check door lock status"""
        try:
            return jsonify({
                'door_locked': face_api.door_locked,
                'unlock_available': face_api.is_unlock_window_valid(),
                'last_recognition': face_api.last_recognition,
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            return jsonify({
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
                not face_api.door_locked  # This condition prevents opening if door is already locked
            )
            
            if should_open:
                person_name = face_api.last_recognition.get('name', 'KNOWN')
                
                # Unlock the door (sets door_locked = False and handles timers)
                face_api.door_locked = False
                
                # Clear the unlock window to prevent multiple opens
                face_api.door_unlock_available_until = 0
                
                # Set up auto-relock timer (5 seconds) - cancel any existing timer first
                if hasattr(face_api, '_esp32_relock_timer') and face_api._esp32_relock_timer:
                    face_api._esp32_relock_timer.cancel()
                
                face_api._esp32_relock_timer = threading.Timer(5.0, face_api.relock_door)
                face_api._esp32_relock_timer.start()
                
                app.logger.info(f"Door unlocked via ESP32 for {person_name}")
                
                response_data = {
                    'open': 1,
                    'message': 'ACCESS_GRANTED',
                    'person': person_name,
                    'door_locked': False
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

    # Dashboard/Web Interface
    @app.route('/dashboard')
    @app.route('/viewer2')  # Keep backward compatibility
    def dashboard():
        """Modern Bootstrap-based dashboard for facial recognition system"""
        return render_template('dashboard.html')