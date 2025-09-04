import os
import logging
import threading
import time
from flask import Flask
from flask_cors import CORS

from api.face_recognition_api import FacialRecognitionAPI
from api.routes import register_routes

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_app():
    """Create and configure the Flask application"""
    app = Flask(__name__)
    CORS(app)  # Enable CORS for web clients
    
    # Initialize the facial recognition API
    face_api = FacialRecognitionAPI()
    
    # Register all routes
    register_routes(app, face_api)
    
    return app, face_api

def start_camera_automatically(face_api):
    """Start camera automatically with a delay"""
    def delayed_start():
        # Wait for Flask to fully initialize
        time.sleep(5)
        
        try:
            logger.info("Attempting to start camera automatically...")
            success = face_api.start_camera_stream(enable_recognition=True)
            
            if success:
                logger.info("Camera started successfully with face recognition")
                logger.info(f"Known faces: {len(face_api.saved_embeddings)}")
            else:
                logger.warning("Failed to start camera automatically")
                
        except Exception as e:
            logger.error(f"Error starting camera automatically: {e}")
    
    # Start in a separate thread to avoid blocking
    camera_thread = threading.Thread(target=delayed_start, daemon=True)
    camera_thread.start()
    return camera_thread

if __name__ == '__main__':
    app, face_api = create_app()
    
    # Start camera automatically
    start_camera_automatically(face_api)
    
    # Run the Flask app
    port = int(os.environ.get('PORT', 5001))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting Facial Recognition API on port {port}")
    logger.info(f"Known faces: {len(face_api.saved_embeddings)}")
    logger.info(f"Device: {face_api.DEVICE}")
    
    app.run(host='0.0.0.0', port=port, debug=debug, threaded=True)