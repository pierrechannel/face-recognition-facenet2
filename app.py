import os
import logging
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

if __name__ == '__main__':
    app, face_api = create_app()
    
    # Run the Flask app
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting Facial Recognition API on port {port}")
    logger.info(f"Known faces: {len(face_api.saved_embeddings)}")
    logger.info(f"Device: {face_api.DEVICE}")
    
    app.run(host='0.0.0.0', port=port, debug=debug, threaded=True)