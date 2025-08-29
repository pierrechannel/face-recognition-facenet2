import os
import torch
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import uuid
from app.model import load_facenet_model
from app.face_utils import read_image_from_bytes, preprocess_face, get_embedding
from app.face_detect import extract_faces
from app.db import save_embedding

MODEL_PATH = "facenet_africain_finetuned.pth"
EMBEDDINGS_DIR = "embeddings"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

def load_model(path):
    model = load_facenet_model()
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'device': DEVICE,
        'model_loaded': os.path.exists(MODEL_PATH)
    })

@app.route('/api/register', methods=['POST'])
def register_face():
    """
    Register multiple face images for a person and save their embeddings.
    Expects multipart/form-data with:
    - name: person's name
    - images: multiple image files
    """
    # Check if name is provided
    name = request.form.get('name')
    if not name:
        return jsonify({'error': 'Name is required'}), 400
    
    # Check if files are provided
    if 'images' not in request.files:
        return jsonify({'error': 'No images provided'}), 400
    
    files = request.files.getlist('images')
    if not files or files[0].filename == '':
        return jsonify({'error': 'No images selected'}), 400
    
    # Load model
    try:
        model = load_model(MODEL_PATH)
    except Exception as e:
        return jsonify({'error': f'Failed to load model: {str(e)}'}), 500
    
    all_embeddings = []
    successful_images = 0
    processed_files = []
    
    for i, file in enumerate(files):
        if file and allowed_file(file.filename):
            try:
                # Save file temporarily
                filename = secure_filename(f"{uuid.uuid4().hex}_{file.filename}")
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                processed_files.append(filepath)
                
                # Read and process image
                with open(filepath, "rb") as f:
                    image_bytes = f.read()

                image = read_image_from_bytes(image_bytes)
                faces = extract_faces(image)

                if not faces:
                    print(f"No face detected in image {i+1}: {file.filename}")
                    continue

                face_tensor = preprocess_face(faces[0])
                embedding = get_embedding(model, face_tensor, DEVICE)
                
                # Convert embedding to numpy array if it's a tensor
                if isinstance(embedding, torch.Tensor):
                    embedding = embedding.cpu().detach().numpy()
                    
                all_embeddings.append(embedding)
                successful_images += 1
                
                print(f"Embedding extracted for {name} from image {i+1}: {file.filename}")
                
            except Exception as e:
                print(f"Error processing image {file.filename}: {str(e)}")
                continue
    
    # Clean up temporary files
    for filepath in processed_files:
        try:
            os.remove(filepath)
        except:
            pass
    
    if not all_embeddings:
        return jsonify({
            'error': 'No faces could be detected in the provided images',
            'successful_images': 0
        }), 400
    
    # Calculate average embedding
    avg_embedding = np.mean(np.stack(all_embeddings), axis=0)
    
    # Convert back to tensor if needed by save_embedding function
    if hasattr(torch, 'from_numpy'):
        avg_embedding_tensor = torch.from_numpy(avg_embedding)
    else:
        avg_embedding_tensor = torch.tensor(avg_embedding)
    
    # Save the embedding
    try:
        save_embedding(name, avg_embedding_tensor)
    except Exception as e:
        return jsonify({'error': f'Failed to save embedding: {str(e)}'}), 500
    
    return jsonify({
        'success': True,
        'message': f'{successful_images} embeddings combined and saved for {name}',
        'person_name': name,
        'successful_images': successful_images,
        'total_images': len(files)
    })

@app.route('/api/register-batch', methods=['POST'])
def register_face_batch():
    """
    Register faces from a list of base64 encoded images or URLs.
    Expects JSON with:
    - name: person's name
    - images: list of base64 strings or URLs
    """
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'JSON data required'}), 400
    
    name = data.get('name')
    if not name:
        return jsonify({'error': 'Name is required'}), 400
    
    images = data.get('images', [])
    if not images:
        return jsonify({'error': 'No images provided'}), 400
    
    # This endpoint would need additional implementation for base64/URL handling
    # You would need to modify the face_utils to handle these formats
    
    return jsonify({
        'error': 'Batch endpoint not fully implemented',
        'hint': 'You need to implement base64/URL image handling'
    }), 501

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large'}), 413

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)