# app/face_utils.py
import torch
import cv2
import numpy as np
from torchvision import transforms
import platform

# Détection de la plateforme
IS_RASPBERRY_PI = platform.machine() in ('armv7l', 'armv6l', 'aarch64')

# Transform avec gestion spéciale pour Raspberry Pi
if IS_RASPBERRY_PI:
    # Transform plus robuste pour Raspberry Pi
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])
else:
    # Transform original pour PC
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])


def read_image_from_bytes(file_bytes):
    """Lire une image depuis des bytes"""
    np_arr = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def preprocess_face(image):
    """Préprocessing d'une image de visage"""
    try:
        # Appliquer les transformations
        tensor = transform(image)
        
        # Vérifier la validité du tensor
        if torch.isnan(tensor).any():
            print("ERREUR: Transform a généré des NaN!")
            # Fallback: transform simple
            simple_transform = transforms.Compose([
                transforms.Resize((160, 160)),
                transforms.ToTensor()
            ])
            tensor = simple_transform(image)
            # Normalisation manuelle
            tensor = (tensor - 0.5) / 0.5
        
        if torch.isinf(tensor).any():
            print("ERREUR: Transform a généré des valeurs infinies!")
            tensor = torch.clamp(tensor, -10, 10)
        
        print(f"Tensor preprocessé: shape={tensor.shape}, min={tensor.min():.4f}, max={tensor.max():.4f}, mean={tensor.mean():.4f}")
        
        return tensor.unsqueeze(0)  # batch of 1
        
    except Exception as e:
        print(f"Erreur dans preprocess_face: {e}")
        # Fallback très simple
        tensor = transforms.ToTensor()(image)
        return tensor.unsqueeze(0)


def safe_tensor_stats(tensor, name="tensor"):
    """Calcul sécurisé des statistiques d'un tensor"""
    try:
        if tensor.numel() == 0:
            return f"{name}: empty tensor"
        
        # Vérifier et corriger les NaN/Inf
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            tensor = torch.nan_to_num(tensor, nan=0.0, posinf=1.0, neginf=-1.0)
        
        min_val = tensor.min().item()
        max_val = tensor.max().item()
        mean_val = tensor.mean().item()
        
        return f"{name}: min={min_val:.6f}, max={max_val:.6f}, mean={mean_val:.6f}"
    
    except Exception as e:
        return f"{name}: error calculating stats - {e}"


def get_embedding(model, image_tensor, device='cpu'):
    """Obtenir l'embedding d'une image de visage"""
    try:
        print(f"get_embedding appelé avec: tensor_shape={image_tensor.shape}, device={device}")
        
        # S'assurer que le modèle est en mode eval
        model.eval()
        
        with torch.no_grad():
            # S'assurer que le tensor est sur le bon device
            if device == 'cuda' and torch.cuda.is_available():
                image_tensor = image_tensor.to('cuda')
                model = model.to('cuda')
            else:
                image_tensor = image_tensor.to('cpu')
                model = model.to('cpu')
            
            print(f"Tensor sur device: {image_tensor.device}, Model sur device: {next(model.parameters()).device}")
            
            # Vérifier l'input avant inférence
            print(safe_tensor_stats(image_tensor, "Input tensor"))
            
            # Inférence
            try:
                output = model(image_tensor)
                print(f"Output du modèle: type={type(output)}, shape={getattr(output, 'shape', 'N/A')}")
                
                # Vérifier si l'output contient des NaN directement du modèle
                if isinstance(output, torch.Tensor):
                    # Vérifier et corriger les NaN/Inf
                    if torch.isnan(output).any() or torch.isinf(output).any():
                        print("ATTENTION: Le modèle génère des NaN/Inf valeurs!")
                        nan_count = torch.isnan(output).sum().item()
                        inf_count = torch.isinf(output).sum().item()
                        print(f"NaN count: {nan_count}, Inf count: {inf_count}")
                        
                        # Corriger les valeurs problématiques
                        output = torch.nan_to_num(output, nan=0.0, posinf=1.0, neginf=-1.0)
                    
                    # Calcul sécurisé des statistiques
                    if output.numel() > 0:
                        output_min = output.min().item()
                        output_max = output.max().item()
                        print(f"Output stats: min={output_min:.6f}, max={output_max:.6f}")
                    else:
                        print("Output vide!")
                
                # Conversion sécurisée en numpy
                if isinstance(output, torch.Tensor):
                    # Déplacer vers CPU avant conversion
                    output_cpu = output.cpu()
                    
                    # Vérifier encore une fois avant conversion numpy
                    if torch.isnan(output_cpu).any() or torch.isinf(output_cpu).any():
                        print("NaN/Inf détectés avant conversion numpy, correction...")
                        output_cpu = torch.nan_to_num(output_cpu, nan=0.0, posinf=1.0, neginf=-1.0)
                    
                    # Conversion en numpy
                    numpy_output = output_cpu.numpy()
                    
                    # Si c'est un batch, prendre le premier élément
                    if len(numpy_output.shape) > 1 and numpy_output.shape[0] == 1:
                        numpy_output = numpy_output[0]
                    
                    # Validation finale
                    if np.isnan(numpy_output).any() or np.isinf(numpy_output).any():
                        print("Correction des NaN/Inf dans l'embedding final")
                        numpy_output = np.nan_to_num(numpy_output, nan=0.0, posinf=1.0, neginf=-1.0)
                    
                    print(f"Embedding final: shape={numpy_output.shape}, type={type(numpy_output)}")
                    print(f"Embedding stats: min={np.min(numpy_output):.6f}, max={np.max(numpy_output):.6f}, mean={np.mean(numpy_output):.6f}")
                    print(f"NaN count: {np.isnan(numpy_output).sum()}, Inf count: {np.isinf(numpy_output).sum()}")
                    
                    return numpy_output
                else:
                    print(f"Output n'est pas un tensor: {type(output)}")
                    # Convertir en numpy array
                    numpy_output = np.array(output)
                    if len(numpy_output.shape) > 1:
                        numpy_output = numpy_output[0]
                    return numpy_output
                    
            except RuntimeError as e:
                print(f"Erreur RuntimeError dans l'inférence: {e}")
                # Essayer avec des paramètres différents
                if "out of memory" in str(e).lower():
                    print("Problème de mémoire - tentative avec batch size réduit")
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
                # Retourner un embedding de fallback
                fallback_embedding = torch.randn(512, dtype=torch.float32)
                fallback_embedding = fallback_embedding / torch.norm(fallback_embedding)
                return fallback_embedding.numpy()
                
    except Exception as e:
        print(f"Erreur générale dans get_embedding: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback final
        print("Utilisation du fallback final")
        fallback_embedding = torch.randn(512, dtype=torch.float32)
        fallback_embedding = fallback_embedding / torch.norm(fallback_embedding)
        return fallback_embedding.numpy()


def validate_embedding(embedding):
    """Valider un embedding"""
    if embedding is None:
        return False
    
    if isinstance(embedding, torch.Tensor):
        embedding = embedding.numpy()
    
    if not isinstance(embedding, np.ndarray):
        return False
    
    if np.isnan(embedding).any():
        print(f"Embedding contient {np.isnan(embedding).sum()} NaN")
        return False
    
    if np.isinf(embedding).any():
        print(f"Embedding contient {np.isinf(embedding).sum()} valeurs infinies")
        return False
    
    if np.allclose(embedding, 0):
        print("Embedding est entièrement nul")
        return False
    
    return True


def debug_model_inference(model, image_tensor, device='cpu'):
    """Fonction de debug pour tester l'inférence du modèle"""
    print("=== DEBUG MODEL INFERENCE ===")
    
    try:
        # Test avec un tensor aléatoire
        print("Test avec tensor aléatoire...")
        random_tensor = torch.randn(1, 3, 160, 160).to(device)
        with torch.no_grad():
            random_output = model(random_tensor)
            print(f"Random output: shape={random_output.shape}, NaN={torch.isnan(random_output).any()}")
        
        # Test avec le vrai tensor
        print("Test avec le vrai tensor...")
        with torch.no_grad():
            real_output = model(image_tensor.to(device))
            print(f"Real output: shape={real_output.shape}, NaN={torch.isnan(real_output).any()}")
            
            if torch.isnan(real_output).any():
                print("Analyse des paramètres du modèle:")
                for name, param in model.named_parameters():
                    if param.numel() > 0:
                        if torch.isnan(param).any():
                            print(f"  Paramètre {name} contient des NaN!")
                        if torch.isinf(param).any():
                            print(f"  Paramètre {name} contient des valeurs infinies!")
        
    except Exception as e:
        print(f"Erreur dans debug_model_inference: {e}")
        import traceback
        traceback.print_exc()
    
    print("=== FIN DEBUG ===")