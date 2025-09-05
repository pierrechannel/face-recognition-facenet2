import cv2
import numpy as np
from PIL import Image, ImageEnhance
import logging

logger = logging.getLogger(__name__)

class LightingAdaptation:
    """Enhanced lighting adaptation for facial recognition system"""
    
    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        self.previous_brightness = None
        self.brightness_history = []
        self.max_history = 30  # Keep last 30 frames for averaging
        
    def analyze_lighting_conditions(self, frame):
        """Analyze current lighting conditions of the frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate various lighting metrics
        mean_brightness = np.mean(gray)
        std_brightness = np.std(gray)
        
        # Calculate histogram for lighting distribution analysis
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_normalized = hist.ravel() / hist.sum()
        
        # Detect lighting conditions
        conditions = {
            'mean_brightness': mean_brightness,
            'std_brightness': std_brightness,
            'is_low_light': mean_brightness < 80,
            'is_high_light': mean_brightness > 180,
            'is_backlit': self._detect_backlighting(gray),
            'has_shadows': std_brightness > 50,
            'uniformity': self._calculate_uniformity(gray)
        }
        
        # Update brightness history for adaptive processing
        self.brightness_history.append(mean_brightness)
        if len(self.brightness_history) > self.max_history:
            self.brightness_history.pop(0)
        
        return conditions
    
    def _detect_backlighting(self, gray_frame):
        """Detect if the subject is backlit"""
        h, w = gray_frame.shape
        
        # Check if center is significantly darker than edges
        center_region = gray_frame[h//4:3*h//4, w//4:3*w//4]
        edge_regions = np.concatenate([
            gray_frame[:h//8, :].flatten(),
            gray_frame[-h//8:, :].flatten(),
            gray_frame[:, :w//8].flatten(),
            gray_frame[:, -w//8:].flatten()
        ])
        
        center_mean = np.mean(center_region)
        edge_mean = np.mean(edge_regions)
        
        return edge_mean - center_mean > 30
    
    def _calculate_uniformity(self, gray_frame):
        """Calculate lighting uniformity across the frame"""
        h, w = gray_frame.shape
        grid_size = 4
        
        # Divide frame into grid and calculate variance
        cell_means = []
        for i in range(grid_size):
            for j in range(grid_size):
                y1, y2 = i * h // grid_size, (i + 1) * h // grid_size
                x1, x2 = j * w // grid_size, (j + 1) * w // grid_size
                cell = gray_frame[y1:y2, x1:x2]
                cell_means.append(np.mean(cell))
        
        return 1.0 / (1.0 + np.std(cell_means))  # Higher value = more uniform
    
    def adaptive_preprocessing(self, frame):
        """Apply adaptive preprocessing based on lighting conditions"""
        conditions = self.analyze_lighting_conditions(frame)
        processed_frame = frame.copy()
        
        # Apply different processing based on conditions
        if conditions['is_low_light']:
            processed_frame = self._enhance_low_light(processed_frame)
        elif conditions['is_high_light']:
            processed_frame = self._reduce_overexposure(processed_frame)
        
        if conditions['is_backlit']:
            processed_frame = self._correct_backlighting(processed_frame)
        
        if not conditions['has_shadows'] and conditions['uniformity'] < 0.7:
            processed_frame = self._improve_uniformity(processed_frame)
        
        # Apply histogram equalization for better contrast
        processed_frame = self._apply_clahe(processed_frame)
        
        # Dynamic range adjustment
        processed_frame = self._adjust_dynamic_range(processed_frame, conditions)
        
        return processed_frame
    
    def _enhance_low_light(self, frame):
        """Enhance images taken in low light conditions"""
        # Convert to LAB color space for better luminance control
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply gamma correction to brighten
        gamma = 1.5
        l_corrected = np.power(l / 255.0, 1.0 / gamma) * 255.0
        l_corrected = np.clip(l_corrected, 0, 255).astype(np.uint8)
        
        # Merge back and convert to BGR
        lab_corrected = cv2.merge([l_corrected, a, b])
        frame_enhanced = cv2.cvtColor(lab_corrected, cv2.COLOR_LAB2BGR)
        
        # Reduce noise that may have been amplified
        frame_enhanced = cv2.bilateralFilter(frame_enhanced, 9, 75, 75)
        
        return frame_enhanced
    
    def _reduce_overexposure(self, frame):
        """Reduce overexposure in bright lighting"""
        # Convert to HSV for better control
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Reduce value channel to control brightness
        v_reduced = cv2.multiply(v, 0.8)
        v_reduced = np.clip(v_reduced, 0, 255).astype(np.uint8)
        
        # Increase saturation slightly to compensate
        s_enhanced = cv2.multiply(s, 1.1)
        s_enhanced = np.clip(s_enhanced, 0, 255).astype(np.uint8)
        
        hsv_corrected = cv2.merge([h, s_enhanced, v_reduced])
        return cv2.cvtColor(hsv_corrected, cv2.COLOR_HSV2BGR)
    
    def _correct_backlighting(self, frame):
        """Correct backlighting issues"""
        # Use unsharp masking to bring out details in dark areas
        gaussian = cv2.GaussianBlur(frame, (0, 0), 2.0)
        unsharp_mask = cv2.addWeighted(frame, 1.5, gaussian, -0.5, 0)
        
        # Apply shadow/highlight adjustment
        lab = cv2.cvtColor(unsharp_mask, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Enhance shadows (lower values) more than highlights
        l_enhanced = np.where(l < 128, 
                             np.power(l / 255.0, 0.7) * 255.0, 
                             l)
        l_enhanced = np.clip(l_enhanced, 0, 255).astype(np.uint8)
        
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    
    def _improve_uniformity(self, frame):
        """Improve lighting uniformity across the frame"""
        # Create a background model using heavy blurring
        background = cv2.GaussianBlur(frame, (0, 0), frame.shape[0] / 20)
        
        # Normalize the frame against the background
        normalized = cv2.divide(frame, background, scale=128)
        
        return normalized
    
    def _apply_clahe(self, frame):
        """Apply Contrast Limited Adaptive Histogram Equalization"""
        # Convert to LAB and apply CLAHE to L channel
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to luminance channel
        l_clahe = self.clahe.apply(l)
        
        # Merge back
        lab_clahe = cv2.merge([l_clahe, a, b])
        return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    
    def _adjust_dynamic_range(self, frame, conditions):
        """Adjust dynamic range based on lighting conditions"""
        if len(self.brightness_history) > 5:
            avg_brightness = np.mean(self.brightness_history)
            
            # Adaptive adjustment based on recent brightness history
            if avg_brightness < 60:  # Very dark
                alpha, beta = 1.3, 30
            elif avg_brightness < 100:  # Dark
                alpha, beta = 1.1, 15
            elif avg_brightness > 200:  # Very bright
                alpha, beta = 0.8, -10
            elif avg_brightness > 160:  # Bright
                alpha, beta = 0.9, -5
            else:  # Normal
                alpha, beta = 1.0, 0
            
            # Apply brightness and contrast adjustment
            adjusted = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
            return adjusted
        
        return frame
    
    def enhance_face_region(self, frame, bbox):
        """Specifically enhance the face region for better recognition"""
        x, y, w, h = bbox
        
        # Extract face region with some padding
        padding = int(min(w, h) * 0.1)
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(frame.shape[1], x + w + padding)
        y2 = min(frame.shape[0], y + h + padding)
        
        face_region = frame[y1:y2, x1:x2].copy()
        
        # Apply targeted enhancements
        enhanced_face = self._enhance_face_specific(face_region)
        
        # Place enhanced face back into frame
        enhanced_frame = frame.copy()
        enhanced_frame[y1:y2, x1:x2] = enhanced_face
        
        return enhanced_frame
    
    def _enhance_face_specific(self, face_region):
        """Apply face-specific enhancements"""
        # Skin tone preservation while enhancing details
        
        # Convert to YUV for better skin tone handling
        yuv = cv2.cvtColor(face_region, cv2.COLOR_BGR2YUV)
        y, u, v = cv2.split(yuv)
        
        # Enhance Y channel (luminance) while preserving color
        y_enhanced = self.clahe.apply(y)
        
        # Slight smoothing to reduce noise while preserving edges
        y_enhanced = cv2.bilateralFilter(y_enhanced, 5, 50, 50)
        
        # Merge back
        yuv_enhanced = cv2.merge([y_enhanced, u, v])
        return cv2.cvtColor(yuv_enhanced, cv2.COLOR_YUV2BGR)


# Integration methods for your existing FacialRecognitionAPI class
class EnhancedFacialRecognitionAPI:
    """Enhanced version with lighting adaptation"""
    
    def __init__(self):
        # ... your existing initialization code ...
        self.lighting_adapter = LightingAdaptation()
    
    def detect_and_recognize_faces(self, frame):
        """Enhanced face detection with lighting adaptation"""
        if frame is None:
            return []
        
        # Apply adaptive preprocessing
        enhanced_frame = self.lighting_adapter.adaptive_preprocessing(frame)
        
        # Resize frame for faster processing if needed
        if self.is_raspberry_pi:
            processing_frame = cv2.resize(enhanced_frame, (320, 240))
            scale_x = enhanced_frame.shape[1] / processing_frame.shape[1]
            scale_y = enhanced_frame.shape[0] / processing_frame.shape[0]
        else:
            processing_frame = enhanced_frame.copy()
            scale_x = scale_y = 1.0
        
        gray = cv2.cvtColor(processing_frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        results = []
        for (x, y, w, h) in faces:
            # Scale back to original frame size
            x, y, w, h = int(x * scale_x), int(y * scale_y), int(w * scale_x), int(h * scale_y)
            
            # Enhance face region specifically
            face_enhanced_frame = self.lighting_adapter.enhance_face_region(
                enhanced_frame, (x, y, w, h))
            
            # Recognize face using enhanced frame
            name, distance = self.recognize_face(face_enhanced_frame, (x, y, w, h))
            
            results.append({
                'bbox': [x, y, w, h],
                'name': name,
                'confidence': (1 - distance) * 100 if distance < 1.0 else 0,
                'distance': distance,
                'recognized': name != "UNKNOWN"
            })
        
        return results
    
    def annotate_frame_with_recognition(self, frame):
        """Enhanced annotation with lighting info"""
        # Get lighting conditions for display
        conditions = self.lighting_adapter.analyze_lighting_conditions(frame)
        
        # Process faces with lighting adaptation
        faces = self.detect_and_recognize_faces(frame)
        
        # ... your existing annotation code ...
        
        # Add lighting condition info to overlay
        lighting_info = []
        if conditions['is_low_light']:
            lighting_info.append("LOW LIGHT")
        if conditions['is_high_light']:
            lighting_info.append("BRIGHT LIGHT")
        if conditions['is_backlit']:
            lighting_info.append("BACKLIT")
        if conditions['has_shadows']:
            lighting_info.append("SHADOWS")
        
        # Add lighting info to system overlay
        overlay_text = [
            f"Door: {'LOCKED' if self.door_locked else 'UNLOCKED'}",
            f"Faces: {len(faces)}",
            f"Known: {len(self.saved_embeddings)}",
            f"Light: {', '.join(lighting_info) if lighting_info else 'NORMAL'}",
            f"Brightness: {conditions['mean_brightness']:.0f}",
            f"Time: {datetime.now().strftime('%H:%M:%S')}"
        ]
        
        # ... rest of your annotation code ...
        
        return frame