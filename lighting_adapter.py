import cv2
import numpy as np
import argparse
from typing import Tuple, Optional

class AdaptiveLightingProcessor:
    def __init__(self):
        self.previous_frame = None
        
    def adjust_gamma(self, image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        """Adjust image gamma correction"""
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 
                         for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)
    
    def calculate_brightness(self, image: np.ndarray) -> float:
        """Calculate average brightness of the image"""
        if len(image.shape) == 3:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        return np.mean(gray)
    
    def adaptive_histogram_equalization(self, image: np.ndarray) -> np.ndarray:
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
        if len(image.shape) == 3:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            
            # Merge channels and convert back to BGR
            merged = cv2.merge((cl, a, b))
            return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
        else:
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            return clahe.apply(image)
    
    def adjust_exposure(self, image: np.ndarray, target_brightness: float = 127) -> np.ndarray:
        """Automatically adjust exposure based on brightness"""
        current_brightness = self.calculate_brightness(image)
        
        if current_brightness < 30:  # Very dark
            # Brighten the image
            gamma = 0.5
            result = self.adjust_gamma(image, gamma)
        elif current_brightness > 200:  # Very bright
            # Darken the image
            gamma = 1.5
            result = self.adjust_gamma(image, gamma)
        else:
            # Normal lighting, use histogram equalization
            result = self.adaptive_histogram_equalization(image)
        
        return result
    
    def remove_shadows(self, image: np.ndarray) -> np.ndarray:
        """Remove shadows using morphological operations"""
        if len(image.shape) == 3:
            rgb_planes = cv2.split(image)
            result_planes = []
            
            for plane in rgb_planes:
                dilated = cv2.dilate(plane, np.ones((7,7), np.uint8))
                bg = cv2.medianBlur(dilated, 21)
                diff = 255 - cv2.absdiff(plane, bg)
                norm = cv2.normalize(diff, None, alpha=0, beta=255, 
                                   norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
                result_planes.append(norm)
            
            return cv2.merge(result_planes)
        return image
    
    def enhance_details(self, image: np.ndarray) -> np.ndarray:
        """Enhance image details using sharpening"""
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        return cv2.filter2D(image, -1, kernel)
    
    def white_balance(self, image: np.ndarray) -> np.ndarray:
        """Simple white balance adjustment"""
        result = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        avg_a = np.average(result[:, :, 1])
        avg_b = np.average(result[:, :, 2])
        result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
        result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
        return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Main processing pipeline"""
        # Store original frame for comparison
        original = frame.copy()
        
        # Calculate current brightness
        brightness = self.calculate_brightness(frame)
        
        # Apply appropriate processing based on lighting conditions
        if brightness < 50:  # Low light conditions
            # Brighten and reduce noise
            frame = self.adjust_gamma(frame, 0.6)
            frame = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)
            frame = self.adaptive_histogram_equalization(frame)
            
        elif brightness > 180:  # Overexposed conditions
            # Darken and enhance details
            frame = self.adjust_gamma(frame, 1.8)
            frame = self.enhance_details(frame)
            
        else:  # Normal lighting
            # Standard enhancement
            frame = self.adaptive_histogram_equalization(frame)
            frame = self.white_balance(frame)
            frame = self.enhance_details(frame)
        
        # Remove shadows in all conditions
        frame = self.remove_shadows(frame)
        
        return frame
    
    def process_image(self, image_path: str, output_path: Optional[str] = None) -> np.ndarray:
        """Process a single image file"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        processed = self.process_frame(image)
        
        if output_path:
            cv2.imwrite(output_path, processed)
        
        return processed
    
    def process_video(self, video_path: str, output_path: Optional[str] = None):
        """Process video file or webcam feed"""
        cap = cv2.VideoCapture(video_path if video_path != '0' else 0)
        
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_path, fourcc, 20.0, 
                                (int(cap.get(3)), int(cap.get(4))))
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            processed = self.process_frame(frame)
            
            # Display brightness information
            brightness = self.calculate_brightness(frame)
            cv2.putText(processed, f"Brightness: {brightness:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show original and processed side by side
            comparison = np.hstack((frame, processed))
            cv2.imshow('Original vs Processed', comparison)
            
            if output_path:
                out.write(processed)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Adaptive Lighting Correction with OpenCV')
    parser.add_argument('--input', type=str, default='0', 
                       help='Input image/video path or 0 for webcam')
    parser.add_argument('--output', type=str, 
                       help='Output file path (optional)')
    parser.add_argument('--mode', type=str, choices=['image', 'video'], default='video',
                       help='Processing mode: image or video')
    
    args = parser.parse_args()
    
    processor = AdaptiveLightingProcessor()
    
    try:
        if args.mode == 'image':
            processor.process_image(args.input, args.output)
            print(f"Image processed successfully!")
            if args.output:
                print(f"Output saved to: {args.output}")
        else:
            processor.process_video(args.input, args.output)
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()