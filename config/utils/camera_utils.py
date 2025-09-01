import cv2
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class CameraManager:
    """Utility class for camera management"""
    
    def __init__(self, is_raspberry_pi=False):
        self.is_raspberry_pi = is_raspberry_pi
        self.available_cameras = self.detect_cameras()
    
    def detect_cameras(self):
        """Detect available cameras"""
        available = []
        for i in range(5):  # Check first 5 camera indices
            if self.is_raspberry_pi:
                cap = cv2.VideoCapture(i)
            else:
                cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            
            if cap.isOpened():
                available.append(i)
                cap.release()
        
        logger.info(f"Available cameras: {available}")
        return available
    
    @contextmanager
    def get_camera(self, camera_id=None, width=640, height=480, fps=30):
        """Context manager for camera operations"""
        cap = None
        try:
            # Use first available camera if none specified
            if camera_id is None:
                if not self.available_cameras:
                    raise Exception("No cameras available")
                camera_id = self.available_cameras[0]
            
            # Initialize camera
            if self.is_raspberry_pi:
                cap = cv2.VideoCapture(camera_id)
            else:
                cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
            
            if not cap.isOpened():
                raise Exception(f"Cannot open camera {camera_id}")
            
            # Set camera properties
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, fps)
            
            # Verify settings
            actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            actual_fps = cap.get(cv2.CAP_PROP_FPS)
            
            logger.info(f"Camera {camera_id} initialized: {actual_width}x{actual_height} @ {actual_fps}fps")
            
            yield cap
            
        finally:
            if cap:
                cap.release()
    
    def test_camera(self, camera_id, duration=5):
        """Test a camera for a specified duration"""
        try:
            with self.get_camera(camera_id) as cap:
                frames_captured = 0
                import time
                start_time = time.time()
                
                while time.time() - start_time < duration:
                    ret, frame = cap.read()
                    if ret:
                        frames_captured += 1
                    time.sleep(0.1)
                
                fps = frames_captured / duration
                logger.info(f"Camera {camera_id} test: {frames_captured} frames in {duration}s ({fps:.1f} fps)")
                return True, fps
                
        except Exception as e:
            logger.error(f"Camera {camera_id} test failed: {e}")
            return False, 0