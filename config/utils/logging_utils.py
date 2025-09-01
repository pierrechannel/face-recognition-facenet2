import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler

def setup_logging(log_level='INFO', log_dir='logs'):
    """Setup application logging"""
    
    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),  # Console output
            RotatingFileHandler(
                os.path.join(log_dir, 'app.log'),
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5
            )
        ]
    )
    
    # Create specific loggers
    loggers = {
        'access': setup_access_logger(log_dir),
        'recognition': setup_recognition_logger(log_dir),
        'camera': setup_camera_logger(log_dir),
        'door': setup_door_logger(log_dir)
    }
    
    return loggers

def setup_access_logger(log_dir):
    """Setup access-specific logger"""
    logger = logging.getLogger('access')
    handler = RotatingFileHandler(
        os.path.join(log_dir, 'access.log'),
        maxBytes=5*1024*1024,  # 5MB
        backupCount=10
    )
    handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(message)s'
    ))
    logger.addHandler(handler)
    return logger

def setup_recognition_logger(log_dir):
    """Setup recognition-specific logger"""
    logger = logging.getLogger('recognition')
    handler = RotatingFileHandler(
        os.path.join(log_dir, 'recognition.log'),
        maxBytes=5*1024*1024,  # 5MB
        backupCount=5
    )
    handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    ))
    logger.addHandler(handler)
    return logger

def setup_camera_logger(log_dir):
    """Setup camera-specific logger"""
    logger = logging.getLogger('camera')
    handler = RotatingFileHandler(
        os.path.join(log_dir, 'camera.log'),
        maxBytes=2*1024*1024,  # 2MB
        backupCount=3
    )
    handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    ))
    logger.addHandler(handler)
    return logger

def setup_door_logger(log_dir):
    """Setup door control specific logger"""
    logger = logging.getLogger('door')
    handler = RotatingFileHandler(
        os.path.join(log_dir, 'door.log'),
        maxBytes=2*1024*1024,  # 2MB
        backupCount=5
    )
    handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    ))
    logger.addHandler(handler)
    return logger

class AccessLogger:
    """Specialized logger for access events"""
    
    def __init__(self, log_file='access_log.txt'):
        self.log_file = log_file
        self.logger = logging.getLogger('access')
    
    def log_access_attempt(self, name, status, confidence, distance=None):
        """Log an access attempt"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Create structured log entry
        log_entry = {
            'timestamp': timestamp,
            'name': name,
            'status': status,
            'confidence': confidence,
            'distance': distance
        }
        
        # Log to structured logger
        self.logger.info(f"Access: {name} - {status} - {confidence}%")
        
        # Also log to simple text file for backward compatibility
        with open(self.log_file, 'a') as f:
            f.write(f"[{timestamp}] {name} - {status} - {confidence}\n")
        
        return log_entry
    
    def log_door_action(self, action, user=None, method='automatic'):
        """Log door lock/unlock actions"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        door_logger = logging.getLogger('door')
        
        message = f"Door {action}"
        if user:
            message += f" for {user}"
        message += f" ({method})"
        
        door_logger.info(message)
        
        with open(self.log_file, 'a') as f:
            f.write(f"[{timestamp}] DOOR {action.upper()} - {method}\n")
    
    def get_recent_logs(self, limit=50):
        """Get recent access logs"""
        try:
            with open(self.log_file, 'r') as f:
                lines = f.readlines()
            return lines[-limit:] if lines else []
        except FileNotFoundError:
            return []