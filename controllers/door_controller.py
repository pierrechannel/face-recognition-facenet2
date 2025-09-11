import time
import threading
import logging
import subprocess  # Added for eSpeak
from datetime import datetime

logger = logging.getLogger(__name__)

class DoorController:
    """Handles door locking/unlocking logic and timers"""
    
    def __init__(self, config):
        self.config = config
        self.door_locked = True
        self.unlock_available_until = 0
        self.last_recognition = None
        
        # Timers
        self._relock_timer = None
        self._esp32_relock_timer = None
        
        # Callbacks for external notifications
        self.on_door_locked = None
        self.on_door_unlocked = None
        
        # eSpeak settings
        self.espeak_voice = getattr(config, 'ESPEAK_VOICE', 'en-us')
        self.espeak_speed = getattr(config, 'ESPEAK_SPEED', 170)
        self.espeak_pitch = getattr(config, 'ESPEAK_PITCH', 50)
    
    def speak(self, text, voice=None, speed=None, pitch=None):
        """Use eSpeak to speak the given text aloud."""
        try:
            voice = voice or self.espeak_voice
            speed = speed or self.espeak_speed
            pitch = pitch or self.espeak_pitch
            
            command = [
                'espeak',
                '-v', voice,
                '-s', str(speed),
                '-p', str(pitch),
                text
            ]
            # Use subprocess.Popen to run asynchronously so it doesn't block
            subprocess.Popen(command)
            logger.debug(f"Spoke: {text}")
        except Exception as e:
            logger.error(f"eSpeak error: {e}")
    
    def unlock_door(self, person_name, confidence, distance, method='recognition'):
        """Unlock the door for a recognized person"""
        # Cancel any existing timers
        self._cancel_timers()
        
        # Unlock the door
        self.door_locked = False
        
        # Update last recognition
        self.last_recognition = {
            'name': person_name,
            'timestamp': datetime.now().isoformat(),
            'confidence': confidence,
            'status': 'GRANTED',
            'method': method
        }
        
        logger.info(f"Door unlocked for {person_name} via {method} (confidence: {confidence:.1f}%)")
        
        # SPEAK: Access granted message
        welcome_message = f"Welcome home {person_name}. Door unlocked."
        self.speak(welcome_message)
        
        # Set unlock window expiration
        self.unlock_available_until = time.time() + self.config.DOOR_UNLOCK_DURATION
        
        # Start auto-relock sequence
        self._start_auto_relock()
        
        # Notify external systems
        if self.on_door_unlocked:
            self.on_door_unlocked(person_name, method)
        
        return True
    
    def lock_door(self, method='manual'):
        """Lock the door"""
        self._cancel_timers()
        
        self.door_locked = True
        self.unlock_available_until = 0
        
        logger.info(f"Door locked via {method}")
        
        # SPEAK: Door locked message (only if not automatic relock to avoid spam)
        if method != 'automatic':
            self.speak("Door locked")
        else:
            self.speak("Door automatically locked")
        
        # Notify external systems
        if self.on_door_locked:
            self.on_door_locked(method)
        
        return True
    
    def is_unlock_window_valid(self):
        """Check if the unlock window is still valid"""
        return time.time() <= self.unlock_available_until
    
    def get_door_status(self):
        """Get current door status"""
        return {
            'locked': self.door_locked,
            'unlock_window_valid': self.is_unlock_window_valid(),
            'unlock_expires_at': self.unlock_available_until if self.unlock_available_until > 0 else None,
            'last_recognition': self.last_recognition
        }
    
    def process_access_request(self, recognized_faces, threshold):
        """Process access request based on recognized faces"""
        access_granted = False
        recognized_person = None
        
        for face in recognized_faces:
            if face['recognized'] and face['confidence'] > (1 - threshold) * 100:
                access_granted = True
                recognized_person = face['name']
                self.unlock_door(
                    person_name=face['name'],
                    confidence=face['confidence'],
                    distance=face['distance'],
                    method='facial_recognition'
                )
                break
        
        if not access_granted and recognized_faces:
            self._deny_access()
        
        return {
            'access_granted': access_granted,
            'person': recognized_person,
            'faces_detected': len(recognized_faces),
            'recognized_faces': [f for f in recognized_faces if f['recognized']],
            'unlock_window_valid': self.is_unlock_window_valid()
        }
    
    def handle_esp32_request(self):
        """Handle ESP32 door status request"""
        current_time = time.time()
        
        # Check if access was recently granted and window is still valid
        should_open = (
            self.last_recognition and 
            self.last_recognition.get('status') == 'GRANTED' and
            current_time <= self.unlock_available_until and
            not self.door_locked
        )
        
        if should_open:
            person_name = self.last_recognition.get('name', 'KNOWN')
            
            # Clear the unlock window to prevent multiple opens
            self.unlock_available_until = 0
            
            # Set up ESP32-specific auto-relock timer (shorter duration)
            self._esp32_relock_timer = threading.Timer(5.0, self._auto_relock)
            self._esp32_relock_timer.start()
            
            logger.info(f"Door access granted to ESP32 for {person_name}")
            
            return {
                'open': 1,
                'message': 'ACCESS_GRANTED',
                'person': person_name,
                'door_locked': False
            }
        else:
            # SPEAK: Access denied to ESP32 request
            if not self.is_unlock_window_valid():
                self.speak("Access not authorized")
            
            return {
                'open': 0,
                'message': 'WAITING',
                'door_locked': self.door_locked
            }
    
    def _start_auto_relock(self):
        """Start the auto-relock sequence"""
        # Wait initial delay before starting countdown
        self._relock_timer = threading.Timer(
            self.config.DOOR_AUTO_RELOCK_DELAY, 
            self._start_relock_countdown
        )
        self._relock_timer.start()
    
    def _start_relock_countdown(self):
        """Start the final relock countdown"""
        self._relock_timer = threading.Timer(
            self.config.DOOR_RELOCK_COUNTDOWN, 
            self._auto_relock
        )
        self._relock_timer.start()
        
        # SPEAK: Countdown warning
        countdown_msg = f"Door will lock in {int(self.config.DOOR_RELOCK_COUNTDOWN)} seconds"
        self.speak(countdown_msg)
        
        logger.info(f"Door will auto-relock in {self.config.DOOR_RELOCK_COUNTDOWN} seconds")
    
    def _auto_relock(self):
        """Automatically relock the door"""
        self.lock_door(method='automatic')
    
    def _deny_access(self):
        """Handle access denial"""
        self.last_recognition = {
            'name': 'UNKNOWN',
            'timestamp': datetime.now().isoformat(),
            'confidence': 0,
            'status': 'DENIED',
            'method': 'facial_recognition'
        }
        
        # SPEAK: Access denied
        self.speak("Access denied. Person not recognized.")
        
        logger.info("Access denied - unknown person")
    
    def _cancel_timers(self):
        """Cancel all active timers"""
        if self._relock_timer and self._relock_timer.is_alive():
            self._relock_timer.cancel()
            self._relock_timer = None
        
        if self._esp32_relock_timer and self._esp32_relock_timer.is_alive():
            self._esp32_relock_timer.cancel()
            self._esp32_relock_timer = None
    
    def test_speech(self, message="Testing speech synthesis"):
        """Test method to verify eSpeak is working"""
        self.speak(message)
        return f"Spoke: {message}"
    
    def cleanup(self):
        """Cleanup resources"""
        self._cancel_timers()
        logger.info("Door controller cleaned up")