import time
import threading
import logging
import subprocess
import os
import tempfile
from datetime import datetime
from gtts import gTTS
import pygame

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
        
        # Google TTS settings with defaults
        self.tts_language = getattr(config, 'TTS_LANGUAGE', 'en')
        self.tts_slow = getattr(config, 'TTS_SLOW', False)
        self.tts_domain = getattr(config, 'TTS_DOMAIN', 'com')  # com, co.uk, ca, etc.
        
        # Audio playback settings
        self.audio_volume = getattr(config, 'AUDIO_VOLUME', 0.7)
        
        # Initialize pygame mixer for audio playback
        self._init_audio()
        
        # Verify Google TTS is working
        self._verify_gtts()
    
    def _init_audio(self):
        """Initialize pygame mixer for audio playback"""
        try:
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
            pygame.mixer.music.set_volume(self.audio_volume)
            logger.info("Audio system initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize audio system: {e}")
            return False
    
    def _verify_gtts(self):
        """Check if Google TTS is working"""
        try:
            # Test with a simple phrase
            test_tts = gTTS(text="Test", lang=self.tts_language, slow=self.tts_slow, tld=self.tts_domain)
            
            # Create a temporary file to test
            with tempfile.NamedTemporaryFile(delete=True, suffix='.mp3') as temp_file:
                test_tts.save(temp_file.name)
                logger.info("Google TTS is working correctly")
                return True
                
        except Exception as e:
            logger.error(f"Google TTS verification failed: {e}")
            logger.warning("Make sure you have internet connection and gtts package installed")
            return False
    
    def speak(self, text, language=None, slow=None, domain=None):
        """Use Google TTS to speak text aloud (non-blocking)"""
        try:
            # Use provided parameters or fall back to defaults
            lang = language or self.tts_language
            is_slow = slow if slow is not None else self.tts_slow
            tld = domain or self.tts_domain
            
            # Create TTS object
            tts = gTTS(text=text, lang=lang, slow=is_slow, tld=tld)
            
            # Create temporary file for audio
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
                temp_filename = temp_file.name
                
            # Save TTS audio to temporary file
            tts.save(temp_filename)
            
            # Play audio in a separate thread to avoid blocking
            def play_audio():
                try:
                    pygame.mixer.music.load(temp_filename)
                    pygame.mixer.music.play()
                    
                    # Wait for playback to finish
                    while pygame.mixer.music.get_busy():
                        time.sleep(0.1)
                    
                    # Clean up temporary file
                    try:
                        os.unlink(temp_filename)
                    except:
                        pass  # Ignore cleanup errors
                        
                except Exception as e:
                    logger.error(f"Audio playback error: {e}")
                    # Clean up on error
                    try:
                        os.unlink(temp_filename)
                    except:
                        pass
            
            # Start playback in background thread
            audio_thread = threading.Thread(target=play_audio, daemon=True)
            audio_thread.start()
            
            logger.debug(f"Speaking: {text}")
            return True
            
        except Exception as e:
            logger.error(f"Google TTS error: {e}")
            # Fallback to system notification sound or beep if available
            self._fallback_audio_notification()
            return False
    
    def _fallback_audio_notification(self):
        """Fallback audio notification when TTS fails"""
        try:
            # Try to play system bell/beep
            subprocess.run(['paplay', '/usr/share/sounds/alsa/Front_Left.wav'], 
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=2)
        except:
            try:
                # Alternative system beep
                subprocess.run(['aplay', '/usr/share/sounds/alsa/Front_Left.wav'], 
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=2)
            except:
                # Last resort - console beep
                print('\a')  # ASCII bell character
    
    def speak_async(self, text, **kwargs):
        """Speak text asynchronously without blocking the main thread"""
        def async_speak():
            self.speak(text, **kwargs)
        
        thread = threading.Thread(target=async_speak, daemon=True)
        thread.start()
        return thread
    
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
        
        # SPEAK: Access granted message with Google TTS
        welcome_message = f"Welcome home {person_name}. Door unlocked."
        self.speak_async(welcome_message)
        
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
            lock_message = "Door locked"
        else:
            lock_message = "Door automatically locked"
        
        self.speak_async(lock_message)
        
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
                self.speak_async("Access not authorized")
            
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
        countdown_seconds = int(self.config.DOOR_RELOCK_COUNTDOWN)
        if countdown_seconds > 5:  # Only announce if it's a meaningful duration
            countdown_msg = f"Door will lock in {countdown_seconds} seconds"
            self.speak_async(countdown_msg)
        
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
        self.speak_async("Access denied. Person not recognized.")
        
        logger.info("Access denied - unknown person")
    
    def _cancel_timers(self):
        """Cancel all active timers"""
        if self._relock_timer and self._relock_timer.is_alive():
            self._relock_timer.cancel()
            self._relock_timer = None
        
        if self._esp32_relock_timer and self._esp32_relock_timer.is_alive():
            self._esp32_relock_timer.cancel()
            self._esp32_relock_timer = None
    
    def test_speech(self, message="Testing Google text to speech synthesis"):
        """Test method to verify Google TTS is working"""
        try:
            success = self.speak(message)
            return f"Google TTS test {'succeeded' if success else 'failed'}: {message}"
        except Exception as e:
            return f"Google TTS test failed with error: {e}"
    
    def test_speech_async(self, message="Testing Google text to speech synthesis"):
        """Test method to verify Google TTS is working asynchronously"""
        thread = self.speak_async(message)
        return f"Google TTS async test started: {message}"
    
    def set_voice_settings(self, language=None, slow=None, domain=None, volume=None):
        """Update TTS voice settings"""
        if language:
            self.tts_language = language
        if slow is not None:
            self.tts_slow = slow
        if domain:
            self.tts_domain = domain
        if volume is not None:
            self.audio_volume = max(0.0, min(1.0, volume))  # Clamp between 0 and 1
            pygame.mixer.music.set_volume(self.audio_volume)
        
        logger.info(f"Voice settings updated: lang={self.tts_language}, slow={self.tts_slow}, domain={self.tts_domain}, volume={self.audio_volume}")
    
    def cleanup(self):
        """Cleanup resources"""
        self._cancel_timers()
        
        # Stop any playing audio
        try:
            pygame.mixer.music.stop()
            pygame.mixer.quit()
        except:
            pass
        
        logger.info("Door controller cleaned up")