import time
import threading
import logging
import os
import io
from datetime import datetime
from google.cloud import texttospeech  # Added for Google TTS
import pygame  # Added for audio playback

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
        
        # Google TTS client initialization
        self.tts_client = None
        self.pygame_initialized = False
        self._init_google_tts()
        
        # Audio playback queue
        self.audio_queue = []
        self.is_playing = False
    
    def _init_google_tts(self):
        """Initialize Google TTS client"""
        try:
            # Set up Google Cloud credentials (ensure GOOGLE_APPLICATION_CREDENTIALS is set)
            if hasattr(self.config, 'GOOGLE_APPLICATION_CREDENTIALS'):
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = self.config.GOOGLE_APPLICATION_CREDENTIALS
            
            self.tts_client = texttospeech.TextToSpeechClient()
            
            # Initialize pygame for audio playback
            pygame.mixer.init()
            self.pygame_initialized = True
            logger.info("Google TTS client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Google TTS: {e}")
            self.tts_client = None
    
    def synthesize_speech(self, text, voice_name=None, language_code="en-US"):
        """Synthesize speech using Google TTS"""
        if not self.tts_client:
            logger.warning("Google TTS client not available")
            return None
        
        try:
            # Set up the text input
            synthesis_input = texttospeech.SynthesisInput(text=text)
            
            # Build the voice request
            voice = texttospeech.VoiceSelectionParams(
                language_code=language_code,
                name=voice_name or getattr(self.config, 'TTS_VOICE', 'en-US-Wavenet-D'),
                ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
            )
            
            # Select the type of audio file
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3,
                speaking_rate=getattr(self.config, 'TTS_SPEAKING_RATE', 1.0),
                pitch=getattr(self.config, 'TTS_PITCH', 0.0),
                volume_gain_db=getattr(self.config, 'TTS_VOLUME_GAIN', 0.0)
            )
            
            # Perform the text-to-speech request
            response = self.tts_client.synthesize_speech(
                input=synthesis_input, voice=voice, audio_config=audio_config
            )
            
            return response.audio_content
            
        except Exception as e:
            logger.error(f"Google TTS synthesis failed: {e}")
            return None
    
    def play_audio(self, audio_content):
        """Play audio content using pygame"""
        if not self.pygame_initialized or not audio_content:
            return False
        
        try:
            # Create a temporary audio file in memory
            audio_file = io.BytesIO(audio_content)
            pygame.mixer.music.load(audio_file)
            pygame.mixer.music.play()
            
            # Wait for playback to finish
            while pygame.mixer.music.get_busy():
                pygame.time.wait(100)
                
            return True
            
        except Exception as e:
            logger.error(f"Audio playback failed: {e}")
            return False
    
    def speak(self, text, async_playback=True):
        """Speak text using Google TTS"""
        if not self.tts_client:
            logger.warning("Cannot speak - TTS client not available")
            return False
        
        try:
            # Synthesize speech
            audio_content = self.synthesize_speech(text)
            if not audio_content:
                return False
            
            if async_playback:
                # Play audio in a separate thread to avoid blocking
                def play_async():
                    self.play_audio(audio_content)
                
                thread = threading.Thread(target=play_async)
                thread.daemon = True
                thread.start()
                return True
            else:
                # Play audio synchronously
                return self.play_audio(audio_content)
                
        except Exception as e:
            logger.error(f"Speech synthesis failed: {e}")
            return False
    
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
        welcome_message = f"Welcome home {person_name}. The door is now unlocked."
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
        
        # SPEAK: Door locked message with Google TTS
        if method != 'automatic':
            lock_message = "Door is now locked"
        else:
            lock_message = "Door has been automatically locked"
        
        self.speak(lock_message)
        
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
            # SPEAK: Access denied to ESP32 request with Google TTS
            if not self.is_unlock_window_valid():
                self.speak("Access not authorized at this time")
            
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
        
        # SPEAK: Countdown warning with Google TTS
        countdown_seconds = int(self.config.DOOR_RELOCK_COUNTDOWN)
        countdown_msg = f"The door will lock in {countdown_seconds} seconds"
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
        
        # SPEAK: Access denied with Google TTS
        self.speak("Access denied. Person not recognized in the system.")
        
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
        success = self.speak(message, async_playback=False)
        return f"Speech test {'succeeded' if success else 'failed'}: {message}"
    
    def cleanup(self):
        """Cleanup resources"""
        self._cancel_timers()
        if self.pygame_initialized:
            pygame.mixer.quit()
        logger.info("Door controller cleaned up")