

# Option 2: Dùng gTTS (Google TTS - cần internet, giọng tự nhiên hơn)
# pip install gtts pygame

from gtts import gTTS
import pygame
import tempfile
import os

class GoogleTTS:
    def __init__(self, lang='vi', speed=None, slow=False):
        self.lang = lang
        self.slow = slow
        pygame.mixer.init()
    
    def speak(self, text):
        tts = gTTS(text=text, lang=self.lang, slow=self.slow)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
            temp_file = fp.name
            tts.save(temp_file)
        
        pygame.mixer.music.load(temp_file)
        pygame.mixer.music.play()
        
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        
        pygame.mixer.music.unload()
        os.unlink(temp_file)

    
    def stop(self):
        """Dừng phát"""
        pygame.mixer.music.stop()
