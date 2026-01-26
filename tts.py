

# Option 2: Dùng gTTS (Google TTS - cần internet, giọng tự nhiên hơn)
# pip install gtts pygame

from gtts import gTTS
import pygame
import tempfile
import os

class GoogleTTS:
    def __init__(self, lang='vi', slow=True):
        """
        lang: ngôn ngữ ('vi', 'en', ...)
        slow: đọc chậm hay không
        """
        self.lang = lang
        self.slow = slow
        pygame.mixer.init()
    
    def speak(self, text):
        """Đọc text qua loa bằng Google TTS"""
        # Tạo file tạm
        tts = gTTS(text=text, lang=self.lang, slow=self.slow)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
            temp_file = fp.name
            tts.save(temp_file)
        
        # Phát audio
        pygame.mixer.music.load(temp_file)
        pygame.mixer.music.play()
        
        # Chờ phát xong
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        
        # Xóa file tạm
        pygame.mixer.music.unload()
        os.unlink(temp_file)
    
    def stop(self):
        """Dừng phát"""
        pygame.mixer.music.stop()
