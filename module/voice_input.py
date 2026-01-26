import speech_recognition as sr

class VoiceInput:
    """Nhận input bằng giọng nói (tiếng Việt)"""
    def __init__(self, language='vi-VN'):
        self.recognizer = sr.Recognizer()
        self.language = language
        
        # Điều chỉnh cho noise
        self.recognizer.energy_threshold = 4000
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.8
    
    def listen(self, timeout=5):
        try:
            with sr.Microphone() as source:
                print(" Đang lắng nghe... (nói 'hủy' để thoát)")
                
                # Điều chỉnh noise
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                
                # Lắng nghe
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=10)
                
                print("Đang xử lý...")
                
                # Nhận dạng với Google Speech API (FREE)
                text = self.recognizer.recognize_google(audio, language=self.language)
                
                return text.strip()
                
        except sr.WaitTimeoutError:
            print("⚠️  Timeout - không nghe thấy gì")
            return None
        except sr.UnknownValueError:
            print("⚠️  Không nhận dạng được giọng nói")
            return None
        except sr.RequestError as e:
            print(f"⚠️  Lỗi kết nối Google API: {e}")
            return None
        except Exception as e:
            print(f"⚠️  Lỗi: {e}")
            return None
    
    def test_microphone(self):
        """Test xem mic có hoạt động không"""
        try:
            with sr.Microphone() as source:
                print("✓ Microphone detected!")
                return True
        except Exception as e:
            print(f"✗ Microphone error: {e}")
            return False