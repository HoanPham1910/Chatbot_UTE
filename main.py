import time
from pathlib import Path
from module.config import *
from module.cache import OCRCache, AnswerCache
from module.search import PDFSearch
from module.llm import OllamaLLM, GroqLLM
from module.tts import GoogleTTS
from module.voice_input import VoiceInput

class MakerSpaceQA:
    def __init__(self, enable_tts=False, enable_voice=False):
        # Load components
        self.search = PDFSearch(INDEX_DIR)
        self.ocr_cache = OCRCache(OCR_CACHE_FILE)
        self.answer_cache = AnswerCache(ANSWER_CACHE_FILE)
        
        if not Path(OCR_CACHE_FILE).exists():
            self.ocr_cache.precache_all(self.search.image_embeddings)
        
        # Initialize LLM
        if USE_GROQ:
            self.llm = GroqLLM(api_key=GROQ_API_KEY, model=GROQ_MODEL)
            if not self.llm.check():
                raise Exception("Groq API key invalid! Get key at: https://console.groq.com")
        else:
            self.llm = OllamaLLM()
            if not self.llm.check():
                raise Exception(f"Model {OLLAMA_MODEL} not found. Run: ollama pull {OLLAMA_MODEL}")
            self.llm.warmup()
        
        # Initialize TTS & Voice
        tts_speed = TTS_SPEED if 'TTS_SPEED' in globals() else 1.5
        self.tts = GoogleTTS(lang='vi', speed=tts_speed) if enable_tts else None
        
        if enable_voice:
            voice = VoiceInput(language='vi-VN')
            self.voice = voice if voice.test_microphone() else None
        else:
            self.voice = None
    
    def ask(self, question, k=SEARCH_TOP_K, manual_context=None):
        # Build context from search or manual input
        if manual_context:
            context = manual_context
        else:
            results = self.search.search(question, k=k)
            if not results:
                return None, False
            
            context = ""
            for r in results:
                page = r["image"].path.split('/')[-1].replace('page_', '').replace('.png', '')
                text = self.ocr_cache.get(r["image"].path)
                context += f"--- Trang {page} ---\n{text}\n\n"
        
        # Check cache or get new answer
        cached = self.answer_cache.get(question, context)
        if cached:
            return cached, True
        
        answer = self.llm.ask(question, context)
        self.answer_cache.set(question, context, answer)
        return answer, False
    
    def speak(self, text):
        if self.tts:
            self.tts.speak(text)
    
    def listen(self):
        return self.voice.listen() if self.voice else None

def main():
    # Initialize system
    qa = MakerSpaceQA(enable_tts=True, enable_voice=True)
    
    tts_enabled = qa.tts is not None
    voice_enabled = qa.voice is not None
    
    while True:
        # Get input (voice or text)
        print("Question (Enter=mic): " if voice_enabled else "Question: ", end='', flush=True)
        q = input().strip()
        
        if not q and voice_enabled:
            q = qa.listen()
            if q:
                print(f"🎤 {q}")
        
        if not q:
            continue
        
        # Handle exit command
        if q.lower() in ['exit', 'quit', 'q', 'thoát']:
            break
        
        # Process question
        start = time.time()
        answer, from_cache = qa.ask(q)
        
        if answer is None:
            print("\n Không tìm thấy. Nhập context? (y/n): ", end='')
            if input().strip().lower() == 'y':
                print("Nhập context (dòng trống để kết thúc):")
                lines = []
                while True:
                    line = input()
                    if not line:
                        break
                    lines.append(line)
                
                if lines:
                    answer, from_cache = qa.ask(q, manual_context="\n".join(lines))
                    print(f"\n✓ {answer}")
                    print(f"{time.time()-start:.2f}s - Đã lưu cache!\n")
        else:
            print(f"\n{answer}")
            print(f"{time.time()-start:.2f}s [{'CACHE' if from_cache else 'LLM'}]\n")
        
        if tts_enabled and answer:
            qa.speak(answer)

if __name__ == "__main__":
    main()