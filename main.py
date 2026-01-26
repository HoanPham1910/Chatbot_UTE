import time
from pathlib import Path
from config import *
from cache import OCRCache, AnswerCache
from search import PDFSearch
from llm import OllamaLLM, GroqLLM
from tts import GoogleTTS

class MakerSpaceQA:
    def __init__(self, enable_tts=False, tts_type='gtts'):
        print("Loading index...")
        self.search = PDFSearch(INDEX_DIR)
        
        print("Loading caches...")
        self.ocr_cache = OCRCache(OCR_CACHE_FILE)
        self.answer_cache = AnswerCache(ANSWER_CACHE_FILE)
        
        if not Path(OCR_CACHE_FILE).exists():
            print("Pre-caching OCR...")
            self.ocr_cache.precache_all(self.search.image_embeddings)
        
        # Initialize LLM based on config
        print("Checking LLM...")
        if USE_GROQ:
            print(f"Using Groq API: {GROQ_MODEL}")
            self.llm = GroqLLM(api_key=GROQ_API_KEY, model=GROQ_MODEL)
            if not self.llm.check():
                raise Exception(f"Groq API key không hợp lệ! Kiểm tra GROQ_API_KEY trong config.py\nLấy key tại: https://console.groq.com")
            print("✓ Groq API connected!")
        else:
            print(f"Using Ollama: {OLLAMA_MODEL}")
            self.llm = OllamaLLM()
            if not self.llm.check():
                raise Exception(f"Model {OLLAMA_MODEL} not found. Run: ollama pull {OLLAMA_MODEL}")
            print("Warming up LLM...")
            self.llm.warmup()
        
        # TTS
        self.tts = None
        if enable_tts:
            print(f"Initializing TTS ({tts_type})...")
            self.tts = GoogleTTS(lang='vi')
                
        print("Ready\n")
    
    def ask(self, question, k=SEARCH_TOP_K, manual_context=None):
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
        
        cached = self.answer_cache.get(question, context)
        if cached:
            return cached, True
        
        answer = self.llm.ask(question, context)
        self.answer_cache.set(question, context, answer)
        return answer, False
    
    def speak(self, text):
        """Đọc text qua loa nếu TTS được bật"""
        if self.tts:
            self.tts.speak(text)

def main():
    try:
        # Bật TTS: enable_tts=True
        qa = MakerSpaceQA(enable_tts=True, tts_type='gtts')
    except Exception as e:
        print(f"Error: {e}")
        return
    
    llm_type = "Groq (Cloud)" if USE_GROQ else f"Ollama (Local)"
    llm_model = GROQ_MODEL if USE_GROQ else OLLAMA_MODEL
    
    print("="*60)
    print(f"MakerSpace Q&A - {llm_type}: {llm_model}")
    print("Commands: 'test', 'stats', 'tts on/off', 'exit'")
    print(f"TTS: {'ON' if qa.tts else 'OFF'}")
    print("="*60 + "\n")
    
    test_questions = [
        "Ai quản lý khu MakerSpace?",
        "MakerSpace mở cửa mấy giờ?",
        "Ở MakerSpace có được ngủ lại không?",
        "CISAT là gì?",
    ]
    
    tts_enabled = qa.tts is not None
    
    while True:
        try:
            q = input("Question: ").strip()
            if not q:
                continue
            
            if q.lower() in ['exit', 'quit', 'q']:
                stats = qa.answer_cache.stats()
                print(f"\nStats: {stats['total']} questions, {stats['hits']} cached ({stats['hit_rate']})")
                break
            
            # Toggle TTS
            if q.lower() == 'tts on':
                tts_enabled = True
                print("🔊 TTS enabled\n")
                continue
            
            if q.lower() == 'tts off':
                tts_enabled = False
                print("🔇 TTS disabled\n")
                continue
            
            if q.lower() == 'stats':
                s = qa.answer_cache.stats()
                print(f"Total: {s['total']}, Hits: {s['hits']}, Misses: {s['misses']}, Rate: {s['hit_rate']}, Cached: {s['cached']}\n")
                continue
            
            if q.lower() == 'test':
                print("\nRunning tests...\n")
                for test_q in test_questions:
                    print(f"Q: {test_q}")
                    start = time.time()
                    ans, cached = qa.ask(test_q)
                    
                    if ans is None:
                        print("A: [Không tìm thấy trang liên quan]")
                        print("Time: 0.00s [NO RESULT]\n")
                    else:
                        elapsed = time.time() - start
                        print(f"A: {ans}")
                        print(f"Time: {elapsed:.2f}s [{'CACHE' if cached else 'LLM'}]\n")
                        
                        if tts_enabled and qa.tts:
                            qa.speak(ans)
                continue
            
            # Xử lý câu hỏi thông thường
            start = time.time()
            answer, from_cache = qa.ask(q)
            
            # Xử lý trường hợp không tìm thấy
            if answer is None:
                msg = "Không tìm thấy trang liên quan trong tài liệu."
                print(f"\n⚠️  {msg}")
                
                if tts_enabled and qa.tts:
                    qa.speak(msg)
                
                print("Bạn có muốn cung cấp thông tin để trả lời? (y/n): ", end='')
                
                choice = input().strip().lower()
                if choice == 'y':
                    print("\n📝 Nhập context/thông tin (nhập dòng trống để kết thúc):")
                    print("-" * 60)
                    lines = []
                    while True:
                        line = input()
                        if not line:
                            break
                        lines.append(line)
                    
                    manual_context = "\n".join(lines)
                    if manual_context:
                        print("\n🤔 Đang xử lý với thông tin bạn cung cấp...")
                        start = time.time()
                        answer, from_cache = qa.ask(q, manual_context=manual_context)
                        elapsed = time.time() - start
                        
                        print(f"\n✓ {answer}")
                        print(f"{elapsed:.2f}s [{'CACHE' if from_cache else 'LLM'}]")
                        print("💾 Đã lưu vào cache cho lần sau!\n")
                        
                        if tts_enabled and qa.tts:
                            qa.speak(answer)
                    else:
                        print("⚠️  Không có context. Bỏ qua.\n")
                        continue
                else:
                    print()
                    continue
            else:
                elapsed = time.time() - start
                print(f"\n{answer}")
                print(f"{elapsed:.2f}s [{'CACHE' if from_cache else 'LLM'}]\n")
                
                # Đọc câu trả lời
                if tts_enabled and qa.tts:
                    print("🔊 Speaking...")
                    qa.speak(answer)
            
            print("-"*60 + "\n")
            
        except KeyboardInterrupt:
            print("\nBye!")
            break
        except Exception as e:
            print(f"Error: {e}\n")

if __name__ == "__main__":
    main()