import time
from pathlib import Path
from config import *
from cache import OCRCache, AnswerCache
from search import PDFSearch
from llm import OllamaLLM

class MakerSpaceQA:
    def __init__(self):
        print("Loading index...")
        self.search = PDFSearch(INDEX_DIR)
        
        print("Loading caches...")
        self.ocr_cache = OCRCache(OCR_CACHE_FILE)
        self.answer_cache = AnswerCache(ANSWER_CACHE_FILE)
        
        if not Path(OCR_CACHE_FILE).exists():
            print("Pre-caching OCR...")
            self.ocr_cache.precache_all(self.search.image_embeddings)
        
        print("Checking LLM...")
        self.llm = OllamaLLM()
        if not self.llm.check():
            raise Exception(f"Model {OLLAMA_MODEL} not found. Run: ollama pull {OLLAMA_MODEL}")
        
        print("Warming up LLM...")
        self.llm.warmup()
        print("Ready\n")
    
    def ask(self, question, k=SEARCH_TOP_K):
        results = self.search.search(question, k=k)
        if not results:
            return "Không tìm thấy trang liên quan", False
        
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


def main():
    try:
        qa = MakerSpaceQA()
    except Exception as e:
        print(f"Error: {e}")
        return
    
    # print("="*60)
    # print(f"MakerSpace Q&A - Model: {OLLAMA_MODEL}")
    # print("Commands: 'test', 'stats', 'exit'")
    # print("="*60 + "\n")
    
    
    while True:
        try:
            q = input("Question: ").strip()
            if not q:
                continue
            
            if q.lower() in ['exit', 'quit', 'q']:
                stats = qa.answer_cache.stats()
                print(f"\nStats: {stats['total']} questions, {stats['hits']} cached ({stats['hit_rate']})")
                break
            
            if q.lower() == 'stats':
                s = qa.answer_cache.stats()
                print(f"Total: {s['total']}, Hits: {s['hits']}, Misses: {s['misses']}, Rate: {s['hit_rate']}, Cached: {s['cached']}\n")
                continue
            
            start = time.time()
            answer, from_cache = qa.ask(q)
            elapsed = time.time() - start
            
            print(f"\n{answer}")
            print(f"{elapsed:.2f}s [{'CACHE' if from_cache else 'LLM'}]\n")
            print("-"*60 + "\n")
            
        except KeyboardInterrupt:
            print("\nBye!")
            break
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    main()