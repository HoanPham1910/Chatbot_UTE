import json
import hashlib
import pytesseract
from PIL import Image
from pathlib import Path

class OCRCache:
    def __init__(self, cache_file):
        self.cache_file = cache_file
        self.cache = self._load()
    
    def _load(self):
        if Path(self.cache_file).exists():
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def _save(self):
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=2)
    
    def get(self, img_path):
        if img_path in self.cache:
            return self.cache[img_path]
        
        img = Image.open(img_path)
        text = pytesseract.image_to_string(img, lang="vie").strip()
        self.cache[img_path] = text
        self._save()
        return text
    
    def precache_all(self, image_files):
        for img_file, _ in image_files:
            if img_file.path not in self.cache:
                self.get(img_file.path)


class AnswerCache:
    def __init__(self, cache_file):
        self.cache_file = cache_file
        data = self._load()
        self.cache = data.get('cache', {})
        self.hit_count = data.get('stats', {}).get('hits', 0)
        self.miss_count = data.get('stats', {}).get('misses', 0)
    
    def _load(self):
        if Path(self.cache_file).exists():
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def _save(self):
        data = {
            'stats': {
                'hits': self.hit_count,
                'misses': self.miss_count,
                'hit_rate': f"{self.hit_count/(self.hit_count+self.miss_count)*100:.1f}%" 
                           if (self.hit_count+self.miss_count) > 0 else "0%"
            },
            'cache': self.cache
        }
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def _normalize(self, question):
        q = question.lower().strip()
        q = q.replace('?', '').replace('!', '').replace('.', '')
        return ' '.join(q.split())
    
    def _key(self, question, context):
        norm_q = self._normalize(question)
        ctx_hash = hashlib.md5(context.encode()).hexdigest()[:8]
        return hashlib.md5(f"{norm_q}:{ctx_hash}".encode()).hexdigest()
    
    def get(self, question, context):
        key = self._key(question, context)
        if key in self.cache:
            self.hit_count += 1
            self._save()
            return self.cache[key]['answer']
        self.miss_count += 1
        return None
    
    def set(self, question, context, answer):
        key = self._key(question, context)
        self.cache[key] = {
            'question': question,
            'answer': answer,
            'context_hash': hashlib.md5(context.encode()).hexdigest()[:8]
        }
        self._save()
    
    def stats(self):
        total = self.hit_count + self.miss_count
        return {
            'hits': self.hit_count,
            'misses': self.miss_count,
            'total': total,
            'hit_rate': f"{self.hit_count/total*100:.1f}%" if total > 0 else "0%",
            'cached': len(self.cache)
        }