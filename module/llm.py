import requests
from module.config import *

class OllamaLLM:
    def __init__(self, model=OLLAMA_MODEL, url=OLLAMA_URL):
        self.model = model
        self.url = url
    
    def check(self):
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        models = [m['name'] for m in response.json().get('models', [])]
        return self.model in models
    
    def warmup(self):
        requests.post(
            self.url,
            json={"model": self.model, "prompt": "Test", "stream": False, "options": {"num_predict": 10}},
            timeout=OLLAMA_TIMEOUT
        )
    
    def ask(self, question, context):
        prompt = f"""Bạn là trợ lý MakerSpace HCMUTE, hãy trò chuyện tự nhiên và thân thiện.

Thông tin có sẵn:
{context}

Câu hỏi: {question}

Hướng dẫn:
- Trả lời ngắn gọn, tự nhiên như đang nói chuyện
- Nếu có thông tin: trả lời trực tiếp
- Nếu KHÔNG có thông tin: nói "Xin lỗi, mình chưa có thông tin về [chủ đề] này."
- Không nhắc đến "tài liệu", "PDF", hay "dữ liệu"

Trả lời:"""
        
        response = requests.post(
            self.url,
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": LLM_TEMPERATURE, "num_predict": LLM_MAX_TOKENS}
            },
            timeout=OLLAMA_TIMEOUT
        )
        
        return response.json()['response'].strip() if response.status_code == 200 else "Error"


class GroqLLM:
    def __init__(self, api_key, model="llama-3.1-8b-instant"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.groq.com/openai/v1"
    
    def check(self):
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json={"model": self.model, "messages": [{"role": "user", "content": "test"}], "max_tokens": 5},
            timeout=5
        )
        return response.status_code == 200
    
    def warmup(self):
        return True
    
    def ask(self, question, context):
        prompt = f"""Bạn là trợ lý MakerSpace HCMUTE, hãy trò chuyện tự nhiên và thân thiện.

Thông tin có sẵn:
{context}

Câu hỏi: {question}

Hướng dẫn:
- Trả lời ngắn gọn, tự nhiên như đang nói chuyện
- Nếu có thông tin: trả lời trực tiếp
- Nếu KHÔNG có thông tin: nói "Xin lỗi, mình chưa có thông tin về [chủ đề] này."
- Không nhắc đến "tài liệu", "PDF", hay "dữ liệu"

Trả lời:"""
        
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": LLM_TEMPERATURE,
                "max_tokens": LLM_MAX_TOKENS
            },
            timeout=30
        )
        
        return response.json()['choices'][0]['message']['content'].strip() if response.status_code == 200 else "Error"