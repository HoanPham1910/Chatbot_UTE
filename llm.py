import requests
from config import OLLAMA_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT, LLM_TEMPERATURE, LLM_MAX_TOKENS

class OllamaLLM:
    def __init__(self, model=OLLAMA_MODEL, url=OLLAMA_URL):
        self.model = model
        self.url = url
    
    def check(self):
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            if response.status_code == 200:
                models = [m['name'] for m in response.json().get('models', [])]
                return self.model in models
        except:
            pass
        return False
    
    def warmup(self):
        try:
            requests.post(
                self.url,
                json={"model": self.model, "prompt": "Test", "stream": False, "options": {"num_predict": 10}},
                timeout=OLLAMA_TIMEOUT
            )
            return True
        except:
            return False
    
    def ask(self, question, context):
        prompt = f"""Bạn là trợ lý hỗ trợ thông tin về MakerSpace tại trường HCMUTE.

Dựa vào thông tin từ tài liệu dưới đây, hãy trả lời câu hỏi:

<document>
{context}
</document>

Câu hỏi: {question}

Yêu cầu:
- Trả lời ngắn gọn, chính xác bằng tiếng Việt
- Chỉ dùng thông tin có trong tài liệu
- Nếu không tìm thấy, nói "Không tìm thấy thông tin trong tài liệu"

Trả lời:"""

        try:
            response = requests.post(
                self.url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": LLM_TEMPERATURE,
                        "num_predict": LLM_MAX_TOKENS,
                    }
                },
                timeout=OLLAMA_TIMEOUT
            )
            
            if response.status_code == 200:
                return response.json()['response'].strip()
            return f"Ollama error: {response.status_code}"
        
        except requests.exceptions.Timeout:
            return "Timeout - thử lại"
        except Exception as e:
            return f"Error: {str(e)}"