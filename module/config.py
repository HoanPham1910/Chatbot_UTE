# === PATHS ===
INDEX_DIR = "pdf_index"
OCR_CACHE_FILE = "ocr_cache.json"
ANSWER_CACHE_FILE = "answer_cache.json"

# === OLLAMA (Local) ===
OLLAMA_MODEL = "llama3.2:1b"  # Đổi thành model nhỏ hơn để nhanh hơn
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_TIMEOUT = 60

# === GROQ (Cloud - Ultra Fast) ===
USE_GROQ = True  # Đổi thành True để dùng Groq
GROQ_API_KEY = "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"  # Paste API key từ https://console.groq.com
GROQ_MODEL = "llama-3.1-8b-instant"  # hoặc "llama-3.3-70b-versatile"

# === FALLBACK ===
ENABLE_FALLBACK = True  # Tự động chuyển sang Ollama khi Groq hết quota

# === SEARCH & LLM ===
SEARCH_TOP_K = 3
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 200

# === VOICE INPUT ===
ENABLE_VOICE_INPUT = True  # Bật/tắt voice input
VOICE_LANGUAGE = 'vi-VN'   # Ngôn ngữ nhận dạng