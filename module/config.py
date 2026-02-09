# === PATHS ===        # ← THÊM DÒNG NÀY
INDEX_DIR = "pdf_index"
OCR_CACHE_FILE = "ocr_cache.json"
ANSWER_CACHE_FILE = "answer_cache.json"

# === OLLAMA (Local) ===
OLLAMA_MODEL = "llama3.2:1b"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_TIMEOUT = 60

# === GROQ (Cloud - Ultra Fast) ===
USE_GROQ = True
GROQ_API_KEY = "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"
GROQ_MODEL = "llama-3.1-8b-instant"

# === FALLBACK ===
ENABLE_FALLBACK = True

# === SEARCH & LLM ===
SEARCH_TOP_K = 3
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 200

# === VOICE INPUT ===
ENABLE_VOICE_INPUT = True
VOICE_LANGUAGE = 'vi-VN'

# ==================== MEMORY CONFIG ====================
MEMORY_WINDOW_SIZE = 5  # Nhớ 5 cặp Q&A gần nhất
MEMORY_MAX_TOKEN_LIMIT = 2000  # Cho ConversationSummaryBufferMemory

# ==================== NAVIGATION CONFIG ====================
NAVIGATION_KEYWORDS = [
    "dẫn", "đưa", "chỉ đường", "đi đến", "navigate",
    "hướng dẫn", "chỉ cho", "đường đến", "take me",
    "di chuyển", "đưa tôi"
]

# ==================== MAKERSPACE INFO ====================
MAKERSPACE_INFO = """
THÔNG TIN MAKERSPACE HCMUTE:
- Địa chỉ: Đối diện tòa Việt Đức, gần bãi xe khu A
- Nhà vệ sinh: Cuối hành lang, đi thẳng sẽ thấy bảng chỉ dẫn
- Giờ mở cửa: 8h00 - 17h00 (Thứ 2 - Thứ 6)
- Phòng họp: Tầng 2
- Khu vực thiết bị: Tầng 1, phía bên phải
"""