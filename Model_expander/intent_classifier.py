import re

# Keyword-based rules (fast, no API call)
LIGHT_PATTERNS = [
    r"\b(bật|tắt|mở|đóng)\s*(đèn|đèn điện|đèn phòng|bóng đèn)",
    r"\b(turn\s*(on|off)|switch\s*(on|off))\s*(light|lamp|bulb)",
    r"\bđèn\b.*(bật|tắt|mở)",
    r"(bật|tắt).*(đèn)",
]

INTENTS = ["giao_tiep", "on_off_den"]

def classify_intent(text: str) -> str:
    """
    Trả về 'on_off_den' hoặc 'giao_tiep'.
    Dùng regex trước, fallback sang LLM nếu không rõ.
    """
    lower = text.lower().strip()
    for pattern in LIGHT_PATTERNS:
        if re.search(pattern, lower):
            return "on_off_den"
    return "giao_tiep"

def parse_light_command(text: str) -> bool:
    """
    Trả về True (bật) hoặc False (tắt).
    Gọi sau khi đã xác định intent là on_off_den.
    """
    lower = text.lower()
    off_words = ["tắt", "turn off", "switch off", "off", "đóng"]
    for w in off_words:
        if w in lower:
            return False
    return True  # mặc định bật nếu không rõ