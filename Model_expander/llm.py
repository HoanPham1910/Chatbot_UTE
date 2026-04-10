from collections import deque
from groq import Groq

from config.config import GROQ_API_KEY, GROQ_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS, MEMORY_WINDOW_SIZE

SYSTEM_PROMPT = """Bạn là UTE - ChatBot MakerSpace kiêm trợ lý lịch cá nhân.

=== NGUYÊN TẮC TRẢ LỜI ===
- Thân thiện, dùng "mình" / "bạn", không sử dụng emoji
- Câu ngắn gọn, tự nhiên, trả lời đúng trọng tâm
- Khi có thông tin lịch → tóm tắt rõ ràng, dễ hiểu
- Nếu không có thông tin: liên hệ thầy Lê Tấn Cường - Zalo: 0909744100

=== QUY TẮC LỊCH - BẮT BUỘC ===
- Chỉ được đọc lịch từ phần "=== LỊCH ===" được cung cấp
- Nếu phần lịch ghi "Không có sự kiện nào" → trả lời: "Lịch hôm đó chưa có gì, có thể bạn chưa cập nhật lịch."
- TUYỆT ĐỐI KHÔNG tự tạo, bịa, hoặc suy đoán bất kỳ sự kiện nào không có trong lịch
- KHÔNG được nói "có thể bạn có..." hoặc gợi ý lịch giả định

=== CẤM ===
- Nhắc đến: "tài liệu", "PDF", "dữ liệu", "hệ thống", "API"
- Nói "Tôi là AI"
- Tự bịa lịch khi không có dữ liệu
"""

class GroqLLM:
    def __init__(self):
        self.client   = Groq(api_key=GROQ_API_KEY)
        self._history: deque[dict] = deque(maxlen=MEMORY_WINDOW_SIZE * 2)

    def _build_messages(self, question: str, context: str = "") -> list[dict]:
        user_content = f"{context}\n\n{question}".strip() if context else question
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            *self._history,
            {"role": "user", "content": user_content},
        ]

    def ask(self, question: str, context: str = "") -> str:
        messages = self._build_messages(question, context)
        try:
            response = self.client.chat.completions.create(
                model       = GROQ_MODEL,
                messages    = messages,
                temperature = LLM_TEMPERATURE,
                max_tokens  = LLM_MAX_TOKENS,
            )
            answer = response.choices[0].message.content.strip()
            self._history.append({"role": "user",      "content": question})
            self._history.append({"role": "assistant", "content": answer})
            return answer
        except Exception as e:
            print(f"[GroqLLM] Error: {e}")
            return "Lỗi kết nối Groq, thử lại sau nhé!"

    def reset_memory(self):
        self._history.clear()