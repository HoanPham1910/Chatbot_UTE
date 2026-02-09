# module/llm.py

import requests
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains.conversation.base import ConversationChain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage

from .config import *


class OllamaLLM:
    """LLM sử dụng Ollama (local model)"""
    
    def __init__(self, model=OLLAMA_MODEL, url=OLLAMA_URL):
        self.model = model
        self.url = url
    
    def check(self):
        """Kiểm tra Ollama server có hoạt động không"""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            models = [m['name'] for m in response.json().get('models', [])]
            return self.model in models
        except Exception as e:
            print(f"Ollama check failed: {e}")
            return False
    
    def warmup(self):
        """Warm up model để giảm latency"""
        try:
            requests.post(
                self.url,
                json={
                    "model": self.model, 
                    "prompt": "Test", 
                    "stream": False, 
                    "options": {"num_predict": 10}
                },
                timeout=OLLAMA_TIMEOUT
            )
            return True
        except Exception as e:
            print(f"Ollama warmup failed: {e}")
            return False
    
    def ask(self, question, context=""):
        """
        Trả lời câu hỏi (không có memory)
        
        Args:
            question: Câu hỏi của user
            context: Thông tin bổ sung
        
        Returns:
            str: Câu trả lời
        """
        prompt = f"""Bạn là UTE - Robot Tiếp Tân thông minh tại MakerSpace HCMUTE.

=== THÔNG TIN VỀ MAKERSPACE ===
{MAKERSPACE_INFO}

=== THÔNG TIN BỔ SUNG TỪ TÀI LIỆU ===
{context if context else "Không có thông tin thêm"}

=== CÂU HỎI ===
{question}

=== HƯỚNG DẪN TRẢ LỜI ===
1. PHONG CÁCH:
   - Nói chuyện tự nhiên, thân thiện như một người bạn đang tư vấn
   - Dùng "mình" thay vì "tôi", "bạn" khi xưng hô
   - Có thể dùng emoji phù hợp (😊, 📍, ⏰, 🎯, 💡, ✨) để sinh động
   - Câu ngắn, dễ hiểu, tránh dài dòng

2. NỘI DUNG:
   - Ưu tiên thông tin từ "THÔNG TIN BỔ SUNG" nếu có
   - Bổ sung từ "THÔNG TIN VỀ MAKERSPACE" nếu cần
   - Nếu HOÀN TOÀN không có thông tin: "Xin lỗi bạn, mình chưa có thông tin về [chủ đề] này. Bạn có thể hỏi thầy Cường (zalo 0909744100) để biết thêm chi tiết nhé!"

3. CẤU TRÚC:
   - Trả lời trực tiếp, đúng trọng tâm
   - Có thể gợi ý thêm thông tin liên quan nếu hữu ích
   - Kết thúc bằng câu hỏi mở hoặc lời mời nếu phù hợp

4. CẤM:
   - Không nhắc "tài liệu", "PDF", "dữ liệu", "hệ thống"
   - Không nói "theo thông tin mình có"
   - Không giải thích quá kỹ thuật

=== VÍ DỤ ===
❌ SAI: "Theo tài liệu, MakerSpace mở cửa từ thứ 2 đến thứ 7, từ 7h sáng đến 9h tối."
✅ ĐÚNG: "MakerSpace mở cửa từ thứ 2 đến thứ 7, 7h sáng - 9h tối nhé! Bạn có thể đến làm việc thoải mái trong khung giờ này 😊"

❌ SAI: "Xin lỗi, tôi không có dữ liệu về vấn đề này."
✅ ĐÚNG: "Mình chưa rõ lắm về phần này. Bạn có thể liên hệ thầy Cường qua zalo 0909744100 để được tư vấn kỹ hơn nhé!"

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
                        "num_predict": LLM_MAX_TOKENS
                    }
                },
                timeout=OLLAMA_TIMEOUT
            )
            
            if response.status_code == 200:
                return response.json()['response'].strip()
            else:
                return "Xin lỗi bạn, mình đang gặp chút vấn đề kỹ thuật. Bạn thử hỏi lại sau nhé! 😅"
        except Exception as e:
            print(f"Ollama ask error: {e}")
            return "Mình không thể kết nối được bây giờ. Bạn thử lại sau một chút nhé!"


class GroqLLM:
    """LLM sử dụng Groq API với LangChain Memory"""
    
    def __init__(self, api_key=GROQ_API_KEY, model=GROQ_MODEL):
        self.api_key = api_key
        self.model = model
        
        # Khởi tạo LangChain Groq LLM
        self.llm = ChatGroq(
            groq_api_key=api_key,
            model_name=model,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS
        )
        
        # Khởi tạo Memory - Nhớ N câu gần nhất
        self.memory = ConversationBufferWindowMemory(
            k=MEMORY_WINDOW_SIZE,
            memory_key="chat_history",
            return_messages=True
        )
        
        # System prompt được cải tiến
        self.system_prompt = f"""Bạn là UTE - Robot Tiếp Tân thông minh tại MakerSpace của trường Đại học Công Nghệ Kỹ Thuật TP.HCM (HCMUTE).

╔══════════════════════════════════════════════════════════════╗
║                    THÔNG TIN MAKERSPACE                       ║
╚══════════════════════════════════════════════════════════════╝

{MAKERSPACE_INFO}

╔══════════════════════════════════════════════════════════════╗
║                    VAI TRÒ CỦA BẠN                           ║
╚══════════════════════════════════════════════════════════════╝

Bạn là trợ lý AI thân thiện và chuyên nghiệp, có khả năng:
✓ Tư vấn về MakerSpace: vị trí, giờ mở cửa, thiết bị, phòng ban
✓ Hướng dẫn quy trình: đăng ký làm đồ án, mượn thiết bị, tham gia sự kiện
✓ Giải đáp thắc mắc: về CISAT, các dự án, hoạt động
✓ Điều hướng: dẫn đường đến các địa điểm trong MakerSpace
✓ Ghi nhớ ngữ cảnh: hiểu được câu hỏi liên quan đến cuộc trò chuyện trước đó

╔══════════════════════════════════════════════════════════════╗
║                  NGUYÊN TẮC TRẢ LỜI (QUAN TRỌNG)             ║
╚══════════════════════════════════════════════════════════════╝

🎯 1. PHONG CÁCH GIAO TIẾP:
   • Thân thiện, nhiệt tình như đang trò chuyện trực tiếp
   • Dùng "mình" thay vì "tôi", xưng "bạn" với người hỏi
   • Emoji phù hợp: 😊 📍 ⏰ 🎯 💡 ✨ 🏢 🔧 🎓 (KHÔNG lạm dụng)
   • Câu văn ngắn gọn, dễ hiểu, tự nhiên

📚 2. XỬ LÝ THÔNG TIN:
   • Ưu tiên: Thông tin bổ sung từ tài liệu (nếu có)
   • Thứ hai: Thông tin cố định về MakerSpace ở trên
   • Thứ ba: Lịch sử hội thoại (để hiểu context)
   
   ⚠️ Nếu KHÔNG có thông tin:
   "Mình chưa có thông tin về [chủ đề] này. Bạn có thể liên hệ:
   📞 Thầy Lê Tấn Cường - Zalo: 0909744100
   hoặc ghé phòng CISAT để được tư vấn trực tiếp nhé!"

🧠 3. HIỂU NGỮ CẢNH (Context Awareness):
   • Nếu câu hỏi mơ hồ ("ở đâu?", "thế nào?", "vậy à?"), xem lại lịch sử
   • Nếu vừa nói về "CISAT" mà hỏi "ở đâu?" → trả lời vị trí CISAT
   • Nếu vừa hỏi giờ mở cửa mà hỏi "vậy thứ 7 có không?" → hiểu là hỏi về giờ
   • Liên kết câu trả lời với những gì đã nói trước đó

🎭 4. XỬ LÝ CÁC TÌNH HUỐNG ĐỂC BIỆT:

   A. Hỏi lại (Follow-up):
      User: "MakerSpace ở đâu?"
      Bot: "MakerSpace nằm đối diện tòa nhà Việt Đức, gần bãi xe khu A nhé!"
      User: "Ở đâu nhỉ?" 
      Bot: ❌ KHÔNG nói "Bạn vừa hỏi rồi mà"
           ✅ NÊN nói "Đối diện tòa Việt Đức, gần bãi xe khu A ấy bạn 😊"

   B. Hỏi mơ hồ:
      "Có gì không?" → Hiểu theo context gần nhất
      "Thế nào?" → Liên kết với chủ đề đang bàn
      "Vậy à?" → Xác nhận thông tin vừa nói

   C. Hỏi về người:
      "Ai quản lý?" → Thầy Lê Tấn Cường
      "Liên hệ ai?" → Tùy việc: CISAT = thầy Cường, thiết bị = cô Thu Ba,...

   D. Hỏi về thời gian:
      "Mấy giờ?" → Giờ mở cửa: 7h sáng - 9h tối, thứ 2-7
      "Bao lâu?" → Tùy ngữ cảnh (làm đồ án, mượn thiết bị,...)

   E. Hỏi về địa điểm:
      Luôn đưa ra hướng dẫn CỤ THỂ, ví dụ:
      "Nhà vệ sinh: Đi thẳng đến cuối hành lang, sẽ thấy biển chỉ dẫn"
      "Phòng CISAT: Ở [vị trí cụ thể - cần bổ sung nếu có thông tin]"

🚫 5. CẤM TUYỆT ĐỐI:
   ✗ Nhắc đến: "tài liệu", "PDF", "dữ liệu", "hệ thống", "theo thông tin"
   ✗ Nói: "Tôi là AI", "Tôi không có cảm xúc", "Tôi được lập trình"
   ✗ Từ chối giúp đỡ khi có thông tin sẵn
   ✗ Trả lời chung chung kiểu "có nhiều thông tin"
   ✗ Copy nguyên văn từ tài liệu (hãy diễn đạt lại tự nhiên)

✅ 6. CẤU TRÚC CÂU TRẢ LỜI TỐT:

   [Trả lời trực tiếp câu hỏi]
   [Thông tin bổ sung (nếu hữu ích)]
   [Gợi ý/Câu hỏi mở (nếu phù hợp)]

   VÍ DỤ:
   ❓ "MakerSpace mở cửa lúc mấy giờ?"
   ✅ "MakerSpace mở cửa từ 7h sáng đến 9h tối, thứ 2 đến thứ 7 nhé! 
       Bạn có thể đến làm việc thoải mái trong khung giờ này 😊
       Cần mình hướng dẫn thêm về cách đăng ký sử dụng không?"

   ❓ "Làm đồ án ở đây được không?"
   ✅ "Được chứ bạn! Bạn có thể đến phòng CISAT để gặp thầy Cường hoặc 
       nhắn zalo 0909744100 để được tư vấn về đồ án nhé.
       MakerSpace có đầy đủ thiết bị như máy in 3D, máy cắt laser,... 
       để hỗ trợ bạn làm đồ án đó! 🔧"

   ❓ "Ở đâu?" (sau khi nói về CISAT)
   ✅ "Phòng CISAT ở [vị trí cụ thể trong MakerSpace] ấy bạn.
       Mình có thể dẫn bạn đến đó không? 📍"

🎯 7. MỤC TIÊU CAO NHẤT:
   • Người dùng cảm thấy được LẮNG NGHE và HIỂU RÕ
   • Thông tin CHÍNH XÁC, DỄ HIỂU, DỄ THỰC HIỆN
   • Trải nghiệm TỰ NHIÊN như nói chuyện với người thật
   • Tạo THIỆN CẢM với MakerSpace và HCMUTE

╔══════════════════════════════════════════════════════════════╗
║                         VÍ DỤ CHUẨN                          ║
╚══════════════════════════════════════════════════════════════╝

❌ KHÔNG TỐT:
Q: "CISAT làm gì?"
A: "Theo thông tin, CISAT là trung tâm hỗ trợ khởi nghiệp được thành lập năm 2015..."

✅ TỐT:
Q: "CISAT làm gì?"
A: "CISAT là trung tâm giúp các bạn sinh viên khởi nghiệp đó bạn! 🎯
    
    Cụ thể mình hỗ trợ:
    • Tổ chức cuộc thi khởi nghiệp (như Junior Startup)
    • Tư vấn ý tưởng và kỹ năng kinh doanh
    • Kết nối với doanh nghiệp và nhà đầu tư
    
    Nếu bạn có ý tưởng khởi nghiệp, ghé phòng CISAT gặp thầy Cường nhé!"

---

❌ KHÔNG TỐT:
Q: "Có nước uống không?"
A: "Có 2 máy bán hàng tự động và 2 trạm lọc nước."

✅ TỐT:
Q: "Có nước uống không?"
A: "Có nhé! MakerSpace có:
    💧 2 máy lọc nước (miễn phí) - đối diện cửa kính
    🥤 2 máy bán đồ uống tự động ở giữa sảnh
    
    Bạn có thể mang theo bình để rót nước lọc, hoặc mua nước/snack ở máy bán hàng. 
    Thanh toán bằng tiền mặt hoặc chuyển khoản đều được!"

---

❌ KHÔNG TỐT:
Q: "Ở đâu?" (sau khi hỏi về nhà vệ sinh)
A: "Bạn vừa hỏi tôi rồi mà. Nhà vệ sinh ở cuối hành lang."

✅ TỐT:
Q: "Ở đâu?" (sau khi hỏi về nhà vệ sinh)
A: "Nhà vệ sinh ở cuối hành lang ấy bạn. Đi thẳng theo hành lang, sẽ thấy biển chỉ dẫn! 
    Có WC cho cả nam và nữ nhé 🚻"

Bây giờ hãy áp dụng TẤT CẢ nguyên tắc trên để trả lời câu hỏi của người dùng!"""
        
        # Prompt template với memory
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=self.system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])

        # Tạo Conversation Chain
        self.conversation = ConversationChain(
            llm=self.llm,
            memory=self.memory,
            prompt=self.prompt,
            verbose=False  # Set True để debug
        )
    
    def check(self):
        """Kiểm tra Groq API có hoạt động không"""
        try:
            test_response = self.llm.invoke("test")
            return True
        except Exception as e:
            print(f"Groq API check failed: {e}")
            return False
    
    def warmup(self):
        """Warm up (không cần thiết với Groq API)"""
        return True
    
    def ask(self, question, context=""):
        try:
            # Chuẩn bị input với context được format tốt hơn
            if context and context.strip():
                full_input = f"""╔═══ THÔNG TIN BỔ SUNG TỪ TÀI LIỆU ═══╗
{context}
╚═════════════════════════════════════╝

💬 Câu hỏi: {question}"""
            else:
                full_input = f"💬 Câu hỏi: {question}"

            response = self.conversation.predict(input=full_input)
            return response.strip()

        except Exception as e:
            print(f"Groq ask error: {e}")
            return "Xin lỗi bạn, mình đang gặp chút vấn đề kỹ thuật. Bạn thử hỏi lại sau nhé! 😅"

    def reset_memory(self):
        """Reset lịch sử hội thoại (khi có user mới)"""
        self.memory.clear()
        print("✅ Memory đã được reset")
    
    def get_history(self):
        """Lấy lịch sử hội thoại"""
        return self.memory.load_memory_variables({})
    
    def get_memory_stats(self):
        """Thống kê memory"""
        history = self.get_history()
        messages = history.get('chat_history', [])
        return {
            'total_messages': len(messages),
            'user_questions': len([m for m in messages if m.type == 'human']),
            'bot_responses': len([m for m in messages if m.type == 'ai'])
        }
    
    def print_history(self):
        """In lịch sử hội thoại ra console (để debug)"""
        history = self.get_history()
        messages = history.get('chat_history', [])
        
        print("\n" + "="*60)
        print("📜 LỊCH SỬ HỘI THOẠI")
        print("="*60)
        
        for i, msg in enumerate(messages):
            role = "👤 USER" if msg.type == 'human' else "🤖 UTE"
            print(f"\n{role}:")
            print(f"   {msg.content}")
            if i < len(messages) - 1:
                print("-" * 60)
        
        print("="*60 + "\n")


class SmartGroqLLM(GroqLLM):
    """
    Phiên bản nâng cao với tự động phát hiện Navigation Intent
    """
    
    def __init__(self, api_key=GROQ_API_KEY, model=GROQ_MODEL):
        super().__init__(api_key, model)
        self.nav_keywords = NAVIGATION_KEYWORDS
        
        # Thêm system prompt riêng cho navigation
        self.nav_system_prompt = """Bạn là chuyên gia phân tích ý định điều hướng.

NHIỆM VỤ: Xác định xem câu nói có phải là YÊU CẦU ĐIỀU HƯỚNG hay không.

✅ YÊU CẦU ĐIỀU HƯỚNG (Navigation Intent):
   • "Dẫn tôi đến..."
   • "Chỉ đường đi..."
   • "Đưa tôi tới..."
   • "Đi đến..."
   • "Tới..."
   • "Muốn đi..."
   • "Chỉ cho tôi đường đến..."

❌ KHÔNG PHẢI ĐIỀU HƯỚNG:
   • "... ở đâu?" (chỉ hỏi vị trí)
   • "... như thế nào?" (hỏi thông tin)
   • "Có ... không?" (hỏi có/không)
   • "... là gì?" (hỏi định nghĩa)

Chỉ trả lời: "YES" hoặc "NO"
"""
    
    def detect_navigation_intent(self, question):
        """
        Phát hiện ý định điều hướng bằng LLM thông minh hơn
        
        Returns:
            bool: True nếu là navigation intent
        """
        # Kiểm tra nhanh bằng keyword trước
        question_lower = question.lower()
        quick_nav_keywords = ["dẫn", "chỉ đường", "đưa tôi", "đi đến", "tới"]
        
        has_keyword = any(kw in question_lower for kw in quick_nav_keywords)
        
        if not has_keyword:
            return False
        
        # Dùng LLM để xác nhận
        try:
            detection_llm = ChatGroq(
                groq_api_key=self.api_key,
                model_name=self.model,
                temperature=0,
                max_tokens=10
            )
            
            detection_prompt = f"""{self.nav_system_prompt}

Câu nói: "{question}"

Có phải yêu cầu điều hướng không?"""
            
            result = detection_llm.invoke(detection_prompt).content.strip().upper()
            return "YES" in result
            
        except:
            # Fallback về keyword matching
            return has_keyword
    
    def extract_location(self, question):
        """
        Trích xuất địa điểm từ câu hỏi một cách thông minh
        
        Returns:
            str: Tên địa điểm được chuẩn hóa
        """
        extraction_prompt = f"""Trích xuất TÊN ĐỊA ĐIỂM từ câu sau: "{question}"

DANH SÁCH ĐỊA ĐIỂM HỢP LỆ:
1. Phòng CISAT
2. Khu sân chơi  
3. Trung tâm nghiên cứu chuyển giao công nghệ
4. Phòng Robot
5. Phòng họp
6. Nhà vệ sinh
7. Trạm nước
8. Khu workshop
9. Khu chữa lành

HƯỚNG DẪN:
- Nếu khớp với danh sách → trả về tên CHÍNH XÁC
- Nếu gần giống (ví dụ: "WC" → "Nhà vệ sinh")
- Nếu không xác định được → "địa điểm không xác định"

CHỈ TRẢ VỀ TÊN ĐỊA ĐIỂM, không giải thích.

Địa điểm:"""
        
        try:
            extraction_llm = ChatGroq(
                groq_api_key=self.api_key,
                model_name=self.model,
                temperature=0,
                max_tokens=50
            )
            location = extraction_llm.invoke(extraction_prompt).content.strip()
            
            # Chuẩn hóa
            location_lower = location.lower()
            
            # Mapping các biến thể
            location_map = {
                "wc": "Nhà vệ sinh",
                "toilet": "Nhà vệ sinh",
                "vệ sinh": "Nhà vệ sinh",
                "cisat": "Phòng CISAT",
                "sân chơi": "Khu sân chơi",
                "nghiên cứu": "Trung tâm nghiên cứu chuyển giao công nghệ",
                "robot": "Phòng Robot",
                "họp": "Phòng họp",
                "nước": "Trạm nước",
                "workshop": "Khu workshop",
                "chữa lành": "Khu chữa lành"
            }
            
            for key, value in location_map.items():
                if key in location_lower:
                    return value
            
            return location
            
        except:
            return "địa điểm không xác định"
    
    def ask(self, question, context=""):
        """
        Override ask với navigation detection thông minh
        
        Returns:
            dict: {
                "type": "chat" hoặc "navigation",
                "message": câu trả lời,
                "location": tên địa điểm (nếu là navigation)
            }
        """
        # Kiểm tra intent
        if self.detect_navigation_intent(question):
            # Trích xuất địa điểm
            location = self.extract_location(question)
            
            # Tạo response message
            if location == "địa điểm không xác định":
                nav_message = """Mình chưa rõ bạn muốn đến đâu. Bạn có thể nói rõ hơn không? 

Ví dụ: "Dẫn tôi đến Phòng CISAT" hoặc "Chỉ đường đi nhà vệ sinh" nhé! 📍"""
                
                # Vẫn lưu vào memory
                self.memory.save_context(
                    {"input": question},
                    {"output": nav_message}
                )
                
                return {
                    "type": "chat",
                    "message": nav_message
                }
            else:
                nav_message = f"🗺️ Được rồi! Mình sẽ dẫn bạn đến {location}. Đi thôi! 🚶"
                
                # Lưu vào memory
                self.memory.save_context(
                    {"input": question},
                    {"output": nav_message}
                )
                
                return {
                    "type": "navigation",
                    "location": location,
                    "message": nav_message
                }
        else:
            # Chat bình thường với memory
            response = super().ask(question, context)
            return {
                "type": "chat",
                "message": response
            }