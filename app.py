# Trong /Chatbot_UTE
from flask import Flask, request, jsonify
from flask_cors import CORS       
import paho.mqtt.client as mqtt
import json
import time
from pathlib import Path
from module.config import *
from module.cache import OCRCache, AnswerCache
from module.search import PDFSearch
from module.llm import OllamaLLM, GroqLLM

app = Flask(__name__)
CORS(app)

# MQTT Configuration
MQTT_BROKER = "192.168.1.115"
# MQTT_BROKER = "192.168.68.119"
MQTT_PORT = 1234
MQTT_USERNAME = None
MQTT_PASSWORD = None
MQTT_TOPIC_QUESTION = "makerspace/question"
MQTT_TOPIC_ANSWER = "makerspace/answer"
MQTT_CLIENT_ID = "makerspace_server"

# Initialize QA System   
class MakerSpaceQA:
    def __init__(self):
        self.search = PDFSearch(INDEX_DIR)
        self.ocr_cache = OCRCache(OCR_CACHE_FILE)
        self.answer_cache = AnswerCache(ANSWER_CACHE_FILE)
        
        if not Path(OCR_CACHE_FILE).exists():
            self.ocr_cache.precache_all(self.search.image_embeddings)
        
        # Initialize LLM
        if USE_GROQ:
            self.llm = GroqLLM(api_key=GROQ_API_KEY, model=GROQ_MODEL)
            if not self.llm.check():
                raise Exception("Groq API key invalid!")
        else:
            self.llm = OllamaLLM()
            if not self.llm.check():
                raise Exception(f"Model {OLLAMA_MODEL} not found!")
            self.llm.warmup()
    
    def ask(self, question, k=SEARCH_TOP_K):
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

print("🚀 Initializing MakerSpace QA System...")
qa_system = MakerSpaceQA()
print("✅ QA System ready!")

# MQTT Setup
mqtt_client = mqtt.Client(client_id=MQTT_CLIENT_ID)
mqtt_connected = False

def on_connect(client, userdata, flags, rc):
    global mqtt_connected
    if rc == 0:
        print(f"✅ Connected to MQTT Broker: {MQTT_BROKER}")
        mqtt_connected = True
        client.subscribe(MQTT_TOPIC_QUESTION)
        print(f"📡 Subscribed to: {MQTT_TOPIC_QUESTION}")
    else:
        print(f"❌ Failed to connect, rc={rc}")
        mqtt_connected = False

def on_message(client, userdata, msg):
    try:
        payload = msg.payload.decode('utf-8')
        data = json.loads(payload)
        question = data.get('question', '')
        device_id = data.get('device_id', 'unknown')
        
        print(f"\n📩 Question from ESP32 [{device_id}]: {question}")
        
        start = time.time()
        answer, from_cache = qa_system.ask(question)
        duration = time.time() - start
        
        if answer is None:
            answer = "Xin lỗi, tôi không tìm thấy thông tin liên quan trong tài liệu."
        
        response = {
            'answer': answer,
            'from_cache': from_cache,
            'duration': round(duration, 2),
            'device_id': device_id,
            'timestamp': int(time.time())
        }
        
        client.publish(MQTT_TOPIC_ANSWER, json.dumps(response, ensure_ascii=False))
        print(f"✅ Answer sent [{duration:.2f}s, {'CACHE' if from_cache else 'LLM'}]")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        error_response = {
            'answer': f"Lỗi xử lý: {str(e)}",
            'error': True
        }
        client.publish(MQTT_TOPIC_ANSWER, json.dumps(error_response, ensure_ascii=False))

def on_disconnect(client, userdata, rc):
    global mqtt_connected
    mqtt_connected = False
    print(f"⚠️ Disconnected (code: {rc})")

mqtt_client.on_connect = on_connect
mqtt_client.on_message = on_message
mqtt_client.on_disconnect = on_disconnect

if MQTT_USERNAME and MQTT_PASSWORD:
    mqtt_client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
    print(f"🔐 Using authentication: {MQTT_USERNAME}")

try:
    print(f"🔌 Connecting to MQTT Broker: {MQTT_BROKER}:{MQTT_PORT}")
    mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
    mqtt_client.loop_start()
except Exception as e:
    print(f"❌ MQTT connection failed: {e}")

# API Endpoints
@app.route('/api/ask', methods=['POST'])
def api_ask():
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({'error': 'Question is required'}), 400
        
        print(f"\n API question: {question}")
        
        start = time.time()
        answer, from_cache = qa_system.ask(question)
        duration = time.time() - start
        
        if answer is None:
            return jsonify({
                'answer': 'Xin lỗi, tôi không tìm thấy thông tin liên quan trong tài liệu.',
                'found': False,
                'duration': round(duration, 2),
                'from_cache': False
            })
        
        return jsonify({
            'answer': answer,
            'from_cache': from_cache,
            'duration': round(duration, 2),
            'found': True
        })
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/status', methods=['GET'])
def api_status():
    return jsonify({
        'status': 'online',
        'mqtt_connected': mqtt_connected,
        'mqtt_broker': MQTT_BROKER,
        'mqtt_port': MQTT_PORT,
        'question_topic': MQTT_TOPIC_QUESTION,
        'answer_topic': MQTT_TOPIC_ANSWER
    })

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': int(time.time())
    })

if __name__ == '__main__':
    print("\n" + "="*50)
    print("MakerSpace QA API Server")
    print("="*50)
    print(f"MQTT Broker: {MQTT_BROKER}:{MQTT_PORT}")
    print(f"Question Topic: {MQTT_TOPIC_QUESTION}")
    print(f"Answer Topic: {MQTT_TOPIC_ANSWER}")
    print(f"API Server: http://0.0.0.0:5000")
    print(f"Endpoints:")
    print(f"   - POST /api/ask")
    print(f"   - GET  /api/status")
    print(f"   - GET  /health")
    print("="*50 + "\n")
    
    app.run(host='0.0.0.0', port=1234, debug=False)