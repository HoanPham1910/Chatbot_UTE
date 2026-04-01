import threading
import time
import os

from flask import Flask, request, jsonify, app as flask_app

from llm              import GroqLLM
from calendar_service import get_events, format_events, get_upcoming_reminders, get_today_info
from TTS              import text_to_wav, TTS_OUTPUT_WAV
from config import (
    REMINDER_CHECK_INTERVAL,
    HTTP_HOST, HTTP_PORT,
)

app = Flask(__name__)
app.json.ensure_ascii = False
bot = GroqLLM()

_reminded: set[str] = set()
_pending_reminders: list[str] = []
_reminder_lock = threading.Lock()

# ── Calendar intent detection ─────────────────────────────

def _detect_calendar_intent(text: str) -> dict:
    t         = text.lower()
    date_info = get_today_info()

    is_tomorrow = any(w in t for w in ["ngày mai", "mai ", "tomorrow"])
    is_week     = any(w in t for w in ["tuần này", "tuần tới", "week"])
    is_calendar = any(w in t for w in [
        "lịch", "sự kiện", "có gì", "làm gì", "kế hoạch",
        "họp", "meeting", "deadline", "nhắc", "schedule",
        "hôm nay", "ngày mai", "tuần",
    ])

    if not is_calendar:
        return {"fetch": False, "date_info": date_info}

    return {
        "fetch":       True,
        "days_offset": 1 if is_tomorrow else 0,
        "days_ahead":  6 if is_week else 0,
        "date_info":   date_info,
    }


def _build_calendar_context(intent: dict) -> str:
    events = get_events(
        days_offset=intent["days_offset"],
        days_ahead=intent["days_ahead"],
    )

    if intent["days_offset"] == 1:
        label = "NGÀY MAI"
    elif intent["days_ahead"] >= 6:
        label = "TUẦN NÀY"
    else:
        label = "HÔM NAY"

    return (
        f"=== THÔNG TIN NGÀY GIỜ ===\n{intent['date_info']}\n\n"
        f"=== LỊCH {label} ===\n{format_events(events)}"
    )


# ── Routes ────────────────────────────────────────────────

@app.get("/")
def health_check():
    return jsonify({"status": "ok", "message": "UTE Chatbot đang chạy"})


@app.post("/ask")
def ask():
    data     = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    tts      = data.get("tts", True)

    if not question:
        return jsonify({"error": "Câu hỏi không được để trống"}), 400

    print(f"\n❓ {question}")

    intent = _detect_calendar_intent(question)
    if intent["fetch"]:
        try:
            context = _build_calendar_context(intent)
        except Exception as e:
            print(f"[Calendar] Lỗi: {e}")
            context = f"=== THÔNG TIN NGÀY GIỜ ===\n{intent['date_info']}"
    else:
        context = f"=== THÔNG TIN NGÀY GIỜ ===\n{intent['date_info']}"

    answer = bot.ask(question, context)
    print(f"💬 {answer}\n")

    wav_url = None
    if tts:
        ok = text_to_wav(answer)
        if ok:
            wav_url = "/audio/wav"

    return jsonify({"answer": answer, "wav_url": wav_url })


@app.get("/audio/wav")
def get_audio_wav():
    """Trả về file WAV cho ESP32. Xóa file sau khi gửi xong."""
    if not os.path.exists(TTS_OUTPUT_WAV):
        return jsonify({"error": "Chưa có audio WAV"}), 404

    with open(TTS_OUTPUT_WAV, "rb") as f:
        wav_data = f.read()

    try:
        os.remove(TTS_OUTPUT_WAV)
        print(f"[Audio] Đã xóa {TTS_OUTPUT_WAV} sau khi gửi")
    except Exception as e:
        print(f"[Audio] Không xóa được {TTS_OUTPUT_WAV}: {e}")

    return app.response_class(
        response=wav_data,
        mimetype="audio/wav",
    )


@app.get("/reminders")
def get_reminders():
    """Client polling để lấy reminder mới. Danh sách tự xoá sau khi đọc."""
    with _reminder_lock:
        reminders = list(_pending_reminders)
        _pending_reminders.clear()
    return jsonify({"reminders": reminders})


# ── Reminder loop ─────────────────────────────────────────

def reminder_loop():
    print(f"🔔 Reminder loop khởi động (interval={REMINDER_CHECK_INTERVAL}s)")
    while True:
        try:
            for event in get_upcoming_reminders():
                eid   = event.get("id")
                title = event.get("summary", "(Không có tiêu đề)")
                start = event["start"].get("dateTime") or event["start"].get("date")
                if eid not in _reminded:
                    _reminded.add(eid)
                    msg = f"🔔 Sắp có: {title} lúc {start}"
                    print(f"[Reminder] {msg}")
                    with _reminder_lock:
                        _pending_reminders.append(msg)
        except Exception as e:
            print(f"[Reminder] Lỗi: {e}")

        time.sleep(REMINDER_CHECK_INTERVAL)


# ── Main ──────────────────────────────────────────────────

if __name__ == "__main__":
    threading.Thread(target=reminder_loop, daemon=True).start()
    app.run(host=HTTP_HOST, port=HTTP_PORT)