import threading
import time
import os
from Service.on_off_light import TuyaController
from flask import Flask, request, jsonify, Response
from Model_expander.intent_classifier import classify_intent, parse_light_command
from Model_expander.llm import GroqLLM
from Service.calendar_service import get_events, format_events, get_upcoming_reminders, get_today_info
from Model_expander.TTS_chunk_opus import text_to_opus, start_render_async, get_chunk, store_status, TTS_OUTPUT_OPUS
from config.config import REMINDER_CHECK_INTERVAL, HTTP_HOST, HTTP_PORT
import colorama


colorama.init(wrap=False)
app = Flask(__name__)
app.json.ensure_ascii = False

bot = GroqLLM()
tuya = TuyaController()

_reminded = set()
_pending_reminders = []
_reminder_lock = threading.Lock()

_pending_light = {}
_light_lock = threading.Lock()

CONFIRM_YES = ["có", "đúng", "ừ", "ok", "được", "yes", "bật đi", "tắt đi", "xác nhận"]
CONFIRM_NO = ["không", "thôi", "no", "hủy", "cancel", "không cần"]


def _detect_calendar_intent(text: str) -> dict:
    t = text.lower()
    date_info = get_today_info()

    is_tomorrow = any(w in t for w in ["ngày mai", "mai ", "tomorrow"])
    is_next_week = any(w in t for w in ["tuần sau", "next week"])
    is_week = any(w in t for w in ["tuần này", "tuần tới", "week"]) and not is_next_week
    is_calendar = any(w in t for w in [
        "lịch", "sự kiện", "có gì", "làm gì", "kế hoạch",
        "họp", "meeting", "deadline", "nhắc", "schedule",
        "hôm nay", "ngày mai", "tuần",
    ])

    if not is_calendar and not is_next_week:
        return {"fetch": False, "date_info": date_info}

    if is_next_week:
        import datetime
        today = datetime.date.today()
        days_offset = (7 - today.weekday()) % 7 or 7
        days_ahead = 6
    elif is_tomorrow:
        days_offset, days_ahead = 1, 0
    else:
        days_offset, days_ahead = 0, (6 if is_week else 0)

    return {
        "fetch": True,
        "days_offset": days_offset,
        "days_ahead": days_ahead,
        "date_info": date_info,
        "is_next_week": is_next_week,
    }


def _build_calendar_context(intent: dict) -> str:
    events = get_events(
        days_offset=intent["days_offset"],
        days_ahead=intent["days_ahead"]
    )

    label = (
        "TUẦN SAU" if intent.get("is_next_week")
        else "NGÀY MAI" if intent["days_offset"] == 1
        else "TUẦN NÀY" if intent["days_ahead"] >= 6
        else "HÔM NAY"
    )

    return (
        f"=== THÔNG TIN NGÀY GIỜ ===\n{intent['date_info']}\n\n"
        f"=== LỊCH {label} ===\n{format_events(events)}"
    )


def _make_response(answer: str, tts: bool):
    wav_url = None
    total_chunks = 0

    if tts:
        total_chunks = start_render_async(answer)
        wav_url = "/audio/chunk/1"      # client bắt đầu từ chunk 1

    return jsonify({
        "answer":       answer,
        "wav_url":      wav_url,        # URL chunk đầu tiên
        "total_chunks": total_chunks,   # client loop đến total_chunks
    })


@app.get("/")
def health_check():
    return jsonify({"status": "ok", "message": "UTE Chatbot đang chạy"})


@app.post("/ask")
def ask():
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    tts = data.get("tts", True)

    if not question:
        return jsonify({"error": "Câu hỏi không được để trống"}), 400

    print(question)
    t = question.lower()

    with _light_lock:
        pending = dict(_pending_light)

    if pending:
        if any(w in t for w in CONFIRM_YES):
            with _light_lock:
                _pending_light.clear()

            turn_on = pending["turn_on"]

            try:
                tuya.switch(turn_on)
                answer = f"Đã {'bật' if turn_on else 'tắt'} đèn rồi nhé!"
            except Exception as e:
                print(e)
                answer = "Lỗi kết nối thiết bị, thử lại sau nhé!"

        elif any(w in t for w in CONFIRM_NO):
            with _light_lock:
                _pending_light.clear()
            answer = "Mình hủy rồi, cần gì thì báo mình nhé!"

        else:
            state = "bật" if pending["turn_on"] else "tắt"
            answer = f"Bạn trả lời '{question}' mình chưa hiểu. Bạn có muốn {state} đèn không?"

        print(answer)
        return _make_response(answer, tts)

    intent_type = classify_intent(question)

    if intent_type == "on_off_den":
        turn_on = parse_light_command(question)

        with _light_lock:
            _pending_light["turn_on"] = turn_on

        answer = f"Bạn có muốn {'bật' if turn_on else 'tắt'} đèn không?"
        print(answer)
        return _make_response(answer, tts)

    intent = _detect_calendar_intent(question)

    if intent["fetch"]:
        try:
            context = _build_calendar_context(intent)
        except Exception as e:
            print(e)
            context = f"=== THÔNG TIN NGÀY GIỜ ===\n{intent['date_info']}"
    else:
        context = f"=== THÔNG TIN NGÀY GIỜ ===\n{intent['date_info']}"

    answer = bot.ask(question, context)

    print(answer)
    return _make_response(answer, tts)


@app.get("/audio/chunk/<int:index>")
def get_chunk_by_index(index: int):
    """
    Client fetch tuần tự: /audio/chunk/1, /audio/chunk/2, ..., /audio/chunk/N
    Trả về Opus bytes của chunk đó, sau đó xóa khỏi RAM.
    204 nếu timeout (chunk bị lỗi render).
    """
    opus = get_chunk(index, timeout=30.0)

    if opus is None:
        return Response(status=204)

    from Model_expander.TTS_chunk import _chunk_total
    return Response(
        opus,
        status=200,
        mimetype="audio/ogg; codecs=opus",
        headers={
            "Content-Length":  str(len(opus)),
            "Cache-Control":   "no-cache, no-store",
            "X-Chunk-Index":   str(index),
            "X-Chunk-Total":   str(_chunk_total),
        }
    )


@app.get("/audio/chunk/status")
def chunk_status():
    return jsonify(store_status())


@app.get("/audio/opus")
def get_audio_opus():
    """Trả về toàn bộ file Opus một lần (dùng thay thế cho /audio/wav cũ)."""
    if not os.path.exists(TTS_OUTPUT_OPUS):
        return jsonify({"error": "Chưa có audio Opus"}), 404

    with open(TTS_OUTPUT_OPUS, "rb") as f:
        data = f.read()

    try:
        os.remove(TTS_OUTPUT_OPUS)
    except Exception:
        pass

    return Response(
        data,
        mimetype="audio/ogg; codecs=opus",
        headers={"Content-Length": str(len(data))}
    )


@app.get("/reminders")
def get_reminders():
    with _reminder_lock:
        reminders = list(_pending_reminders)
        _pending_reminders.clear()

    return jsonify({"reminders": reminders})


def reminder_loop():
    print(f"Reminder loop interval={REMINDER_CHECK_INTERVAL}s")

    while True:
        try:
            for event in get_upcoming_reminders():
                eid = event.get("id")
                title = event.get("summary", "(Không có tiêu đề)")
                start = event["start"].get("dateTime") or event["start"].get("date")

                if eid not in _reminded:
                    _reminded.add(eid)
                    msg = f"Sắp có: {title} lúc {start}"

                    print(msg)

                    with _reminder_lock:
                        _pending_reminders.append(msg)

        except Exception as e:
            print(e)

        time.sleep(REMINDER_CHECK_INTERVAL)

def _preload_tts():
    try:
        from Model_expander.TTS_chunk_opus import _check_deps
        print("[APP] pre-loading TTS model...")
        _check_deps()
        print("[APP] TTS model ready ✅")
    except Exception as e:
        print(f"[APP] TTS preload error: {e}")

if __name__ == "__main__":
    threading.Thread(target=reminder_loop, daemon=True).start()
    threading.Thread(target=_preload_tts, daemon=True).start()
    app.run(
        host=HTTP_HOST,
        port=HTTP_PORT,
        debug=False,
        use_reloader=False,
        use_debugger=False
    )