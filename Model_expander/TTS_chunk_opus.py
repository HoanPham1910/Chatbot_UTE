import os
import io
import re
import subprocess
import tempfile
import threading
import time

# ── Config ────────────────────────────────────────────────
TTS_OUTPUT_OPUS = "answer.opus"

LANG          = "vi"
SPEED         = 1.25
PCM_RATE      = 48000   # Opus chuẩn 48kHz
PCM_CHANNELS  = 1       # mono — nhỏ hơn, đủ cho giọng nói
OPUS_BITRATE  = "32k"   # 24-48k là đủ cho TTS
CHUNK_TIMEOUT = 30.0    # giây tối đa chờ 1 chunk


# ── Chunk store (shared state) ────────────────────────────
_chunk_store: dict[int, bytes] = {}
_chunk_total: int = 0
_render_done: bool = False
_store_lock = threading.Lock()


# =====================================================
# INTERNAL HELPERS
# =====================================================

def _build_atempo(speed: float) -> str:
    """Xây chuỗi atempo filter cho ffmpeg, hỗ trợ speed ngoài [0.5, 2.0]."""
    filters = []
    remaining = speed
    while remaining > 2.0:
        filters.append("atempo=2.0")
        remaining /= 2.0
    while remaining < 0.5:
        filters.append("atempo=0.5")
        remaining /= 0.5
    filters.append(f"atempo={remaining:.4f}")
    return ",".join(filters)


def _gtts_to_opus_bytes(text: str) -> bytes | None:
    """Gọi gTTS → MP3 → Opus bytes qua ffmpeg. Trả None nếu lỗi."""
    try:
        from gtts import gTTS
    except ImportError:
        print("[TTS] ❌ gTTS chưa cài. Chạy: pip install gtts")
        return None

    try:
        # Bước 1: gTTS → MP3 bytes trong RAM
        tts_obj = gTTS(text=text, lang=LANG, slow=False)
        mp3_buf = io.BytesIO()
        tts_obj.write_to_fp(mp3_buf)
        mp3_buf.seek(0)
        mp3_bytes = mp3_buf.read()

        # Bước 2: ffmpeg MP3 → Opus (pipe)
        atempo = _build_atempo(SPEED)
        cmd = [
            "ffmpeg", "-y",
            "-f", "mp3", "-i", "pipe:0",
            "-filter:a", atempo,
            "-ar", str(PCM_RATE),
            "-ac", str(PCM_CHANNELS),
            "-c:a", "libopus",
            "-b:a", OPUS_BITRATE,
            "-f", "ogg",        
            "pipe:1",
        ]
        result = subprocess.run(
            cmd,
            input=mp3_bytes,
            capture_output=True,
        )

        if result.returncode != 0:
            print(f"[TTS] ffmpeg error: {result.stderr[-300:].decode(errors='replace')}")
            return None

        return result.stdout

    except Exception as e:
        print(f"[TTS] _gtts_to_opus_bytes error: {e}")
        return None


def _split_sentences(text: str) -> list[str]:
    """Tách text thành các đoạn nhỏ để render song song."""
    parts = re.split(r'(?<=[.!?,;:\n])\s+', text.strip())

    result = []
    buf = ""
    for p in parts:
        p = p.strip()
        if not p:
            continue
        buf = (buf + " " + p).strip() if buf else p
        if len(buf) >= 15:          # chunk tối thiểu 15 ký tự
            result.append(buf)
            buf = ""
    if buf:
        result.append(buf)

    return result or [text]


# =====================================================
# PUBLIC API — khớp 100% với main.py
# =====================================================

def text_to_opus(text: str, opus_file: str = TTS_OUTPUT_OPUS) -> bool:
    """Blocking TTS: render toàn bộ text → lưu ra file Opus."""
    print(f"[TTS] text_to_opus: '{text[:60]}{'...' if len(text) > 60 else ''}'")
    opus_bytes = _gtts_to_opus_bytes(text)
    if opus_bytes is None:
        return False
    try:
        with open(opus_file, "wb") as f:
            f.write(opus_bytes)
        print(f"[TTS] ✅ saved → {opus_file} ({len(opus_bytes)//1024} KB)")
        return True
    except Exception as e:
        print(f"[TTS] write error: {e}")
        return False


def start_render_async(text: str) -> int:
    global _chunk_store, _chunk_total, _render_done

    sentences = _split_sentences(text)
    n = len(sentences)

    with _store_lock:
        _chunk_store.clear()
        _chunk_total = n
        _render_done = False

    print(f"[TTS] async render {n} chunk(s) | speed={SPEED}x | format=opus")

    def worker():
        global _render_done
        for i, sentence in enumerate(sentences):
            t0 = time.perf_counter()
            opus_bytes = _gtts_to_opus_bytes(sentence)
            elapsed = time.perf_counter() - t0

            if opus_bytes:
                with _store_lock:
                    _chunk_store[i + 1] = opus_bytes
                print(
                    f"[TTS] chunk {i+1}/{n} ready | "
                    f"{elapsed:.2f}s | '{sentence[:40]}'"
                )
            else:
                print(f"[TTS] chunk {i+1} FAILED: '{sentence[:40]}'")

        _render_done = True
        print("[TTS] ✅ all chunks done")

    threading.Thread(target=worker, daemon=True).start()
    return n


def get_chunk(index: int, timeout: float = CHUNK_TIMEOUT) -> bytes | None:
    """
    Chờ và trả về Opus bytes của chunk `index` (1-based).
    Xóa khỏi RAM sau khi trả. Trả None nếu timeout hoặc index sai.
    """
    if index < 1 or index > _chunk_total:
        return None

    deadline = time.time() + timeout
    while time.time() < deadline:
        with _store_lock:
            if index in _chunk_store:
                return _chunk_store.pop(index)
        if _render_done:
            return None
        time.sleep(0.05)

    print(f"[TTS] get_chunk({index}) timeout")
    return None


def store_status() -> dict:
    """Debug: trạng thái chunk store hiện tại."""
    with _store_lock:
        return {
            "total":       _chunk_total,
            "stored":      list(_chunk_store.keys()),
            "render_done": _render_done,
        }


# =====================================================
# PRELOAD CHECK (chạy khi import)
# =====================================================
def _check_deps():
    try:
        from gtts import gTTS  # noqa
        print("[TTS] gTTS ✅")
    except ImportError:
        print("[TTS] ❌ gTTS chưa cài — chạy: pip install gtts")

    result = subprocess.run(
        ["ffmpeg", "-version"],
        capture_output=True
    )
    if result.returncode == 0:
        print("[TTS] ffmpeg ✅")
    else:
        print("[TTS] ❌ ffmpeg không tìm thấy trong PATH")

    # Kiểm tra libopus
    result = subprocess.run(
        ["ffmpeg", "-codecs"],
        capture_output=True
    )
    output = result.stdout.decode(errors="replace") + result.stderr.decode(errors="replace")
    if "libopus" in output:
        print("[TTS] libopus ✅")
    else:
        print("[TTS] ❌ libopus không có — cài ffmpeg đầy đủ: sudo apt install ffmpeg")


_check_deps()