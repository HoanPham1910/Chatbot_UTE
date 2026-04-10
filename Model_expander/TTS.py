import io
import re
import time
import threading
import numpy as np
import soundfile as sf
import os
import tempfile

from vieneu import Vieneu

# ── Config ────────────────────────────────────────────
TTS_OUTPUT_WAV = "answer.wav"
PITCH_SHIFT = 0

# ── RVC Config ────────────────────────────────────────
RVC_ENABLED = True
RVC_MODEL_PATH  = r"C:\Users\ThinkPad\Idruino_Hoan\Chatbot_GGC\config\RVC\bichphuong-p1_e500_s5000.pth"
RVC_INDEX_PATH  = r"C:\Users\ThinkPad\Idruino_Hoan\Chatbot_GGC\config\RVC\added_IVF533_Flat_nprobe_1_bichphuong-p1_v2.index"
RVC_PITCH = 1 # semitones: 0 = giữ nguyên cao độ, dương = cao hơn, âm = thấp hơn 
RVC_INDEX_RATE = 0.25 # 0.0–1.0: mức ảnh hưởng của index file (càng cao càng giống giọng gốc) 
RVC_FILTER_RADIUS = 3 # median filter radius để khử artifacts (0 = tắt) 
RVC_RESAMPLE_SR = 0 # resample output 
RVC_RMS_MIX_RATE = 0.25 # blend loudness gốc vs converted 
RVC_PROTECT = 0.33 # bảo vệ consonants, 0–0.5
# ──────────────────────────────────────────────────────

_tts_instance: Vieneu = None

_active_ref_audio: str = None
_active_ref_text: str = None

# RVC instance (lazy load)
_rvc_instance = None
_rvc_lock = threading.Lock()

# chunk storage
_chunk_store: dict[int, bytes] = {}
_chunk_total: int = 0
_render_done: bool = False
_store_lock = threading.Lock()


# =====================================================
# RVC INIT (lazy, thread-safe)
# =====================================================
def _get_rvc():
    global _rvc_instance

    if not RVC_ENABLED:
        return None

    with _rvc_lock:
        if _rvc_instance is not None:
            return _rvc_instance

        try:
            from rvc_python.infer import RVCInference
            print("[RVC] loading model...")
            rvc = RVCInference(device="cpu")
            rvc.load_model(RVC_MODEL_PATH, index_path=RVC_INDEX_PATH)
            _rvc_instance = rvc
            print("[RVC] ✅ model loaded")
        except ImportError:
            print("[RVC] ❌ rvc-python chưa cài. Chạy: pip install rvc-python")
            _rvc_instance = None
        except Exception as e:
            print(f"[RVC] ❌ lỗi load model: {e}")
            _rvc_instance = None

    return _rvc_instance


def _apply_rvc(wav_bytes: bytes) -> bytes:
    rvc = _get_rvc()
    if rvc is None:
        return wav_bytes

    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_in:
            tmp_in.write(wav_bytes)
            tmp_in_path = tmp_in.name

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_out:
            tmp_out_path = tmp_out.name

        t0 = time.perf_counter()
        rvc.infer_file(tmp_in_path, tmp_out_path)

        with open(tmp_out_path, "rb") as f:
            result = f.read()

        print(f"[RVC] convert done | {time.perf_counter() - t0:.2f}s")
        return result

    except Exception as e:
        print(f"[RVC] ⚠️ lỗi convert, dùng audio gốc: {e}")
        return wav_bytes

    finally:
        for p in (tmp_in_path, tmp_out_path):
            try:
                os.remove(p)
            except Exception:
                pass


# =====================================================
# TTS INIT
# =====================================================
def _get_tts() -> Vieneu:
    global _tts_instance

    if _tts_instance is not None:
        return _tts_instance

    print("[TTS] loading model...")
    _tts_instance = Vieneu()
    return _tts_instance


# =====================================================
# AUDIO HELPERS
# =====================================================
def _shift_pitch(audio_np: np.ndarray, sr: int, semitones: float) -> np.ndarray:
    if semitones == 0:
        return audio_np
    try:
        import librosa
        return librosa.effects.pitch_shift(audio_np, sr=sr, n_steps=semitones)
    except ImportError:
        print("[TTS] librosa missing → skip pitch shift")
        return audio_np


def _audio_spec_to_wav_bytes(tts: Vieneu, audio_spec) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        tts.save(audio_spec, tmp_path)

        if PITCH_SHIFT != 0:
            audio_np, sr = sf.read(tmp_path, dtype="float32")
            if audio_np.ndim == 2:
                audio_np = audio_np.mean(axis=1)
            audio_np = _shift_pitch(audio_np, sr, PITCH_SHIFT)
            buf = io.BytesIO()
            sf.write(buf, audio_np, sr, format="WAV", subtype="PCM_16")
            return buf.getvalue()

        with open(tmp_path, "rb") as f:
            return f.read()

    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


# =====================================================
# TEXT SPLIT
# =====================================================
def _split_sentences(text: str):
    parts = re.split(r'(?<=[.!?,;:\n])\s+', text.strip())

    result = []
    buf = ""

    for p in parts:
        p = p.strip()
        if not p:
            continue
        buf = (buf + " " + p).strip() if buf else p
        if len(buf) >= 10:
            result.append(buf)
            buf = ""

    if buf:
        result.append(buf)

    return result or [text]


# =====================================================
# INFER HELPER
# =====================================================
def _infer(tts: Vieneu, text: str):
    if _active_ref_audio is not None:
        return tts.infer(
            text=text,
            ref_audio=_active_ref_audio,
            ref_text=_active_ref_text,
        )
    return tts.infer(text=text)


# =====================================================
# BLOCKING TTS  (có RVC)
# =====================================================
def text_to_wav(text: str, wav_file=TTS_OUTPUT_WAV) -> bool:
    try:
        tts = _get_tts()
        audio_spec = _infer(tts, text)
        wav_bytes = _audio_spec_to_wav_bytes(tts, audio_spec)
        wav_bytes = _apply_rvc(wav_bytes)

        with open(wav_file, "wb") as f:
            f.write(wav_bytes)

        print(f"[TTS] done → {wav_file}")
        return True
    except Exception as e:
        print("[TTS] error:", e)
        return False


# =====================================================
# ASYNC CHUNK RENDER  (có RVC per-chunk)
# =====================================================
def start_render_async(text: str) -> int:
    global _chunk_store, _chunk_total, _render_done

    sentences = _split_sentences(text)
    n = len(sentences)

    with _store_lock:
        _chunk_store.clear()
        _chunk_total = n
        _render_done = False

    print(f"[TTS] async render {n} chunks (RVC={'ON' if RVC_ENABLED else 'OFF'})")

    def worker():
        global _render_done

        tts = _get_tts()

        for i, sentence in enumerate(sentences):
            try:
                t0 = time.perf_counter()

                audio_spec = _infer(tts, sentence)
                wav_bytes  = _audio_spec_to_wav_bytes(tts, audio_spec)
                t_tts = time.perf_counter() - t0

                wav_bytes = _apply_rvc(wav_bytes)
                t_total = time.perf_counter() - t0

                with _store_lock:
                    _chunk_store[i + 1] = wav_bytes

                print(
                    f"[TTS] chunk {i+1}/{n} ready | "
                    f"tts={t_tts:.2f}s total={t_total:.2f}s | "
                    f"'{sentence[:40]}'"
                )

            except Exception as e:
                print(f"[TTS] chunk {i+1} error: {e}")

        _render_done = True
        print("[TTS] all chunks done ✅")

    threading.Thread(target=worker, daemon=True).start()
    return n


# =====================================================
# CHUNK FETCH
# =====================================================
def get_chunk(index: int, timeout: float = 30.0):
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

    return None


def store_status():
    with _store_lock:
        return {
            "total":       _chunk_total,
            "stored":      list(_chunk_store.keys()),
            "render_done": _render_done,
        }


# =====================================================
# PRELOAD
# =====================================================
print("[TTS] preload...")
_get_tts()

threading.Thread(target=_get_rvc, daemon=True).start()