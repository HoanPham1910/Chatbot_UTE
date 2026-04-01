"""
Google TTS (gTTS) - Chuyển text sang WAV Stereo 16-bit
Miễn phí, không cần API key
Yêu cầu: pip install gtts | ffmpeg trong PATH

Output: file .wav, signed 16-bit little-endian, stereo, 44100 Hz
"""

import os
import subprocess
from gtts import gTTS


TTS_OUTPUT_WAV = "answer.wav"

# Thông số output
PCM_RATE     = 44100
PCM_CHANNELS = 2

SPEED = 1.25


def _build_atempo(speed: float) -> str:
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


def text_to_wav(
    text:     str,
    wav_file: str   = TTS_OUTPUT_WAV,
    lang:     str   = "vi",
    rate:     int   = PCM_RATE,
    channels: int   = PCM_CHANNELS,
    speed:    float = SPEED,
) -> bool:
    try:
        print(f"🎙️  TTS → WAV: '{text[:60]}{'...' if len(text) > 60 else ''}' | speed={speed}x")
        tmp_mp3 = wav_file.replace(".wav", "_tmp.mp3")
        atempo  = _build_atempo(speed)

        # Bước 1: gTTS → MP3 tạm
        tts = gTTS(text=text, lang=lang, slow=False)
        tts.save(tmp_mp3)

        # Bước 2: ffmpeg MP3 → WAV
        r = subprocess.run(
            ["ffmpeg", "-y", "-i", tmp_mp3,
             "-filter:a", atempo,
             "-ar", str(rate), "-ac", str(channels),
             wav_file],
            capture_output=True, text=True
        )

        os.remove(tmp_mp3)

        if r.returncode != 0:
            print(f"[TTS] ffmpeg WAV lỗi: {r.stderr}")
            return False

        wav_kb   = os.path.getsize(wav_file) / 1024
        duration = (os.path.getsize(wav_file) - 44) / (rate * channels * 2)
        print(f"[TTS] ✅ WAV: {wav_file} ({wav_kb:.1f} KB) | {duration:.1f}s")
        return True

    except Exception as e:
        print(f"[TTS] Lỗi: {e}")
        return False


# ── Chạy độc lập để test ──────────────────────────────────

if __name__ == "__main__":
    TEXT = "Xin chào! Đây là ví dụ chuyển văn bản thành giọng nói tiếng Việt dạng WAV stereo."
    LANG = "vi"

    MY_SPEED = 1.5

    ok = text_to_wav(TEXT, "output.wav", LANG, speed=MY_SPEED)
    if ok:
        print(f"🎉 Hoàn thành! speed={MY_SPEED}x")
    else:
        print("❌ Tạo audio thất bại.")