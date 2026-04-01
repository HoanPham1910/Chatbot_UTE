"""
PCM → WAV converter (debug tool)
Chuyển raw PCM s16le stereo sang WAV để nghe thử.
Yêu cầu: ffmpeg trong PATH
"""

import os
import subprocess
import sys


PCM_RATE     = 44100   # Hz
PCM_CHANNELS = 2       # 2 = stereo
PCM_FORMAT   = "s16le" 



def pcm_to_wav(
    input_file:  str,
    output_file: str = None,
    rate:        int = PCM_RATE,
    channels:    int = PCM_CHANNELS,
    fmt:         str = PCM_FORMAT,
) -> bool:

    if not os.path.exists(input_file):
        print(f"❌ Không tìm thấy file: {input_file}")
        return False

    if output_file is None:
        output_file = os.path.splitext(input_file)[0] + ".wav"

    size_kb  = os.path.getsize(input_file) / 1024
    n_frames = os.path.getsize(input_file) // (2 * channels)
    duration = n_frames / rate

    print(f"📂 Input  : {input_file} ({size_kb:.1f} KB)")
    print(f"⚙️  Thông số: {fmt} | {rate}Hz | {channels}ch | {duration:.2f}s")
    print(f"📁 Output : {output_file}")

    result = subprocess.run(
        [
            "ffmpeg", "-y",
            "-f",       fmt,           # format PCM đầu vào
            "-ar",      str(rate),     # sample rate
            "-ac",      str(channels), # số kênh
            "-i",       input_file,    # file PCM
            output_file,               # file WAV đầu ra
        ],
        capture_output=True, text=True
    )

    if result.returncode != 0:
        print(f"❌ ffmpeg lỗi:\n{result.stderr}")
        return False

    out_kb = os.path.getsize(output_file) / 1024
    print(f"✅ Hoàn thành! {output_file} ({out_kb:.1f} KB)")
    return True


# ── Main ──────────────────────────────────────────────────

if __name__ == "__main__":
    # Dùng argument nếu có: python pcm_to_wav.py answer.pcm
    if len(sys.argv) >= 2:
        INPUT  = sys.argv[1]
        OUTPUT = sys.argv[2] if len(sys.argv) >= 3 else None
    else:
        INPUT  = "downloaded.pcm"   # ← đổi tên file PCM ở đây
        OUTPUT = None           # None = tự tạo tên: answer.wav

    ok = pcm_to_wav(INPUT, OUTPUT)
    if not ok:
        sys.exit(1)