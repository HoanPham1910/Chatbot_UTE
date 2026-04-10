import requests
import time
import sys
import wave
import io
import numpy as np

try:
    import sounddevice as sd
    HAS_AUDIO = True
except ImportError:
    HAS_AUDIO = False


SERVER_IP   = "192.168.1.34"
SERVER_PORT = 8001
BASE_URL    = f"http://{SERVER_IP}:{SERVER_PORT}"
RETRY_INTERVAL = 0.3

# Chỉ định thẳng device output nếu auto-detect sai
# Đặt None để dùng default, hoặc đặt số device (vd: 3)
OUTPUT_DEVICE = None


def play_wav_bytes(wav_bytes: bytes, chunk_idx: int, chunk_total: int):
    print(f"   [DBG] recv {len(wav_bytes)} bytes")

    if not HAS_AUDIO:
        fname = f"chunk_{chunk_idx:02d}_of_{chunk_total}.wav"
        with open(fname, "wb") as f:
            f.write(wav_bytes)
        print(f"   saved: {fname} ({len(wav_bytes)/1024:.1f} KB)")
        return

    if len(wav_bytes) == 0:
        print(f"   [WARN] chunk {chunk_idx} rỗng, bỏ qua")
        return

    try:
        buf = io.BytesIO(wav_bytes)
        with wave.open(buf, 'rb') as wf:
            sr       = wf.getframerate()
            channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            nframes  = wf.getnframes()
            frames   = wf.readframes(nframes)
            print(f"   [DBG] WAV: sr={sr} ch={channels} sw={sampwidth} frames={nframes}")

        dtype = np.int16 if sampwidth == 2 else np.int32
        audio_np = np.frombuffer(frames, dtype=dtype).astype(np.float32)
        audio_np /= float(np.iinfo(dtype).max)

        if channels > 1:
            audio_np = audio_np.reshape(-1, channels)

        dur = (len(audio_np) if audio_np.ndim == 1 else audio_np.shape[0]) / sr
        print(f"   playing chunk {chunk_idx}/{chunk_total} | {dur:.1f}s | {sr}Hz | device={OUTPUT_DEVICE}")

        sd.play(audio_np, samplerate=sr, device=OUTPUT_DEVICE, blocking=True)
        print(f"   done chunk {chunk_idx}")

    except Exception as e:
        import traceback
        print(f"   play error: {e}")
        traceback.print_exc()

        # Lưu lại để debug thủ công
        fname = f"chunk_{chunk_idx:02d}_debug.wav"
        with open(fname, "wb") as f:
            f.write(wav_bytes)
        print(f"   saved for debug: {fname}")


def step1_ask(question: str) -> dict:
    print(f"\n{'='*55}")
    print(f"question: {question}")
    print(f"{'='*55}")

    t0   = time.time()
    resp = requests.post(
        f"{BASE_URL}/ask",
        json={"question": question, "tts": True},
        timeout=120,
    )
    elapsed = time.time() - t0

    if resp.status_code != 200:
        print(f"/ask error {resp.status_code}: {resp.text}")
        sys.exit(1)

    data = resp.json()
    print(f"/ask done {elapsed:.2f}s")
    print(f"answer      : {data.get('answer', '')}")
    print(f"total_chunks: {data.get('total_chunks', '?')}")
    return data


def step2_fetch_chunks(total_chunks: int):
    print(f"\n[2] fetching {total_chunks} chunks...")
    print(f"{'─'*55}")

    t_start = time.time()

    for i in range(1, total_chunks + 1):
        url = f"{BASE_URL}/audio/chunk/{i}"

        while True:
            try:
                resp = requests.get(url, timeout=35)
            except Exception as e:
                print(f"request error chunk {i}: {e}")
                time.sleep(RETRY_INTERVAL)
                continue

            if resp.status_code == 200:
                wav_bytes = resp.content
                print(f"\nchunk {i}/{total_chunks} | {len(wav_bytes)/1024:.1f} KB")
                play_wav_bytes(wav_bytes, i, total_chunks)
                break

            elif resp.status_code == 204:
                sys.stdout.write(f"\rwaiting chunk {i}/{total_chunks}...")
                sys.stdout.flush()
                time.sleep(RETRY_INTERVAL)
                continue

            else:
                print(f"HTTP {resp.status_code} chunk {i}: {resp.text[:100]}")
                break

    print(f"\ndone | total time: {time.time()-t_start:.2f}s")


if __name__ == "__main__":
    print(f"\nESP32 Simulator - Chunk Index Mode")
    print(f"   server      : {BASE_URL}")
    print(f"   audio       : {'sounddevice ok' if HAS_AUDIO else 'NO sounddevice - save to file'}")
    if HAS_AUDIO:
        try:
            info = sd.query_devices(OUTPUT_DEVICE, 'output')
            print(f"   output dev  : [{OUTPUT_DEVICE}] {info['name']}")
        except Exception as e:
            print(f"   output dev  : error - {e}")
    print(f"   type 'quit' to exit\n")

    while True:
        try:
            question = input("question: ").strip()
        except KeyboardInterrupt:
            print("\nexiting.")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q", "thoat"):
            print("exiting.")
            break

        data         = step1_ask(question)
        wav_url      = data.get("wav_url")
        total_chunks = data.get("total_chunks", 0)

        if not wav_url or total_chunks == 0:
            print("no audio in response")
            continue

        step2_fetch_chunks(total_chunks)
        print()