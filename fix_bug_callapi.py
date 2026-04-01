"""
GET PCM từ server về lưu file local
"""

import requests

SERVER = "http://192.168.1.84:8000"  # ← đổi IP máy tính ở đây
OUTPUT = "downloaded.pcm"


def download_pcm(server: str = SERVER, output: str = OUTPUT) -> bool:
    try:
        print(f"📡 Đang tải PCM từ {server}/audio ...")
        response = requests.get(f"{server}/audio", timeout=10)

        if response.status_code != 200:
            print(f"❌ Lỗi HTTP {response.status_code}: {response.text}")
            return False

        with open(output, "wb") as f:
            f.write(response.content)

        size_kb = len(response.content) / 1024
        print(f"✅ Đã lưu: {output} ({size_kb:.1f} KB)")
        return True

    except requests.exceptions.ConnectionError:
        print(f"❌ Không kết nối được tới {server}")
        return False
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        return False


if __name__ == "__main__":
    download_pcm()