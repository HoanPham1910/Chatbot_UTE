from vieneu import Vieneu

tts = Vieneu()

# Clone từ file audio (wav/mp3/flac)
my_voice = tts.clone_voice(
    audio_path="giong_mau.wav",   # file 5-10s giọng bạn muốn
    text="nội dung trong file audio đó",  # transcript chính xác
    name="MyVoice"                # đặt tên để tái sử dụng
)

# Dùng ngay
audio = tts.infer(text="Xin chào!", voice=my_voice)
tts.save(audio, "output.wav")