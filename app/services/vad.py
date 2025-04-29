# app/services/vad.py

import webrtcvad
import collections
import sounddevice as sd
import numpy as np
import time
import tempfile
import soundfile as sf

def record_audio_vad(sample_rate=16000, aggressiveness=2, max_record_duration=30):
    vad = webrtcvad.Vad(aggressiveness)
    duration_ms = 30
    num_padding_frames = int(300 / duration_ms)
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    triggered = False
    voiced_frames = []
    silence_counter = 0
    silence_threshold = 20

    print("🎙️ Listening... Speak now.")

    with sd.InputStream(samplerate=sample_rate, channels=1, dtype='int16',
                        blocksize=int(sample_rate * duration_ms / 1000)) as stream:
        start_time = time.time()
        while True:
            if time.time() - start_time > max_record_duration:
                print("⏱️ Max recording time reached.")
                break

            audio_block, _ = stream.read(int(sample_rate * duration_ms / 1000))
            frame = audio_block[:, 0].tobytes()

            if vad.is_speech(frame, sample_rate):
                if not triggered:
                    triggered = True
                    print("🔊 Voice detected, recording...")
                ring_buffer.clear()
                voiced_frames.append(frame)
                silence_counter = 0
            elif triggered:
                ring_buffer.append(frame)
                silence_counter += 1
                if silence_counter > silence_threshold:
                    print("🤫 Silence detected, stopping recording.")
                    break

    audio_data = b"".join(voiced_frames)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, np.frombuffer(audio_data, dtype=np.int16).reshape(-1, 1), sample_rate)
        print(f"💾 Audio saved: {f.name}")
        return f.name
