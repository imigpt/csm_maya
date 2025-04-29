from app.services.vad import record_audio_vad
from app.services.whisper_stt import transcribe_audio
from app.services.gpt import generate_response
from app.services.audio_gen import generate_audio_response
import torchaudio
import logging
import sounddevice as sd
import os

# Set up logging
logging.basicConfig(level=logging.DEBUG)

def play_audio(file_path):
    waveform, sample_rate = torchaudio.load(file_path)
    sd.play(waveform.squeeze(), samplerate=sample_rate)
    sd.wait()

def interact_with_voice_assistant():
    logging.debug("Starting voice interaction...")

    audio_path = record_audio_vad()
    print("📥 Transcribing...")
    user_input = transcribe_audio(audio_path)

    if user_input.strip().lower() in ["exit", "quit", "bye"]:
        print("👋 Exiting session.")
        return "exit"

    print("🧠 Generating GPT reply...")
    reply_text = generate_response(user_input)

    print("🎧 Generating audio reply...")
    response_path = generate_audio_response(reply_text)

    play_audio(response_path)

    logging.debug("Voice interaction complete.")
    return response_path
