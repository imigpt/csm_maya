# app/services/whisper_stt.py
import torch
import whisper
# Load model on GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_model = whisper.load_model("base").to(device)
# Load once at the top level
whisper_model = whisper.load_model("base")

def transcribe_audio(audio_path: str) -> str:
    print(f"📝 Transcribing audio: {audio_path}")
    result = whisper_model.transcribe(audio_path, task="transcribe", language="hi")
    text = result["text"].strip()
    print(f"🗣️ Transcribed Text: {text}")
    return text
