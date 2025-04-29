from app.services.vad import record_audio_vad
from app.services.whisper_stt import transcribe_audio
from app.services.gpt import generate_response
from app.services.audio_gen import generate_audio_response
import torchaudio
import logging
import sounddevice as sd


# Set up logging
logging.basicConfig(level=logging.DEBUG)
def play_audio(file_path):
    waveform, sample_rate = torchaudio.load(file_path)
    sd.play(waveform.squeeze(), samplerate=sample_rate)
    sd.wait()

def interact_with_voice_assistant():
    logging.debug("Entering interact_with_voice_assistant")

    audio_path = record_audio_vad()

    print("📥 Transcribing...")
    user_input = transcribe_audio(audio_path)

    if user_input.strip().lower() in ["exit", "quit", "bye"]:
        return "exit"

    print("🧠 Generating GPT reply...")
    reply_text = generate_response(user_input)

    print("🎧 Generating audio reply...")
    output_filename, audio_tensor = generate_audio_response(reply_text)

        # Save as 'response.wav'
    response_path = "output/response.wav"
    torchaudio.save(response_path, audio_tensor.unsqueeze(0), 24000)
    print(f"🔊 Audio saved to {response_path}")

    # Play it
    play_audio(response_path)

    torchaudio.save(output_filename, audio_tensor.unsqueeze(0), 24000)
    print(f"🔊 Audio saved to {output_filename}")

    logging.debug("Exiting interact_with_voice_assistant")
    return output_filename
