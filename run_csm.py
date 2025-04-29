# 🛠️ Here's the updated script with dynamic long reply audio generation

import os
import torch
import torchaudio
import bitsandbytes
import moshi
import silentcipher
import whisper
import torchao
import sounddevice as sd
import numpy as np
import tempfile
import scipy.io.wavfile
import soundfile as sf
import webrtcvad
import collections
import time
import speech_recognition as sr
from openai import OpenAI
from dotenv import load_dotenv

from huggingface_hub import hf_hub_download
from generator import load_csm_1b, Segment
import re

# Load environment variables
load_dotenv()
client = OpenAI()

whisper_model = whisper.load_model("base")
os.environ["NO_TORCH_COMPILE"] = "1"

# Load speaker prompts
prompt_filepath_conversational_a = hf_hub_download("sesame/csm-1b", "prompts/conversational_a.wav")
prompt_filepath_conversational_b = hf_hub_download("sesame/csm-1b", "prompts/conversational_b.wav")

SPEAKER_PROMPTS = {
    "conversational_a": {
        "text": "like revising for an exam I'd have to try and...",
        "audio": prompt_filepath_conversational_a
    },
    "conversational_b": {
        "text": "like a super Mario level. Like it's very...",
        "audio": prompt_filepath_conversational_b
    }
}

def split_text_by_sentences(text, max_chars=400):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current = ""
    for s in sentences:
        if len(current) + len(s) < max_chars:
            current += " " + s
        else:
            chunks.append(current.strip())
            current = s
    if current:
        chunks.append(current.strip())
    return chunks

def generate_full_audio(text, generator, speaker, context_segments, sample_rate):
    chunks = split_text_by_sentences(text, max_chars=400)
    audios = []
    for i, chunk in enumerate(chunks):
        print(f"🔊 Generating chunk {i+1}/{len(chunks)}: {chunk}")
        trimmed_context = context_segments[-6:]  # Reduce context to stay safe
        audio_chunk = generator.generate(
            text=chunk,
            speaker=speaker,
            context=trimmed_context,
            max_audio_length_ms=10000
        )
        audios.append(audio_chunk.cpu())
        context_segments.append(Segment(text=chunk, speaker=speaker, audio=audio_chunk))
    return torch.cat(audios, dim=-1)

def generate_response(prompt):
    try:
        print("📝 Prompt sent to ChatGPT:", prompt)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful, friendly voice assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
        )
        reply = response.choices[0].message.content.strip()
        print("💬 ChatGPT Reply:", reply)
        return reply
    except Exception as e:
        print("❌ ChatGPT Error:", e)
        return "Sorry, I had trouble generating a response."

def load_prompt_audio(audio_path: str, target_sample_rate: int) -> torch.Tensor:
    audio_tensor, sample_rate = torchaudio.load(audio_path)
    audio_tensor = audio_tensor.squeeze(0)
    return torchaudio.functional.resample(audio_tensor, orig_freq=sample_rate, new_freq=target_sample_rate)

def prepare_prompt(text: str, speaker: int, audio_path: str, sample_rate: int) -> Segment:
    audio_tensor = load_prompt_audio(audio_path, sample_rate)
    return Segment(text=text, speaker=speaker, audio=audio_tensor)

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
        return f.name

def play_audio(filename):
    data, samplerate = sf.read(filename)
    sd.play(data, samplerate)
    sd.wait()

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    generator = load_csm_1b(device)

    prompt_a = prepare_prompt(SPEAKER_PROMPTS["conversational_a"]["text"], 0, SPEAKER_PROMPTS["conversational_a"]["audio"], generator.sample_rate)
    prompt_b = prepare_prompt(SPEAKER_PROMPTS["conversational_b"]["text"], 1, SPEAKER_PROMPTS["conversational_b"]["audio"], generator.sample_rate)
    prompt_segments = [prompt_a]
    generated_segments = []

    index = 0
    print("\n🎤 Speak to the AI! Say 'exit' to stop.\n")

    while True:
        audio_path = record_audio_vad()
        result = whisper_model.transcribe(audio_path, task="transcribe", language="hi")
        user_input = result["text"]
        print(f"🗣️ You said: {user_input}")

        if user_input.strip().lower() in ["exit", "quit", "bye"]:
            print("👋 Conversation ended.")
            break

        print("🤖 Sending to ChatGPT for smart reply...")
        reply_text = generate_response(user_input)

        if reply_text.strip().lower() == user_input.strip().lower():
            print("⚠️ Warning: ChatGPT returned the same text as input. Possibly failed.")
        else:
            print("✅ Smart Reply:", reply_text)

        print("🎧 Generating AI voice...")
        audio_tensor = generate_full_audio(
            text=reply_text,
            generator=generator,
            speaker=1,
            context_segments=prompt_segments + generated_segments,
            sample_rate=generator.sample_rate,
        )

        filename = f"response_{index}.wav"
        torchaudio.save(filename, audio_tensor.unsqueeze(0), generator.sample_rate)
        print(f"🔊 Voice response saved as {filename}")

        play_audio(filename)

        generated_segments.append(Segment(text=reply_text, speaker=1, audio=audio_tensor))
        index += 1

if __name__ == "__main__":
    main()