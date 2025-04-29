# app/services/audio_gen.py

import os
import torch
import torchaudio
import re
from generator import load_csm_1b, Segment
from huggingface_hub import hf_hub_download

# Initialize generator
device = "cuda" if torch.cuda.is_available() else "cpu"
generator = load_csm_1b(device)

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

# Prepare prompt segments
def load_prompt(audio_path, text, speaker_id=0):
    audio_tensor, sample_rate = torchaudio.load(audio_path)
    audio_tensor = audio_tensor.squeeze(0)
    return Segment(text=text, speaker=speaker_id, audio=audio_tensor)

prompt_a = load_prompt(SPEAKER_PROMPTS["conversational_a"]["audio"], SPEAKER_PROMPTS["conversational_a"]["text"], 0)
prompt_b = load_prompt(SPEAKER_PROMPTS["conversational_b"]["audio"], SPEAKER_PROMPTS["conversational_b"]["text"], 1)

prompt_segments = [prompt_a, prompt_b]
generated_segments = []

# Split text by sentences
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

# Generate audio response
def generate_audio_response(text: str, speaker: int = 1):
    chunks = split_text_by_sentences(text)
    audios = []

    for idx, chunk in enumerate(chunks):
        print(f"🎛️ Generating audio chunk {idx+1}/{len(chunks)}: {chunk}")
        trimmed_context = (prompt_segments + generated_segments)[-6:]
        audio_chunk = generator.generate(
            text=chunk,
            speaker=speaker,
            context=trimmed_context,
            max_audio_length_ms=10000
        )
        audios.append(audio_chunk.cpu())
        generated_segments.append(Segment(text=chunk, speaker=speaker, audio=audio_chunk))

    # Combine all audio chunks
    full_audio = torch.cat(audios, dim=-1)

    # Normalize audio
    full_audio = full_audio / full_audio.abs().max()

    # Save audio
    os.makedirs("output", exist_ok=True)
    output_path = "output/response.wav"
    torchaudio.save(output_path, full_audio.unsqueeze(0), generator.sample_rate)

    print(f"🔊 Audio saved at {output_path}")

    return output_path
