import os
import torch
import torchaudio
import re
from generator import load_csm_1b, Segment
from huggingface_hub import hf_hub_download

# Initialize the voice generator
device = "cuda" if torch.cuda.is_available() else "cpu"
generator = load_csm_1b(device)

# Download prompt from HF (on first run, caches locally)
prompt_audio_path = hf_hub_download("sesame/csm-1b", "prompts/conversational_b.wav")

# Prompt details
SPEAKER_PROMPTS = {
    "text": "like a super Mario level. Like it's very...",
    "audio_path": prompt_audio_path
}

# Split large text into manageable audio chunks
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

# Load the reference audio for speaker tone
def load_prompt(audio_path, text, speaker_id=1):
    audio_tensor, sample_rate = torchaudio.load(audio_path)
    audio_tensor = audio_tensor.squeeze(0)
    return Segment(text=text, speaker=speaker_id, audio=audio_tensor)

# Prepare prompt segment once
prompt_segment = load_prompt(SPEAKER_PROMPTS["audio_path"], SPEAKER_PROMPTS["text"])
context_segments = [prompt_segment]
generated_segments = []

# Main function to generate audio from text
def generate_audio_response(text: str, speaker: int = 1):
    chunks = split_text_by_sentences(text)
    audios = []

    for i, chunk in enumerate(chunks):
        print(f"🎛️ Generating audio chunk {i+1}/{len(chunks)}...")
        trimmed_context = (context_segments + generated_segments)[-6:]  # Max 6 context items
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

    # Save the audio to a file
    output_path = "output/response_audio.wav"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torchaudio.save(output_path, full_audio.unsqueeze(0), generator.sample_rate)

    return output_path
