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

# Load and prepare speaker prompts
def load_prompt(audio_path, text, speaker_id=0):
    audio_tensor, sample_rate = torchaudio.load(audio_path)
    
    # Resample if needed to 24000 Hz
    target_sample_rate = 24000
    if sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        audio_tensor = resampler(audio_tensor)
    
    # Make sure tensor is 1D
    audio_tensor = audio_tensor.squeeze(0)
    
    return Segment(text=text, speaker=speaker_id, audio=audio_tensor)

# Download speaker prompts
prompt_filepath_conversational_a = hf_hub_download("sesame/csm-1b", "prompts/conversational_a.wav")
prompt_filepath_conversational_b = hf_hub_download("sesame/csm-1b", "prompts/conversational_b.wav")

# Create speaker prompt segments
SPEAKER_PROMPTS = {
    "conversational_a": {
        "text": (
            "like revising for an exam I'd have to try and like keep up the momentum because I'd "
            "start really early I'd be like okay I'm gonna start revising now and then like "
            "you're revising for ages and then I just like start losing steam I didn't do that "
            "for the exam we had recently to be fair that was a more of a last minute scenario "
            "but like yeah I'm trying to like yeah I noticed this yesterday that like Mondays I "
            "sort of start the day with this not like a panic but like a"
        ),
        "audio": prompt_filepath_conversational_a
    },
    "conversational_b": {
        "text": (
            "like a super Mario level. Like it's very like high detail. And like, once you get "
            "into the park, it just like, everything looks like a computer game and they have all "
            "these, like, you know, if, if there's like a, you know, like in a Mario game, they "
            "will have like a question block. And if you like, you know, punch it, a coin will "
            "come out. So like everyone, when they come into the park, they get like this little "
            "bracelet and then you can go punching question blocks around."
        ),
        "audio": prompt_filepath_conversational_b
    }
}

prompt_a = load_prompt(SPEAKER_PROMPTS["conversational_a"]["audio"], SPEAKER_PROMPTS["conversational_a"]["text"], speaker_id=0)
prompt_b = load_prompt(SPEAKER_PROMPTS["conversational_b"]["audio"], SPEAKER_PROMPTS["conversational_b"]["text"], speaker_id=1)

# Function to split text nicely
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

# Main function to generate audio
def generate_audio_response(text: str, speaker: int = 0):
    chunks = split_text_by_sentences(text)
    audios = []
    generated_segments = []  # reset per request
    
    # Select prompt according to speaker
    if speaker == 0:
        prompt_segments = [prompt_a]
    else:
        prompt_segments = [prompt_b]

    for idx, chunk in enumerate(chunks):
        print(f"🎛️ Generating audio chunk {idx+1}/{len(chunks)}: {chunk}")
        
        trimmed_context = (prompt_segments + generated_segments)[-6:]
        
        audio_chunk = generator.generate(
            text=chunk,
            speaker=speaker,
            context=trimmed_context,
            max_audio_length_ms=60000
        )
        audios.append(audio_chunk.cpu())
        generated_segments.append(Segment(text=chunk, speaker=speaker, audio=audio_chunk))

    # Combine all chunks
    full_audio = torch.cat(audios, dim=-1)

    # Normalize
    full_audio = full_audio / full_audio.abs().max()

    # Save
    os.makedirs("output", exist_ok=True)
    output_path = "output/response.wav"
    torchaudio.save(output_path, full_audio.unsqueeze(0), generator.sample_rate)

    print(f"🔊 Audio saved at {output_path}")

    return output_path
