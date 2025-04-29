FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Install system packages
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip git ffmpeg \
    libportaudio2 libsndfile1 libasound2

# Set Python alias
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Install Python deps
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Install Whisper (since it's not in requirements.txt)
RUN pip install git+https://github.com/openai/whisper.git

# Expose port (not used here but in case needed later)
EXPOSE 7860

# Run your assistant
CMD ["python", "run_csm.py"]
