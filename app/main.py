# app/main.py

import logging
import os
import warnings
import time
from fastapi import FastAPI, WebSocket
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import torchaudio
import tempfile

# Ignore future warnings from whisper
warnings.filterwarnings("ignore", category=FutureWarning, module="whisper")

# Import services
from app.services.whisper_stt import transcribe_audio
from app.services.gpt import generate_response
from app.services.audio_gen import generate_audio_response

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Voice Assistant API",
    description="API to interact with a real-time voice assistant using frontend mic input.",
    version="1.0",
)

# Serve static files from "output/" directory
os.makedirs("output", exist_ok=True)
app.mount("/audio", StaticFiles(directory="output"), name="audio")

# API route: POST endpoint (optional, if you want)
@app.post("/interact")
def interact():
    return JSONResponse(content={"message": "Use WebSocket connection at /ws/assistant"})

# API route: WebSocket endpoint
@app.websocket("/ws/assistant")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_text("🟢 Assistant started! Send audio.")

    try:
        while True:
            # 1. Receive audio binary data
            audio_bytes = await websocket.receive_bytes()

            # 2. Save it to a temp WAV file
            timestamp = int(time.time())
            temp_wav_path = f"output/user_audio_{timestamp}.wav"
            with open(temp_wav_path, "wb") as f:
                f.write(audio_bytes)

            logger.debug(f"🎙️ Received and saved audio at {temp_wav_path}")

            # 3. Transcribe the audio to text
            user_input = transcribe_audio(temp_wav_path)

            if not user_input.strip():
                await websocket.send_text("❌ Could not transcribe anything. Please try again.")
                continue

            if user_input.strip().lower() in ["exit", "quit", "bye"]:
                await websocket.send_text("👋 Goodbye!")
                break

            await websocket.send_text(f"👤 You said: {user_input}")

            # 4. Generate ChatGPT reply
            reply_text = generate_response(user_input)

            await websocket.send_text(f"🤖 Assistant says: {reply_text}")

            # 5. Generate audio from reply
            output_audio_path = generate_audio_response(reply_text)

            # 6. Rename output audio with timestamp
            final_audio_path = f"output/response_{timestamp}.wav"
            os.rename(output_audio_path, final_audio_path)

            # 7. Send JSON with text + audio URL
            await websocket.send_json({
                "text": reply_text,
                "audio_url": f"/audio/response_{timestamp}.wav"
            })

    except Exception as e:
        logger.error(f"Error during WebSocket session: {str(e)}")
        await websocket.send_text(f"❌ Server error: {str(e)}")

    finally:
        await websocket.close()

if __name__ == "__main__":
    logger.debug("Starting Uvicorn server...")
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000)
