import logging
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="whisper")

from app.endpoints import interact_with_voice_assistant

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Voice Assistant API",
    description="API to interact with a real-time voice assistant using mic input.",
    version="1.0",
)

@app.post("/interact")
def interact():
    logger.debug("Received POST request at '/interact'")
    try:
        while True:
            logger.debug("Calling interact_with_voice_assistant()")
            result = interact_with_voice_assistant()
            if result == "exit":
                logger.debug("User said exit. Stopping interaction.")
                break
        return JSONResponse(content={"message": "User exited assistant"})
    except Exception as e:
        logger.error("Error during interaction: %s", str(e))
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    logger.debug("Starting Uvicorn server...")
    try:
        uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
    except Exception as e:
        logger.error("Error while starting Uvicorn: %s", str(e))
