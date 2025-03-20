import os
import logging
import time
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from whispercpp import Whisper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("whisper-api")

app = FastAPI()

# Initialize the Whisper model
model = Whisper.from_pretrained("medium")

@app.post("/transcribe/")
async def transcribe(file: UploadFile = File(...)):
    try:
        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file.write(await file.read())
            temp_file_path = temp_file.name

        start_time = time.time()

        # Transcribe the audio file
        result = model.transcribe(temp_file_path)

        end_time = time.time()
        duration = end_time - start_time

        # Remove the temporary file
        os.remove(temp_file_path)

        return {
            "transcription": result["text"],
            "processing_time": duration,
        }
    except Exception as e:
        logger.error(f"Error during transcription: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
