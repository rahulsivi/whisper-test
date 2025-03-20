import os
import logging
import time
import gc
from fastapi import FastAPI, UploadFile, File, HTTPException
import whisperx
import tempfile

# # Set CUDA environment
# os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda-12.4/lib:" + os.environ.get("LD_LIBRARY_PATH", "")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("whisperx-api")

app = FastAPI()

device = "cuda"
batch_size = 16  # Reduce if low on GPU memory
compute_type = "int8"  # Change to "int8" if low on GPU memory

# Load WhisperX model
model = whisperx.load_model("medium", device, compute_type=compute_type)

def transcribe_audio(file_path: str):
    audio = whisperx.load_audio(file_path)
    result = model.transcribe(audio, batch_size=batch_size)
    
    # Step 2: Align transcription
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    aligned_result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
    
    # Cleanup alignment model
    del model_a
    gc.collect()
    
    return aligned_result

@app.post("/transcribe/")
async def transcribe(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
            temp_file.write(await file.read())
            temp_file_path = temp_file.name

        start = time.time()
        result = transcribe_audio(temp_file_path)
        end = time.time()
        os.remove(temp_file_path)

        return {
            "transcription": result["segments"],
            "processing_time": end - start,
        }
    except Exception as e:
        logger.error(f"Error during transcription: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
