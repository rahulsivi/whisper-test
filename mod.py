import os
import logging
import time
from fastapi import FastAPI, UploadFile, File, HTTPException
from faster_whisper import WhisperModel
import tempfile
import os
import nvidia.cublas.lib
import nvidia.cudnn.lib

os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda-12.4/lib:"+os.environ.get("LD_LIBRARY_PATH", "")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("whisper-api")

app = FastAPI()

# Initialize the model with optimized settings for T4 GPU
model = WhisperModel(
    "medium",
    device="cuda",
    compute_type="float16",  # Use float16 for T4 tensor cores
    download_root="./models" # Cache models locally
)

@app.post("/transcribe/")
async def transcribe(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file.write(await file.read())
            temp_file_path = temp_file.name

        start = time.time()
        
        # Optimized settings for faster inference
        segments, info = model.transcribe(
            temp_file_path,
            language="en",           # Explicitly set language to avoid detection overhead
            beam_size=1,             # Reduce beam search complexity
            vad_filter=True,         # Filter out non-speech
            vad_parameters=dict(min_silence_duration_ms=500),  # Optimize VAD
            initial_prompt=None,     # No conditioning on previous text
            word_timestamps=False,   # Disable word timestamps for speed
            condition_on_previous_text=False  # Disable for faster inference
        )
        
        # Process results
        result = []
        for segment in segments:
            result.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip()
            })

        end = time.time()
        duration = end - start

        os.remove(temp_file_path)

        return {
            "transcription": result,
            "processing_time": duration,
        }
    except Exception as e:
        logger.error(f"Error during transcription: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
