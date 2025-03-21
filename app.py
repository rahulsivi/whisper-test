import os
import logging
import time
from fastapi import FastAPI, UploadFile, File, HTTPException
from whisperplus import SpeechToTextPipeline
from transformers import BitsAndBytesConfig, HqqConfig
import torch
import tempfile

logging.basicConfig()
logging.getLogger("whisperplus").setLevel(logging.DEBUG)

app = FastAPI()

hqq_config = HqqConfig(
    nbits=4,
    group_size=64,
    quant_zero=False,
    quant_scale=False,
    axis=0,
    offload_meta=False,
) 
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="fp4",  # Faster than nf4
    bnb_4bit_compute_dtype=torch.float16,  # Faster than bfloat16 on some GPUs
    bnb_4bit_use_double_quant=False,  # Disabling for speed
)


pipeline = SpeechToTextPipeline(
    model_id="Systran/faster-whisper-medium",
    quant_config=bnb_config,
    flash_attention_2=True,
)
pipeline.model.generation_config.task = "transcribe"
pipeline.model.generation_config.forced_decoder_ids = None

@app.post("/transcribe/")
async def transcribe(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(await file.read())
            temp_file_path = temp_file.name

        start = time.time()
        transcript = pipeline(
            audio_path=temp_file_path,
            chunk_length_s=30,  # Smaller chunks for faster processing
            stride_length_s=2,  # Higher stride to reduce redundant computation
            max_new_tokens=100,  # Lower max tokens to speed up decoding
            batch_size=64,  # Reduce batch size for better efficiency
            language="english",
            return_timestamps=True,
        )

        end = time.time()
        duration = end - start

        os.remove(temp_file_path)

        return {
            "transcription": transcript,
            "processing_time": duration,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
