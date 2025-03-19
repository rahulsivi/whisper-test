import os
import logging
from time import time
from locust import HttpUser, task, between

# Configure logging
logging.basicConfig(
    filename="locust_metrics.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
)

AUDIO_FOLDER = "/Users/rahul/Desktop/Audio"

class InferenceTaskSet(HttpUser):
    host = "http://ec2-3-109-158-143.ap-south-1.compute.amazonaws.com"  
    wait_time = between(1, 2)  

    @task
    def post_audio_files(self):
        # Fetch only .mp3 files from the folder
        audio_files = [f for f in os.listdir(AUDIO_FOLDER) if f.endswith(".wav")]

        if not audio_files:
            logging.error("No MP3 audio files found in the folder: %s", AUDIO_FOLDER)
            return

        for file_name in audio_files:
            file_path = os.path.join(AUDIO_FOLDER, file_name)
            try:
                with open(file_path, "rb") as audio:
                    files = {"file": (file_name, audio, "audio/wav")}
                    logging.info(f"Sending request with file: {file_name}")

                    start = time()
                    response = self.client.post("/inference", files=files)
                    elapsed_time = time() - start

                    if response.status_code == 200:
                        logging.info(f"Sent file: {file_name} | Response: {response.status_code}")
                        logging.info(f"Response body: {response.text}")
                        logging.info(f"Time taken: {elapsed_time:.2f}s")
                    else:
                        logging.error(f"Error sending {file_name}: {response.text}")

            except FileNotFoundError:
                logging.error(f"File not found: {file_path}")
            except Exception as e:
                logging.error(f"Error sending file {file_path}: {e}")
