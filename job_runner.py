# job_runner.py
import os
import json
import base64
import logging
from app import _run_video_generation
import asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# This is the main function that runs when the Cloud Run job is triggered
async def run_job():
    # Pub/Sub sends the message data in an environment variable
    try:
        # The Pub/Sub message body is base64 encoded
        message_data_b64 = os.environ.get("K_DATA")
        if not message_data_b64:
            logger.error("No Pub/Sub message data found in K_DATA environment variable.")
            return

        message_data_str = base64.b64decode(message_data_b64).decode("utf-8")
        message_data = json.loads(message_data_str)
        
        # The actual task data is nested under 'data' in the Pub/Sub message
        task_data_b64 = message_data.get('data')
        if not task_data_b64:
             logger.error("Missing 'data' field in Pub/Sub message.")
             return

        task_data_str = base64.b64decode(task_data_b64).decode("utf-8")
        task_data = json.loads(task_data_str)

        task_id = task_data["task_id"]
        image_url = task_data["image_url"]
        audio_url = task_data["audio_url"]
        output_dir = task_data["output_dir"]
        quality = task_data["quality"]
        
        logger.info(f"Cloud Run Job starting for task: {task_id}")
        
        # Call your existing function
        await _run_video_generation(task_id, image_url, audio_url, output_dir, quality)
        
        logger.info(f"Cloud Run Job finished for task: {task_id}")

    except Exception as e:
        logger.error(f"Error processing Cloud Run job: {e}")
        # Return a non-zero exit code to signal failure
        exit(1)

# Run the job
if __name__ == "__main__":
    asyncio.run(run_job())