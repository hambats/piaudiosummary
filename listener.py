import os
import queue
import sys
import sounddevice as sd
import numpy as np
import whisper
import tensorflow_hub as hub
import pandas as pd
import datetime
from scipy.io.wavfile import write
from langdetect import detect
from model_manager import ensure_models

# Initialize models
ensure_models()
whisper_model = whisper.load_model("small")
yamnet_model = hub.load(os.getenv("YAMNET_MODEL", 'https://tfhub.dev/google/yamnet/1'))
YAMNET_CLASSES = pd.read_csv(
    hub.resolve("https://storage.googleapis.com/audioset/yamnet/yamnet_class_map.csv")
)

q = queue.Queue()

# Capture 5-second audio segments
SEGMENT_DURATION = 5  # seconds
# Threshold for loud events
LOUD_THRESHOLD = 0.5

def audio_callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    q.put(indata.copy())

def classify_and_transcribe(audio_data, samplerate):
    """Classify audio and transcribe segments containing speech."""

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    wav_path = f"data/{timestamp}.wav"
    write(wav_path, samplerate, audio_data)

    # Detect loud noises
    if np.max(np.abs(audio_data)) > LOUD_THRESHOLD:
        top_class = "loud_noise"
    else:
        # Classify with YAMNet
        scores, _, _ = yamnet_model(audio_data)
        scores_np = scores.numpy()
        top_class = YAMNET_CLASSES['display_name'][scores_np.mean(axis=0).argmax()]

    # Only transcribe if speech is detected
    if "speech" not in top_class.lower():
        event = {"time": timestamp, "class": top_class, "lang": "", "transcript": ""}
        with open("data/log.csv", "a") as logf:
            logf.write(f"{event['time']},{event['class']},{event['lang']},{event['transcript']}\n")
        return

    result = whisper_model.transcribe(wav_path)
    transcript = result['text'].strip()
    if not transcript:
        return
    lang = detect(transcript)

    summary = {
        "time": timestamp,
        "class": top_class,
        "lang": lang,
        "transcript": transcript
    }

    print(summary)
    with open("data/log.csv", "a") as logf:
        logf.write(
            f"{summary['time']},{summary['class']},{summary['lang']},{summary['transcript']}\n"
        )

def main():
    os.makedirs("data", exist_ok=True)
    samplerate = 16000  # Hz
    blocksize = int(samplerate * SEGMENT_DURATION)

    print("[Listener] Starting...")
    with sd.InputStream(
        samplerate=samplerate, channels=1, blocksize=blocksize, callback=audio_callback
    ):
        while True:
            audio_chunk = q.get()
            audio_chunk = np.squeeze(audio_chunk)
            classify_and_transcribe(audio_chunk, samplerate)

if __name__ == "__main__":
    main()
